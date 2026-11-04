"""Grouped damped-oscillator SSM with certified selective input writes (H001, SPEC §1–4).

Real coordinates throughout: state [batch, G, m, 2].

Execution modes of `forward`:
  dense       all writes; parallel scan (training / batched evaluation); bounds None.
  masked_ref  streaming loop; FULL projection computed then multiplied by the mask. This is
              only a correctness/quality reference - it never omits work.
  selective   streaming loop; genuinely computes W_i u_t only for written groups (per-group
              matrix-vector calls); omitted projections are never issued.
Policies (how the mask is chosen in masked_ref / selective):
  certified   SPEC §4 budget policy (bounds reported).
  oracle      certified ledger but beta replaced by the exact ||b_i|| (requires computing every
              write: diagnostic only, never an efficiency result).
  threshold   heuristic: omit when c_i*beta_i < tau (no ledger, bounds None).
  delta       Delta-network-style baseline: cached projection reused unless ||u_t - u_cache|| > tau.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .bounds import BoundCache, output_bound, update_error, write_bound
from .policy import choose_omissions


@dataclass
class ForwardResult:
    predictions: Tensor  # [batch, time, output_dim]
    bounds: Tensor | None  # [batch, time]
    evaluated_writes: Tensor  # [batch, time] number of group projections evaluated
    numerical_status: str


def complex_scan(ar: Tensor, ai: Tensor, br: Tensor, bi: Tensor) -> tuple[Tensor, Tensor]:
    """Inclusive scan of h_t = a_t h_{t-1} + b_t along dim 1 (Hillis–Steele, log2 T steps).
    All tensors [B, T, ...]; |a| <= 1 keeps every partial product bounded."""
    T = ar.shape[1]
    k = 1
    while k < T:
        par, pai = ar[:, k:], ai[:, k:]
        qar, qai = ar[:, :-k], ai[:, :-k]
        qbr, qbi = br[:, :-k], bi[:, :-k]
        nbr = par * qbr - pai * qbi + br[:, k:]
        nbi = par * qbi + pai * qbr + bi[:, k:]
        nar = par * qar - pai * qai
        nai = par * qai + pai * qar
        br = torch.cat([br[:, :k], nbr], 1)
        bi = torch.cat([bi[:, :k], nbi], 1)
        ar = torch.cat([ar[:, :k], nar], 1)
        ai = torch.cat([ai[:, :k], nai], 1)
        k *= 2
    return br, bi


class OscillatorSSM(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_model: int = 64, d_hidden: int = 64, groups: int = 8,
                 modes: int = 8, zero_frequency: bool = False, omega_max: float = 3.0,
                 gamma_init: tuple[float, float] = (0.005, 0.5), seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.G, self.m, self.d, self.p = groups, modes, d_model, d_out
        self.zero_frequency = zero_frequency
        self.register_buffer("x_mean", torch.zeros(d_in))
        self.register_buffer("x_std", torch.ones(d_in))
        self.enc1 = nn.Linear(d_in, d_hidden)
        self.enc2 = nn.Linear(d_hidden, d_model)
        self.gate = nn.Linear(d_model, groups)
        self.W = nn.Parameter(torch.randn(groups, 2 * modes, d_model, generator=gen) / d_model ** 0.5 * 0.5)
        self.C = nn.Parameter(torch.randn(groups, d_out, 2 * modes, generator=gen) / (groups * modes) ** 0.5)
        self.D = nn.Parameter(torch.zeros(d_out, d_model))
        self.out_bias = nn.Parameter(torch.zeros(d_out))
        lo, hi = gamma_init
        g0 = torch.exp(torch.empty(groups, modes).uniform_(torch.log(torch.tensor(lo)).item(), torch.log(torch.tensor(hi)).item(), generator=gen))
        self.gamma_raw = nn.Parameter(torch.log(torch.expm1(g0)))  # softplus^-1
        if zero_frequency:
            self.register_buffer("omega", torch.zeros(groups, modes))
        else:
            self.omega = nn.Parameter(torch.empty(groups, modes).uniform_(-omega_max, omega_max, generator=gen))
        self._cache: BoundCache | None = None

    # ---- components -----------------------------------------------------------------------
    @property
    def gamma(self) -> Tensor:
        return F.softplus(self.gamma_raw)

    def set_normalization(self, mean: Tensor, std: Tensor) -> None:
        self.x_mean.copy_(mean)
        self.x_std.copy_(std.clamp_min(1e-6))

    def encode(self, x: Tensor) -> Tensor:
        z = (x - self.x_mean) / self.x_std
        return self.enc2(F.gelu(self.enc1(z), approximate="tanh"))

    def gates(self, u: Tensor) -> Tensor:
        return torch.sigmoid(self.gate(u))

    def coeffs(self, dt: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """dt [...]-> ar, ai [..., G, m], rho [..., G]."""
        if (dt < 0).any():
            raise ValueError("dt must be nonnegative")
        g = self.gamma
        mag = torch.exp(-g * dt[..., None, None])
        ang = self.omega * dt[..., None, None]
        return mag * torch.cos(ang), mag * torch.sin(ang), torch.exp(-g.amin(-1) * dt[..., None])

    def readout(self, hr: Tensor, hi: Tensor, u: Tensor) -> Tensor:
        h = torch.stack([hr, hi], -1).flatten(-2)  # [..., G, 2m]
        return torch.einsum("...gk,gpk->...p", h, self.C) + u @ self.D.T + self.out_bias

    def freeze_bounds(self) -> BoundCache:
        with torch.no_grad():
            self._cache = BoundCache(self.W.detach(), self.C.detach())
        return self._cache

    def bounds(self) -> BoundCache:
        if self._cache is None:
            raise RuntimeError("call freeze_bounds() after training")
        self._cache.check(self.W.detach(), self.C.detach())
        return self._cache

    # ---- forward ---------------------------------------------------------------------------
    def forward(self, inputs: Tensor, dt: Tensor, mode: str = "dense", epsilon: Tensor | float | None = None,
                policy: str = "certified", tau: float | None = None) -> ForwardResult:
        if mode == "dense":
            return self._dense(inputs, dt)
        if mode not in ("masked_ref", "selective"):
            raise ValueError(mode)
        return self._stream(inputs, dt, mode, epsilon, policy, tau)

    def _dense(self, x: Tensor, dt: Tensor) -> ForwardResult:
        B, T, _ = x.shape
        u = self.encode(x)  # [B, T, d]
        g = self.gates(u)  # [B, T, G]
        b = torch.einsum("gkd,btd->btgk", self.W, u) * g[..., None]
        b = b.view(B, T, self.G, self.m, 2)
        ar, ai, _ = self.coeffs(dt)
        hr, hi = complex_scan(ar, ai, b[..., 0], b[..., 1])
        y = self.readout(hr, hi, u)
        ev = torch.full((B, T), self.G, dtype=torch.int64, device=x.device)
        return ForwardResult(y, None, ev, "dense")

    @torch.no_grad()
    def _stream(self, x: Tensor, dt: Tensor, mode: str, epsilon, policy: str, tau) -> ForwardResult:
        B, T, _ = x.shape
        G, m = self.G, self.m
        cache = self.bounds()
        wbar, c = cache.wbar.to(x.dtype), cache.c.to(x.dtype)
        if policy in ("certified", "oracle"):
            if epsilon is None:
                raise ValueError("certified/oracle policy needs epsilon")
            eps = torch.as_tensor(epsilon, dtype=x.dtype).expand(B).clone()
            if (eps < 0).any() or not torch.isfinite(eps).all():
                raise ValueError("epsilon must be finite and nonnegative")
        elif tau is None:
            raise ValueError(f"{policy} policy needs tau")
        hr = x.new_zeros(B, G, m)
        hi = x.new_zeros(B, G, m)
        e = x.new_zeros(B, G)
        ys, bs, evs = [], [], []
        u_cache = None
        b_cache = None
        status = "exact-arithmetic bound; float observed only" if policy in ("certified", "oracle") else "no bound"
        for t in range(T):
            u = self.encode(x[:, t])
            g = self.gates(u)
            ar, ai, rho = self.coeffs(dt[:, t])
            # homogeneous dynamics: every group, every step
            hr, hi = ar * hr - ai * hi, ar * hi + ai * hr
            if policy == "certified":
                beta = write_bound(g, wbar, u)
                om = choose_omissions(e, rho, beta, c, eps)
                e = update_error(e, rho, beta, om)
            elif policy == "oracle":
                full = torch.einsum("gkd,bd->bgk", self.W, u) * g[..., None]
                beta = torch.linalg.vector_norm(full, dim=-1)
                om = choose_omissions(e, rho, beta, c, eps)
                e = update_error(e, rho, beta, om)
            elif policy == "threshold":
                beta = write_bound(g, wbar, u)
                om = c[None] * beta < tau
            elif policy == "delta":
                if u_cache is None:
                    om = torch.zeros(B, G, dtype=torch.bool, device=x.device)
                else:
                    stale = torch.linalg.vector_norm(u - u_cache, dim=-1) <= tau  # [B]
                    om = stale[:, None].expand(B, G)
            else:
                raise ValueError(policy)
            keep = ~om
            if policy == "delta":
                # recompute projection only when input moved; reuse cached b otherwise (gate fresh)
                if b_cache is None:
                    b_cache = x.new_zeros(B, G, 2 * m)
                    u_cache = u.clone()
                if mode == "masked_ref" or B > 1:
                    full = torch.einsum("gkd,bd->bgk", self.W, u)
                    refresh = keep.any(-1)
                    b_cache = torch.where(refresh[:, None, None], full, b_cache)
                    u_cache = torch.where(refresh[:, None], u, u_cache)
                elif keep.any():
                    b_cache = torch.einsum("gkd,bd->bgk", self.W, u)
                    u_cache = u.clone()
                bb = (b_cache * g[..., None]).view(B, G, m, 2)
                hr, hi = hr + bb[..., 0], hi + bb[..., 1]
                n_eval = keep.sum(-1) if mode == "selective" else torch.full((B,), G, device=x.device)
            elif mode == "masked_ref":
                full = torch.einsum("gkd,bd->bgk", self.W, u) * g[..., None]
                bb = (full * keep[..., None]).view(B, G, m, 2)
                hr, hi = hr + bb[..., 0], hi + bb[..., 1]
                n_eval = torch.full((B,), G, device=x.device)
            else:  # selective: genuinely skip omitted projections
                n_eval = keep.sum(-1)
                for bi_, gi in keep.nonzero().tolist():
                    w = (torch.mv(self.W[gi], u[bi_]) * g[bi_, gi]).view(m, 2)
                    hr[bi_, gi] += w[:, 0]
                    hi[bi_, gi] += w[:, 1]
            ys.append(self.readout(hr, hi, u))
            evs.append(n_eval)
            if policy in ("certified", "oracle"):
                bs.append(output_bound(e, c))
        y = torch.stack(ys, 1)
        bounds = torch.stack(bs, 1) if bs else None
        return ForwardResult(y, bounds, torch.stack(evs, 1), status)
