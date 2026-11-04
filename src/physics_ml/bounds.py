"""Error-ledger bounds for selective input writes (SPEC §2–3, theorem T1).

Shapes (real coordinates):
    e, rho, beta : [batch, groups]
    c            : [groups]
    omitted      : bool [batch, groups]  (True = write omitted)

All bounds are exact-arithmetic statements (SPEC §5). Nothing here is a machine-arithmetic
certificate.
"""
from __future__ import annotations

import hashlib

import torch
from torch import Tensor


def _check_finite(name: str, x: Tensor) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains nonfinite values")


def _check_nonneg(name: str, x: Tensor) -> None:
    _check_finite(name, x)
    if (x < 0).any():
        raise ValueError(f"{name} must be nonnegative")


def transition_coefficients(gamma: Tensor, omega: Tensor, dt: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """a = exp((-gamma + i omega) dt) in real coordinates.

    gamma, omega: [groups, modes]; dt: [batch]. Returns (ar, ai) [batch, groups, modes] and
    rho [batch, groups] = max_j |a_ij| = exp(-min_j gamma_ij dt).
    """
    _check_nonneg("gamma", gamma)
    _check_finite("omega", omega)
    _check_nonneg("dt", dt)
    dtb = dt[:, None, None]
    mag = torch.exp(-gamma[None] * dtb)
    ang = omega[None] * dtb
    ar, ai = mag * torch.cos(ang), mag * torch.sin(ang)
    rho = mag.amax(dim=-1)
    return ar, ai, rho


def write_bound(gate: Tensor, wbar: Tensor, u: Tensor) -> Tensor:
    """beta_i = |g_i(u)| * wbar_i * ||u||_2, computed before any W_i u product.

    gate: [batch, groups]; wbar: [groups]; u: [batch, input_dim].
    """
    _check_finite("gate", gate)
    _check_nonneg("wbar", wbar)
    _check_finite("u", u)
    return gate.abs() * wbar[None] * torch.linalg.vector_norm(u, dim=-1, keepdim=True)


def update_error(e: Tensor, rho: Tensor, beta: Tensor, omitted: Tensor) -> Tensor:
    """e_t = rho_t e_{t-1} + (1 - m_t) beta_t. Applied exactly once per step."""
    _check_nonneg("e", e)
    _check_nonneg("rho", rho)
    _check_nonneg("beta", beta)
    if (rho > 1).any():
        raise ValueError("rho must be <= 1 (requires gamma >= 0, dt >= 0)")
    if omitted.dtype != torch.bool:
        raise TypeError("omitted must be a bool tensor")
    return rho * e + omitted.to(beta.dtype) * beta


def output_bound(e: Tensor, c: Tensor) -> Tensor:
    """B = sum_i c_i e_i -> [batch]."""
    _check_nonneg("c", c)
    return (e * c[None]).sum(dim=-1)


def frobenius_bounds(W: Tensor) -> Tensor:
    """Sound per-group upper bound on the spectral norm: ||W_i||_F >= ||W_i||_2.

    W: [groups, out, in] -> [groups].
    """
    _check_finite("W", W)
    return torch.linalg.matrix_norm(W, ord="fro", dim=(-2, -1))


def weight_fingerprint(*tensors: Tensor) -> str:
    """Hash of exact weight bytes; cached bounds are invalid if this changes."""
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class BoundCache:
    """Fixed-weight bounds (wbar, c) tied to a weight fingerprint."""

    def __init__(self, W: Tensor, C: Tensor):
        # W: [groups, 2m, d]; C: [groups, p, 2m]
        self.fingerprint = weight_fingerprint(W, C)
        self.wbar = frobenius_bounds(W)
        self.c = frobenius_bounds(C)
        m = W.shape[1] // 2
        self.wmode = mode_write_norms(W, m).to(W.dtype)  # [G, m]
        cs, cm = readout_norms(C, m)
        self.c_spec, self.c_mode = cs.to(C.dtype), cm.to(C.dtype)

    def check(self, W: Tensor, C: Tensor) -> None:
        if weight_fingerprint(W, C) != self.fingerprint:
            raise RuntimeError("weights changed since bounds were cached; recompute BoundCache")


# ---------------------------------------------------------------- per-mode ledger (H001b, SPEC §3c)
_MARGIN = 1e-9  # relative safety margin on float64-computed spectral norms (not a machine certificate)


def mode_write_norms(W: Tensor, m: int) -> Tensor:
    """Spectral norms of each mode's 2 x d real write block, closed form (2x2 Gram), float64.

    W: [G, 2m, d] (rows interleaved re, im) -> [G, m]."""
    _check_finite("W", W)
    Wd = W.detach().double().view(W.shape[0], m, 2, W.shape[-1])
    a = (Wd[:, :, 0] ** 2).sum(-1)
    b = (Wd[:, :, 0] * Wd[:, :, 1]).sum(-1)
    c = (Wd[:, :, 1] ** 2).sum(-1)
    lam = 0.5 * (a + c) + torch.sqrt(0.25 * (a - c) ** 2 + b ** 2)
    return torch.sqrt(lam) * (1 + _MARGIN)


def readout_norms(C: Tensor, m: int) -> tuple[Tensor, Tensor]:
    """Per-group spectral norm of C_i (float64 SVD + margin) and per-mode spectral norms of the
    p x 2 blocks. C: [G, p, 2m] -> ([G], [G, m])."""
    _check_finite("C", C)
    Cd = C.detach().double()
    c_spec = torch.linalg.matrix_norm(Cd, ord=2) * (1 + _MARGIN)
    blocks = Cd.view(C.shape[0], C.shape[1], m, 2).permute(0, 2, 1, 3)  # [G, m, p, 2]
    c_mode = torch.linalg.matrix_norm(blocks, ord=2) * (1 + _MARGIN)
    return c_spec, c_mode


def group_bounds_modes(e: Tensor, c_spec: Tensor, c_mode: Tensor) -> Tensor:
    """B_i = min(||C_i||_2 ||e_i||_2, sum_j ||C_ij||_2 e_ij), e: [B, G, m] -> [B, G]."""
    return torch.minimum(c_spec * torch.linalg.vector_norm(e, dim=-1), (c_mode * e).sum(-1))


def update_error_modes(e: Tensor, amag: Tensor, beta: Tensor, omitted: Tensor) -> Tensor:
    """e_ij <- |a_ij| e_ij + (1 - m_i) beta_ij. e, amag, beta: [B, G, m]; omitted: [B, G]."""
    _check_nonneg("e", e)
    _check_nonneg("beta", beta)
    if (amag > 1).any() or (amag < 0).any():
        raise ValueError("|a| must lie in [0, 1]")
    return amag * e + omitted[..., None].to(beta.dtype) * beta
