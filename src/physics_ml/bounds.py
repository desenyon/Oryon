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

    def check(self, W: Tensor, C: Tensor) -> None:
        if weight_fingerprint(W, C) != self.fingerprint:
            raise RuntimeError("weights changed since bounds were cached; recompute BoundCache")
