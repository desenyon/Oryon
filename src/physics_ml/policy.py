"""Causal budget policy P1 (SPEC §4)."""
from __future__ import annotations

import torch
from torch import Tensor

from .bounds import _check_finite, _check_nonneg, output_bound, update_error


def choose_omissions(
    e: Tensor,
    rho: Tensor,
    beta: Tensor,
    c: Tensor,
    epsilon: Tensor,
    costs: Tensor | None = None,
) -> Tensor:
    """Choose which writes to omit at this step without mutating e.

    e: bound after the previous step [batch, groups]; rho, beta: [batch, groups];
    c: [groups]; epsilon: [batch]; costs: optional [groups] measured per-group write cost.
    Returns bool [batch, groups], True = omit.

    Equal costs: omit the longest ascending prefix of c_i beta_i fitting in eps - B_base
    (maximizes the omitted count). Unequal costs: prefix in descending cost/contribution order.
    A deterministic final check removes the last-added omissions if rounding broke the budget.
    """
    _check_nonneg("epsilon", epsilon)
    _check_nonneg("e", e)
    _check_nonneg("rho", rho)
    _check_nonneg("beta", beta)
    _check_nonneg("c", c)
    base = output_bound(rho * e, c)  # [batch]
    # zero-contribution omissions are always free, even if rounding left base above epsilon
    room = (epsilon - base).clamp_min(0.0)
    contrib = c[None] * beta  # [batch, groups]
    if costs is None:
        order = torch.argsort(contrib, dim=-1, stable=True)
    else:
        _check_nonneg("costs", costs)
        ratio = costs[None] / contrib.clamp_min(torch.finfo(contrib.dtype).tiny)
        ratio = torch.where(contrib == 0, torch.full_like(ratio, float("inf")), ratio)
        order = torch.argsort(-ratio, dim=-1, stable=True)
    sorted_contrib = torch.gather(contrib, -1, order)
    csum = torch.cumsum(sorted_contrib, dim=-1)
    take_sorted = csum <= room[:, None]
    # enforce prefix property (cumsum is monotone, so already a prefix; keep explicit)
    take_sorted = torch.cumprod(take_sorted.to(torch.int8), dim=-1).bool()
    omitted = torch.zeros_like(take_sorted)
    omitted.scatter_(-1, order, take_sorted)
    # deterministic final budget check (direct recomputation, not via cumsum)
    n_take = take_sorted.sum(-1)
    for _ in range(contrib.shape[-1] + 1):
        spent = (contrib * omitted.to(contrib.dtype)).sum(-1)
        bad = (spent > 0) & (base + spent > epsilon) & (n_take > 0)
        if not bad.any():
            break
        idx = bad.nonzero(as_tuple=True)[0]
        last = torch.gather(order[idx], -1, (n_take[idx] - 1)[:, None]).squeeze(-1)
        omitted[idx, last] = False
        n_take[idx] -= 1
    return omitted


class BudgetLedger:
    """Per-sequence epsilon with a running ledger. Epsilon is fixed per sequence; a request to
    lower it below the already-committed bound is rejected (no ledger reset)."""

    def __init__(self, c: Tensor, epsilon: Tensor, groups: int):
        _check_nonneg("epsilon", epsilon)
        _check_nonneg("c", c)
        self.c = c
        self.epsilon = epsilon.clone()
        self.e = torch.zeros(epsilon.shape[0], groups, dtype=c.dtype, device=c.device)

    @property
    def bound(self) -> Tensor:
        return output_bound(self.e, self.c)

    def set_epsilon(self, new_eps: Tensor) -> None:
        _check_nonneg("epsilon", new_eps)
        if (new_eps < self.bound).any():
            raise ValueError("requested epsilon is below the committed bound; unsupported mid-sequence change")
        self.epsilon = new_eps.clone()

    def step(self, rho: Tensor, beta: Tensor, costs: Tensor | None = None, force_compute: Tensor | None = None) -> Tensor:
        omitted = choose_omissions(self.e, rho, beta, self.c, self.epsilon, costs)
        if force_compute is not None:
            omitted = omitted & ~force_compute
        self.e = update_error(self.e, rho, beta, omitted)
        return omitted
