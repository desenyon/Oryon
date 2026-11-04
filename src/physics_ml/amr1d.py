"""Minimal 1D two-level block-AMR finite-volume solver (float64) for the pivot equivalence check.

Periodic domain [0,1), Burgers f(u)=u^2/2, refinement ratio 2 with subcycling (2 fine steps per
coarse step), piecewise-constant conservative prolongation, averaging restriction.

Two coarse-fine coupling schemes for ANY face-flux function F(uL, uR) (classical or learned):
  reflux  Berger-Colella: coarse step with coarse interface fluxes, fine substeps, then correct
          the coarse cells adjacent to the patch with (fine time-summed flux - coarse flux).
  joint   coarse interface flux *defined* as the time-average of the fine subcycle fluxes
          (the "jointly represented" hierarchical flux of pivot_review.md Direction 2 (a)).
Fine ghost cells use coarse data at time n (`ghost="n"`), or, for reflux only, linear time
interpolation between coarse n and n+1 (`ghost="interp"`), which `joint` cannot use because its
coarse n+1 depends on the fine fluxes.
"""
from __future__ import annotations

import numpy as np


def rusanov(uL, uR):
    return 0.25 * (uL * uL + uR * uR) - 0.5 * np.maximum(np.abs(uL), np.abs(uR)) * (uR - uL)


def mlp_flux(seed: int = 0, width: int = 16):
    """A random 'learned' flux network F(uL, uR) (untrained; conservation is structural)."""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((2, width)); b1 = rng.standard_normal(width) * 0.1
    W2 = rng.standard_normal(width) / width

    def F(uL, uR):
        z = np.tanh(np.stack([uL, uR], -1) @ W1 + b1)
        return rusanov(uL, uR) + 0.1 * (z @ W2)

    return F


def _fine_substeps(uf, gl, gr, dtf, dxf, F):
    """Two fine substeps on a patch with constant ghost lists gl[k], gr[k]; returns new uf and
    the interface fluxes at the left/right patch boundary for each substep."""
    FL, FR = [], []
    for k in range(2):
        ext = np.concatenate([[gl[k]], uf, [gr[k]]])
        fl = F(ext[:-1], ext[1:])  # faces of the patch (len n+1)
        FL.append(fl[0]); FR.append(fl[-1])
        uf = uf - dtf / dxf * (fl[1:] - fl[:-1])
    return uf, FL, FR


def step(uc, uf, a, b, dtc, dxc, F, scheme="reflux", ghost="n"):
    """One coarse step. uc: coarse cells [N]; uf: fine cells covering coarse [a, b) (len 2(b-a));
    patch must not touch the periodic seam (1 <= a < b <= N-1)."""
    N = len(uc)
    dxf, dtf = dxc / 2, dtc / 2
    faces = F(np.roll(uc, 1), uc)  # faces[i] = flux at left face of coarse cell i
    if scheme == "reflux":
        ucn = uc - dtc / dxc * (np.roll(faces, -1) - faces)
        if ghost == "n":
            gl, gr = [uc[a - 1]] * 2, [uc[b]] * 2
        else:
            gl = [uc[a - 1] + 0.5 * k * (ucn[a - 1] - uc[a - 1]) for k in range(2)]
            gr = [uc[b] + 0.5 * k * (ucn[b] - uc[b]) for k in range(2)]
        ufn, FL, FR = _fine_substeps(uf, gl, gr, dtf, dxf, F)
        # reflux: replace coarse interface flux by fine time-summed flux in the neighbours
        ucn[a - 1] += dtc / dxc * faces[a] - dtf / dxc * sum(FL)
        ucn[b] += -dtc / dxc * faces[b] + dtf / dxc * sum(FR)
    elif scheme == "joint":
        if ghost != "n":
            raise ValueError("joint scheme cannot use time-interpolated ghosts (circular dependence)")
        ufn, FL, FR = _fine_substeps(uf, [uc[a - 1]] * 2, [uc[b]] * 2, dtf, dxf, F)
        faces = faces.copy()
        faces[a] = 0.5 * (FL[0] + FL[1])
        faces[b] = 0.5 * (FR[0] + FR[1])
        ucn = uc - dtc / dxc * (np.roll(faces, -1) - faces)
    else:
        raise ValueError(scheme)
    ucn[a:b] = ufn.reshape(-1, 2).mean(1)  # restriction
    return ucn, ufn


def regrid(uc, uf, a, b, threshold, min_cells=4):
    """Conservative regrid: tag the coarse cells with large |grad u|, choose one patch [a2, b2);
    coarse values from the current composite state, new fine cells by piecewise-constant
    prolongation except where old fine data exist (copied)."""
    N = len(uc)
    comp = uc.copy()
    comp[a:b] = uf.reshape(-1, 2).mean(1)
    grad = np.abs(np.roll(comp, -1) - np.roll(comp, 1))
    tagged = np.flatnonzero(grad > threshold)
    tagged = tagged[(tagged >= 2) & (tagged <= N - 3)]
    if len(tagged) == 0:
        a2, b2 = N // 2 - min_cells // 2, N // 2 + min_cells // 2
    else:
        a2, b2 = max(1, tagged.min() - 1), min(N - 1, tagged.max() + 2)
        if b2 - a2 < min_cells:
            b2 = min(N - 1, a2 + min_cells)
    new_f = np.repeat(comp[a2:b2], 2)
    lo, hi = max(a, a2), min(b, b2)
    if lo < hi:  # keep existing fine data where the patches overlap
        new_f[2 * (lo - a2):2 * (hi - a2)] = uf[2 * (lo - a):2 * (hi - a)]
    return comp, new_f, a2, b2


def rollout(u0c, F, steps, dtc, scheme, ghost="n", regrid_every=10, threshold=0.05):
    N = len(u0c)
    dxc = 1.0 / N
    uc, a, b = u0c.copy(), N // 4, N // 4 + 8
    uf = np.repeat(uc[a:b], 2)
    mass = [uc.sum() * dxc]
    for n in range(steps):
        if n % regrid_every == 0:
            uc, uf, a, b = regrid(uc, uf, a, b, threshold)
        uc, uf = step(uc, uf, a, b, dtc, dxc, F, scheme, ghost)
        mass.append(uc.sum() * dxc)
    return uc, uf, (a, b), np.array(mass)
