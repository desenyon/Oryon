"""Pivot check (D014): the 'joint' hierarchical flux of pivot Direction 2 (a) is algebraically
identical to Berger-Colella refluxing with matched ghost treatment, for classical and learned
fluxes, through dynamic regridding; both conserve mass to round-off."""
import numpy as np
import pytest

from physics_ml.amr1d import mlp_flux, rollout, rusanov


def _ic(seed, N=64):
    rng = np.random.default_rng(seed)
    x = (np.arange(N) + 0.5) / N
    return 0.5 + sum(rng.uniform(-0.4, 0.4) * np.sin(2 * np.pi * k * x + rng.uniform(0, 6.3)) for k in (1, 2, 3))


@pytest.mark.parametrize("flux", ["rusanov", "mlp"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_joint_equals_reflux_with_matched_ghosts(flux, seed):
    F = rusanov if flux == "rusanov" else mlp_flux(seed)
    u0 = _ic(seed)
    dtc = 0.4 / 64 / 1.5  # CFL-safe for |u| <= 1.5
    r = rollout(u0, F, 200, dtc, "reflux", "n", regrid_every=7)
    j = rollout(u0, F, 200, dtc, "joint", "n", regrid_every=7)
    assert r[2] == j[2]
    assert np.max(np.abs(r[0] - j[0])) < 1e-12 and np.max(np.abs(r[1] - j[1])) < 1e-12
    for res in (r, j):
        assert np.max(np.abs(res[3] - res[3][0])) < 1e-12  # conservation to round-off


def test_interpolated_ghosts_are_the_only_difference():
    u0 = _ic(5)
    dtc = 0.4 / 64 / 1.5
    r = rollout(u0, rusanov, 200, dtc, "reflux", "interp", regrid_every=7)
    j = rollout(u0, rusanov, 200, dtc, "joint", "n", regrid_every=7)
    assert np.max(np.abs(r[3] - r[3][0])) < 1e-12
    diff = np.max(np.abs(r[0] - j[0]))
    assert diff > 1e-8  # differs only through the classical ghost-cell time interpolation
    with pytest.raises(ValueError):
        rollout(u0, rusanov, 2, dtc, "joint", "interp")
