"""Tests of T1/L1 assumptions, the ledger, and the budget policy (float64)."""
import numpy as np
import pytest
import torch

from physics_ml import reference as ref
from physics_ml.bounds import (
    BoundCache, frobenius_bounds, output_bound, transition_coefficients, update_error, write_bound,
)
from physics_ml.policy import BudgetLedger, choose_omissions

torch.set_default_dtype(torch.float64)


def _setup(seed=0, G=4, m=3, d=5, p=2, T=60, gamma_zero=False, dt=None, u_scale=1.0):
    rng = np.random.default_rng(seed)
    s = ref.random_system(rng, G, m, d, p, undamped_frac=0.0)
    if gamma_zero:
        s.gamma[:] = 0.0
    u, dt_ = ref.random_inputs(rng, T, d)
    u = u * u_scale
    if dt is not None:
        dt_ = np.full(T, dt)
    Wr, Cr = torch.tensor(s.real_W()), torch.tensor(s.real_C())
    wbar, c = frobenius_bounds(Wr), frobenius_bounds(Cr)
    gates = torch.tensor(np.stack([s.gates(x) for x in u]))
    beta = write_bound(gates, wbar, torch.tensor(u))
    _, _, rho = transition_coefficients(torch.tensor(s.gamma), torch.tensor(s.omega), torch.tensor(dt_))
    return s, u, dt_, wbar, c, beta, rho


def _ledger(beta, rho, omitted, c):
    e = torch.zeros(1, beta.shape[1])
    E, B = [], []
    for t in range(beta.shape[0]):
        e = update_error(e, rho[t:t + 1], beta[t:t + 1], torch.tensor(omitted[t:t + 1]))
        E.append(e[0].numpy())
        B.append(float(output_bound(e, c)[0]))
    return np.array(E), np.array(B)


def _check_bound(s, u, dt, omitted, E, B, tol=1e-10):
    hs, ys = ref.run_dense(s, u, dt)
    ha, ya = ref.run_masked(s, u, dt, omitted)
    st = np.linalg.norm(hs - ha, axis=-1)
    out = np.linalg.norm(ys - ya, axis=-1)
    assert (st <= E + tol).all(), (st - E).max()
    assert (out <= B + tol).all(), (out - B).max()
    return st, out


def test_frobenius_is_upper_bound_on_spectral():
    W = torch.randn(6, 8, 5)
    assert (frobenius_bounds(W) >= torch.linalg.matrix_norm(W, ord=2) - 1e-12).all()


def test_all_write_equivalence():
    s, u, dt, wbar, c, beta, rho = _setup()
    omitted = np.zeros((len(u), len(c)), bool)
    E, B = _ledger(beta, rho, omitted, c)
    assert (E == 0).all() and (B == 0).all()
    hs, _ = ref.run_dense(s, u, dt)
    ha, _ = ref.run_masked(s, u, dt, omitted)
    assert np.array_equal(hs, ha)


def test_zero_input_gives_zero_beta_and_free_omission():
    s, u, dt, wbar, c, beta, rho = _setup(u_scale=0.0)
    assert (beta == 0).all()
    om = choose_omissions(torch.zeros(1, len(c)), rho[:1], beta[:1], c, torch.tensor([0.0]))
    assert om.all()  # zero-cost omissions fit even at eps = 0


def test_random_masks_respect_bound_positive_damping():
    for seed in range(5):
        s, u, dt, wbar, c, beta, rho = _setup(seed=seed)
        assert (s.gamma > 0).all() and (rho < 1).any()
        omitted = np.random.default_rng(seed).random(beta.shape) < 0.5
        _check_bound(s, u, dt, omitted, *_ledger(beta, rho, omitted, c))


def test_zero_damping_error_does_not_replenish():
    s, u, dt, wbar, c, beta, rho = _setup(gamma_zero=True)
    assert torch.all(rho == 1)
    omitted = np.random.default_rng(1).random(beta.shape) < 0.3
    E, B = _ledger(beta, rho, omitted, c)
    cum = np.cumsum(beta.numpy() * omitted, axis=0)
    assert np.allclose(E, cum)  # pure accumulation
    assert (np.diff(E, axis=0) >= -1e-15).all()
    _check_bound(s, u, dt, omitted, E, B)


def test_zero_dt_gives_identity_transition():
    ar, ai, rho = transition_coefficients(torch.rand(2, 3), torch.randn(2, 3), torch.zeros(4))
    assert torch.all(ar == 1) and torch.all(ai == 0) and torch.all(rho == 1)


def test_variable_dt_bound():
    s, u, dt, wbar, c, beta, rho = _setup(seed=3, T=100)
    assert len(np.unique(dt)) > 3
    omitted = np.random.default_rng(3).random(beta.shape) < 0.6
    _check_bound(s, u, dt, omitted, *_ledger(beta, rho, omitted, c))


def test_negative_dt_rejected():
    with pytest.raises(ValueError):
        transition_coefficients(torch.rand(2, 3), torch.randn(2, 3), torch.tensor([0.1, -0.01]))
    s = ref.random_system(np.random.default_rng(0), 2, 2, 2, 1)
    with pytest.raises(ValueError):
        s.coeffs(-1.0)


def test_negative_gamma_and_nonfinite_rejected():
    with pytest.raises(ValueError):
        transition_coefficients(-torch.rand(2, 3), torch.randn(2, 3), torch.ones(1))
    with pytest.raises(ValueError):
        update_error(torch.zeros(1, 2), torch.ones(1, 2), torch.tensor([[float("nan"), 0.0]]), torch.zeros(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError):
        update_error(torch.zeros(1, 2), torch.full((1, 2), 1.5), torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))


def test_negative_epsilon_rejected():
    with pytest.raises(ValueError):
        choose_omissions(torch.zeros(1, 2), torch.ones(1, 2), torch.ones(1, 2), torch.ones(2), torch.tensor([-1e-3]))


def test_policy_budget_and_feasibility():
    for seed in range(6):
        s, u, dt, wbar, c, beta, rho = _setup(seed=seed, T=120, gamma_zero=(seed % 2 == 0))
        eps = float((c[None] * beta).mean()) * [0.0, 0.3, 3.0][seed % 3]
        led = BudgetLedger(c, torch.tensor([eps]), len(c))
        om_all, E, B = [], [], []
        for t in range(len(u)):
            assert float(output_bound(rho[t:t + 1] * led.e, c)[0]) <= eps + 1e-12  # compute-all feasible
            om = led.step(rho[t:t + 1], beta[t:t + 1])
            om_all.append(om[0].numpy()); E.append(led.e[0].numpy()); B.append(float(led.bound[0]))
            assert B[-1] <= eps + 1e-12
        om_all = np.array(om_all)
        _check_bound(s, u, dt, om_all, np.array(E), np.array(B))
        if eps == 0:
            # only zero-contribution writes may be omitted at eps = 0
            assert ((c[None] * beta).numpy()[om_all] == 0).all()


def test_accumulated_error_persists_after_writes_resume():
    s, u, dt, wbar, c, beta, rho = _setup(gamma_zero=True, T=80)
    omitted = np.zeros(beta.shape, bool)
    omitted[:20] = True  # omit early, then compute everything
    E, B = _ledger(beta, rho, omitted, c)
    assert np.allclose(E[20:], E[19])  # writes do not reset the ledger (rho = 1)
    st, _ = _check_bound(s, u, dt, omitted, E, B)
    assert st[-1].max() > 0  # actual deviation persists too


def test_mixed_damping_group_has_unit_rho():
    gamma = torch.tensor([[0.0, 0.5, 2.0], [0.3, 0.5, 2.0]])
    _, _, rho = transition_coefficients(gamma, torch.randn(2, 3), torch.tensor([1.0]))
    assert rho[0, 0] == 1.0 and rho[0, 1] < 1.0


def test_zero_and_unequal_readout_contributions():
    c = torch.tensor([0.0, 1.0, 4.0])
    beta = torch.tensor([[5.0, 1.0, 1.0]])
    om = choose_omissions(torch.zeros(1, 3), torch.ones(1, 3), beta, c, torch.tensor([1.5]))
    # contributions [0, 1, 4] -> omit group 0 (free) and group 1 (fits), not group 2
    assert om.tolist() == [[True, True, False]]


def test_cost_aware_policy_and_final_check():
    c = torch.ones(3)
    beta = torch.tensor([[1.0, 1.0, 1.0]])
    costs = torch.tensor([1.0, 10.0, 1.0])
    om = choose_omissions(torch.zeros(1, 3), torch.ones(1, 3), beta, c, torch.tensor([1.0]), costs)
    assert om.tolist() == [[False, True, False]]  # expensive group omitted first
    # final check: base already above epsilon (e.g. rounding) -> nothing omitted except free writes
    om = choose_omissions(torch.tensor([[1.0, 0.0, 0.0]]), torch.ones(1, 3), torch.tensor([[0.0, 1.0, 0.0]]), c, torch.tensor([0.5]))
    assert om.tolist() == [[True, False, True]]


def test_mid_sequence_epsilon_decrease_rejected():
    led = BudgetLedger(torch.ones(2), torch.tensor([1.0]), 2)
    led.step(torch.ones(1, 2), torch.tensor([[0.4, 0.4]]))
    assert float(led.bound[0]) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        led.set_epsilon(torch.tensor([0.5]))
    led.set_epsilon(torch.tensor([2.0]))  # increases are allowed and do not reset the ledger
    assert float(led.bound[0]) == pytest.approx(0.8)


def test_bound_cache_invalidation():
    W, C = torch.randn(2, 4, 3), torch.randn(2, 1, 4)
    cache = BoundCache(W, C)
    cache.check(W, C)
    W2 = W.clone(); W2[0, 0, 0] += 1e-6
    with pytest.raises(RuntimeError):
        cache.check(W2, C)


def test_bound_is_frequency_independent():
    gamma = torch.rand(2, 3)
    _, _, r1 = transition_coefficients(gamma, torch.zeros(2, 3), torch.tensor([0.7]))
    _, _, r2 = transition_coefficients(gamma, 5 * torch.randn(2, 3), torch.tensor([0.7]))
    assert torch.equal(r1, r2)
