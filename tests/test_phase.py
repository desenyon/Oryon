"""Counterexamples showing why SPEC assumptions A2 (phase always advances, state-independent
transitions) are required. Each test exhibits a ledger violation when the assumption is broken."""
import numpy as np


def test_freezing_phase_breaks_ledger():
    """Skip-RNN-style state copy: on a skipped step the state is frozen instead of advanced.
    With zero input (beta = 0, no write omitted) the ledger stays 0, but the frozen state differs
    from the dense state by 2|h| for an undamped mode with omega*dt = pi."""
    a = np.exp(1j * np.pi)  # gamma = 0, omega dt = pi
    h0 = 1.0 + 0.0j
    dense = a * h0 + 0.0  # zero input
    frozen = h0  # phase frozen during skip
    ledger = 1.0 * 0.0 + 0.0  # rho * e_prev + (1 - m) beta, with e_prev = 0, beta = 0
    assert abs(dense - frozen) == 2.0
    assert abs(dense - frozen) > ledger  # violation: freezing is not covered by T1


def test_energy_preservation_is_not_incremental_contraction():
    """A norm-preserving but state-dependent rotation h -> R(theta(|h|)) h keeps |h| constant
    (energy preserved, 'rho = 1') but expands differences between nearby states, so the ledger
    e_t = 1 * e_{t-1} under-bounds the deviation. T1 needs incremental nonexpansion (A2)."""

    def f(h):
        return h * np.exp(1j * 3.0 * abs(h) ** 2)

    h, ht = 1.0 + 0j, 1.05 + 0j
    e = abs(h - ht)
    for _ in range(5):
        h, ht = f(h), f(ht)
        assert np.isclose(abs(h), 1.0) and np.isclose(abs(ht), 1.05)  # energies preserved
        e = 1.0 * e  # ledger with rho = max |a| = 1 and no new omission
    assert abs(h - ht) > e * 5  # deviation far exceeds the ledger


def test_mixed_damping_counterexample_for_min_rho():
    """Using the *minimum* or average |a| of a group as its contraction factor is unsound when the
    group contains an undamped mode: the error stored in that mode does not decay."""
    a = np.array([1.0, 0.2])  # undamped + damped mode, both with zero phase for clarity
    d = np.array([1.0, 0.0])  # all deviation sits in the undamped mode
    wrong_rho = a.mean()
    e_wrong = wrong_rho * np.linalg.norm(d)
    e_right = a.max() * np.linalg.norm(d)
    assert np.linalg.norm(a * d) > e_wrong
    assert np.linalg.norm(a * d) <= e_right
