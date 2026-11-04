"""P3: selective execution omits work and matches the masked reference."""
import numpy as np
import pytest
import torch

from physics_ml.model import OscillatorSSM


def _model(**kw):
    torch.manual_seed(0)
    m = OscillatorSSM(d_in=3, d_out=2, d_model=16, d_hidden=16, groups=4, modes=3, seed=0, **kw)
    m.eval()
    m.freeze_bounds()
    return m


def _data(B=2, T=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, 3, generator=g) * (torch.rand(B, T, 1, generator=g) < 0.3)
    dt = torch.rand(B, T, generator=g) * 0.5
    return x, dt


def test_dense_scan_matches_streaming_all_writes():
    m = _model()
    x, dt = _data()
    dense = m(x, dt, mode="dense").predictions
    # epsilon = 0 with certified policy computes every nonzero write
    stream = m(x, dt, mode="masked_ref", epsilon=0.0).predictions
    assert torch.allclose(dense, stream, atol=1e-5)


@pytest.mark.parametrize("policy,kw", [("certified", {"epsilon": 0.5}), ("threshold", {"tau": 3.0}), ("delta", {"tau": 0.5})])
def test_selective_matches_masked_ref(policy, kw):
    m = _model()
    x, dt = _data(B=1)
    a = m(x, dt, mode="masked_ref", policy=policy, **kw)
    b = m(x, dt, mode="selective", policy=policy, **kw)
    assert torch.allclose(a.predictions, b.predictions, atol=1e-5)
    if a.bounds is not None:
        assert torch.allclose(a.bounds, b.bounds)
    assert (b.evaluated_writes <= m.G).all()
    assert (b.evaluated_writes < m.G).any()  # something was actually omitted


def test_certified_bound_holds_against_dense():
    m = _model().double()
    m.freeze_bounds()
    x, dt = _data(B=3, T=60)
    x, dt = x.double(), dt.double()
    dense = m(x, dt, mode="dense").predictions
    for eps in [0.0, 0.05, 0.5, 5.0]:
        r = m(x, dt, mode="selective" if x.shape[0] == 1 else "masked_ref", epsilon=eps)
        dev = torch.linalg.vector_norm(r.predictions - dense, dim=-1)
        assert (dev <= r.bounds + 1e-9).all()
        assert (r.bounds <= eps + 1e-12).all()


def test_selective_does_not_issue_omitted_projections(monkeypatch):
    m = _model()
    x, dt = _data(B=1, T=50)
    calls = {"n": 0}
    real_mv = torch.mv

    def counting_mv(*a, **k):
        calls["n"] += 1
        return real_mv(*a, **k)

    monkeypatch.setattr(torch, "mv", counting_mv)
    r = m(x, dt, mode="selective", epsilon=1.0)
    assert calls["n"] == int(r.evaluated_writes.sum())
    assert calls["n"] < m.G * x.shape[1]


def test_bounds_must_be_frozen_and_invalidate():
    m = OscillatorSSM(d_in=3, d_out=2, d_model=8, d_hidden=8, groups=2, modes=2)
    x, dt = _data(B=1, T=5)
    with pytest.raises(RuntimeError):
        m(x, dt, mode="selective", epsilon=1.0)
    m.freeze_bounds()
    with torch.no_grad():
        m.W[0, 0, 0] += 0.1
    with pytest.raises(RuntimeError):
        m(x, dt, mode="selective", epsilon=1.0)


def test_zero_frequency_control_has_no_rotation():
    m = _model(zero_frequency=True)
    ar, ai, _ = m.coeffs(torch.tensor([0.7]))
    assert torch.all(ai == 0)
    assert "omega" not in dict(m.named_parameters())


def test_negative_dt_and_epsilon_rejected():
    m = _model()
    x, dt = _data(B=1, T=5)
    with pytest.raises(ValueError):
        m(x, -dt - 0.1, mode="dense")
    with pytest.raises(ValueError):
        m(x, dt, mode="selective", epsilon=-1.0)
