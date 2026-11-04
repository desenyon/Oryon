"""NumPy streaming engine agrees with the torch model's masked reference for every policy."""
import numpy as np
import pytest
import torch

from physics_ml.engine import StreamEngine
from physics_ml.model import OscillatorSSM


@pytest.fixture(scope="module")
def setup():
    m = OscillatorSSM(d_in=3, d_out=2, d_model=16, d_hidden=16, groups=4, modes=3, seed=1).eval()
    m.freeze_bounds()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 60, 3, generator=g) * (torch.rand(1, 60, 1, generator=g) < 0.3)
    dt = torch.rand(1, 60, generator=g) * 0.5
    return m, StreamEngine(m), x, dt


@pytest.mark.parametrize("policy,kw", [("certified", {"epsilon": 0.0}), ("certified", {"epsilon": 1.0}),
                                       ("oracle", {"epsilon": 1.0}), ("threshold", {"tau": 3.0}), ("delta", {"tau": 0.5})])
def test_engine_matches_model(setup, policy, kw):
    m, eng, x, dt = setup
    ref = m(x, dt, mode="masked_ref", policy=policy, **kw)
    y, b, w = eng.run(x[0].numpy(), dt[0].numpy(), policy=policy, **kw)
    assert np.allclose(y, ref.predictions[0].numpy(), atol=1e-4)
    if policy in ("certified",):
        assert np.allclose(b, ref.bounds[0].numpy(), rtol=1e-4, atol=1e-6)
        sel = m(x, dt, mode="selective", policy=policy, **kw)
        assert np.array_equal(w, sel.evaluated_writes[0].numpy())


def test_engine_dense(setup):
    m, eng, x, dt = setup
    y, _, w = eng.run(x[0].numpy(), dt[0].numpy(), policy="dense")
    assert np.allclose(y, m(x, dt).predictions[0].detach().numpy(), atol=1e-4)
    assert (w == m.G).all()


@pytest.mark.parametrize("policy,kw", [("dense", {}), ("dense_fused", {}), ("certified", {"epsilon": 0.0}), ("certified", {"epsilon": 1.0}),
                                       ("oracle", {"epsilon": 1.0}), ("threshold", {"tau": 3.0}), ("delta", {"tau": 0.5})])
def test_compiled_engine_matches_numpy_engine(setup, policy, kw):
    from physics_ml.engine_nb import CompiledEngine

    m, eng, x, dt = setup
    ce = CompiledEngine(m)
    y0, b0, w0 = eng.run(x[0].numpy(), dt[0].numpy(), policy=policy.replace("_fused", ""), **kw)
    y1, b1, w1 = ce.run(x[0].numpy(), dt[0].numpy(), policy=policy, **kw)
    assert np.allclose(y0, y1, atol=1e-4)
    assert np.array_equal(w0, w1)
    if b0 is not None:
        assert np.allclose(b0, b1, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("eps", [0.0, 0.5, 5.0])
def test_cdelta_compiled_matches_model_and_bound_holds(setup, eps):
    from physics_ml.engine_nb import CompiledEngine

    m, eng, x, dt = setup
    ref = m(x, dt, mode="masked_ref", policy="cdelta", epsilon=eps)
    sel = m(x, dt, mode="selective", policy="cdelta", epsilon=eps)
    y, b, w = CompiledEngine(m).run(x[0].numpy(), dt[0].numpy(), policy="cdelta", epsilon=eps)
    assert torch.allclose(ref.predictions, sel.predictions, atol=1e-5)
    assert np.allclose(y, ref.predictions[0].numpy(), atol=1e-4)
    assert np.array_equal(w, sel.evaluated_writes[0].numpy())
    assert np.allclose(b, ref.bounds[0].numpy(), rtol=1e-4, atol=1e-6)
    dense = m(x, dt).predictions.detach()
    dev = torch.linalg.vector_norm(ref.predictions - dense, dim=-1)
    assert (dev <= ref.bounds + 1e-4).all() and (ref.bounds <= eps + 1e-6).all()


@pytest.mark.parametrize("policy,eps", [("certified_pm", 0.5), ("certified_pm", 5.0), ("cdelta_pm", 0.0), ("cdelta_pm", 0.5)])
def test_per_mode_compiled_matches_model(setup, policy, eps):
    from physics_ml.engine_nb import CompiledEngine

    m, eng, x, dt = setup
    ref = m(x, dt, mode="masked_ref", policy=policy, epsilon=eps)
    y, b, w = CompiledEngine(m).run(x[0].numpy(), dt[0].numpy(), policy=policy, epsilon=eps)
    assert np.allclose(y, ref.predictions[0].numpy(), atol=1e-4)
    assert np.array_equal(w, ref.selected_writes[0].numpy())
    assert np.allclose(b, ref.bounds[0].numpy(), rtol=1e-4, atol=1e-6)
