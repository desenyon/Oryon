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
