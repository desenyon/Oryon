import numpy as np

from physics_ml.data import SPLIT_SEEDS, make, train_normalization


def test_splits_are_distinct_realizations():
    for task in ("resonator", "recall"):
        a = make(task, "train", 20)
        b = make(task, "val", 20)
        c = make(task, "test", 20)
        assert not np.array_equal(a.x, b.x) and not np.array_equal(b.x, c.x)
        assert len(set(SPLIT_SEEDS.values())) == len(SPLIT_SEEDS)
        assert np.array_equal(make(task, "train", 20).x, a.x)  # deterministic


def test_resonator_target_is_causal_and_delayed():
    s = make("resonator", "train", 4)
    d = s.meta["delay"]
    assert np.all(s.y[:, :d] == 0)
    assert not s.mask[:, :d].any()
    # quiet stretches exist
    assert (np.abs(s.x[..., 0]) < 1e-9).mean() > 0.5


def test_shift_condition_differs():
    tr, sh = make("resonator", "train", 2), make("resonator", "shift", 2)
    assert tr.meta["chirp_band"] != sh.meta["chirp_band"]
    r = make("recall", "shift", 3)
    assert r.meta["delay"].startswith("long")


def test_recall_label_present_once_marked():
    s = make("recall", "val", 10)
    K = s.meta["K"]
    for i in range(10):
        marked = np.flatnonzero(s.x[i, :, K])
        assert len(marked) == 1
        assert s.x[i, marked[0], s.y[i]] == 1.0
        assert s.x[i, -1, K + 1] == 1.0


def test_normalization_uses_train_only():
    tr = make("resonator", "train", 5)
    m, sd = train_normalization(tr)
    assert m.shape == (2,) and (sd > 0).all()
