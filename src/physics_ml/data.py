"""Controlled pilot tasks (P4). Splits are separated by generation seed (whole realizations).

Task "resonator" (regression, phase-sensitive): the input is a sparse stream of pulses and
occasional chirp bursts (channel 0) plus an independent distractor channel (channel 1) that the
target ignores. The target is the response of a fixed hidden bank of damped resonators driven by
channel 0, delayed by `delay` steps. Long quiet intervals and unpredictable event times.
Shift condition: chirp frequencies drawn from a band excluded from training, and sparser events.

Task "recall" (classification, delayed information with distractors): a marked one-hot symbol
appears early; unmarked distractor symbols follow at random; at the final step a query flag is
raised and the class of the marked symbol must be output. Shift condition: longer delay.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPLIT_SEEDS = {"train": 1001, "val": 2002, "test": 3003, "shift": 4004}


@dataclass
class Split:
    x: np.ndarray  # [N, T, d_in] float32
    dt: np.ndarray  # [N, T] float32
    y: np.ndarray  # resonator: [N, T, p] float32; recall: [N] int64
    mask: np.ndarray  # [N, T] bool: steps where the loss/metric is evaluated
    meta: dict


# ---------------------------------------------------------------- resonator task
RES_OMEGA = np.array([0.35, 0.9, 1.7])  # hidden resonator frequencies (rad / step)
RES_GAMMA = np.array([0.02, 0.05, 0.03])  # hidden damping (1 / step)


def _resonator_inputs(rng, T, event_rate, chirp_band, chirp_prob):
    x = np.zeros((T, 2), np.float64)
    ev = rng.random(T) < event_rate
    x[ev, 0] = rng.choice([-1.0, 1.0], ev.sum()) * rng.uniform(0.5, 1.5, ev.sum())
    t = 0
    while t < T:
        if rng.random() < chirp_prob:
            L = int(rng.integers(20, 60))
            w0, w1 = rng.uniform(*chirp_band, 2)
            tt = np.arange(min(L, T - t))
            phase = w0 * tt + 0.5 * (w1 - w0) / max(L, 1) * tt ** 2
            x[t : t + len(tt), 0] += 0.6 * np.sin(phase) * np.hanning(len(tt) + 2)[1:-1]
            t += L + int(rng.integers(40, 160))
        else:
            t += int(rng.integers(40, 160))
    dis = rng.random(T) < event_rate
    x[dis, 1] = rng.standard_normal(dis.sum())
    return x


def _resonate(drive, delay):
    T = drive.shape[0]
    a = np.exp(-RES_GAMMA + 1j * RES_OMEGA)
    h = np.zeros(3, np.complex128)
    y = np.zeros((T, 3))
    for t in range(T):
        h = a * h + drive[t]
        y[t] = h.real
    out = np.zeros_like(y)
    out[delay:] = y[: T - delay]
    return out


def make_resonator(split: str, n: int, T: int = 512, delay: int = 10) -> Split:
    rng = np.random.default_rng(SPLIT_SEEDS[split])
    shift = split == "shift"
    band = (1.2, 2.2) if shift else (0.2, 1.1)  # training band excludes (1.2, 2.2)
    rate = 0.01 if shift else 0.02
    xs, ys = [], []
    for _ in range(n):
        x = _resonator_inputs(rng, T, rate, band, 0.25)
        xs.append(x)
        ys.append(_resonate(x[:, 0], delay))
    x = np.stack(xs).astype(np.float32)
    y = np.stack(ys).astype(np.float32)
    mask = np.ones((n, T), bool)
    mask[:, :delay] = False
    return Split(x, np.ones((n, T), np.float32), y, mask,
                 {"task": "resonator", "split": split, "seed": SPLIT_SEEDS[split], "T": T, "delay": delay,
                  "chirp_band": band, "event_rate": rate})


# ---------------------------------------------------------------- recall task
def make_recall(split: str, n: int, T: int = 512, K: int = 8) -> Split:
    """Channels: K symbol channels, marker, query."""
    rng = np.random.default_rng(SPLIT_SEEDS[split] + 17)
    shift = split == "shift"
    x = np.zeros((n, T, K + 2), np.float32)
    y = rng.integers(0, K, n)
    for i in range(n):
        t0 = int(rng.integers(0, T // 8)) if shift else int(rng.integers(T // 8, T // 2))
        x[i, t0, y[i]] = 1.0
        x[i, t0, K] = 1.0
        n_dis = int(rng.integers(3, 12))
        for td in rng.choice(np.setdiff1d(np.arange(T - 1), [t0]), n_dis, replace=False):
            x[i, td, rng.integers(0, K)] = 1.0
        x[i, T - 1, K + 1] = 1.0
    mask = np.zeros((n, T), bool)
    mask[:, -1] = True
    return Split(x, np.ones((n, T), np.float32), y.astype(np.int64), mask,
                 {"task": "recall", "split": split, "seed": SPLIT_SEEDS[split] + 17, "T": T, "K": K,
                  "delay": "long (t0 in [0, T/8))" if shift else "t0 in [T/8, T/2)"})


def make(task: str, split: str, n: int, **kw) -> Split:
    return {"resonator": make_resonator, "recall": make_recall}[task](split, n, **kw)


def train_normalization(train: Split) -> tuple[np.ndarray, np.ndarray]:
    """Input normalization from the training split only."""
    flat = train.x.reshape(-1, train.x.shape[-1])
    return flat.mean(0), flat.std(0) + 1e-6
