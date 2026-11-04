"""Low-overhead NumPy float32 batch-one streaming engine for timing (P3/P5).

Executes the COMPLETE model each step: normalization, encoder, gates, transition coefficients,
pre-write bounds, ledger, selection, oscillator evolution, genuinely selected projections, and
readout. Omitted projections are never computed: kept groups use per-group views W[i] of a
contiguous array (no gather copy). Correctness vs OscillatorSSM: tests/test_engine.py.

Policies: dense | certified | oracle (diagnostic: computes all writes) | threshold | delta |
nowrite (ceiling only: removes all projection work; not a valid model).
"""
from __future__ import annotations

import math

import numpy as np

from .model import OscillatorSSM

_K = math.sqrt(2.0 / math.pi)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(_K * (x + 0.044715 * x * x * x)))


class StreamEngine:
    def __init__(self, model: OscillatorSSM, dtype=np.float32):
        f = lambda t: np.ascontiguousarray(t.detach().cpu().numpy().astype(dtype))
        cache = model.bounds()
        self.dtype = dtype
        self.mean, self.inv_std = f(model.x_mean), f(1.0 / model.x_std)
        self.E1, self.e1 = f(model.enc1.weight), f(model.enc1.bias)
        self.E2, self.e2 = f(model.enc2.weight), f(model.enc2.bias)
        self.Wg, self.bg = f(model.gate.weight), f(model.gate.bias)
        self.G, self.m, self.p = model.G, model.m, model.p
        self.W = f(model.W)  # [G, 2m, d]
        self.Wflat = self.W.reshape(self.G * 2 * self.m, -1)
        self.Cflat = f(model.C.permute(1, 0, 2).reshape(model.p, -1))  # [p, G*2m]
        self.D, self.ob = f(model.D), f(model.out_bias)
        self.gamma, self.omega = f(model.gamma), f(model.omega)
        self.gmin = self.gamma.min(-1)
        self.wbar, self.c = f(cache.wbar), f(cache.c)

    def run(self, x: np.ndarray, dt: np.ndarray, policy: str = "dense", epsilon: float = 0.0, tau: float = 0.0):
        """x: [T, d_in] float32, dt: [T]. Returns y [T, p], bounds [T] or None, writes [T]."""
        T = x.shape[0]
        G, m = self.G, self.m
        h = np.zeros((G, m, 2), self.dtype)
        e = np.zeros(G, self.dtype)
        y = np.empty((T, self.p), self.dtype)
        writes = np.empty(T, np.int32)
        bounds = np.empty(T, self.dtype) if policy in ("certified", "oracle") else None
        W, Wflat, E1, E2, Wg, c, wbar = self.W, self.Wflat, self.E1, self.E2, self.Wg, self.c, self.wbar
        u_cache = b_cache = None
        for t in range(T):
            u = E2 @ gelu_tanh(E1 @ ((x[t] - self.mean) * self.inv_std) + self.e1) + self.e2
            g = 1.0 / (1.0 + np.exp(-(Wg @ u + self.bg)))
            d = dt[t]
            mag = np.exp(-self.gamma * d)
            ang = self.omega * d
            ar, ai = mag * np.cos(ang), mag * np.sin(ang)
            hr, hi = h[..., 0], h[..., 1]
            h = np.stack([ar * hr - ai * hi, ar * hi + ai * hr], -1)
            if policy == "dense":
                h += (Wflat @ u).reshape(G, m, 2) * g[:, None, None]
                writes[t] = G
            elif policy == "nowrite":
                writes[t] = 0
            elif policy in ("certified", "oracle"):
                rho = np.exp(-self.gmin * d)
                if policy == "certified":
                    beta = wbar * g * np.sqrt(u @ u)  # g in (0,1): |g| = g
                else:
                    full = (Wflat @ u).reshape(G, m, 2) * g[:, None, None]
                    beta = np.sqrt((full * full).sum((1, 2)))
                contrib = c * beta
                er = rho * e
                base = float(c @ er)
                room = max(epsilon - base, 0.0)
                order = np.argsort(contrib, kind="stable")
                cs = np.cumsum(contrib[order])
                k = int(np.searchsorted(cs, room, side="right"))
                while k > 0 and cs[k - 1] > 0 and base + float(contrib[order[:k]].sum()) > epsilon:
                    k -= 1
                om = np.zeros(G, bool)
                om[order[:k]] = True
                e = er + om * beta
                if policy == "certified":
                    for i in order[k:]:
                        h[i] += (W[i] @ u).reshape(m, 2) * g[i]
                else:
                    h += full * (~om)[:, None, None]
                writes[t] = G - k
                bounds[t] = float(c @ e)
            elif policy == "threshold":
                contrib = c * wbar * g * np.sqrt(u @ u)
                n = 0
                for i in np.flatnonzero(contrib >= tau):
                    h[i] += (W[i] @ u).reshape(m, 2) * g[i]
                    n += 1
                writes[t] = n
            elif policy == "delta":
                if u_cache is None or float(np.sqrt(((u - u_cache) ** 2).sum())) > tau:
                    b_cache = (Wflat @ u).reshape(G, m, 2)
                    u_cache = u.copy()
                    writes[t] = G
                else:
                    writes[t] = 0
                h += b_cache * g[:, None, None]
            else:
                raise ValueError(policy)
            y[t] = self.Cflat @ h.reshape(-1) + self.D @ u + self.ob
        return y, bounds, writes
