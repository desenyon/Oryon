"""Compiled (numba) batch-one streaming engine: the declared timing regime from R2 on.

Same semantics as engine.StreamEngine, but the complete step (encoder, gates, coefficients,
bounds, ledger, stable selection, oscillator evolution, genuinely selected projections, readout)
is one compiled loop, so timing reflects arithmetic and memory rather than interpreter overhead.
Kept groups and the dense path use the same matvec kernel. Policies:
0 dense | 1 nowrite (ceiling only) | 2 certified | 3 threshold | 4 delta | 5 oracle (diagnostic).
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np

from .model import OscillatorSSM

POLICIES = {"dense": 0, "nowrite": 1, "certified": 2, "threshold": 3, "delta": 4, "oracle": 5}


@nb.njit(cache=True, inline="always")
def _mv(W, r0, r1, u, out):
    d = u.shape[0]
    for r in range(r0, r1):
        s = np.float32(0.0)
        for k in range(d):
            s += W[r, k] * u[k]
        out[r - r0] = s


@nb.njit(cache=True)
def _run(x, dt, mean, inv_std, E1, e1, E2, e2, Wg, bg, Wflat, Cflat, D, ob, gamma, omega, gmin, wbar, c,
         G, m, policy, epsilon, tau, y, bounds, writes):
    T, d_in = x.shape
    H, d = E1.shape[0], E2.shape[0]
    p = D.shape[0]
    M2 = 2 * m
    h = np.zeros(G * M2, np.float32)
    e = np.zeros(G, np.float32)
    z = np.empty(d_in, np.float32)
    a1 = np.empty(H, np.float32)
    u = np.empty(d, np.float32)
    g = np.empty(G, np.float32)
    buf = np.empty(M2, np.float32)
    full = np.empty(G * M2, np.float32)
    beta = np.empty(G, np.float32)
    contrib = np.empty(G, np.float32)
    order = np.empty(G, np.int64)
    om = np.zeros(G, np.bool_)
    u_cache = np.zeros(d, np.float32)
    b_cache = np.zeros(G * M2, np.float32)
    have_cache = False
    K = np.float32(math.sqrt(2.0 / math.pi))
    for t in range(T):
        for k in range(d_in):
            z[k] = (x[t, k] - mean[k]) * inv_std[k]
        _mv(E1, 0, H, z, a1)
        for k in range(H):
            v = a1[k] + e1[k]
            a1[k] = np.float32(0.5) * v * (np.float32(1.0) + math.tanh(K * (v + np.float32(0.044715) * v * v * v)))
        _mv(E2, 0, d, a1, u)
        unorm2 = np.float32(0.0)
        for k in range(d):
            u[k] += e2[k]
            unorm2 += u[k] * u[k]
        unorm = math.sqrt(unorm2)
        _mv(Wg, 0, G, u, g)
        for i in range(G):
            g[i] = np.float32(1.0) / (np.float32(1.0) + math.exp(-(g[i] + bg[i])))
        dtt = dt[t]
        # homogeneous dynamics for every group
        for i in range(G):
            for j in range(m):
                mag = math.exp(-gamma[i, j] * dtt)
                ang = omega[i, j] * dtt
                ar, ai = mag * math.cos(ang), mag * math.sin(ang)
                q = i * M2 + 2 * j
                hr, hi = h[q], h[q + 1]
                h[q] = ar * hr - ai * hi
                h[q + 1] = ar * hi + ai * hr
        nw = 0
        if policy == 0:
            for i in range(G):
                _mv(Wflat, i * M2, (i + 1) * M2, u, buf)
                for k in range(M2):
                    h[i * M2 + k] += buf[k] * g[i]
            nw = G
        elif policy == 2 or policy == 5:
            if policy == 5:
                for i in range(G):
                    _mv(Wflat, i * M2, (i + 1) * M2, u, buf)
                    s = np.float32(0.0)
                    for k in range(M2):
                        full[i * M2 + k] = buf[k] * g[i]
                        s += full[i * M2 + k] * full[i * M2 + k]
                    beta[i] = math.sqrt(s)
            else:
                for i in range(G):
                    beta[i] = wbar[i] * g[i] * unorm
            base = np.float32(0.0)
            for i in range(G):
                e[i] = math.exp(-gmin[i] * dtt) * e[i]
                base += c[i] * e[i]
                contrib[i] = c[i] * beta[i]
                order[i] = i
            # stable insertion sort by contrib (G is small)
            for a in range(1, G):
                key = order[a]
                b = a - 1
                while b >= 0 and contrib[order[b]] > contrib[key]:
                    order[b + 1] = order[b]
                    b -= 1
                order[b + 1] = key
            room = epsilon - base
            if room < 0:
                room = 0.0
            kk = 0
            cs = 0.0
            while kk < G and cs + contrib[order[kk]] <= room:
                cs += contrib[order[kk]]
                kk += 1
            while kk > 0 and cs > 0 and base + cs > epsilon:  # deterministic final check
                kk -= 1
                cs -= contrib[order[kk]]
            for i in range(G):
                om[i] = False
            for a in range(kk):
                om[order[a]] = True
            bsum = np.float32(0.0)
            for i in range(G):
                if om[i]:
                    e[i] += beta[i]
                else:
                    if policy == 2:
                        _mv(Wflat, i * M2, (i + 1) * M2, u, buf)
                        for k in range(M2):
                            h[i * M2 + k] += buf[k] * g[i]
                    else:
                        for k in range(M2):
                            h[i * M2 + k] += full[i * M2 + k]
                    nw += 1
                bsum += c[i] * e[i]
            bounds[t] = bsum
        elif policy == 3:
            for i in range(G):
                if c[i] * wbar[i] * g[i] * unorm >= tau:
                    _mv(Wflat, i * M2, (i + 1) * M2, u, buf)
                    for k in range(M2):
                        h[i * M2 + k] += buf[k] * g[i]
                    nw += 1
        elif policy == 4:
            dist2 = np.float32(0.0)
            for k in range(d):
                dist2 += (u[k] - u_cache[k]) ** 2
            if (not have_cache) or math.sqrt(dist2) > tau:
                for i in range(G):
                    _mv(Wflat, i * M2, (i + 1) * M2, u, buf)
                    for k in range(M2):
                        b_cache[i * M2 + k] = buf[k]
                for k in range(d):
                    u_cache[k] = u[k]
                have_cache = True
                nw = G
            for i in range(G):
                for k in range(M2):
                    h[i * M2 + k] += b_cache[i * M2 + k] * g[i]
        writes[t] = nw
        for r in range(p):
            s = ob[r]
            for k in range(G * M2):
                s += Cflat[r, k] * h[k]
            for k in range(d):
                s += D[r, k] * u[k]
            y[t, r] = s


class CompiledEngine:
    def __init__(self, model: OscillatorSSM):
        f = lambda t: np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32))
        cache = model.bounds()
        self.G, self.m, self.p = model.G, model.m, model.p
        self.args = (
            f(model.x_mean), f(1.0 / model.x_std), f(model.enc1.weight), f(model.enc1.bias), f(model.enc2.weight),
            f(model.enc2.bias), f(model.gate.weight), f(model.gate.bias), f(model.W.reshape(model.G * 2 * model.m, -1)),
            f(model.C.permute(1, 0, 2).reshape(model.p, -1)), f(model.D), f(model.out_bias), f(model.gamma),
            f(model.omega), f(model.gamma.amin(-1)), f(cache.wbar), f(cache.c),
        )

    def run(self, x: np.ndarray, dt: np.ndarray, policy: str = "dense", epsilon: float = 0.0, tau: float = 0.0):
        x = np.ascontiguousarray(x, np.float32)
        dt = np.ascontiguousarray(dt, np.float32)
        T = x.shape[0]
        y = np.empty((T, self.p), np.float32)
        bounds = np.zeros(T, np.float32)
        writes = np.empty(T, np.int32)
        _run(x, dt, *self.args, self.G, self.m, POLICIES[policy], float(epsilon), float(tau), y, bounds, writes)
        return y, (bounds if policy in ("certified", "oracle") else None), writes
