"""Compiled (numba) batch-one streaming engine: the declared timing regime from R2 on.

Same semantics as engine.StreamEngine, but the complete step (encoder, gates, coefficients,
bounds, ledger, stable selection, oscillator evolution, genuinely selected projections, readout)
is one compiled loop, so timing reflects arithmetic and memory rather than interpreter overhead.
Kept groups and the dense path use the same Accelerate sgemv kernel (per-group calls). Policies:
0 dense | 1 nowrite (ceiling only) | 2 certified | 3 threshold | 4 delta | 5 oracle (diagnostic) |
7 cdelta (certified delta baseline) | 9 certified_pm, 10 cdelta_pm (H001b per-mode ledger) | 6 dense_fused (one sgemv over all groups: the fastest dense reference, used as speedup denominator).
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np

from .model import OscillatorSSM

POLICIES = {"dense": 0, "nowrite": 1, "certified": 2, "threshold": 3, "delta": 4, "oracle": 5, "dense_fused": 6, "cdelta": 7, "certified_pm": 9, "cdelta_pm": 10}


import ctypes  # noqa: E402

# Accelerate cblas_sgemv (single-threaded via VECLIB_MAXIMUM_THREADS=1 in the benchmark), called
# directly from compiled code so dense and selective projections use the same optimized kernel.
_ACC = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
_sgemv = _ACC.cblas_sgemv
_sgemv.restype = None
_sgemv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_int,
                   ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_int]


@nb.njit(inline="always")
def _mv(W, r0, r1, u, out):
    """out[:r1-r0] = W[r0:r1] @ u (row-major float32, contiguous)."""
    n = W.shape[1]
    _sgemv(101, 111, r1 - r0, n, np.float32(1.0), W.ctypes.data + r0 * n * 4, n, u.ctypes.data, 1,
           np.float32(0.0), out.ctypes.data, 1)


@nb.njit
def _run(x, dt, mean, inv_std, E1, e1, E2, e2, Wg, bg, Wflat, Cflat, D, ob, gamma, omega, gmin, wbar, c, wmode, c_spec, c_mode,
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
    uh = np.zeros((G, d), np.float32)
    MAG = np.empty(G * m, np.float32)
    em = np.zeros((G, m), np.float32)
    bm = np.zeros((G, m), np.float32)
    nrm = np.zeros(G, np.float32)
    AR = np.empty(G * m, np.float32)
    AI = np.empty(G * m, np.float32)
    rho_c = np.empty(G, np.float32)
    last_dt = np.float32(0.0)
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
        # transition coefficients, recomputed only when dt changes (shared by every policy)
        if t == 0 or dtt != last_dt:
            for i in range(G):
                rho_c[i] = math.exp(-gmin[i] * dtt)
                for j in range(m):
                    mag = math.exp(-gamma[i, j] * dtt)
                    MAG[i * m + j] = mag
                    ang = omega[i, j] * dtt
                    AR[i * m + j] = mag * math.cos(ang)
                    AI[i * m + j] = mag * math.sin(ang)
            last_dt = dtt
        # homogeneous dynamics for every group
        for q2 in range(G * m):
            ar, ai = AR[q2], AI[q2]
            q = 2 * q2
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
        elif policy == 6:
            _mv(Wflat, 0, G * M2, u, full)
            for i in range(G):
                for k in range(M2):
                    h[i * M2 + k] += full[i * M2 + k] * g[i]
            nw = G
        elif policy == 2 or policy == 5 or policy == 7 or policy == 9 or policy == 10:
            pm = policy >= 9
            cd = policy == 7 or policy == 10
            if policy == 5:
                for i in range(G):
                    _mv(Wflat, i * M2, (i + 1) * M2, u, buf)
                    s = np.float32(0.0)
                    for k in range(M2):
                        full[i * M2 + k] = buf[k] * g[i]
                        s += full[i * M2 + k] * full[i * M2 + k]
                    beta[i] = math.sqrt(s)
            elif cd:
                for i in range(G):
                    s2 = np.float32(0.0)
                    for k in range(d):
                        dd = u[k] - uh[i, k]
                        s2 += dd * dd
                    nrm[i] = math.sqrt(s2) if t > 0 else np.float32(np.inf)
                    beta[i] = wbar[i] * g[i] * nrm[i]
            else:
                for i in range(G):
                    beta[i] = wbar[i] * g[i] * unorm
            base = np.float32(0.0)
            if pm:
                for i in range(G):
                    ni = nrm[i] if cd else unorm
                    s_old = 0.0
                    l_old = 0.0
                    s_new = 0.0
                    l_new = 0.0
                    for j in range(m):
                        dec = MAG[i * m + j] * em[i, j]
                        em[i, j] = dec
                        bb_ = g[i] * wmode[i, j] * ni
                        bm[i, j] = bb_
                        s_old += dec * dec
                        l_old += c_mode[i, j] * dec
                        s_new += (dec + bb_) * (dec + bb_)
                        l_new += c_mode[i, j] * (dec + bb_)
                    b_old = min(c_spec[i] * math.sqrt(s_old), l_old)
                    b_new = min(c_spec[i] * math.sqrt(s_new), l_new)
                    base += b_old
                    dlt = b_new - b_old
                    contrib[i] = dlt if dlt > 0 else 0.0
                    order[i] = i
            else:
                for i in range(G):
                    e[i] = rho_c[i] * e[i]
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
                    if pm:
                        for j in range(m):
                            em[i, j] += bm[i, j]
                    else:
                        e[i] += beta[i]
                    if cd:
                        for k in range(M2):
                            h[i * M2 + k] += b_cache[i * M2 + k] * g[i]
                else:
                    if cd:
                        _mv(Wflat, i * M2, (i + 1) * M2, u, buf)
                        for k in range(M2):
                            b_cache[i * M2 + k] = buf[k]
                            h[i * M2 + k] += buf[k] * g[i]
                        for k in range(d):
                            uh[i, k] = u[k]
                    elif policy == 2 or policy == 9:
                        _mv(Wflat, i * M2, (i + 1) * M2, u, buf)
                        for k in range(M2):
                            h[i * M2 + k] += buf[k] * g[i]
                    else:
                        for k in range(M2):
                            h[i * M2 + k] += full[i * M2 + k]
                    nw += 1
                if pm:
                    s2 = 0.0
                    l2 = 0.0
                    for j in range(m):
                        s2 += em[i, j] * em[i, j]
                        l2 += c_mode[i, j] * em[i, j]
                    bsum += min(c_spec[i] * math.sqrt(s2), l2)
                else:
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
            f(model.omega), f(model.gamma.amin(-1)), f(cache.wbar), f(cache.c), f(cache.wmode), f(cache.c_spec),
            f(cache.c_mode),
        )

    def run(self, x: np.ndarray, dt: np.ndarray, policy: str = "dense", epsilon: float = 0.0, tau: float = 0.0):
        x = np.ascontiguousarray(x, np.float32)
        dt = np.ascontiguousarray(dt, np.float32)
        T = x.shape[0]
        y = np.empty((T, self.p), np.float32)
        bounds = np.zeros(T, np.float32)
        writes = np.empty(T, np.int32)
        _run(x, dt, *self.args, self.G, self.m, POLICIES[policy], float(epsilon), float(tau), y, bounds, writes)
        return y, (bounds if policy in ("certified", "oracle", "cdelta", "certified_pm", "cdelta_pm") else None), writes
