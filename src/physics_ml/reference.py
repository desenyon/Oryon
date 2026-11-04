"""Independent float64 reference (NumPy, complex coordinates, CPU).

Deliberately shares no code with bounds.py / policy.py / model.py: it uses complex128 arithmetic,
explicit spectral norms, and a plain Python step loop, so agreement is evidence about the
torch real-pair implementation rather than a tautology.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class System:
    """One random system: G groups of m complex modes, input dim d, output dim p."""

    gamma: np.ndarray  # [G, m] >= 0
    omega: np.ndarray  # [G, m]
    W: np.ndarray  # [G, m, d] complex
    C: np.ndarray  # [G, p, m] complex (y = Re sum_i C_i h_i + D u)
    D: np.ndarray  # [p, d]
    gate_w: np.ndarray  # [G, d] gate = tanh(gate_w @ u + gate_b)
    gate_b: np.ndarray  # [G]

    def gates(self, u: np.ndarray) -> np.ndarray:
        return np.tanh(self.gate_w @ u + self.gate_b)

    def coeffs(self, dt: float) -> np.ndarray:
        if dt < 0:
            raise ValueError("dt must be nonnegative")
        return np.exp((-self.gamma + 1j * self.omega) * dt)

    def readout(self, h: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.real(np.einsum("gpm,gm->p", self.C, h)) + self.D @ u

    def real_W(self) -> np.ndarray:
        """Real-coordinate write maps [G, 2m, d] (rows interleaved re, im)."""
        G, m, d = self.W.shape
        out = np.empty((G, m, 2, d))
        out[:, :, 0], out[:, :, 1] = self.W.real, self.W.imag
        return out.reshape(G, 2 * m, d)

    def real_C(self) -> np.ndarray:
        """Real-coordinate readout [G, p, 2m] so that Re(C h) = C_R [Re h, Im h] interleaved."""
        G, p, m = self.C.shape
        out = np.empty((G, p, m, 2))
        out[..., 0], out[..., 1] = self.C.real, -self.C.imag
        return out.reshape(G, p, 2 * m)


def spectral_norms_real(M: np.ndarray) -> np.ndarray:
    """Exact (LAPACK SVD) spectral norms of [G, r, c] real matrices."""
    return np.linalg.norm(M, ord=2, axis=(-2, -1))


def run_dense(sys: System, u: np.ndarray, dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """u: [T, d], dt: [T]. Returns states [T, G, m] complex and outputs [T, p]."""
    T = u.shape[0]
    G, m, _ = sys.W.shape
    h = np.zeros((G, m), dtype=np.complex128)
    hs, ys = np.empty((T, G, m), np.complex128), np.empty((T, sys.C.shape[1]))
    for t in range(T):
        a = sys.coeffs(float(dt[t]))
        b = sys.gates(u[t])[:, None] * np.einsum("gmd,d->gm", sys.W, u[t])
        h = a * h + b
        hs[t], ys[t] = h, sys.readout(h, u[t])
    return hs, ys


def run_masked(sys: System, u: np.ndarray, dt: np.ndarray, omitted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate execution with a given omission mask omitted[T, G] (True = omit)."""
    T = u.shape[0]
    G, m, _ = sys.W.shape
    h = np.zeros((G, m), dtype=np.complex128)
    hs, ys = np.empty((T, G, m), np.complex128), np.empty((T, sys.C.shape[1]))
    for t in range(T):
        a = sys.coeffs(float(dt[t]))
        b = sys.gates(u[t])[:, None] * np.einsum("gmd,d->gm", sys.W, u[t])
        h = a * h + np.where(omitted[t][:, None], 0.0, b)
        hs[t], ys[t] = h, sys.readout(h, u[t])
    return hs, ys


def random_system(rng: np.random.Generator, G: int, m: int, d: int, p: int, undamped_frac: float = 0.25) -> System:
    gamma = rng.exponential(0.3, size=(G, m))
    # some fully undamped groups and some mixed-damping groups with one undamped mode
    for g in range(G):
        r = rng.random()
        if r < undamped_frac:
            gamma[g] = 0.0
        elif r < 2 * undamped_frac:
            gamma[g, rng.integers(m)] = 0.0
    omega = rng.uniform(-np.pi, np.pi, size=(G, m)) * rng.choice([0.1, 1.0, 5.0])
    scale_w = rng.choice([0.1, 1.0, 3.0])
    W = scale_w * (rng.standard_normal((G, m, d)) + 1j * rng.standard_normal((G, m, d))) / np.sqrt(2 * d)
    C = (rng.standard_normal((G, p, m)) + 1j * rng.standard_normal((G, p, m))) / np.sqrt(2 * m)
    C *= rng.choice([0.0, 0.5, 1.0, 2.0], size=(G, 1, 1), p=[0.1, 0.3, 0.3, 0.3])  # zero / unequal readouts
    D = rng.standard_normal((p, d)) / np.sqrt(d)
    gate_w = rng.standard_normal((G, d)) / np.sqrt(d)
    gate_b = rng.standard_normal(G)
    return System(gamma, omega, W, C, D, gate_w, gate_b)


def random_inputs(rng: np.random.Generator, T: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Bursty inputs with quiet stretches and variable (including zero) dt."""
    u = rng.standard_normal((T, d))
    active = rng.random(T) < rng.uniform(0.05, 0.6)
    u = u * np.where(active, 1.0, rng.choice([0.0, 0.01]))[:, None]
    dt = rng.choice([0.0, 0.05, 0.1, 0.5, 1.0], size=T, p=[0.05, 0.25, 0.4, 0.2, 0.1])
    dt = dt * rng.uniform(0.5, 1.5, size=T)
    return u, dt
