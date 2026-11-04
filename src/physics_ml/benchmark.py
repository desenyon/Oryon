"""P3 profiling: dense write fraction f, ideal ceiling 1/(1-f), and controller overhead.

Declared timing regime (SPEC/plan §8): batch-one streaming, float32, single-threaded CPU
(Accelerate/BLAS threads = 1), NumPy StreamEngine, complete model per step.

    uv run python -m physics_ml.benchmark --config configs/benchmark.yaml
"""
from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from .engine import StreamEngine  # noqa: E402
from .engine_nb import CompiledEngine  # noqa: E402
from .model import OscillatorSSM  # noqa: E402
from .records import Run, load_config  # noqa: E402


def time_methods(eng: StreamEngine, xs, dts, methods: dict, reps: int, rng) -> dict:
    """methods: name -> kwargs for eng.run. Returns name -> array of per-sequence seconds."""
    out = {k: [] for k in methods}
    names = list(methods)
    for _ in range(2):  # warmup
        for k in names:
            eng.run(xs[0], dts[0], **methods[k])
    for r in range(reps):
        rng.shuffle(names)  # randomized method order
        x, dt = xs[r % len(xs)], dts[r % len(dts)]
        for k in names:
            t0 = time.perf_counter()
            eng.run(x, dt, **methods[k])
            out[k].append(time.perf_counter() - t0)
    return {k: np.array(v) for k, v in out.items()}


def ratio_ci(num: np.ndarray, den: np.ndarray, rng, n_boot: int = 2000) -> tuple[float, float, float]:
    """Median(num)/median(den) with a percentile bootstrap over sequences."""
    est = float(np.median(num) / np.median(den))
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(num), len(num))
        j = rng.integers(0, len(den), len(den))
        boots.append(np.median(num[i]) / np.median(den[j]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi)


def bursty_inputs(rng, T: int, d_in: int, active: float):
    x = rng.standard_normal((T, d_in)).astype(np.float32) * (rng.random((T, 1)) < active)
    dt = np.full(T, 0.1, np.float32)
    return x, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(1)
    run = Run(f"p3-bench-{cfg.get('engine', 'numpy')}", "H001", f"profile write fraction f, ceiling, controller overhead; engine={cfg.get('engine', 'numpy')} (untrained weights)", cfg)
    rng = np.random.default_rng(cfg["seed"])
    results = []
    for shp in cfg["shapes"]:
        model = OscillatorSSM(d_in=cfg["d_in"], d_out=cfg["d_out"], d_model=shp["d_model"], d_hidden=shp["d_hidden"],
                              groups=shp["groups"], modes=shp["modes"], seed=0).eval()
        model.freeze_bounds()
        eng = CompiledEngine(model) if cfg.get("engine", "numpy") == "compiled" else StreamEngine(model)
        data = [bursty_inputs(rng, cfg["steps"], cfg["d_in"], cfg["active_frac"]) for _ in range(4)]
        xs, dts = [d[0] for d in data], [d[1] for d in data]
        methods = {
            "dense": {"policy": "dense_fused"} if cfg.get("engine") == "compiled" else {"policy": "dense"},
            "dense_pergroup": {"policy": "dense"},
            "nowrite": {"policy": "nowrite"},
            "cert_all_write": {"policy": "certified", "epsilon": 0.0},
            "cert_all_omit": {"policy": "certified", "epsilon": 1e30},
        }
        tm = time_methods(eng, xs, dts, methods, cfg["reps"], rng)
        G, m, d, hdn = shp["groups"], shp["modes"], shp["d_model"], shp["d_hidden"]
        macs = {"write": G * 2 * m * d, "encoder": cfg["d_in"] * hdn + hdn * d, "gate": G * d,
                "readout": cfg["d_out"] * (G * 2 * m + d)}
        f = 1.0 - np.median(tm["nowrite"]) / np.median(tm["dense"])
        ceil = ratio_ci(tm["dense"], tm["nowrite"], rng)
        best = ratio_ci(tm["dense"], tm["cert_all_omit"], rng)
        worst = ratio_ci(tm["dense"], tm["cert_all_write"], rng)
        row = {
            **shp, "step_us_dense": float(np.median(tm["dense"]) / cfg["steps"] * 1e6),
            "step_us_nowrite": float(np.median(tm["nowrite"]) / cfg["steps"] * 1e6),
            "step_us_cert_all_omit": float(np.median(tm["cert_all_omit"]) / cfg["steps"] * 1e6),
            "step_us_cert_all_write": float(np.median(tm["cert_all_write"]) / cfg["steps"] * 1e6),
            "write_fraction_f": float(f), "ceiling_speedup": ceil, "cert_all_omit_speedup": best,
            "cert_all_write_speedup": worst, "macs": macs,
            "mac_write_fraction": macs["write"] / sum(macs.values()),
        }
        results.append(row)
        run.log(f"shape {shp}: dense {row['step_us_dense']:.1f}us/step f={f:.3f} ceiling {ceil[0]:.2f}x "
                f"[{ceil[1]:.2f},{ceil[2]:.2f}] cert-all-omit {best[0]:.2f}x cert-all-write {worst[0]:.2f}x "
                f"MAC write frac {row['mac_write_fraction']:.2f}")
    best_shape = max(results, key=lambda r: r["cert_all_omit_speedup"][0])
    key = (f"max cert-all-omit speedup {best_shape['cert_all_omit_speedup'][0]:.2f}x at d_model={best_shape['d_model']} "
           f"modes={best_shape['modes']} (f={best_shape['write_fraction_f']:.2f}); "
           f"min cert-all-write {min(r['cert_all_write_speedup'][0] for r in results):.2f}x")
    run.log(key)
    run.finalize("passed", {"shapes": results}, key)


if __name__ == "__main__":
    main()


def time_paired(eng, xs, dts, methods: dict, reps: int, rng) -> dict:
    """Protocol A2: every DISTINCT sequence is timed under every method `reps` times in randomized
    order; returns name -> per-sequence median seconds (array over sequences)."""
    names = list(methods)
    for k in names:  # warmup
        eng.run(xs[0], dts[0], **methods[k])
    per = {k: np.empty((len(xs), reps)) for k in names}
    for r in range(reps):
        for i in range(len(xs)):
            rng.shuffle(names)
            for k in names:
                t0 = time.perf_counter()
                eng.run(xs[i], dts[i], **methods[k])
                per[k][i, r] = time.perf_counter() - t0
    return {k: np.median(v, axis=1) for k, v in per.items()}


def paired_speedup(den: np.ndarray, num: np.ndarray, rng, n_boot: int = 4000) -> tuple[float, float, float]:
    """Total-time ratio sum(den)/sum(num) with a paired bootstrap over sequences."""
    est = float(den.sum() / num.sum())
    idx = rng.integers(0, len(den), (n_boot, len(den)))
    boots = den[idx].sum(1) / num[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi)
