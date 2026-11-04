"""Aggregate pilot runs across independent training seeds (C8 and C7), validation metric only.

    uv run python -m physics_ml.aggregate

Independent seeds are the unit of uncertainty (time steps are never treated as samples).
Reports per-seed values, mean, sd, and a two-sided 95% t-interval for the per-seed difference
(osc - zerofreq), paired by seed (same data, same seed index, independent training).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .records import ROOT, Run

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}  # t_{0.975, n-1} keyed by n


def collect() -> dict:
    out: dict = {}
    for d in sorted((ROOT / "runs").glob("p4-*-s*-*")):
        man = json.loads((d / "manifest.json").read_text())
        cfg = man["config"]
        if man["status"] != "passed" or cfg.get("n_train") != 4000 or cfg.get("steps") != 1500:
            continue  # smoke/interrupted runs excluded
        met = json.loads((d / "metrics.json").read_text())
        key = met["best_val"].get("nrmse", met["best_val"].get("acc"))
        out.setdefault(cfg["task"], {}).setdefault(cfg["model"], {})[cfg["seed"]] = {
            "val": key, "test": (met.get("test") or {}), "shift": (met.get("shift") or {}), "run": d.name}
    return out


def ci_diff(a: dict, b: dict) -> dict:
    seeds = sorted(set(a) & set(b))
    diff = np.array([a[s]["val"] - b[s]["val"] for s in seeds])
    n = len(diff)
    res = {"seeds": seeds, "diff_per_seed": diff.tolist(), "mean": float(diff.mean()) if n else None}
    if n >= 2:
        sd = float(diff.std(ddof=1))
        h = T95[n] * sd / np.sqrt(n)
        res.update(sd=sd, ci95=[float(diff.mean() - h), float(diff.mean() + h)])
    return res


def main() -> None:
    data = collect()
    run = Run("p4-aggregate", "H001", "aggregate pilot seeds: C7 baseline and C8 oscillator vs zero-frequency", {"source": "runs/p4-*"})
    summary = {}
    for task, models in data.items():
        s = {}
        for m, seeds in models.items():
            v = np.array([x["val"] for x in seeds.values()])
            s[m] = {"per_seed": {k: x["val"] for k, x in seeds.items()}, "mean": float(v.mean()),
                    "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "n": len(v),
                    "runs": [x["run"] for x in seeds.values()]}
        if "osc" in models and "zerofreq" in models:
            s["osc_minus_zerofreq"] = ci_diff(models["osc"], models["zerofreq"])
        if "osc" in models and "gru" in models:
            s["osc_minus_gru"] = ci_diff(models["osc"], models["gru"])
        summary[task] = s
        run.log(f"{task}: " + json.dumps({m: (round(x['mean'], 4), x['sd'] and round(x['sd'], 4), x['n']) for m, x in s.items() if 'n' in x})
                + f" | osc-zf {json.dumps(s.get('osc_minus_zerofreq'))}")
    key = "; ".join(f"{t}: " + ", ".join(f"{m} {x['mean']:.4f}±{(x['sd'] or 0):.4f} (n={x['n']})" for m, x in s.items() if "n" in x) for t, s in summary.items())  # model entries only (diff dicts have "mean" but no "n")
    run.finalize("passed", summary, key)


if __name__ == "__main__":
    main()
