"""P5 policy evaluation on VALIDATION data with frozen weights.

    uv run python -m physics_ml.evaluate --config configs/policy_eval.yaml

For each policy and tolerance: write-selection fraction, realized deviation from the frozen dense
model, certified bound and its tightness, task metric change, and (for classification) the
fraction of sequences whose label agreement is certified by the margin condition
    dense top-1 margin > sqrt(2) * B_T     (||delta logits||_2 <= B_T  =>  any gap moves <= sqrt(2) B_T).
Quality uses the batched masked reference (mathematically identical masks/outputs to the genuine
selective path, tests/test_execution.py); latency uses the compiled engine (genuine omission) on
batch-one sequences against the fastest dense implementation (D005).
"""
from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from .benchmark import paired_speedup, ratio_ci, time_methods, time_paired  # noqa: E402
from .model import CERTIFIED  # noqa: E402
from .data import make  # noqa: E402
from .engine_nb import CompiledEngine  # noqa: E402
from .records import ROOT, Run, load_config  # noqa: E402
from .train import build  # noqa: E402


def load_trained(run_dir: Path):
    man = json.loads((run_dir / "manifest.json").read_text())
    cfg = man["config"]
    met = json.loads((run_dir / "metrics.json").read_text())
    task = cfg["task"]
    tr = make(task, "train", 8)
    d_in = tr.x.shape[-1]
    d_out = tr.y.shape[-1] if task == "resonator" else tr.meta["K"]
    model = build(cfg, d_in, d_out)
    model.load_state_dict(torch.load(run_dir / "best.pt"))
    model.eval()
    return model, cfg, torch.tensor(met["y_scale"]), man


def task_metric(task, pred, y, mask, y_scale):
    if task == "resonator":
        err = (pred - y) / y_scale
        return {"nrmse": float(((err ** 2 * mask[..., None]).sum() / (mask.sum() * y.shape[-1])).sqrt())}
    return {"acc": float((pred[:, -1].argmax(-1) == y).float().mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config)
    import yaml

    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    torch.set_num_threads(cfg.get("threads", 4))
    src = ROOT / cfg["model_run"]
    model, mcfg, y_scale, man = load_trained(src)
    task = mcfg["task"]
    model.freeze_bounds()
    run = Run(f"p5-eval-{task}-{mcfg['model']}" + (f"-noise{cfg['input_noise']}" if cfg.get("input_noise") else ""), cfg.get("hypothesis", "H001"),
              f"policy sweep on validation for {cfg['model_run']}", {**cfg, "model_manifest_sha": man.get("config_sha256")})
    va = make(task, "val", cfg["n_val"])
    if cfg.get("input_noise", 0.0) > 0:
        # sensor-noise condition: additive Gaussian noise on every input sample (fixed seed); the
        # dense reference sees the same noisy input, targets unchanged
        nrng = np.random.default_rng(cfg["seed"] + 5)
        va.x = (va.x + cfg["input_noise"] * nrng.standard_normal(va.x.shape)).astype(np.float32)
    x = torch.from_numpy(va.x); dt = torch.from_numpy(va.dt)
    y = torch.from_numpy(va.y); mask = torch.from_numpy(va.mask).float()
    # output units: declared tolerances are multiples of the training-target RMS (regression) or
    # logit units (classification)
    unit = float(torch.linalg.vector_norm(y_scale)) if task == "resonator" else 1.0
    with torch.no_grad():
        dense = model(x, dt, mode="dense").predictions
        # same streaming arithmetic as the policies, all writes (tau=-inf keeps every group): the
        # reference for raw bound excess, separating float32 scan-vs-loop ordering from policy error
        sdense = model(x, dt, mode="masked_ref", policy="threshold", tau=-float("inf")).predictions
    scan_vs_stream = float(torch.linalg.vector_norm(sdense - dense, dim=-1).max())
    dm = task_metric(task, dense, y, mask, y_scale)
    run.log(f"dense val {dm}; output unit {unit:.4f}; scan-vs-stream dense max dev {scan_vs_stream:.3e}")
    if task == "recall":
        top2 = dense[:, -1].topk(2, -1).values
        margin = top2[:, 0] - top2[:, 1]
    G = model.G
    rows = []
    sweeps = []
    for pol in cfg["policies"]:
        name, grid = pol["name"], pol["grid"]
        for val in grid:
            kw = {"epsilon": val * unit} if name in CERTIFIED else {"tau": val * (unit if name == "threshold" else 1.0)}
            r = model(x, dt, mode="masked_ref", policy=name, **kw)
            dev = torch.linalg.vector_norm(r.predictions - dense, dim=-1)  # [B, T]
            m = task_metric(task, r.predictions, y, mask, y_scale)
            row = {"policy": name, "param": val, "write_frac": float(r.selected_writes.float().mean() / G),
                   "dev_max": float(dev.max()) / unit, "dev_mean": float(dev.mean()) / unit, **m}
            sdev = torch.linalg.vector_norm(r.predictions - sdense, dim=-1)
            row["dev_max_vs_stream_dense"] = float(sdev.max()) / unit
            if r.bounds is not None:
                b = r.bounds
                row["stream_bound_pos_excess_count"] = int((sdev > b).sum())
                row["stream_max_raw_excess"] = float((sdev - b).max())
                row["bound_max"] = float(b.max()) / unit
                pos = b > 0
                row["tightness_median"] = float((dev[pos] / b[pos]).median()) if pos.any() else None
                row["bound_violations_raw"] = int((dev > b).sum())
                row["max_raw_excess"] = float((dev - b).max())
                if task == "recall":
                    row["margin_certified_frac"] = float((margin > math.sqrt(2) * b[:, -1]).float().mean())
            if task == "resonator":
                row["rel_rmse_increase"] = m["nrmse"] / dm["nrmse"] - 1.0
            else:
                row["acc_drop_pp"] = 100 * (dm["acc"] - m["acc"])
            rows.append(row)
            run.log(json.dumps({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in row.items()}))  # sig. figs: tiny excesses stay visible
        sweeps.append(name)
    # operating points: fewest writes meeting the preregistered quality tolerance on validation
    ok = (lambda r: r["rel_rmse_increase"] <= cfg["tol_rel_rmse"]) if task == "resonator" else (lambda r: r["acc_drop_pp"] <= cfg["tol_acc_pp"])
    ops = {}
    for name in sweeps:
        cands = [r for r in rows if r["policy"] == name and ok(r)]
        ops[name] = min(cands, key=lambda r: r["write_frac"]) if cands else None
    run.log(f"operating points: {json.dumps(ops)}")
    # latency at operating points (compiled engine, batch one, genuine omission)
    timing = {}
    if cfg.get("time_ops", True):
        eng = CompiledEngine(model)
        rng = np.random.default_rng(cfg["seed"])
        xs = [va.x[i] for i in range(cfg["n_time_seqs"])]
        dts = [va.dt[i] for i in range(cfg["n_time_seqs"])]
        methods = {"dense_fused": {"policy": "dense_fused"}}
        for name, op in ops.items():
            if op is None or name == "oracle":
                continue
            kw = {"epsilon": op["param"] * unit} if name in CERTIFIED else {"tau": op["param"] * (unit if name == "threshold" else 1.0)}
            methods[name] = {"policy": name, **kw}
        T = va.x.shape[1]
        if cfg.get("timing_protocol", "A1") == "A2":
            tm = time_paired(eng, xs, dts, methods, cfg["timing_reps"], rng)
            for k, v in tm.items():
                timing[k] = {"median_us_per_step": float(np.median(v) / T * 1e6), "protocol": "A2 paired"}
                if k != "dense_fused":
                    timing[k]["speedup_vs_dense_fused"] = paired_speedup(tm["dense_fused"], v, rng)
        else:
            tm = time_methods(eng, xs, dts, methods, cfg["timing_reps"], rng)
            for k, v in tm.items():
                timing[k] = {"median_us_per_step": float(np.median(v) / T * 1e6)}
                if k != "dense_fused":
                    timing[k]["speedup_vs_dense_fused"] = ratio_ci(tm["dense_fused"], v, rng)
        # engine/masked-ref agreement on the timed sequences (writes per step must match)
        for name in methods:
            if name == "dense_fused":
                continue
            _, _, w = eng.run(xs[0], dts[0], **methods[name])
            timing[name]["engine_write_frac_seq0"] = float(w.mean() / G)
        run.log(f"timing: {json.dumps(timing)}")
    metrics = {"dense_val": dm, "unit": unit, "scan_vs_stream_dense_max_dev": scan_vs_stream, "rows": rows, "operating_points": ops, "timing": timing}
    best = {k: v.get("speedup_vs_dense_fused", [None])[0] for k, v in timing.items() if k != "dense_fused"}
    key = f"{task}/{mcfg['model']}: dense {dm}; op write_frac " + ", ".join(
        f"{k}={ops[k]['write_frac']:.3f}" if ops.get(k) else f"{k}=none" for k in sweeps) + f"; speedups {best}"
    run.log(key)
    run.finalize("passed", metrics, key)


if __name__ == "__main__":
    main()
