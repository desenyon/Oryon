"""P4 dense training with ordinary task loss (no compute-aware loss).

    uv run python -m physics_ml.train --config configs/pilot.yaml [--set model=zerofreq seed=1]
    uv run python -m physics_ml.train --resume runs/<run_id>     # continue an interrupted run

Models: osc (oscillator SSM), zerofreq (independently trained, omega fixed to 0), gru (baseline).
Model selection by validation metric only; test/shift splits are evaluated once at the end and
recorded but must not be used for design decisions.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import make, train_normalization
from .model import OscillatorSSM
from .records import Run, load_config, sha256_file


class GRUBaseline(nn.Module):
    """Simple baseline: normalized input -> GRU -> linear readout."""

    def __init__(self, d_in: int, d_out: int, hidden: int):
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(d_in))
        self.register_buffer("x_std", torch.ones(d_in))
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.out = nn.Linear(hidden, d_out)

    def set_normalization(self, mean, std):
        self.x_mean.copy_(mean)
        self.x_std.copy_(std.clamp_min(1e-6))

    def predict(self, x, dt):
        z = (x - self.x_mean) / self.x_std
        return self.out(self.gru(z)[0])


def build(cfg: dict, d_in: int, d_out: int) -> nn.Module:
    kind = cfg["model"]
    mc = cfg["model_cfg"]
    if kind in ("osc", "zerofreq"):
        return OscillatorSSM(d_in, d_out, d_model=mc["d_model"], d_hidden=mc["d_hidden"], groups=mc["groups"],
                             modes=mc["modes"], zero_frequency=(kind == "zerofreq"), omega_max=mc["omega_max"],
                             gamma_init=tuple(mc["gamma_init"]), seed=cfg["seed"])
    if kind == "gru":
        return GRUBaseline(d_in, d_out, cfg["gru_hidden"])
    raise ValueError(kind)


def predict(model, x, dt):
    if isinstance(model, OscillatorSSM):
        return model(x, dt, mode="dense").predictions
    return model.predict(x, dt)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def loss_and_metric(task, pred, y, mask, y_scale):
    if task == "resonator":
        err = (pred - y) / y_scale
        mse = (err ** 2 * mask[..., None]).sum() / (mask.sum() * y.shape[-1])
        return mse, {"nrmse": float(mse.detach().sqrt())}
    logits = pred[:, -1]
    ce = F.cross_entropy(logits, y)
    acc = float((logits.argmax(-1) == y).float().mean())
    return ce, {"acc": acc, "ce": float(ce.detach())}


@torch.no_grad()
def evaluate_split(model, split, task, y_scale, bs=100):
    model.eval()
    tot, n, sq, cnt, correct = 0.0, 0, 0.0, 0, 0
    for i in range(0, len(split.x), bs):
        x = torch.from_numpy(split.x[i:i + bs]); dt = torch.from_numpy(split.dt[i:i + bs])
        y = torch.from_numpy(split.y[i:i + bs]); mask = torch.from_numpy(split.mask[i:i + bs]).float()
        pred = predict(model, x, dt)
        if task == "resonator":
            err = (pred - y) / y_scale
            sq += float((err ** 2 * mask[..., None]).sum()); cnt += float(mask.sum()) * y.shape[-1]
        else:
            correct += int((pred[:, -1].argmax(-1) == y).sum()); n += len(y)
            tot += float(F.cross_entropy(pred[:, -1], y, reduction="sum"))
    model.train()
    if task == "resonator":
        return {"nrmse": math.sqrt(sq / cnt)}
    return {"acc": correct / n, "ce": tot / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--resume")
    args = ap.parse_args()
    if args.resume:
        rdir = Path(args.resume)
        cfg = json.loads((rdir / "manifest.json").read_text())["config"]
    else:
        cfg = load_config(args.config)
        for kv in args.set:
            k, v = kv.split("=", 1)
            cfg[k] = type(cfg[k])(v) if k in cfg and not isinstance(cfg[k], bool) else v
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg.get("threads", 4))
    task = cfg["task"]
    tr = make(task, "train", cfg["n_train"]); va = make(task, "val", cfg["n_val"])
    d_in = tr.x.shape[-1]
    d_out = tr.y.shape[-1] if task == "resonator" else tr.meta["K"]
    mean, std = train_normalization(tr)
    y_scale = torch.tensor(tr.y[tr.mask].std(0) if task == "resonator" else 1.0, dtype=torch.float32)
    model = build(cfg, d_in, d_out)
    model.set_normalization(torch.from_numpy(mean), torch.from_numpy(std))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"], pct_start=0.05)
    if args.resume:
        run = Run.reopen(rdir)
        ck = torch.load(rdir / "last.pt", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step0, best, hist = ck["step"], ck["best"], ck["hist"]
        torch.set_rng_state(ck["rng"])
        run.log(f"resumed at step {step0}")
    else:
        run = Run(f"p4-{task}-{cfg['model']}-s{cfg['seed']}", "H001", f"dense pilot training: {task} / {cfg['model']}", cfg)
        step0, best, hist = 0, None, []
    run.log(f"params {n_params(model)}; d_in {d_in}; d_out {d_out}; train {len(tr.x)} val {len(va.x)}")
    key_metric = "nrmse" if task == "resonator" else "acc"
    better = (lambda a, b: a < b) if task == "resonator" else (lambda a, b: a > b)
    rng = np.random.default_rng(cfg["seed"] + 99)
    t_start = time.time()
    model.train()
    status = "passed"
    for step in range(step0, cfg["steps"]):
        idx = rng.integers(0, len(tr.x), cfg["batch_size"])
        x = torch.from_numpy(tr.x[idx]); dt = torch.from_numpy(tr.dt[idx])
        y = torch.from_numpy(tr.y[idx]); mask = torch.from_numpy(tr.mask[idx]).float()
        loss, _ = loss_and_metric(task, predict(model, x, dt), y, mask, y_scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        opt.step(); sched.step()
        if not math.isfinite(float(loss)):
            status = "failed"; run.log(f"nonfinite loss at step {step}"); break
        last = step + 1 == cfg["steps"]
        if (step + 1) % cfg["eval_every"] == 0 or last:
            m = evaluate_split(model, va, task, y_scale)
            hist.append({"step": step + 1, "train_loss": float(loss), **{f"val_{k}": v for k, v in m.items()},
                         "elapsed_s": time.time() - t_start})
            if best is None or better(m[key_metric], best[key_metric]):
                best = {**m, "step": step + 1}
                torch.save(model.state_dict(), run.dir / "best.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step + 1, "best": best, "hist": hist, "rng": torch.get_rng_state()}, run.dir / "last.pt")
            run.log(f"step {step + 1} loss {float(loss):.4f} val {m} best {best}")
        if (time.time() - t_start) / 60 > cfg["max_minutes"]:
            status = "interrupted"; run.log("time limit reached; resume with --resume"); break
    model.load_state_dict(torch.load(run.dir / "best.pt"))
    final = {"best_val": best, "params": n_params(model), "history": hist}
    if status == "passed":
        te = make(task, "test", cfg["n_val"]); sh = make(task, "shift", cfg["n_val"])
        final["test"] = evaluate_split(model, te, task, y_scale)  # recorded once; not used for design
        final["shift"] = evaluate_split(model, sh, task, y_scale)
    final["y_scale"] = y_scale.tolist()
    final["best_ckpt_sha256"] = sha256_file(run.dir / "best.pt")
    key = f"{task}/{cfg['model']}/s{cfg['seed']}: best val {key_metric} {best[key_metric]:.4f} @ {best['step']}; params {n_params(model)}"
    run.log(key)
    run.finalize(status, final, key)


if __name__ == "__main__":
    main()
