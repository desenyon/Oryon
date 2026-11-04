"""P2 algebra check: independent new check of T1/L1 on random systems.

This is a NEW check implemented from SPEC §3–4, not a reproduction of the historical review
(whose script was not supplied).

    uv run python -m physics_ml.reproduce --config configs/algebra.yaml
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from . import reference as ref
from .bounds import frobenius_bounds, output_bound, transition_coefficients, update_error, write_bound
from .policy import BudgetLedger
from .records import Run, load_config


def _torch_masked_float32(sys_: ref.System, u: np.ndarray, dt: np.ndarray, omitted: np.ndarray) -> np.ndarray:
    """Float32 real-pair torch execution (the implementation style used by the model)."""
    G, m, d = sys_.W.shape
    W = torch.tensor(sys_.real_W(), dtype=torch.float32)  # [G, 2m, d]
    gamma = torch.tensor(sys_.gamma, dtype=torch.float32)
    omega = torch.tensor(sys_.omega, dtype=torch.float32)
    gw = torch.tensor(sys_.gate_w, dtype=torch.float32)
    gb = torch.tensor(sys_.gate_b, dtype=torch.float32)
    h = torch.zeros(G, m, 2)
    out = np.empty((u.shape[0], G, m), np.complex128)
    for t in range(u.shape[0]):
        ut = torch.tensor(u[t], dtype=torch.float32)
        ar, ai, _ = transition_coefficients(gamma, omega, torch.tensor([dt[t]], dtype=torch.float32))
        ar, ai = ar[0], ai[0]
        hr = ar * h[..., 0] - ai * h[..., 1]
        hi = ar * h[..., 1] + ai * h[..., 0]
        h = torch.stack([hr, hi], -1)
        keep = ~torch.tensor(omitted[t])
        if keep.any():
            g = torch.tanh(gw[keep] @ ut + gb[keep])
            b = (g[:, None] * (W[keep] @ ut)).view(-1, m, 2)
            h[keep] = h[keep] + b
        out[t] = h[..., 0].double().numpy() + 1j * h[..., 1].double().numpy()
    return out


def check_system(rng: np.random.Generator, cfg: dict, mode: str) -> dict:
    G, m, p, T = cfg["groups"], cfg["modes"], cfg["outputs"], cfg["steps"]
    d = int(rng.choice(cfg["input_dims"]))
    s = ref.random_system(rng, G, m, d, p, cfg["undamped_frac"])
    u, dt = ref.random_inputs(rng, T, d)
    # implementation under test (torch float64)
    Wr = torch.tensor(s.real_W())
    Cr = torch.tensor(s.real_C())
    wbar, c = frobenius_bounds(Wr), frobenius_bounds(Cr)
    spec_w, spec_c = ref.spectral_norms_real(s.real_W()), ref.spectral_norms_real(s.real_C())
    assert np.all(wbar.numpy() >= spec_w - 1e-12) and np.all(c.numpy() >= spec_c - 1e-12)
    gates = torch.tensor(np.stack([s.gates(u[t]) for t in range(T)]))  # [T, G]
    beta = write_bound(gates, wbar, torch.tensor(u))  # [T, G]
    _, _, rho = transition_coefficients(torch.tensor(s.gamma), torch.tensor(s.omega), torch.tensor(dt))  # [T, G]
    contrib_scale = float((c[None] * beta).mean())
    if mode == "policy":
        eps_frac = math.exp(rng.uniform(math.log(cfg["eps_frac_min"]), math.log(cfg["eps_frac_max"])))
        eps_val = 0.0 if rng.random() < cfg["eps_zero_prob"] else eps_frac * max(contrib_scale, 1e-12)
        led = BudgetLedger(c, torch.tensor([eps_val], dtype=torch.float64), G)
        omitted, E, B = np.zeros((T, G), bool), np.zeros((T, G)), np.zeros(T)
        for t in range(T):
            om = led.step(rho[t : t + 1], beta[t : t + 1])
            omitted[t], E[t], B[t] = om[0].numpy(), led.e[0].numpy(), float(led.bound[0])
    else:  # random mask, ledger by update_error
        eps_val = None
        omitted = rng.random((T, G)) < rng.uniform(0.0, 1.0)
        e = torch.zeros(1, G, dtype=torch.float64)
        E, B = np.zeros((T, G)), np.zeros(T)
        for t in range(T):
            e = update_error(e, rho[t : t + 1], beta[t : t + 1], torch.tensor(omitted[t : t + 1]))
            E[t], B[t] = e[0].numpy(), float(output_bound(e, c)[0])
    # independent reference
    hs, ys = ref.run_dense(s, u, dt)
    ha, ya = ref.run_masked(s, u, dt, omitted)
    state_err = np.linalg.norm(hs - ha, axis=-1)  # [T, G]
    out_err = np.linalg.norm(ys - ya, axis=-1)  # [T]
    scale = max(1.0, float(np.abs(hs).max()), float(np.abs(ha).max()))
    tol = cfg["tol"] * scale
    st_ex, out_ex = state_err - E, out_err - B
    res = {
        "d": d, "scale": scale, "tol": tol, "omit_frac": float(omitted.mean()),
        "state_checks": state_err.size, "out_checks": out_err.size,
        "state_max_excess": float(st_ex.max()), "out_max_excess": float(out_ex.max()),
        "state_pos_excess_count": int((st_ex > 0).sum()), "out_pos_excess_count": int((out_ex > 0).sum()),
        "state_viol": int((st_ex > tol).sum()), "out_viol": int((out_ex > tol).sum()),
        "tightness_out_median": float(np.median(out_err[B > 0] / B[B > 0])) if (B > 0).any() else None,
        "frob_over_spec_w_mean": float(np.mean(wbar.numpy() / np.maximum(spec_w, 1e-300))),
    }
    if mode == "policy":
        bud_ex = B - eps_val
        res.update(budget_checks=T, budget_max_excess=float(bud_ex.max()), budget_viol=int((bud_ex > tol).sum()),
                   budget_pos_excess_count=int((bud_ex > 0).sum()), eps=eps_val)
    if cfg.get("float32_every") and rng.random() < 1.0 / cfg["float32_every"]:
        h32 = _torch_masked_float32(s, u, dt, omitted)
        err32 = np.linalg.norm(hs - h32, axis=-1)
        ex32 = err32 - E
        res.update(f32_state_checks=ex32.size, f32_state_max_excess=float(ex32.max()),
                   f32_state_pos_excess_count=int((ex32 > 0).sum()),
                   f32_state_max_excess_rel=float(ex32.max() / scale))
    res["_example"] = {"seed_state": None, "omitted": omitted, "E": E, "B": B} if (res["state_viol"] or res["out_viol"] or res.get("budget_viol")) else None
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.set_default_dtype(torch.float64)
    run = Run("p2-algebra", "H001", "independent new float64 check of T1/L1 bounds (not a reproduction)", cfg)
    summary = {}
    violating = []
    for mode in cfg["modes_run"]:
        rng = np.random.default_rng([cfg["seed"], 0 if mode == "policy" else 1])
        agg: dict = {}
        for k in range(cfg["n_systems"]):
            r = check_system(rng, cfg, mode)
            ex = r.pop("_example")
            if ex is not None and len(violating) < 5:
                violating.append({"mode": mode, "system": k, **{kk: v for kk, v in r.items()}})
            for key, v in r.items():
                if v is None:
                    continue
                if key.endswith("_checks") or key.endswith("_count") or key.endswith("_viol"):
                    agg[key] = agg.get(key, 0) + v
                elif "max_excess" in key:
                    agg[key] = max(agg.get(key, -math.inf), v)
                elif key in ("omit_frac", "tightness_out_median", "frob_over_spec_w_mean"):
                    agg.setdefault(key + "_list", []).append(v)
            if (k + 1) % 100 == 0:
                run.log(f"{mode}: {k + 1}/{cfg['n_systems']} systems; viol so far "
                        f"state={agg.get('state_viol', 0)} out={agg.get('out_viol', 0)} budget={agg.get('budget_viol', 0)}")
        for key in [k for k in agg if k.endswith("_list")]:
            vals = np.array(agg.pop(key))
            agg[key[:-5]] = {"mean": float(vals.mean()), "median": float(np.median(vals)), "min": float(vals.min()), "max": float(vals.max())}
        summary[mode] = agg
    run.save_json("violations.json", violating)
    total_viol = sum(v.get("state_viol", 0) + v.get("out_viol", 0) + v.get("budget_viol", 0) for v in summary.values())
    status = "passed" if total_viol == 0 else "failed"
    pol = summary.get("policy", {})
    key = (f"{status}: policy mode state/out/budget checks {pol.get('state_checks')}/{pol.get('out_checks')}/{pol.get('budget_checks')}, "
           f"violations@tol {total_viol}; max raw excess float64 state {pol.get('state_max_excess'):.3e} out {pol.get('out_max_excess'):.3e} "
           f"budget {pol.get('budget_max_excess'):.3e}; float32 state max excess {pol.get('f32_state_max_excess', float('nan')):.3e}")
    run.log(key)
    run.finalize(status, summary, key)


if __name__ == "__main__":
    main()
