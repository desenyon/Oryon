# RESEARCH_PLAN (live, authoritative)

Revision: R4 · 2026-11-04 · Active hypothesis: **H001 — REJECTED (pilot scope, D011)**; next
hypothesis pending pivot feasibility review (plan §9).

Current claim: H001's certified write omission gives no useful saving: at the preregistered
tolerance it omits ~0% of writes (0.88–0.93x speed); a plain delta baseline gives 1.36–1.38x.
Bounds T1/L1 remain valid but are 10²–10³x conservative. Oscillation benefit is task-dependent.

## Evidence status
| Item | Status | Evidence |
|---|---|---|
| T1/L1 bounds | derived; float64 locally verified (new check) | SPEC §3–4; runs/p2-algebra-20261104T171316 |
| Prior art | passed for inspected set; novelty unresolved | PRIOR_ART.md PA2 |
| Selective execution / profile | locally verified | runs/p3-bench-compiled-20261104T172121 (D005) |
| Dense pilot seed 0 | locally verified (exploratory) | runs/p4-*-s0-* (D010) |
| Policy evaluation | locally verified: H001 rejected; cdelta no advantage | runs/p5-eval-* (D011, D012), docs/RESULTS.md |
| Physics comparison C8 | unresolved (1 seed; seeds 1–2 running) | runs/p4-* |

## Tasks
| ID | Task | Status | Dep | Acceptance check | Evidence |
|---|---|---|---|---|---|
| P0–P3 | Bootstrap, prior art, algebra, execution profile | passed | — | see R2 | D001–D006 |
| P1b | Read Delta-SSM methods | blocked | — | full text read | needs user (OpenReview 0b75H3NcHp bot check) |
| P4 | Dense pilot seed 0 | passed (resonator osc; recall zerofreq) | P3 | learns task | D010 |
| P4s | Seeds 1–2 for all 6 pilot configs | active | P4 | 3-seed mean ± sd for C8 | runs/p4-*-s{1,2}-* |
| P5 | Policy sweep + diagnosis | failed (H001 rejected) | P4 | operating point meeting tolerance and 1.2x | D011, D012 |
| P5s | Re-run P5 on seed 1–2 models (robustness of D011) | pending | P4s | same conclusion on all seeds | runs/p5-eval-* |
| P6 | Real-task confirmation of H001 | retired (H001 rejected) | — | — | D011 |
| PV | Pivot feasibility review (2 directions) | active (agent) | D012 | recommendation with falsifiable hypothesis or stop | docs/prior_art/pivot_review.md |
| P7 | Results + disposition | active | P4s, P5s, PV | reproducible; reviewer challenge | docs/RESULTS.md |

## Next three actions
1. **[active]** Wait for seeds 1–2 (job below); then `scripts/run_p5.sh`-style evals on seed 1–2 models (clean, validation) to confirm D011.
2. Read docs/prior_art/pivot_review.md; decide: implement one pivot hypothesis (H002) with its own spec/baseline/budget, or record a supported negative result and stop.
3. Independent reviewer challenge of RESULTS.md (math, prior art, splits, baseline strength, timing fairness).

## Acceptance criteria
Frozen v1.0 targets: SPEC §7; amendment A1 (D005: timing regime + fastest-dense denominator).
H001 failed C5/C6 on validation; test splits recorded once by train.py and never used for design.

## Resources
Tranche 8.0 device-h. Consumed ≈ 0.68 device-h (index.csv, through P5 batch 1).
Running: seeds 1–2 ≈ 1.2 device-h. Remaining after: ≈ 6.1 device-h.

## Owned running jobs
| Job | PID / host | Started (local) | Command | cwd | Log | Checkpoints |
|---|---|---|---|---|---|---|
| P4 seeds 1–2 (12 sequential runs) | 81313 / this Mac (bash driver) | Wed Nov 4 10:08:53 2026 | `scripts/run_seeds.sh` | repo root | logs/pilot_s12.log; runs/p4-*-s{1,2}-*/log.txt | runs/p4-*-s{1,2}-*/last.pt, best.pt |
Reconcile: `ps -o pid,lstart,command -p 81313`. If dead: find the newest runs/p4-*-s{1,2}-* with status running/interrupted → `uv run python -m physics_ml.train --resume <dir>`; then run remaining (task, model, seed) via `uv run python -m physics_ml.train --config configs/pilot.yaml --set task=<t> model=<m> seed=<s>` (completed configs are listed in logs/pilot_s12.log "end" lines).
Agent (not compute): pivot review → docs/prior_art/pivot_review.md.

## Blockers
P1b needs user access. Real-task data (P6/pivot) would need a download → ask user first.

## Next command
`grep "===" logs/pilot_s12.log` (check seed job), then evaluate seed models:
`uv run python -m physics_ml.evaluate --config configs/policy_eval.yaml --set model_run=<runs/p4-resonator-osc-s1-...>`
