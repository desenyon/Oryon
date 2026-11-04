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
| P4s | Seeds 1–2 for all 6 pilot configs | passed (recall/osc/s1 hit the 25-min cap at step 1382, resumed to 1500) | P4 | 3-seed mean ± sd for C8 | runs/p4-*-s{1,2}-* |
| P5 | Policy sweep + diagnosis | failed (H001 rejected) | P4 | operating point meeting tolerance and 1.2x | D011, D012 |
| P5s | Re-run P5 on seeds 0–2 under protocol A2 (D015) | active (job below) | P4s | same conclusion on all seeds | runs/p5-eval-* |
| P6 | Real-task confirmation of H001 | retired (H001 rejected) | — | — | D011 |
| PV | Pivot feasibility review (2 directions) | passed (pivot declined, D014) | D012 | recommendation with falsifiable hypothesis or stop | docs/prior_art/pivot_review.md |
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
| Job | PID / host | Started (UTC) | Command | cwd | Log | Outputs |
|---|---|---|---|---|---|---|
| P5s A2 evals (9 sequential evals + aggregate) | bash driver / this Mac | 2026-11-04 ≈19:45Z | `scripts/run_p5s.sh > logs/p5s_A2.log 2>&1` | repo root | logs/p5s_A2.log | runs/p5-eval-*-2026110[4]T19[4-5]*, runs/p4-aggregate-* |
Reconcile: `pgrep -fl run_p5s.sh`. Evals are stateless (no checkpoint): if dead, rerun the whole script with a fresh log; partial eval dirs from the dead attempt stay in runs/ and are marked superseded in index.csv.
Note: runs/p5-eval-resonator-osc-20261104T1942* is an A2 smoke test (n_val=16, 4 timing seqs); not evidence.
Note: logs/pilot_s12.log `rc=` values are unreliable (they captured `date`'s status; fixed in ce35a76); run status lives in each manifest.json.

## Blockers
P1b needs user access. Real-task data (P6/pivot) would need a download → ask user first.

## Next command
When the job above finishes: `grep -E "^seed|speedups" logs/p5s_A2.log`; read runs/p4-aggregate-*/metrics.json; then append D016 (does D011 hold under A2 on 3 seeds?), write RESULTS RS2, plan R5.
