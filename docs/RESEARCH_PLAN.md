# RESEARCH_PLAN (live, authoritative)

Revision: R5 · 2026-11-04 · Active hypothesis: **none.** H001 REJECTED (pilot scope; D011, re-tested
on 3 seeds under A2, D016); pivot declined (D014). Disposition: supported negative result.

Current claim: no (task, model) passes jointly. Resonator (oscillation helps, −0.972 NRMSE vs
zero-frequency, 3 seeds): certified omission ≈0% (0.88x); cdelta 1.18–1.19x < 1.20x; delta 1.42x.
Recall (oscillation hurts, −85.7 pp): certified omission is fast on the zero-frequency model
(1.41–1.44x on 2/3 seeds) only at a vacuous bound (0% margin-certified); threshold/delta match it.

## Evidence status
| Item | Status | Evidence |
|---|---|---|
| T1/L1 bounds | derived; float64 locally verified (new check) | SPEC §3–4; runs/p2-algebra-20261104T171316 |
| Prior art | passed for inspected set; novelty unresolved | PRIOR_ART.md PA2 |
| Selective execution / profile | locally verified | runs/p3-bench-compiled-20261104T172121 (D005) |
| Dense pilot seed 0 | locally verified (exploratory) | runs/p4-*-s0-* (D010) |
| Policy evaluation | locally verified, 3 seeds, A2: H001 rejected; H001b/cdelta no advantage | runs/p5-eval-*T19* (D016), v1 runs/p5-eval-*T18* (D011, D012), docs/RESULTS.md RS2 |
| Physics comparison C8 | locally verified, 3 seeds: task-dependent (helps resonator, hurts recall) | runs/p4-aggregate-20261104T195054 |

## Tasks
| ID | Task | Status | Dep | Acceptance check | Evidence |
|---|---|---|---|---|---|
| P0–P3 | Bootstrap, prior art, algebra, execution profile | passed | — | see R2 | D001–D006 |
| P1b | Read Delta-SSM methods | blocked | — | full text read | needs user (OpenReview 0b75H3NcHp bot check) |
| P4 | Dense pilot seed 0 | passed (resonator osc; recall zerofreq) | P3 | learns task | D010 |
| P4s | Seeds 1–2 for all 6 pilot configs | passed (recall/osc/s1 hit the 25-min cap at step 1382, resumed to 1500) | P4 | 3-seed mean ± sd for C8 | runs/p4-*-s{1,2}-* |
| P5 | Policy sweep + diagnosis | failed (H001 rejected) | P4 | operating point meeting tolerance and 1.2x | D011, D012 |
| P5s | Re-run P5 on seeds 0–2 under protocol A2 (D015) | passed (rejection retained, D016) | P4s | same conclusion on all seeds | runs/p5-eval-* |
| P6 | Real-task confirmation of H001 | retired (H001 rejected) | — | — | D011 |
| PV | Pivot feasibility review (2 directions) | passed (pivot declined, D014) | D012 | recommendation with falsifiable hypothesis or stop | docs/prior_art/pivot_review.md |
| P7 | Results + disposition | passed (negative result; review D015 addressed) | P4s, P5s, PV | reproducible; reviewer challenge | docs/RESULTS.md RS2 |

## Next three actions (all optional; none required for the disposition)
1. If the user unblocks P1b: read Delta-SSM methods and update PRIOR_ART (could only weaken novelty further).
2. Stronger GRU baseline (tuned lr/steps) to firm up C7; does not affect D016.
3. New hypothesis only with new evidence or user direction (plan §9; D014).

## Acceptance criteria
Frozen v1.0 targets: SPEC §7; amendments A1 (D005: timing regime + fastest-dense denominator) and A2
(D015: wider grids, memoization, per-mode ledger, paired timing; tolerances unchanged).
H001 failed C5/C6 on validation; test splits recorded once by train.py and never used for design.

## Resources
Tranche 8.0 device-h. Consumed ≈ 2.74 device-h (sum of index.csv device_hours, 51 rows). Remaining ≈ 5.3.

## Owned running jobs
None. Last job (P5s A2, scripts/run_p5s.sh → logs/p5s_A2.log) finished 2026-11-04 19:50Z; the aggregate
step crashed (code bug, fixed, rerun passed). Verify with `pgrep -fl "physics_ml|run_"` (expect nothing).
Notes: runs/p5-eval-resonator-osc-20261104T1942* is an A2 smoke test (not evidence). logs/pilot_s12.log
`rc=` values are unreliable (captured `date`'s status; fixed in ce35a76); run status lives in manifest.json.

## Blockers
P1b needs user access. Real-task data (P6/pivot) would need a download → ask user first.

## Next command
None required. To reproduce the A2 evidence: `scripts/run_p5s.sh > logs/p5s_A2_rerun.log 2>&1` (new run dirs).
