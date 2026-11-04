# RESEARCH_PLAN (live, authoritative)

Revision: R3 · 2026-11-04 · Active hypothesis: **H001** (certified selective-write oscillator SSM)

Current claim: *Proposed, partially verified.* Bounds T1/L1 hold (exact derivation; float64
observed). Selective execution genuinely omits work; ideal speed ceiling 1.3–2.0x in the declared
compiled regime. Novelty unresolved; training, accuracy and achievable omission untested.

## Evidence status
| Item | Status | Evidence |
|---|---|---|
| T1/L1 bounds (exact arithmetic) | derived | docs/SPEC.md §3–4 |
| Historical 1,000-system check | imported only (no script supplied) | CODEX_RESEARCH_PLAN.md §2 |
| New 1,000-system float64 check | locally verified (0 violations, 0 raw excess); float32 raw excess ≤1.7e-5 rel | runs/p2-algebra-20261104T171316, D003 |
| Prior-art comparison | passed for inspected set; novelty unresolved (Delta-SSM unread) | docs/PRIOR_ART.md PA2, D004, D006 |
| Selective execution omits work | locally verified | tests/test_execution.py, tests/test_engine.py |
| Write fraction / ceiling | locally verified: f 0.23–0.49, ceiling 1.30–1.97x (compiled, untrained) | runs/p3-bench-compiled-20261104T172121, D005 |
| Trained models / omission rate / speedup | unresolved | — |

## Tasks
| ID | Task | Status | Dep | Acceptance check | Evidence |
|---|---|---|---|---|---|
| P0 | Workspace, docs, env, tracking | passed | — | new session can resume from files | AGENTS.md, D001 |
| P1 | Nearest prior-art collisions | passed (inspected set) | P0 | differences sourced or unresolved | PRIOR_ART.md PA2, review_1.md |
| P1b | Read Delta-SSM methods | blocked | — | full text read | needs user to open OpenReview 0b75H3NcHp |
| P2 | Bounds/policy/reference/tests, 1,000-system check | passed | P0 | 0 violations; raw excess reported | runs/p2-algebra-*, D003 |
| P3 | Selective execution + write-fraction profile | passed | P2 | omits work; matches ref; path to 1.2x needs ≥~58% omission | runs/p3-bench-*, D005 |
| P4 | Dense pilot: tasks, oscillator, zero-freq control, GRU baseline | active | P3 | reference learns tasks competitively | runs/p4-* |
| P5 | Policy sweep: certified vs cdelta vs threshold vs delta vs oracle | pending | P4 | validation operating point meets tolerance + 1.2x | runs/p5-* |
| P6 | Real task confirmation | pending | P5 | frozen protocol, ≥3 seeds | — |
| P7 | Results and disposition | pending | P6 | reproducible central result | docs/RESULTS.md |

## Next three actions
1. **[active]** P4: implement data.py (phase-sensitive pulse/chirp response; delayed recall with distractors), train.py; train oscillator, zero-frequency control, GRU baseline (pilot shape G=8, m=32, d=256, hidden=64).
2. P5: freeze weights; sweep ε on validation for certified/cdelta/threshold/delta/oracle; measure omission, deviation, task loss, compiled latency.
3. P5 diagnosis: bound tightness (realized deviation / bound) and source of conservatism.

## Acceptance criteria
Frozen v1.0 targets: docs/SPEC.md §7. Pilot: approximation ≤0.5 pp acc / ≤2% RMSE increase vs
frozen dense; runtime ≥1.20x median vs fastest dense (D005 amendment A1: regime + denominator).

## Resources
Tranche 8.0 device-h · 1 training process · ≤60 min/run. Consumed ≈ 0.03 device-h before pilot (index.csv); pilot seed 0 in progress (≈0.85).
Next run estimate: P4 smoke ≈ 0.05 device-h; each pilot training run ≤ 0.25 device-h (CPU).

## Owned running jobs
| Job | PID / host | Started (local) | Command | cwd | Log | Checkpoints |
|---|---|---|---|---|---|---|
| P4 pilot seed 0 (6 sequential runs: resonator,recall × osc,zerofreq,gru) | 79236 / this Mac (bash driver) | Wed Nov 4 09:26:51 2026 | `scripts/run_pilot.sh 0` | repo root | logs/pilot_s0.log + runs/p4-<task>-<model>-s0-*/log.txt | runs/p4-*/last.pt, best.pt |
Reconcile: `ps -o pid,lstart,command -p 79236`. If dead mid-run: `uv run python -m physics_ml.train --resume runs/<interrupted id>` then rerun the remaining (task, model) pairs individually. Est. ≈ 50 min ≈ 0.85 device-h.

## Blockers
P1b (Delta-SSM full text) needs the user (bot-check on OpenReview). Does not block P4/P5.

## Next command
`uv run python -m physics_ml.train --config configs/pilot.yaml` (after P4 implementation).
