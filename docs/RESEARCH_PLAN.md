# RESEARCH_PLAN (live, authoritative)

Revision: R1 · 2026-11-04 · Active hypothesis: **H001** (certified selective-write oscillator SSM)

Current claim: *Proposed.* An oscillator SSM can omit input writes online under a cheap
pre-write bound and a causal budget, keeping ||y - y_dense|| <= eps, with measurable speedup.
Novelty unresolved; training, accuracy and performance untested.

## Evidence status
| Item | Status | Evidence |
|---|---|---|
| T1/L1 bounds (exact arithmetic) | derived | docs/SPEC.md §3–4 |
| Historical 1,000-system check (§2 of brief) | imported only (no script supplied) | CODEX_RESEARCH_PLAN.md §2 |
| Float check of bounds | unresolved | — |
| Prior-art comparison | unresolved (3 literature agents running) | docs/prior_art/ |
| Selective execution / speedup | unresolved | — |
| Trained models | unresolved | — |

## Tasks
| ID | Task | Status | Dep | Acceptance check | Evidence |
|---|---|---|---|---|---|
| P0 | Workspace, docs, env, tracking | passed | — | new session can find hypothesis/next action/budget from files | AGENTS.md, this file, D001 |
| P1 | Nearest prior-art collisions | active | P0 | every asserted difference sourced or marked unresolved | docs/PRIOR_ART.md |
| P2 | Float64 reference, bounds, policy, 1,000-system check, tests | pending | P0 | inequalities hold within documented tol; raw excesses reported; counterexamples exposed | runs/, tests/ |
| P3 | True selective execution + profile write fraction f | pending | P2 | omitted work absent; matches masked ref; plausible path to 1.2x | runs/ |
| P4 | Dense pilot, baseline, zero-freq control | pending | P3 | reference learns task competitively | runs/ |
| P5 | Policy sweep vs delta/heuristic/oracle | pending | P4 | validation operating point meeting tolerance + latency | runs/ |
| P6 | Real task confirmation | pending | P5 | frozen protocol, >=3 seeds | — |
| P7 | Results and disposition | pending | P6 | reproducible central result | docs/RESULTS.md |

## Next three actions
1. **[active]** P2: implement bounds/policy/reference (float64 CPU) and tests; run 1,000-system check.
2. P1: consolidate agent reports into docs/PRIOR_ART.md; contribution paragraph + falsification.
3. P3: model with genuine selected projections; profile write fraction f across shapes.

## Acceptance criteria
Frozen v1.0 targets: docs/SPEC.md §7 (copied from CODEX_RESEARCH_PLAN.md §7 P6). Pilot uses the
same approximation (≤0.5 pp / ≤2% RMSE) and runtime (≥1.20x) thresholds on validation.

## Resources
Tranche 8.0 device-h · max 1 training process · ≤60 min/run. Consumed: 0.00 device-h.
Next run estimate: P2 algebra check ≈ 0.02 device-h (CPU).

## Owned running jobs
None (literature agents are not compute jobs; they write only docs/prior_art/batch_{A,B,C}.md).

## Blockers
None.

## Next command
`uv run python -m physics_ml.reproduce --config configs/algebra.yaml` (after P2 implementation).
