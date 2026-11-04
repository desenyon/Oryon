# DECISIONS (append-only)

Format: D### · date · hypothesis · evidence · old → new · rationale · affected tasks.

## D001 · 2026-11-04 · H001 · P0 workspace bootstrap
- Evidence: workspace contained only CODEX_RESEARCH_PLAN.md (no repo, no AGENTS.md, no prior review script). Hardware: Apple M4 (4P+6E cores), 16 GiB RAM, 249 GiB free disk, MPS available without float64; no CUDA. Env: uv 0.13.0, CPython 3.12.15 (system 3.15 has no torch wheels), torch 2.15.0, numpy 2.5.4, pyyaml 6.0.3, pytest 9.2.0 (uv.lock, requirements.lock).
- Old: none → New: git repo initialized here (directory is dedicated to this project); H001 initialized as *proposed*, novelty unresolved, training/performance untested.
- Rationale: plan §5/P0. The historical 1,000-system check (§2) has no supplied script; P2 implements an independent check labelled "new check", not a reproduction.
- Resource tranche: 8 device-hours (CPU job = 1 device; MPS job = 1 device), one training process at a time, ≤60 min per run.
- Affects: P0 (passed), P1, P2.

## D002 · 2026-11-04 · H001 · float64 reference device
- Evidence: `torch.ones(2, dtype=torch.float64, device="mps")` raises (MPS has no float64).
- Decision: all float64 references run on CPU. Float32 model runs may use CPU or MPS; latency claims are device-specific.
- Affects: P2, P3.

## D003 · 2026-11-04 · H001 · P2 algebra check passed (float64); float32 shows raw excesses
- Evidence: runs/p2-algebra-20261104T171316/{metrics.json,log.txt,manifest.json}; tests/test_bounds.py, tests/test_phase.py (21 passed).
- Result: NEW independent check (not a reproduction of the historical review). Policy mode: 2,048,000 state / 256,000 output / 256,000 budget checks, 0 violations at tol 1e-10·max(1,max|h|); raw positive excesses in float64: 0. Random-mask mode: same counts, 0 violations. Float32 torch real-pair path vs float64 reference (~1/10 of systems): max raw state excess 6.75e-4 absolute (1.66e-5 relative to state scale), positive in 49,998/206,848 checks — expected where the bound is 0 or tight; no machine-arithmetic certificate exists.
- Diagnostic: realized output deviation / bound median ≈ 0.020 (policy) on random systems; Frobenius/spectral ratio ≈ 1.8. Strong early sign of ledger conservatism (triangle inequality across groups and time, ignored cancellation), to be re-measured on trained models in P5.
- Policy fix during P2: zero-contribution omissions are allowed even when rounding leaves B_base above ε (room clamped at 0; final check only removes positive-cost omissions). SPEC §4 invariant under rounding: B_t ≤ max(ε, B_t^base).
- Decision: P2 passed; C1–C3 status "exact theorem + observed float64 agreement"; float32 status "observed agreement up to ~1e-5 relative". Proceed to P3.

## D004 · 2026-11-04 · H001 · P1 prior-art consolidation (PA1)
- Evidence: docs/prior_art/batch_{A,B,C}.md, docs/PRIOR_ART.md.
- Result: no inspected source states the full composition, but every component has precedent: ledger = standard perturbation recursion (Quamba Thm 4.1; LAST/AIRE-Prune H∞ bounds), online projection skipping with evolving state = Delta Networks / EventSSM, deviation-triggered updates = event-triggered control. Exact Instrument (2607.11796) gives exact but offline, non-causal subset error. Delta-SSM (abstract only) and Seghairi dissertation (403) unresolved.
- Decision: C9 = "novelty unresolved; mechanism distinguished from inspected prior work only in composition". H001 retained; its contribution now hinges on measured benefit vs the delta-network and threshold baselines at matched realized deviation (P5). Independent review requested (docs/prior_art/review_1.md).
- Affects: P1 (active until review), P5 baseline set (delta baseline mandatory).

## D005 · 2026-11-04 · H001 · P3 timing regime and fairness rule (protocol amendment A1, before any trained-model timing)
- Evidence: runs/p3-bench-20261104T171739 (NumPy engine): write fraction f = 0.11–0.29 although writes are 59–87% of MACs; certified policy with ALL writes omitted 0.73–0.97x (slower than dense) — interpreter overhead dominates. runs/p3-bench-compiled-20261104T171856 (numba, scalar loops): ceiling 1.9–9.3x but its dense path (42 µs/step at d=256) was slower than NumPy+Accelerate (24 µs) → inflated, rejected as a denominator. runs/p3-bench-compiled-20261104T172054 (numba + Accelerate sgemv, fused dense denominator): f = 0.32–0.43. runs/p3-bench-compiled-20261104T172121 (+ coefficient cache when dt unchanged, shared by all policies): f = 0.23–0.49, ceiling 1.30–1.97x, certified all-omit 1.25–1.94x, certified all-write 0.82–0.97x (per-group call + controller penalty).
- Old: timing engine unspecified → New: declared regime = batch-one streaming, float32, single-thread CPU (VECLIB_MAXIMUM_THREADS=1), numba-compiled complete step calling Accelerate cblas_sgemv; speedup denominator = fastest dense implementation measured (fused single-sgemv dense, "dense_fused"). NumPy-engine results retained as evidence that the eager/interpreted regime cannot meet 1.2x.
- Requirement carried to P5: for the pilot shape (G=8, m=32, d=256, hidden=64) the policy must omit ≈ ≥58% of writes at the quality tolerance to reach 1.20x (interpolating all-write 0.89x and all-omit 1.61x); ≈ ≥50% for G=16, m=32, d=512.
- P3 status: passed (selective path omits work — instrumented mv-call count test; matches masked reference; plausible path to 1.2x contingent on omission rate).

## D006 · 2026-11-04 · H001 · independent prior-art review applied (PA2); certified delta baseline added
- Evidence: docs/prior_art/review_1.md; docs/PRIOR_ART.md PA2.
- Result: reviewer spot-checks confirmed AIRE-Prune/Quamba/Exact-Instrument/Delta-net claims (locations corrected). SMR 2405.17534 and EdgeDRNN 2012.13600 resolved (not collisions); Trimpe 1511.05223 Thm 5 verified as conceptual precedent (communication, fixed δ). Delta-SSM still abstract-only (OpenReview 0b75H3NcHp bot-blocked; needs the user to open it manually). No exact match among inspected sources.
- Decision: P1 status passed for the inspected set (every asserted difference sourced or marked unresolved); C9 remains "novelty unresolved". New mandatory baseline `cdelta` (certified delta: cached per-group projection reused, ledger β = |g|·w̄·‖u−û_i‖, same budget policy) implemented in model.py and engine_nb.py with tests. It is a baseline, not a new hypothesis; if it dominates H001 the plan §9 rule applies (reject H001's claimed advantage).
- D007 · 2026-11-04 · plan revision R1 → R2 (P1, P2, P3 passed; P4 active).
