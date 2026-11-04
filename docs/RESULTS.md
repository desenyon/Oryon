# RESULTS

Revision RS1 · 2026-11-04 · All numbers are validation unless marked; pilot results are
exploratory (synthetic tasks). Disposition: **H001 rejected candidate (pilot scope)**.

## Claim-to-evidence table
| ID | Claim | Status | Evidence |
|---|---|---|---|
| C1/C2 | T1 state/output bound (exact arithmetic) | derived; float64 observed 0 violations, 0 raw excess on 2,048,000/256,000 checks (new check, not a reproduction) | docs/SPEC.md §3; runs/p2-algebra-20261104T171316 |
| C3 | L1 budget policy feasibility | derived; 256,000 budget checks, 0 violations | same |
| — | float32 implementation | observed raw excess ≤1.7e-5 relative (algebra check); ≤1.2e-5 output units on trained models; **no machine-arithmetic certificate** | runs/p2-*, runs/p5-eval-* |
| C4 | Selective path truly omits W_i u work | locally verified (instrumented call counts; engine/model agreement) | tests/test_execution.py, tests/test_engine.py |
| C5 | ≥1.20x complete-model streaming speedup | **failed** for H001 (0.88x resonator, 0.93x recall); certified-delta 1.17x / 1.198x (fails); uncertified delta 1.38x / 1.36x | runs/p5-eval-* (D011, D012) |
| C6 | Quality within tolerance at that point | H001 only at ~0% omission | same |
| C7 | Dense reference competitive with simple baseline | resonator: osc 0.0054 vs GRU 0.121 NRMSE (GRU under-trained); recall: osc fails (13.4%) | runs/p4-*-s0-* (D010) |
| C8 | Oscillation helps vs independent zero-frequency control | task-dependent, single seed: helps on resonator (0.0054 vs 0.978), hurts on recall (13.4% vs 100%); seeds 1–2 pending | runs/p4-* |
| C9 | Relationship to prior work | novelty unresolved; components all have precedent; Delta-SSM unread | docs/PRIOR_ART.md PA2 |

## Profiling (P3; untrained shapes; batch-one float32 single-thread CPU)
| Engine | write fraction f | ideal ceiling | certified, all omitted | certified, all written |
|---|---|---|---|---|
| NumPy (interpreted) | 0.11–0.29 | 1.12–1.41x | 0.73–0.97x | 0.46–0.61x |
| numba + Accelerate sgemv, fused-dense denominator, coefficient cache | 0.23–0.49 | 1.30–1.97x | 1.25–1.94x | 0.82–0.97x |
Evidence: runs/p3-bench-20261104T171739, runs/p3-bench-compiled-20261104T172121.

## Policy evaluation (P5, seed 0, validation, tolerance ≤2% rel. RMSE / ≤0.5 pp)
| Model / condition | H001 certified writes / speed | oracle writes | certified delta writes / speed | uncertified delta writes / speed |
|---|---|---|---|---|
| resonator osc, clean | 1.000 / 0.88x | 1.000 | 0.165 (ε=0.3) / 1.17x [1.15,1.19] | 0.164 / 1.38x [1.36,1.42] |
| recall zerofreq, clean | 0.986 / 0.93x | 0.932 | 0.012 (ε=3) / 1.198x [1.196,1.198] | 0.036 / 1.36x [1.35,1.36] |
| resonator osc, input noise 0.01 | 0.998 / 0.88x | 0.950 | 0.960 / 0.77x | 0.827 / 0.92x |
Exact memoization (cdelta at ε=0): 0.168 (resonator), 0.036 (recall) writes — essentially all
of the certified-delta savings. Bound tightness (realized/bound): 6e-4–7e-4 (resonator),
2e-2–5e-2 (recall).

## Failure diagnosis
1. Omission discards information: writes on quiet steps carry the drive u=φ(0)≠0 and gates do
   not vanish; omitting even 0.15% of writes (exact-norm oracle) breaks the 2% tolerance.
2. Worst-case ledger conservatism (triangle inequality across groups and time, Frobenius norms,
   ignored cancellation): 10²–10³x; exact write norms recover only 10–100x.
3. Certified deferral (delta) only pays on exact repeats, which plain memoization captures; with
   sensor noise the certificate blocks reuse.
4. Grouped execution and per-group distances cost 3–18% versus fused dense.

## Not established
No real-task result, no multi-seed uncertainty yet, no device-energy measurement, no novelty.
