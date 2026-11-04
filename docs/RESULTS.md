# RESULTS

Revision RS2 · 2026-11-04 · All numbers are validation unless marked; pilot results are
exploratory (synthetic tasks). Disposition: **H001 rejected (pilot scope; D011, re-tested on 3 seeds under protocol A2, D016)**.
RS1 rows below are kept; superseded statements are marked.

## Claim-to-evidence table
| ID | Claim | Status | Evidence |
|---|---|---|---|
| C1/C2 | T1 state/output bound (exact arithmetic) | derived; float64 observed 0 violations, 0 raw excess on 2,048,000/256,000 checks (new check, not a reproduction) | docs/SPEC.md §3; runs/p2-algebra-20261104T171316 |
| C3 | L1 budget policy feasibility | derived; 256,000 budget checks, 0 violations | same |
| — | float32 implementation | observed raw excess ≤1.7e-5 relative (algebra check); ≤1.2e-5 output units on trained models; **no machine-arithmetic certificate** | runs/p2-*, runs/p5-eval-* |
| C4 | Selective path truly omits W_i u work | locally verified (instrumented call counts; engine/model agreement) | tests/test_execution.py, tests/test_engine.py |
| C5 | ≥1.20x complete-model streaming speedup | resonator: **failed** on 3 seeds (H001 0.88–0.89x, cdelta 1.18–1.19x, delta 1.42x). recall (zerofreq): H001 1.07/1.41/1.44x — passes 2/3 seeds but only at a vacuous bound (ε 300–1000 logits, 0% margin-certified); threshold 1.44x and delta 1.36x match/beat it. v1 "0.93x recall" was a grid artifact (withdrawn, D015) | runs/p5-eval-*T19* (D016); v1 runs/p5-eval-*T18* (D011, D012) |
| C6 | Quality within tolerance at that point | H001 only at ~0% omission | same |
| C7 | Dense reference competitive with simple baseline | resonator: osc 0.0057±0.0003 vs GRU 0.154±0.029 NRMSE (3 seeds; GRU weak); recall: osc 14.3% and GRU 13.1% both ≈ chance | runs/p4-aggregate-20261104T195054 |
| C8 | Oscillation helps vs independent zero-frequency control | task-dependent, 3 seeds: resonator NRMSE osc−zf −0.972 [−0.973, −0.971] (helps); recall acc osc−zf −85.7 pp [−87.7, −83.7] (hurts) | runs/p4-aggregate-20261104T195054 |
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

## Policy evaluation, protocol A2 (P5s; seeds 0/1/2; validation; same tolerances)
Paired timing, 40 sequences × 5 reps; speedup = median ratio vs fused dense, 95% paired bootstrap CI.
| Model / condition | H001 certified | H001b certified_pm | cdelta | exact memo (delta τ=0) / best delta | threshold |
|---|---|---|---|---|---|
| resonator osc, clean | 1.000 writes, 0.88/0.88/0.89x | 1.000, 0.86x | 0.165, 1.19/1.18/1.19x (CI low 1.165–1.171) | 0.177 / 0.164, 1.42–1.43x | 1.000, 0.88x |
| recall zerofreq, clean | 0.538/0.008/0.001, 1.07/1.41/1.44x | 0.216/0.001/0.001, 1.16/1.33/1.37x | 0.004, 1.21/1.21/1.25x | 0.036, 1.35–1.36x | 0.989/0.001/0.002, 0.93/1.44/1.45x |
| resonator osc, noise 0.01 | ≥0.98, 0.89x | ≥0.97, 0.86–0.87x | ≥0.90, 0.78x | 0.83/0.21/0.23, 0.93/1.37/1.36x | 1.000, 0.88x |
Recall certified operating points: ε = 300–1000 logit units; margin-certified fraction 0% (cdelta_pm ≤7.5%):
the certificate does not certify the prediction there. Streaming-reference positive bound excess: 0 in
every A2 operating point (float32; not a machine certificate).

## Failure diagnosis
1. Omission discards information: writes on quiet steps carry the drive u=φ(0)≠0 and gates do
   not vanish; omitting even 0.15% of writes (exact-norm oracle) breaks the 2% tolerance.
2. Worst-case ledger conservatism (triangle inequality across groups and time, Frobenius norms,
   ignored cancellation): 10²–10³x; exact write norms recover only 10–100x.
3. Certified deferral (delta) only pays on exact repeats, which plain memoization captures; with
   sensor noise the certificate blocks reuse.
4. Grouped execution and per-group distances cost 3–18% versus fused dense.

## Not established
No real-task result, no device-energy measurement, no novelty. Seeds give uncertainty for C7/C8 and P5 robustness (3 seeds, synthetic tasks only).

5. (A2) Wider ε recovers omission only where the bound is vacuous; a guarantee loose enough to permit
   savings no longer constrains the prediction, and uncertified thresholds then do equally well.
