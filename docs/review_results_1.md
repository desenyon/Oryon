# Adversarial review 1 of RESULTS RS1 / D011–D012 (H001 rejection)

Reviewer scope: docs/RESULTS.md, docs/DECISIONS.md (D005, D010–D012), docs/SPEC.md §3–5b,
src/physics_ml/{bounds,policy,model,engine_nb,evaluate,data,train,benchmark}.py,
configs/policy_eval.yaml, configs/pilot.yaml, runs/p5-eval-*/metrics.json, runs/p4-*-s0-*/metrics.json.
No project file other than this one was modified. Training was not touched. Extra checks were
quality-only `masked_ref` sweeps (1 torch thread, about 0.5 s each, validation split only, no timing).
The script is at /private/tmp/claude-501/-Users-desenyon--claude-spares-spare-Ybrppr/80701d12-0104-4ddf-b3e2-b325466ee3cd/scratchpad/ext.py.
Numbers from those checks are labelled **[R]** (reviewer-run, validation, seed-0 models, unaudited).

## Verdict

H001 rejection on the **resonator** task holds up. On **recall** it rests on a truncated ε grid.
If the grid is extended, the H001 policy keeps 30% of writes with 0 pp accuracy loss. At that ε
the certificate says nothing about labels, so the policy is acting as a tuned heuristic. Projected
speed is about 1.3x, which is an interpolation and has not been measured. The overall disposition
("reject H001's *certified* advantage") probably stands, but on different grounds than the ones
RESULTS gives. Several of the stated reasons need to be rewritten: C5 recall, failure diagnosis 1,
"oracle keeps ≥93%", and "delta gives far more benefit".

## Critical

### K1. The recall ε grid is truncated. All certified operating points sit at the grid edge.
- configs/policy_eval.yaml:12-14 caps ε at 3 output units. For recall the unit is 1 logit
  (evaluate.py:84). evaluate.py:129-133 picks the fewest writes inside the grid.
- runs/p5-eval-recall-zerofreq-20261104T180609: certified, cdelta and oracle all reach their
  operating point at ε=3.0, the largest grid value. At that point acc_drop is 0 and
  `margin_certified_frac`=1.0, so the tolerance still had slack. The selected point (0.986 writes,
  0.93x) is therefore a grid edge, not the policy's best.
- **[R]** Dense margins: min 7.2 and median 11.6 logits (500 val sequences: min 6.9).
  Extended results (n_val=200; ε=300/500 repeated on n_val=500 with the same numbers):

  | policy | ε / τ | write frac | acc drop | margin-certified |
  |---|---|---|---|---|
  | certified | 5 | 0.983 | 0 | 1.00 |
  | certified | 30 / 100 / 200 | 0.941 / 0.827 / 0.687 | 0 | 0 |
  | certified | 300 / **500** / 700 | 0.538 / **0.304** / 0.125 | 0 / **0** / 1.2 pp | 0 |
  | oracle | 30 / **100** | 0.551 / **0.090** | 0 | 0 |
  | cdelta | 30 | 0.0044 | 0 | 0 |
  | threshold | 0.55 / **0.6** / 0.65 | 0.743 / **0.498** / 0.129 | 0 / **0** / 21 pp | — |

- Effect: by the preregistered selection rule, H001-certified on recall would pick ε≈500 and omit
  about 70% of writes. D005 requires at least 58%. Interpolating between all-write 0.93x and the
  P3 all-omit figure (about 1.6x) gives roughly 1.3x. **This is not measured.** On this evidence
  C5 and C6 for recall go from "failed" to "likely pass, pending timing". Failure diagnosis 1
  ("omission discards information") is false for recall: the exact-norm oracle omits 91% at 0 pp.
- Effect on the disposition: the plan §9 comparison becomes close (delta 1.36x vs an estimated
  ~1.3x) instead of "far more benefit". At ε=500 the bound gives no label guarantee
  (margin-certified = 0). The speed would come from a policy that behaves like a threshold
  heuristic, not from the certificate, so rejecting H001's *certified* value proposition still
  stands. RESULTS and D011 must say so explicitly instead of reporting 0.986 writes / 0.93x.
- Fix: before choosing an operating point, scale each grid to the task. For example, use log-spaced
  ε up to the first failing value, or up to √2·ε = min dense margin for the certified reading.
  Report both the "certified-label" point (ε ≤ min margin/√2 ≈ 5) and the "tolerance" point.
  Flag any operating point that lands on a grid edge as censored.

## Major

### M1. The resonator tolerance is effectively "near-exact equivalence", because the dense model is near-exact.
- The resonator target is a 3-mode damped resonator bank delayed by 10 steps (data.py:31-66),
  which is an exact member of the model class. The osc model reaches NRMSE 0.0055
  (runs/p4-resonator-osc-s0-20261104T172656). A 2% relative RMSE increase therefore allows an RMS
  deviation of roughly 1e-3 output units. Any method that is not exact will fail.
- Supporting evidence: with input noise 0.01, dense NRMSE rises to 0.056, which loosens the
  absolute tolerance about 10x. The oracle then passes with 5% omitted at ε=3. **[R]** cdelta
  passes at ε=10 with 0.904 writes, and certified passes at ε=10 with 0.993 writes.
- The resonator rejection survives. Even at that looser tolerance nothing approaches 58% omission
  (**[R]** delta τ=0.15 gives 0.66 writes and fails at +3.0%). The *diagnosis* in RESULTS:37-38
  should read "the tolerance relative to an in-class near-exact fit leaves no room for lossy
  omission", not "omission discards information". Fix: also report an absolute deviation criterion
  (e.g. dev_mean ≤ 1% of the target RMS).

### M2. The delta baseline wins only on exactly repeated inputs, and the "fastest dense" denominator ignores exact memoization.
- Quiet steps have x exactly 0. Resonator events occur at rate 0.02 plus sparse chirps
  (data.py:37-52). Recall has 4-13 one-hot steps out of 512 (data.py:93-102). So u=φ(0) repeats
  bit-for-bit on about 83% / 96% of steps. delta and cdelta at ε=0 reuse exactly those steps
  (0.168 / 0.036 writes).
- A dense engine that memoizes on `x_t == x_{t-1}` would skip the encoder, gate and projection,
  stay exact, and beat delta. delta still computes E1, E2 and the gate every step
  (engine_nb.py:70-84). "Uncertified delta 1.38x / 1.36x" therefore says nothing about delta
  networks. It is an artifact of the task's exact zeros. The H001 rejection gets *stronger*, but the
  plan §9 "simple delta baseline is better" framing is weaker than claimed: none of the baselines
  beats exact memoization.
- Fix: add an exact-input-memo dense method to engine_nb as the denominator for clean conditions.
  Add a noisy-input or noise-trained task in which exact repeats do not occur.

### M3. The ledger is loose for avoidable, sound reasons. The resonator cannot benefit, but recall might.
- `gamma_init` is log-uniform on [0.005, 0.5] *within each group* (model.py:80-82). After training,
  every group's min γ is about 0.004-0.006 (**[R]**; median γ about 0.05). The group ledger therefore
  treats every group as nearly undamped (ρ≈0.9945), and its per-step replenishment is about 0.55%
  of ε. SPEC §4 mentions grouping by damping but it was never tried.
- **[R]** Sound tightenings that need no oracle:
  - A per-mode ledger, using the same T1 proof per complex mode with |a_ij|, ‖W_ij‖ and ‖C_ij‖, is
    3.5x (recall) to 5.9x (resonator) tighter in steady-state omission cost.
  - Spectral norms instead of Frobenius norms: ‖W_i‖_F/‖W_i‖_2 = 2.5-5.1 and ‖C_i‖_F/‖C_i‖_2 = 1.4-2.3.
  - Exact ‖W_i φ(0)‖, precomputable because φ(0) is a known constant, is about 10-20x below
    wbar·‖u0‖ on quiet steps.
- D012's claim that "a cheaper bound … exact norms close ≤10–100x" does not consider these combined
  sound tightenings (≥40x before exact-u0). For recall, a 40x tighter bound would bring the
  roughly 70%-omission point from ε≈500 down to about 12, close to the label-certified limit of
  min-margin/√2 ≈ 5. That is the one regime where H001's certificate could carry real content.
  It is untested.
- The math itself is fine (see "Mathematics"). The issue is how the bounds were instantiated.

### M4. The timing uncertainty is under-stated, and the cdelta "fail" is not robust.
- benchmark.py:36 times 40 reps over only 20 distinct sequences, so each sequence counts twice.
- benchmark.py:44-53 bootstraps numerator and denominator *independently*. That is unpaired,
  ignores per-rep pairing, and comes from a single process with no between-run replication. The
  docstring says "over sequences", but the bootstrap actually resamples reps.
- cdelta recall at 1.198x [1.196, 1.198] "fails 1.20x" by 0.2%. That gap is smaller than plausible
  session-to-session variance on macOS (no core pinning, P/E migration). It is also smaller than
  fixable overheads: G×d per-group distances (engine_nb.py:126-131) are recomputed even when every
  û_i is identical, kept groups use per-group sgemv, and the readout is a scalar loop shared by all
  methods (engine_nb.py:213-219).
- Fix: use a paired bootstrap over distinct sequences (median of per-sequence ratios), run at least
  3 separate processes, and report "≈1.20x, indistinguishable" rather than "fails". The cdelta
  rejection should rest on the memoization argument in D012, which is sound, not on the 0.002 gap.

### M5. Single seed, and the selection set is the reporting set.
- The operating point is chosen and reported on the same 200 validation sequences
  (evaluate.py:74, 129-133). Those are also the first 200 of the 500 sequences used for checkpoint
  selection (data.py:70 seed 2002, train.py:129).
- The test split is not used for design: train.py:188-190 only records it. That part is fine.
- The bias is small for grids this coarse, but the whole disposition rests on one model per task.
  Fix: evaluate seeds 1-2 when they finish, and confirm the chosen ε on a held-out validation half.

## Minor
- m1. delta is slightly handicapped: on refresh it issues G per-group sgemv calls instead of one
  fused call (engine_nb.py:201-204). Certified with zero omissions also pays the per-group penalty
  (0.88x); a fused fallback would give about 0.97x. Both biases are conservative for the rejection.
- m2. The shared non-write work (the homogeneous loop at engine_nb.py:97-102 and the scalar readout)
  is unoptimized. This compresses every ratio toward 1x. `dense_fused` is the fastest *measured*
  dense path but not an optimized one. The effect is conservative for omission methods, and it
  matters for the 1.17x / 1.198x vs 1.20x margins.
- m3. The table pairs write fractions from 200 masked-ref sequences with speeds from the first 20
  sequences on the engine. For example, cdelta resonator has engine_write_frac_seq0 = 0.234 vs
  0.165 in the table. Report the engine's write fraction over all timed sequences.
- m4. evaluate.py:118 `max_raw_excess` is in raw units while `dev_max` is unit-normalized.
  `bound_violations_raw` counts scan-vs-loop float ordering (101,966 at ε=0). It is disclosed but
  confusing; drop it or rename it.
- m5. VECLIB_MAXIMUM_THREADS is set with `setdefault` (evaluate.py:17-18) and is not recorded in
  the manifest env, so the single-thread regime cannot be audited after the fact.
- m6. Noise is added to every channel of a model trained without noise. Recall noise 0.05 is
  already marked uninformative. **[R]** Recall noise 0.01 is also uninformative: dense accuracy
  falls to 55%. RESULTS:41 ("with sensor noise the certificate blocks reuse") rests on one resonator
  condition, and that condition was itself grid-censored at ε=3 (certified passes at ε=10 **[R]**).
  Fix: train with noise augmentation before drawing noise conclusions.
- m7. GRU baseline: val NRMSE is still falling at 1500 steps (0.141 → 0.125 → 0.121). Recall GRU is
  at chance. There was no learning-rate sweep, and the step counts match but the compute does not
  (~145 s vs ~480 s wall). C7 should read "not established". The resonator zerofreq result of 0.978
  is about the same as predicting zero. That suggests an optimization failure rather than a
  capacity limit, since GRU reaches 0.12. Both C8 outcomes are fixed by how the tasks were built:
  resonator is an in-class oscillator target, and recall uses a random t0 with ω∈[-3,3], which
  scrambles phase. C8 should be labelled task-determined.
- m8. On recall, H001 is evaluated on a zero-frequency model. T1 does not depend on frequency, but
  the *benefit* does: a phase-aware bound has nothing to exploit at ω=0. "Oscillator" H001 is
  therefore only tested on the resonator task.

## Mathematics (Q1): sound
- **T1, real coordinates.** Each 2x2 block is |a|·R(θ), so its operator 2-norm is |a|. The
  block-diagonal norm is max_j |a_j| = exp(−min_j γ_j dt), which holds because exp is monotone and
  dt ≥ 0 (model.py:112, engine_nb.py:89). C_i is a general real [p, 2m] map (model.py:114-116), so
  ‖C_i‖_F ≥ ‖C_i‖_2 is enough. D·u and the bias cancel because u is identical in both runs.
  The implementation matches: decay first, then add β only for omitted groups (engine_nb.py:137,
  167; bounds.py:67). It is an exact-arithmetic bound only. In float32, mag·(cos, sin) can exceed
  mag by a few ulps; this is disclosed and has no certificate.
- **cdelta.** b − b̃ = g(u_t)·W_i(u_t − û_i). The gate is applied fresh (model.py:217,
  engine_nb.py:170). The cache and û_i are refreshed per written group (model.py:216,
  engine_nb.py:173-178). Writes are forced at t=0 (model.py:191-192; β=inf at engine_nb.py:131).
  So β = |g|·wbar·‖u − û_i‖ is sound.
- **Margin condition.** |Δl_i − Δl_j| = |⟨Δl, e_i − e_j⟩| ≤ √2·‖Δl‖ ≤ √2·B_T, so a dense margin
  greater than √2·B_T preserves top-1. evaluate.py:120 implements this correctly. It mixes the scan
  margin with the streaming bound, which is fine in exact arithmetic.

## Q6: Is the rejection supported?
- Resonator: yes. Recall: the rejection is overstated, and the stated reasons are partly artifacts
  (K1, M2, M1).
- Supported disposition: "On these synthetic tasks H001 yields no *certified* benefit. On the
  resonator, no lossy omission fits the near-exact tolerance. On recall, enough omission for
  ≥1.2x appears only at ε ≈ 70x the minimum dense margin, where the bound is vacuous. All
  delta-style savings are exact-input memoization."
- The cheap experiment most likely to change the conclusion takes minutes and needs no training:
  1. Re-run P5 on recall-zerofreq and resonator-osc with a log ε grid up to failure, ending around
     1000 units, **with timing** in an idle machine window.
  2. Add an exact-input-memo dense method to the engine.
  3. Add a per-mode, spectral-norm ledger, i.e. damping-sorted groups.

  If certified recall at ε ≤ 5 (label-certified) reaches ≥58% omission under the tighter ledger and
  times at ≥1.2x **and** above exact memoization, H001 would be revived on recall. Otherwise the
  rejection becomes airtight.
