# SPEC — H001 certified selective-write oscillator SSM

Revision: S2 (2026-11-04; S1 = bootstrap from CODEX_RESEARCH_PLAN.md v1.0 §3; S2 adds §5b and B1).
Changes to this file are logged in docs/DECISIONS.md.

## 1. Model (dense reference)

Causal features `u_t = phi(x_t)` from a per-sample (non-recurrent) encoder with training-set
input normalization; identical in dense and approximate execution. One recurrent bank, G groups,
group i has m_i complex modes. Real coordinates: a complex mode is a pair (re, im); state tensor
`[batch, G, m, 2]`.

    h_{i,t} = A_{i,t} h_{i,t-1} + b_{i,t},     b_{i,t} = g_i(u_t) W_i u_t
    a_{ij,t} = exp[(-gamma_ij + i*omega_ij) dt_t],   gamma_ij >= 0, dt_t >= 0
    rho_{i,t} = max_j |a_{ij,t}| = exp(-min_j gamma_ij * dt_t) <= 1
    y_t = Re sum_i C_i h_{i,t} + D u_t

- `W_i` is a real map R^d -> R^{2 m_i} (real coordinates of the complex write).
- `C_i` is a real map R^{2 m_i} -> R^p (real coordinates of `Re(C_i^complex h)`).
- `g_i` is a scalar gate depending only on u_t (never on h).
- gamma, omega are learned constants (S1); dt may vary per step but is an input, not state-derived.
- No recurrent encoder, deep recurrent stack, output feedback, or state-dependent coefficient.

## 2. Approximate execution

    h~_{i,t} = A_{i,t} h~_{i,t-1} + m_{i,t} b_{i,t},   m_{i,t} in {0,1}

Every group advances its homogeneous dynamics every step; only the input projection `W_i u_t`
(and its gate multiply) is omitted. Full recurrent state is kept: no claim of reduced state or
parameter storage.

Pre-write bound (computed before `W_i u_t`):

    beta_{i,t} = |g_i(u_t)| * wbar_i * ||u_t||_2,   wbar_i >= ||W_i||_2 (Frobenius in S1)
    c_i >= ||C_i||_2 (Frobenius in S1)

Bounds are computed once after freezing weights and cached with a weight fingerprint; any weight
change invalidates them.

Ledger, with e_{i,0} = 0 and identical initial states:

    e_{i,t} = rho_{i,t} e_{i,t-1} + (1 - m_{i,t}) beta_{i,t}
    B_t     = sum_i c_i e_{i,t}

## 3. Theorem T1 (exact arithmetic) and proof

Assumptions A1–A6:
- A1 Same frozen parameters, same inputs x, same dt sequence in both executions; h_0 = h~_0.
- A2 A_{i,t} is diagonal in complex coordinates (block-diagonal 2x2 scaled rotations in real
  coordinates) and its coefficients do not depend on h or h~.
- A3 gamma >= 0, dt_t >= 0, so |a_ij,t| <= 1.
- A4 g_i depends only on u_t; u_t depends only on x_{<=t} (identical in both executions).
- A5 wbar_i >= ||W_i||_2 and c_i >= ||C_i||_2 (sound upper bounds, not estimates).
- A6 Exact real arithmetic.

Claim T1a: `||h_{i,t} - h~_{i,t}||_2 <= e_{i,t}` for all i, t.
Claim T1b: `||y_t - y~_t||_2 <= B_t`.

Proof. Let d_{i,t} = h_{i,t} - h~_{i,t}. Subtracting the recurrences (A1, A2: same A in both):

    d_{i,t} = A_{i,t} d_{i,t-1} + (1 - m_{i,t}) b_{i,t}.

A_{i,t} is diagonal complex, so its operator 2-norm equals max_j |a_ij,t| = rho_{i,t}; in real
coordinates each 2x2 block is |a| times a rotation, which has the same norm, and the real
Euclidean norm of the pair vector equals the complex Euclidean norm. Hence

    ||d_{i,t}|| <= rho_{i,t} ||d_{i,t-1}|| + (1 - m_{i,t}) ||b_{i,t}||.

By A4, A5: ||b_{i,t}|| = |g_i(u_t)| ||W_i u_t|| <= |g_i(u_t)| ||W_i||_2 ||u_t|| <= beta_{i,t}.
Base case ||d_{i,0}|| = 0 = e_{i,0}. Induction: if ||d_{i,t-1}|| <= e_{i,t-1} then, since
rho >= 0, ||d_{i,t}|| <= rho e_{i,t-1} + (1-m) beta = e_{i,t}. (T1a)

y_t - y~_t = sum_i C_i d_{i,t} (the D u_t term cancels by A4; C_i is the real form of Re C h).
By the triangle inequality and A5, ||y_t - y~_t|| <= sum_i c_i ||d_{i,t}|| <= sum_i c_i e_{i,t}.
(T1b) QED.

T1 bounds deviation from the dense execution of the same frozen model, NOT error against ground
truth. It does not use omega: the bound is frequency-independent (claims about oscillation must be
tested separately, see C8).

Non-claims (each would need a new derivation): hidden-state-dependent gates/transitions,
autoregressive feedback, deep recurrent stacks, changing or unbounded readout, nonlinear readout.

## 4. Budget policy P1 (causal, fixed epsilon per sequence)

Before deciding writes at step t:

    B_t^base = sum_i c_i rho_{i,t} e_{i,t-1},   r_t = epsilon - B_t^base.

Omit a set S of groups with sum_{i in S} c_i beta_{i,t} <= r_t. Equal costs: sort c_i beta_i
ascending and take the longest feasible prefix (maximizes omitted count). Unequal measured costs
`k_i`: greedy by k_i / (c_i beta_i) descending, then a deterministic final check
`B_t^base + sum_S c_i beta_i <= epsilon`; drop the last-added group until it holds.

Lemma L1 (feasibility). If B_{t-1} <= epsilon then B_t^base <= B_{t-1} <= epsilon (rho <= 1),
so S = {} (compute everything) is always feasible and B_t <= epsilon is maintained by induction.
Computing a write does not reset accumulated error. A mid-sequence decrease of epsilon may be
infeasible and is rejected (no ledger reset). epsilon < 0 or nonfinite coefficients are rejected.

Undamped groups (rho = 1 always): allowance never replenishes,
`sum_{tau<=t,i} c_i (1-m_{i,tau}) beta_{i,tau} <= epsilon`. A single undamped mode makes the
whole group's rho = 1 (mixed-damping group); grouping by damping is an implementation option.

## 5. Numerical status discipline

Every result reports these separately:
1. Exact-arithmetic theorem (T1, L1) and assumptions A1–A6.
2. Observed agreement in floating-point tests: float32/float64 implementations vs an independent
   float64 CPU reference (MPS lacks float64). Raw positive excesses (observed - bound) are
   recorded before any tolerance is applied.
3. Machine-arithmetic certificate: NOT derived or implemented (status: none). A test tolerance
   is not a certificate.

Runtime measurements include the cost of computing beta, the ledger, and selection.

## 5b. Classification agreement (margin condition)

If ||l_T - l~_T||_2 <= B_T for logit vectors, then for any pair (i, j),
|(l_i - l_j) - (l~_i - l~_j)| <= |d_i| + |d_j| <= sqrt(2) ||d||_2 <= sqrt(2) B_T. Hence the
approximate top-1 equals the dense top-1 whenever the dense top-1 margin exceeds sqrt(2) B_T.
Implemented in evaluate.py as `margin_certified_frac` (exact arithmetic; float observed).

## 6. Claims register

| ID | Claim | Kind | Status |
|---|---|---|---|
| C1 | T1a state bound | theorem (exact) | derived (§3); float check pending P2 |
| C2 | T1b output bound | theorem (exact) | derived (§3); float check pending P2 |
| C3 | L1 policy keeps B_t <= eps; all-write feasible | theorem (exact) | derived (§4); test pending P2 |
| C4 | Selective path truly omits W_i u_t work and matches masked reference | implementation | untested (P3) |
| C5 | >=1.20x complete-model median streaming speedup | measurement | untested (P3/P5/P6) |
| C6 | Quality loss within tolerance at that operating point | measurement | untested (P5/P6) |
| C7 | Dense reference competitive with simple baseline | measurement | untested (P4/P6) |
| C8 | Oscillation helps vs independently trained zero-frequency control | measurement | untested (P4/P6) |
| C9 | Mechanism relationship to prior work | literature | novelty unresolved (P1) |
| B1 | Certified delta baseline `cdelta`: writes g_i W_i u^_i (cached), ledger beta = |g_i| wbar_i ||u_t - u^_i|| | theorem (exact; same proof as T1 with b - b~ = g W (u - u^)) | implemented, tested (D006) |

## 7. Acceptance protocol (frozen v1.0 targets; see CODEX_RESEARCH_PLAN.md §7 P6)

| Dimension | Target |
|---|---|
| Dense quality | within 1 pp of classification baseline / within 5% relative RMSE of regression baseline at comparable budget |
| Approximation quality | accuracy drop <= 0.5 pp vs frozen dense / RMSE increase <= 2% |
| Runtime benefit | >= 1.20x complete-model median streaming speedup, uncertainty excluding 1.0x |
| Correctness | no unexplained bound violations; all raw excesses disclosed |
| Physics-specific | vs independently trained zero-frequency control (matched state/params/training budget): >= 0.5 pp accuracy or >= 2% lower RMSE, uncertainty excluding no improvement |

Pilot (P4/P5) uses the same approximation and runtime thresholds on validation data; pilot
results are exploratory. epsilon is chosen on validation only. Amendments create a new protocol
revision in DECISIONS.md and never convert a prior failure into a pass.
