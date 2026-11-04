# PRIOR_ART — H001 mechanism comparison

Revision PA1 · 2026-11-04. Detailed per-source records (title, authors, version, URL, access
level, sections, verified mechanism, overlap, remaining difference, eliminating finding, search
log): docs/prior_art/batch_A.md (oscillator/skip/delta), batch_B.md (pruning/bounds),
batch_C.md (recent leads, event-based SSMs, event-triggered control). Those were produced by
three bounded literature agents through a page-summarizing fetch tool; equation numbers must be
spot-checked before citation. An independent review of this file is recorded in
docs/prior_art/review_1.md when complete.

**Overall status (C9): novelty unresolved.** Components are individually established; the
inspected sources do not state the specific composition, but the search was bounded and the
closest lead (Delta-SSM) is unread. Absence in a bounded search is not evidence of novelty.

## Mechanism-by-mechanism comparison

| H001 element | Closest inspected prior work (access) | Overlap | Remaining difference |
|---|---|---|---|
| Damped oscillator diagonal SSM backbone | LinOSS 2410.03943 (full text); D-LinOSS 2505.12171 (full text) | Oscillator SSM with learned damping, eigenvalue magnitude ≤ 1 | None for the backbone itself. H001 uses complex-diagonal exp((−γ+iω)dt), not LinOSS IM/IMEX 2×2 blocks. Caveat (batch_A): in LinOSS block coordinates max\|λ\| is not an operator-norm bound (non-normal); H001's ledger relies on the diagonal basis. |
| Online skipping of input-projection work with state still evolving | Delta Networks (Neil et al. 2017, full text, arXiv version); EventSSM 2404.18508 (full text) | Delta nets: cheap threshold test before matrix–vector product, cached products, state keeps evolving; EventSSM: complex-diagonal SSM advancing exp(ΛΔ) between events, Re(Ch)+Du readout | Delta nets threshold input *change* globally, *defer* (cache) contributions, give no stated bound or per-step tolerance; EventSSM omission is sensor-driven, no controller, no bound. |
| Learned/heuristic update skipping | Skip RNN 1708.06834 (full text); EGRU (full text); SkipW (unresolved) | Online skip decisions, compute budget | Skip RNN freezes the whole state (phase frozen; counterexample in tests/test_phase.py), soft budget via loss, no bound. |
| Ledger e_t = ρ e_{t−1} + (1−m)β | Quamba 2410.13229 Thm 4.1/N.1 (full text); LAST 2411.02824 App. B; AIRE-Prune 2602.00534 App. B.5 Eq. 12–17 (full text); event-triggered estimation (unresolved leads) | Same perturbation recursion through a contractive diagonal system (Quamba, for quantization noise); H∞ per-mode bounds ‖C‖‖B‖/(1−\|λ\|) with ρ = max\|λ\| (LAST, AIRE-Prune) are the worst-case infinite-horizon limit | Quamba bounds quantization error, not omission; LAST/AIRE-Prune prune whole modes statically (dynamics removed). H001's ledger is per-step, causal, input-realized. The ledger itself is standard perturbation analysis — no novelty claim. |
| Exact/offline subset error accounting | Exact Instrument 2607.11796 v2 (full text methods + README, batch_C) ; Balanced subspaces 2610.36617 (full text); Sakamoto & Sato 2510.14542 (full text) | Exact per-mode output decomposition and exact subset-drop error vs dense model; input-dependent mode usage | Window-level, first-pass, non-causal (paper states its scheduler is not a causal deployment protocol); drops modes not writes; no compute saving claimed. Names a predictive scheduler as future work → a follow-up could close the gap. |
| Per-unit bounded deviation from dense execution | Event Neural Networks (ECCV 2022, full text per batch_C) | Residual memory keeps each unit within ±h of dense | Per-unit bound only (output bound left as future work); defers rather than omits; not an SSM write budget. |
| Causal budget triggering on a deviation bound | Event-/self-triggered control (Tabuada 2007; send-on-delta estimation) — unresolved leads | Trigger an update when a provable deviation measure would exceed a threshold; open-loop propagation between events | Classical control of plants/estimators, not a neural SSM execution policy; no multi-group shared budget with pre-projection bounds verified. **Likely conceptual precedent.** |
| Delta/sparse SSM for edge | Delta-SSM (Seghairi, Neftci, Bouhadjar; NeurIPS 2026 workshop; abstract only); Seghairi dissertation (ResearchGate 403, unresolved) | Spiking/delta Mamba-2 reducing redundant ops (~40–45% compute claimed) | **Unresolved** — methods unread. Highest-priority remaining collision. |
| SMR 2405.17534 (event-triggered control applied to SSMs) | unresolved lead | Possibly related | Unread. |

## Strongest argument that H001 is already covered

The state bound is the triangle inequality for a perturbed contractive diagonal linear system
(Quamba's recursion, LAST/AIRE-Prune H∞ bounds). Skipping input projections online while the
state evolves is Delta Networks / EventSSM. Triggering updates to keep a provable deviation
under a threshold is event-triggered control. H001's only new ingredients are (a) the
gate-scaled pre-projection bound |g|·w̄·‖u‖, (b) omission rather than deferral, and (c) a
shared per-step output-ε budget across groups. These are small, natural combinations; a reviewer
could reasonably call them an engineering composition rather than a contribution.

## Proposed contribution (conditional) and its falsification

*Proposed:* For gate-driven diagonal SSMs, a single-pass causal execution policy that decides,
before each input projection, which group writes to omit using a sound pre-projection bound, with
a hard per-step bound on output deviation from the dense model, and that yields a measured
complete-model streaming speedup at a tolerance preserving task quality, beyond what a
delta-network or threshold baseline achieves at matched deviation.

*Falsified (novelty) if:* Delta-SSM, SMR, the Seghairi dissertation, a delta-network follow-up
(DeltaRNN/EdgeDRNN/Spartus), or event-triggered estimation literature already states an online
per-step omission policy for linear recurrent filters with a pre-computation bound and a hard
output-deviation budget. *Falsified (usefulness) if:* at matched realized deviation and quality,
the delta or threshold baseline matches its savings (plan §9), or no operating point meets the
≤0.5 pp / ≤2% RMSE tolerance with ≥1.20x speedup.

## Search/access record
See "Search log" sections of batch_A/B/C (all 2026-11-04). Inaccessible: Seghairi dissertation
(HTTP 403). Abstract-only: Delta-SSM, Thakker 2019, Skip-Vision. Unread leads: SMR 2405.17534,
SkipW, DeltaRNN/EdgeDRNN, Phased LSTM details, event-triggered estimation papers, 2603.02226.
