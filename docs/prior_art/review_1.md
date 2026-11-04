# Independent review 1 of PRIOR_ART.md (revision PA1)

Reviewer: independent agent, 2026-11-04. Scope: docs/PRIOR_ART.md, docs/SPEC.md §1–4, and
batch_A/B/C. I read sources with a page-summarizing fetch tool (WebFetch). Equation and section
numbers below are the ones the tool reported, so spot-check them before citing. No file was
downloaded, no code was run, and I did not sign in anywhere. OpenReview showed a bot-verification
page, and I did not try to get around it.

## 1. Spot-checks of consequential claims

| # | Claim in PRIOR_ART.md | Source checked (access) | Verdict |
|---|---|---|---|
| S1 | AIRE-Prune 2602.00534 App. B.5 Eq. 12–17 gives H∞ per-mode bounds ‖C‖‖B‖/(1−\|λ\|) with ρ = max\|λ\|, and prunes whole modes statically | arxiv.org/html/2602.00534v1 (full text, App. B.5) | **Correct, slightly conflated.** Eq. 13 gives ‖G_i‖∞ ≤ α_i/(1−\|λ_i\|), where α_i = ‖C_{:,i}‖‖B_{i,:}‖. ρ = max over the *pruned* modes of \|λ_i\|, and it appears only in κ(ρ) = √((1+ρ)/(1−ρ)) in the energy-form bounds (Eq. 15–17): ε ≤ κ(ρ)·min{Σ√E_i, √\|T\|·√ΣE_i}. Pruning is static and offline, with one global threshold. There is no per-step or input-conditioned budget. The paper lists **input-dependent extension to Mamba (Monte-Carlo over gates)** as future work, and PRIOR_ART.md omits this. That makes AIRE, like Exact Instrument, a line whose follow-up could close the gap. |
| S2 | Quamba 2410.13229 "Thm 4.1/N.1" uses the same perturbation recursion through a contractive system, for quantization noise | arxiv.org/html/2410.13229v2 (full text, both halves) | **Correct in substance, wrong location.** The proof states Δ(t) = A(t)Δ(t−1) + Bδx(t), which matches H001's d_t recursion with (1−m)b_t in place of Bδx. The proof is in **App. N.2** ("Theoretical Error Analysis"). N.1 is "Empirical Analysis of A Matrices". The assumption is ‖A(t)‖₂ ≤ a·e^{t−T} with 0<a<1, and ‖B‖ ≤ b. The bound is a-priori, ‖Δ(T)‖ ≤ εb/(1−a) with a uniform ε. Calibration is offline and empirical (clipping percentile), and the bound is never used at run time. |
| S3 | Exact Instrument 2607.11796 says its scheduler "is not a causal deployment protocol"; it drops modes, works at window level, claims no compute saving, and names a predictive scheduler as future work | arxiv.org/html/2607.11796 (v2, 11 Aug 2026; full text §3–5) | **Correct.** §4.3: "It is not a causal deployment protocol, because its mask is allowed to depend on tokens in the scored window." It uses two passes (measure, then prune the same window to the top-r modes). It drops modes (state dims), not writes. The per-layer error is exact for Mamba-1/Falcon-Mamba and not exact past the Mamba-2 gated RMSNorm. It claims no compute saving. §4.3 and §5 name a predictive scheduler as future work. Note that batch_B.md lists this source as "abstract inspected only", which contradicts batch_C.md and PRIOR_ART.md (full text). Reconcile batch_B. |
| S4 | Delta Networks "threshold input *change* globally, *defer* (cache) contributions, give no stated bound" | ar5iv 1612.05571 (full text, §II–VI) | **Correct but incomplete, and the omission matters.** The threshold applies per element to the change from a stored reference (the last value that triggered an update), with a single Θ symbol per experiment. Sub-threshold change is carried forward, not lost, so after each step \|x_i − x̂_i\| ≤ Θ (an implied invariant the paper does not state formally). No bound is stated, and Θ is swept, not chosen to meet a tolerance. **However, the invariant immediately gives a bound for an SSM:** for b̃_t = g(u_t)W û_t, ‖b_t − b̃_t‖ ≤ \|g\|·‖W‖·Θ√d, so H001's own ledger certifies a delta-network SSM at no extra cost. PRIOR_ART.md should not suggest that delta nets are inherently uncertifiable. |
| S5 (partial) | LAST 2411.02824 uses H∞ subsystem scores | arxiv.org/abs/2411.02824 (abstract only) | Abstract confirms H∞ subsystem norms, layer-wise energy normalization, and pruning without retraining (NeurIPS 2024). The exact per-mode formula is not on the abs page. For a diagonal mode with rank-one numerator, ‖C_iB_i/(1−λ_i z⁻¹)‖∞ equals ‖C_i‖‖B_i‖/(1−\|λ_i\|) exactly, so the claim is consistent. Batch_B says App. B was read, but this review did not re-verify it. |

## 2. Resolution of unread leads

### L1. SMR: State Memory Replay for Long Sequence Modeling
- **Authors:** Biqing Qi, Junqi Gao, Kaiyan Zhang, Dong Li, Jianxing Liu, Ligang Wu, Bowen Zhou.
- **Date/venue:** arXiv:2405.17534, v1 27 May 2024, v2 8 Jun 2024; Findings of ACL 2024.
- **URL:** https://arxiv.org/abs/2405.17534 ; https://arxiv.org/html/2405.17534v2
- **Access:** full-text inspected (methods §3.1–3.3, Prop. 1, Thm 1, Listing 1).
- **Verified mechanism:** Event-triggered control is used only as an *analysis lens* (§3.1, Eq. 7–13: Lyapunov-based triggering for a 2-D illustrative system). The method itself is dense. It uses a learnable causal Conv1d + sigmoid gate on the input, x_k = Ā x_{k−1} + B̄ u_k σ(K_τ ∗ u)_k (Eq. 15), applied to every step. Prop. 1 bounds the output perturbation from sampling deviation by Σ\|λ_max\|^{t−i}·c·b·ε_i, which is unbounded if \|λ_max\| ≥ 1.
- **Overlap with H001:** Prop. 1 is the same geometric perturbation sum as T1 (another instance of "the ledger is standard"). The input gate on B̄u resembles g_i(u).
- **Remaining difference:** No skipping or omission, no run-time trigger, no budget, no hard tolerance against a dense reference. **Not a collision. Close this lead.**

### L2. Delta-SSM: Exact parallelization of spiking State Space Models for Efficient Sequence Modeling
- **Authors:** Abderraouf Seghairi, Emre Neftci, Younes Bouhadjar (confirmed from the NeurIPS listing; batch_C says the attribution is unconfirmed, so update it).
- **Date/venue:** NeurIPS 2026 workshop AXIOM: Foundations of Efficient Deep Learning (poster).
- **URL:** https://neurips.cc/virtual/2026/157011 ; OpenReview https://openreview.net/forum?id=0b75H3NcHp (bot verification blocked access). Two searches found no arXiv version.
- **Access:** abstract inspected. Methods are still unread.
- **Mechanism (abstract-level):** A Mamba-2-based spiking SSM with a custom integer spiking neuron whose sequential dynamics have an exact time-parallel form that "reduces to activation quantization". It reduces redundant computation across time for event-based execution, at about 45% of the original compute with limited loss on audio and language tasks.
- **Overlap with H001:** Event-based skipping of redundant work in an SSM on edge hardware.
- **Remaining difference (provisional):** The abstract describes a quantization/delta-encoding design for parallel *training*. It mentions no deviation bound, per-step tolerance, or budget allocation across groups. The "reduces to activation quantization" wording suggests sigma-delta-style deferral (residual carried) rather than omission. **Still unresolved and still the highest-priority collision.** The user would need to open the OpenReview PDF themselves.

### L3. EdgeDRNN: Recurrent Neural Network Accelerator for Edge Inference
- **Authors:** Chang Gao, Antonio Rios-Navarro, Xi Chen, Shih-Chii Liu, Tobi Delbruck.
- **Date/venue:** arXiv:2012.13600 (25 Dec 2020); IEEE JETCAS 10(4):419–432, Dec 2020.
- **URL:** https://arxiv.org/abs/2012.13600 ; https://ar5iv.labs.arxiv.org/html/2012.13600
- **Access:** full-text inspected (§II-C, II-D, IV-C, VI).
- **Verified mechanism:** DeltaGRU uses per-element thresholds Θx and Θh against stored references (Eq. 2), and accumulates delta memories M_t = W_x Δx_t + W_h Δh_{t−1} + M_{t−1} (Eq. 3). Zero deltas skip weight columns.
- **Error bound or tolerance?** **No.** It derives no bound on hidden-state or output error against the dense GRU. Thresholds are swept empirically (classification: Θ=64 ≈ 0.25 fp, chosen just before WER rises; regression: (Θx,Θh)=(4,8)). It notes that Θ allows a run-time accuracy/latency trade-off and leaves a dynamic, guided threshold search as future work.
- **Overlap:** Online skipping of input- and hidden-projection columns, with a run-time-adjustable threshold.
- **Remaining difference:** No bound, no tolerance-driven threshold choice, deferral not omission, nonlinear GRU (H001's linear-recurrence ledger would not apply directly to the hidden-delta path). **Not a collision. Close this lead.** DeltaRNN and Spartus are probably the same in this respect but were not inspected.

### L4. Representative event-triggered state estimation: Distributed Event-based State Estimation
- **Author:** Sebastian Trimpe (MPI-IS).
- **Date:** arXiv:1511.05223 (Nov 2015, per identifier).
- **URL:** https://ar5iv.arxiv.org/html/1511.05223
- **Access:** full-text inspected (first ~100k characters, including Thm 5; Thm 11 summarized).
- **Verified mechanism:** Agent i transmits y_i(k) iff ‖y_i(k) − C_i x̂(k\|k−1)‖ ≥ δ_i (send-on-delta on the innovation). Inputs use send-on-delta against the last sent value. Between events the estimator propagates the model open-loop (prediction step only). **Thm 5:** the error between the event-based and centralized estimators satisfies ‖e‖ ≤ c‖e(0)‖ + (c/(1−ρ))‖L‖‖δ‖, using ‖((I−LC)A)^k‖ ≤ cρ^k.
- **Overlap with H001:** This is the closest *conceptual* precedent found. Omitting an update in a stable linear recursion while the dynamics keep evolving, with a provable bound on deviation from the always-updating reference that scales with the threshold through a geometric factor 1/(1−ρ). Structurally, this is T1 in worst-case form.
- **Remaining difference:** (a) The trigger inspects the realized innovation, so the omitted quantity is computed, and the saving is communication, not computation. H001 must decide *before* computing W_i u, using a norm surrogate. (b) δ is a design-time tuning knob with a worst-case steady-state bound. Nothing maintains a realized ledger or chooses omissions online to keep a hard per-step budget. (c) There is no shared budget across groups and no learned gate. Lit search snippets (Andrén & Cervin 2016; Hu et al. 2017) show stochastic send-on-delta variants with error-covariance bounds. They were not inspected.

## 3. Verdict

**Exact match:** None of the sources inspected here or in batches A–C (including SMR, EdgeDRNN,
Delta Networks, Trimpe 2015, AIRE-Prune, Quamba, and Exact Instrument) states the full H001 mechanism:
causal per-step omission of input writes in a diagonal SSM, decided *before* the projection with a
sound bound |g|·w̄·‖u‖, a realized ledger e_t = ρe_{t−1} + (1−m)β, and online selection to
keep a hard output budget ε. Delta-SSM's methods are unread, so the verdict "no exact match" is
provisional. Absence in a bounded search is not evidence of novelty, and this review does not
claim novelty.

**Strongest trivial-composition argument (stronger than the one in PRIOR_ART.md):** Take
Delta Networks or EdgeDRNN's reference-memory rule on u_t. Its invariant ‖u − û‖∞ ≤ Θ gives a per-step
write error ≤ |g|·‖W‖·Θ√d. Push that through the contractive diagonal dynamics with Quamba's recursion
(or SMR Prop. 1), and you get exactly Trimpe's Thm-5-style bound, c/(1−ρ) × threshold. That is a
*certified* delta-network SSM, assembled from three cited results with one line of algebra. H001
changes three things: (i) omission instead of deferral, (ii) a realized ledger instead of the
worst-case 1/(1−ρ) bound, and (iii) per-step greedy selection under ε instead of a fixed Θ. Each is a standard move
(online tightening of a worst-case bound, knapsack over a budget). Point (i) is arguably a
*disadvantage*: deferral keeps the input-representation error at ≤ Θ per element, while omission's
error persists in the state and never replenishes for undamped groups (SPEC §4). A reviewer could
call H001 "event-triggered estimation applied to SSM compute, with a cheaper trigger." The
remaining defensible content is empirical. Does the pre-projection trigger with a realized ledger
give a better speedup-versus-certified-deviation frontier than a certified delta baseline on real
hardware?

## 4. Suggested corrections to PRIOR_ART.md (not applied)

- Ledger row: change "Quamba 2410.13229 Thm 4.1/N.1" to "Thm 4.1, proof App. N.2 (N.1 is
  empirical)". Record its assumption ‖A(t)‖ ≤ a·e^{t−T}, 0<a<1, and its a-priori bound εb/(1−a).
- Ledger row: AIRE-Prune 1/(1−|λ_i|) is Eq. 13. ρ = max over *pruned* modes appears in κ(ρ) in
  Eq. 15–17 (energy form). Add that AIRE names input-dependent (Mamba/gate) extension as future work.
- Ledger row: add SMR 2405.17534 Prop. 1 (Σ|λ_max|^{t−i}·c·b·ε_i) as a further instance of the
  same perturbation sum in an SSM.
- Replace the "SMR … unresolved lead / Possibly related / Unread" row with: full text inspected.
  ETC is used only for analysis, the model is dense, with no skipping and no bound against dense. Not a collision.
- Delta-nets row: replace "threshold input *change* globally" with "per-element threshold on change
  from a stored reference (single Θ)". Add that the reference rule implies |x − x̂| ≤ Θ, so a
  delta-network SSM is certifiable with H001's own ledger. State that the matched-deviation baseline (plan §9)
  should therefore be a **certified delta baseline** (delta on u_t with ledger
  β = |g|·w̄·Θ√d, or the exact residual norm), not only an uncertified threshold.
- Add EdgeDRNN 2012.13600 (full text): no bound, Θ swept empirically, dynamic threshold search
  left as future work. Remove EdgeDRNN from the "unread leads" and novelty-falsifier lists (DeltaRNN and
  Spartus remain unread).
- Event-triggered row: replace "unresolved leads" with Trimpe 1511.05223 (full text), Thm 5
  ‖e‖ ≤ c‖e(0)‖ + c/(1−ρ)·‖L‖·δ. Keep "likely conceptual precedent", now verified. Name the
  difference precisely: the trigger computes the realized innovation (it saves communication, not computation),
  δ is fixed at design time, and there is no realized ledger or online budget.
- Delta-SSM row: authors are confirmed by the NeurIPS listing. Add the OpenReview id 0b75H3NcHp (blocked
  by bot verification, so the user should open it manually). Abstract wording is "reduces to activation quantization", ~45%
  compute. Keep it **unresolved, highest priority**. Treat the Seghairi dissertation as a separate item.
- Strongest-argument section: replace it with (or add) the certified-delta-network composition in §3
  above, and note that omission versus deferral is a trade-off, not purely an ingredient.
- Search record: reconcile batch_B's "abstract inspected only" for 2607.11796 with batch_C's full-text
  reading. Move SMR, EdgeDRNN, and one event-triggered estimation paper out of "unread leads".

## 5. Search/fetch log (this review)

| # | Tool | Target | Outcome |
|---|---|---|---|
| 1 | WebFetch | arxiv.org/html/2602.00534v1 | AIRE App. B.5 verified |
| 2–3 | WebFetch | arxiv.org/html/2410.13229v2 (offset 0, 100000) | Quamba Thm 4.1, App. N.2 recursion verified |
| 4 | WebFetch | arxiv.org/html/2607.11796 | §4.3 quote verified |
| 5 | WebFetch | ar5iv 1612.05571 | Delta Networks threshold rule verified |
| 6 | WebFetch | arxiv.org/abs/2411.02824 | LAST abstract only |
| 7–8 | WebFetch | arxiv.org/abs + html 2405.17534v2 | SMR resolved |
| 9–10 | WebFetch | arxiv.org/abs + ar5iv 2012.13600 | EdgeDRNN resolved |
| 11 | WebSearch | Delta-SSM Seghairi Neftci Bouhadjar … | NeurIPS listing only |
| 12 | WebFetch | neurips.cc/virtual/2026/157011 | Authors confirmed; OpenReview link |
| 13–14 | WebFetch | openreview.net forum / pdf 0b75H3NcHp | Bot-verification page; not bypassed |
| 15 | WebFetch | arxiv.org/abs/2604.01295 | Unrelated (PHCSSM, Chiang) |
| 16 | WebSearch | Seghairi Bouhadjar Neftci Mamba-2 integer spiking … arXiv | No arXiv copy found |
| 17 | WebSearch | send-on-delta event-based estimation bounded error | Candidate list |
| 18 | WebFetch | ar5iv 1511.05223 | Trimpe Thm 5 verified |
