# Prior-art batch C: bounded online omission of SSM input writes (H001)

Check date: 2026-11-04. Scope: the H001 execution mechanism. That means a cheap pre-write bound beta_{i,t} = |g_i(u_t)| * wbar_i * ||u_t||, a scalar per-group error ledger e_{i,t} = rho_i e_{i,t-1} + (1 - m_{i,t}) beta_{i,t}, and a causal budget policy that guarantees ||y_t - y_dense_t|| <= epsilon at every step, measured against the dense execution of the same frozen grouped diagonal damped-oscillator SSM.

Access levels used below: full-text inspected / abstract inspected / author-code inspected / unresolved lead. Paraphrases come from WebFetch text extraction (a summarizing model reads the page), not from a human reading the PDF line by line. Equation and section numbers are as the extracted HTML reported them.

---

## 1. Quantization and Sparsity in State Space Models for Efficient Edge Deployment (Delta-SSM)

- **Authors:** listed as Abderraouf Seghairi (thesis). I could not confirm the author's identity in an independent index.
- **Version/date:** ResearchGate listing dated July 2026, per a search-engine snippet.
- **Primary URL:** https://www.researchgate.net/publication/409965202_Quantization_and_Sparsity_in_State_Space_Models_for_Efficient_Edge_Deployment
- **Access level:** unresolved lead. ResearchGate returned HTTP 403. Searches found no arXiv, university-repository or Google Scholar copy. An exact-name search for "Abderraouf Seghairi" returned no relevant person. The only text about the work is a search-engine snippet. It came from a ResearchGate page whose main title is Neil et al. "Delta networks", so it is not even certain the snippet belongs to this thesis.
- **Relevant equation/section:** not available.
- **Mechanism (unverified, snippet only):** Delta-SSM is described as adding spiking or event mechanisms to Mamba-2. It uses temporal redundancy in the inputs, so that updates happen only when activation changes are large enough. It reports about 40% of the original compute on audio and language tasks, with small accuracy loss. I could not verify anything about sparse projections, sigma-delta residuals, threshold rules, or guarantees.
- **Overlap with H001 (potential):** Delta-SSM may skip input-projection work in an SSM using a per-step threshold. That is the same broad family as omitting b_{i,t}.
- **Remaining difference (unverified):** From the snippet, it looks like a Delta-network-style change threshold on activations, with no certified output-error ledger or epsilon budget. This cannot be confirmed without the methods.
- **What would eliminate the difference:** finding all of the following in the thesis methods:
  - a bound on the deviation of the SSM state or output from dense execution that is propagated through the recurrence (for example, geometric decay with |a|);
  - a threshold chosen online from a per-step output budget;
  - a pre-projection cheap bound such as ||u||·||W||.

  Even a sigma-delta residual that keeps the per-element error at or below the threshold, propagated through the diagonal recurrence, would cover most of H001's ledger idea. **This lead must stay open until the full text is obtained.**

## 2. An Exact Instrument for State Usage in Selective State-Space Models, and the Input-Driven Migration It Reveals

- **Authors:** Raktim Bhattacharya.
- **Version/date:** arXiv:2607.11796. v1 was posted 13 Jul 2026; v2 (inspected) was posted 11 Aug 2026. Licence CC BY 4.0.
- **Primary URL:** https://arxiv.org/abs/2607.11796. Code: https://github.com/isrlab/selective-layer-audit
- **Access level:** full-text inspected (arXiv HTML, methods sections) and author-code inspected (README only; no source files read).
- **Relevant equations/sections:** Eq. 1–3 (per-mode decomposition), Eq. 4 (Gram identity), §3.3–3.5, §4.3, Table 3.
- **Verified mechanism:**
  - Because A is diagonal, each channel's output splits exactly into per-mode contributions, y = Σ_i m_{t,i} + f_t (Eq. 3).
  - The paper accumulates a per-(layer, channel, window) Gram tensor G_ij = Σ_t m_{t,i} m_{t,j}. The squared output error from dropping a mode subset is then an exact identity, Σ_{i,j ∉ S} G_ij (Eq. 4), read off offline for any budget.
  - "Dropping" a mode removes that mode's whole output contribution, so the mode is pruned. The paper does not describe omitting individual input writes while keeping the homogeneous dynamics.
  - The input-scheduled pruning uses a first pass over the same window to rank modes, then prunes to a fixed count r per channel.
  - The paper itself says the scheduler "is not a causal deployment protocol." It also claims no compute or memory saving. Exactness holds for Mamba-1 and Falcon-Mamba, but breaks past the gated RMSNorm in Mamba-2.
  - The repo README describes an offline audit plus a pruned forward path and scheduler scripts. It mentions no runtime skipping, error bounds, or certificates.
- **Overlap with H001:**
  - Both exploit diagonal-A decoupling to attribute output error exactly or boundably to per-mode (per-group) terms.
  - Both measure against the dense execution of the same frozen model.
  - Both use input-dependent selection of which modes contribute.
  - The paper also finds that the input-dependent write map B_t drives mode migration, which motivates gating writes per input.
- **Remaining difference:**
  - The paper's error measure is retrospective and windowed (it needs the full window's m_{t,i}). It is not a causal per-step bound.
  - It prunes whole modes rather than omitting writes while the dynamics keep advancing.
  - It has no cheap pre-computation bound that avoids computing W_i u_t.
  - It has no epsilon-budget policy and measures no compute benefit.
- **What would eliminate the difference:** finding any of the following in v2 or later versions, or in the repo's `scheduler.py` / `learned_scheduler.py`:
  - a causal predictive scheduler that skips computation of the write B_t x_t;
  - per-step error accounting with |a|-decay;
  - an online error certificate.

  The paper explicitly lists a predictive scheduler as future work (§4.3), so a follow-up from this author is the most likely source of direct overlap.

## 3. Balanced State Subspaces for Mamba-2 State Compression (BASS + HARP)

- **Authors:** Shang Tingyu, Cao Wenfei.
- **Version/date:** arXiv:2610.36617v1, 27 Oct 2026.
- **Primary URL:** https://arxiv.org/abs/2610.36617
- **Access level:** full-text inspected (arXiv HTML, method sections).
- **Relevant equations/sections:** §3, §5.1 Eq. 3 (empirical Gramians), §5.2 Eq. 4–5 (balanced projection), §5.3 Eq. 6–7 (HARP rank allocation).
- **Verified mechanism:**
  - The paper computes empirical controllability and observability Gramians Wc = (1/T) Σ B_t B_t^T and Wo = (1/T) Σ C_t C_t^T from a calibration pass.
  - It applies balanced truncation through Q = orth(Lc V_{:,1:r}), with B'_t = Q^T B_t and C'_t = Q^T C_t. This is exact under Mamba-2's scalar-identity transition, and lossy only through the truncation.
  - HARP allocates per-layer ranks from the Hankel tail energy, plus a retrieval-sensitivity probe.
  - Everything is fixed after calibration. Every token still writes into the state, through the projected map. The paper states that no guarantee holds for time-varying A_t; the LTI near-optimality is cited only as motivation.
- **Overlap with H001:** Both use control-theoretic, output-aware reasoning about which state directions matter for the output (the write-read subspace). The Hankel and observability weighting is the static analogue of H001's c_i weighting.
- **Remaining difference:**
  - The compression is static and offline: the state dimension shrinks, and no write is ever omitted.
  - It has no per-step bound, no ledger, no causal budget, and no certified output deviation.
- **What would eliminate the difference:** a variant that drops the projected write per token when a bound on its output effect fits a running budget. I found none in v1.

## 4. Event-based / delta / sparse SSM and RNN inference (search hits)

### 4a. Delta Networks for Optimized Recurrent Network Computation

- **Authors:** Neil, Lee, Delbruck, Liu.
- **Version/date:** arXiv:1612.05571 (Dec 2016); published at ICML 2017.
- **Primary URL:** https://arxiv.org/abs/1612.05571
- **Access level:** full-text inspected (via ar5iv).
- **Relevant equations/sections:** Sec. II Eq. 2–5 (exact delta form); Sec. IV Eq. 33–36 (thresholded delta form).
- **Verified mechanism:**
  - The method computes r_t = W Δ + r_{t-1}. An input or hidden element's delta is sent only if |x_t - x̂| > Θ.
  - x̂ is the last value that caused an above-threshold change, which prevents drift.
  - Skipped elements remove whole weight-column MACs. It is applied to GRUs.
  - The paper gives no formal error bound and no propagation through the recurrence. It calls the result "approximately correct" and supports it with empirical accuracy and speedup curves.
- **Overlap with H001:** This is the canonical "skip the input-projection compute when its contribution is small" mechanism for recurrent nets, with a stored reference so that skipped error does not accumulate silently.
- **Remaining difference:**
  - Delta Networks skip by temporal change in the input, not by magnitude.
  - They use a fixed elementwise threshold, not an output-epsilon budget.
  - They have no certified deviation bound through the dynamics.
  - They target nonlinear GRUs rather than a linear diagonal SSM, where geometric error decay can be proved.
- **What would eliminate the difference:** a follow-up (for example, the EdgeDRNN or DeltaRNN hardware papers, not inspected here) that proves a bound on hidden-state deviation as a function of Θ and the recurrent weights, and sets Θ online to meet an output tolerance.

### 4b. Event Neural Networks (EvNets)

- **Authors:** Dutson, Li, Gupta.
- **Version/date:** ECCV 2022; arXiv:2112.00891 v2.
- **Primary URL:** https://arxiv.org/abs/2112.00891
- **Access level:** full-text inspected (arXiv HTML).
- **Relevant equations/sections:** Sec. 3.1 Eq. 3, Sec. 3.2 Eq. 4, Eq. 6 (threshold policy), Supp. F–G.
- **Verified mechanism:**
  - Each neuron keeps an accumulator, a best estimate, and an untransmitted difference d. It transmits when |d| > h (Eq. 6).
  - Eq. 4 shows that d is exactly the gap between the true activation and what has been transmitted, so per-neuron error stays in [-h, h].
  - For h = 0 the output equals the dense network. For h > 0, the paper models per-layer perturbations with ||ε||∞ < h, but explicitly leaves bounding ||y - y'|| as future work.
- **Overlap with H001:**
  - The design is a ledger: a stored residual of untransmitted contribution, with a per-unit bounded error against dense execution of the same frozen model, and threshold-triggered omission.
  - Its argument against Skip-Conv-style gating mirrors H001's motivation for an accumulating ledger.
- **Remaining difference:**
  - EvNets have no network-level or output-level guarantee.
  - They use a fixed threshold rather than a causal epsilon budget that allocates across groups.
  - They target feedforward video CNNs, not a recurrent SSM.
  - Their residual is carried exactly, so omitted contributions are eventually transmitted (deferral). H001 never re-injects omitted writes (omission), and it bounds the resulting drift through rho-decay.
- **What would eliminate the difference:** a follow-up proving an output bound and selecting thresholds from it per frame or step.

### 4c. Accelerating Linear Recurrent Neural Networks for the Edge with Unstructured Sparsity

- **Authors:** Pierro, Abreu, Timcheck, Stratmann, Wild, Shrestha (Intel).
- **Version/date:** arXiv:2502.01330 v1 (Feb 2025); ICML 2025.
- **Primary URL:** https://arxiv.org/abs/2502.01330
- **Access level:** full-text inspected (arXiv HTML).
- **Relevant sections:** §2.1, §2.3–2.5, §3.2–3.3, App. A.
- **Verified mechanism:**
  - The model is an S5-style diagonal complex linear RNN, with iterative magnitude weight pruning and "ReLU-fication" for activation sparsity (including ReLU on the real part of the hidden state).
  - It uses W8A16 QAT and is deployed on Loihi 2, where only nonzero activation and weight pairs are computed.
  - Sigma-delta coding is mentioned only in related work.
  - It has no input-magnitude or temporal-change skipping and no error bound against the dense model; results are empirical Pareto fronts.
- **Overlap with H001:** It is the same model class (diagonal complex linear recurrence) and it skips input-projection MACs at inference on edge hardware.
- **Remaining difference:** Its sparsity is a static architectural or trained property. It has no online bound-driven omission and no guarantee.
- **What would eliminate the difference:** a Loihi or Intel follow-up adding a sigma-delta threshold on SSM inputs with a derived state or output bound.

### 4d. Scalable Event-by-event Processing of Neuromorphic Sensory Signals With Deep State-Space Models

- **Authors:** Schöne, Mohan Sushma, Zhuge, Mayr, Subramoney, Kappel.
- **Version/date:** arXiv:2404.18508 v3, Oct 2024.
- **Primary URL:** https://arxiv.org/abs/2404.18508
- **Access level:** full-text inspected (arXiv HTML).
- **Relevant equations/sections:** Sec. 3.2, Eq. 13–19.
- **Verified mechanism:**
  - Each sensor event writes once: x_k = exp(Λ δ Δ_k) x_{k-1} + B̄ u_k, with irregular Δ_k and an associative scan.
  - Event pooling subsamples outputs by a fixed factor.
  - It has no threshold-based skipping and no error bounds.
- **Overlap with H001:** Homogeneous diagonal dynamics advance in closed form between sparse input writes. That is exactly the "dynamics always advance, writes are sparse" execution pattern.
- **Remaining difference:** Sparsity comes from the sensor. Nothing is omitted relative to a dense model, so there is no approximation and no bound.
- **What would eliminate the difference:** none needed within this paper; it is complementary.

### 4e. Spiking and Event-driven Neuromorphic Mamba Models for Efficient Speech Recognition

- **Authors:** Ahmed, Sun, Castrillon, Vadivel, Tang.
- **Version/date:** arXiv:2606.01135 v1 (31 May 2026); IJCNN 2026.
- **Primary URL:** https://arxiv.org/abs/2606.01135
- **Access level:** abstract inspected.
- **Mechanism (from the abstract):**
  - Event-driven SpeechMamba uses a FATReLU threshold activation, with more than 60% activation sparsity and less than 1% accuracy loss.
  - There is also a spiking variant and a cycle-accurate event-driven simulator.
  - The abstract does not say whether the thresholding gates the input or state projections. No formal bound is stated.
- **Overlap with H001:** Threshold-based omission of small activations feeding Mamba projections at inference.
- **Remaining difference:** Probably no deviation ledger or budget, but the **methods are not inspected**.
- **What would eliminate the difference:** the methods showing thresholds on SSM input writes with a derived error bound or a budgeted threshold.

### 4f. A Deep State-Space Model Compression Method using Upper Bound on Output Error

- **Authors:** Sakamoto, Sato (University of Tokyo).
- **Version/date:** arXiv:2510.14542 v1, Oct 2025.
- **Primary URL:** https://arxiv.org/abs/2510.14542
- **Access level:** full-text inspected (arXiv HTML).
- **Relevant equations/sections:** Theorem 1 (Eq. 23–26), Corollary 1 (Eq. 39–41), Algorithm 1.
- **Verified mechanism:**
  - The method does offline model-order reduction of diagonal-A linear (LQO) blocks in a deep SSM.
  - It gives a provable finite-horizon output error bound: final ℓ∞ output error ≤ a constant times the sum of layerwise h2 errors, with an input-norm constant b (justified via LayerNorm).
  - The bound is not evaluated per step, and no input write is skipped.
- **Overlap with H001:** A certified output-deviation bound against the uncompressed model for a diagonal SSM, built from input norms and system norms. This is the closest example found of the "provable output error vs dense" style in SSMs.
- **Remaining difference:** It is static and offline compression with an H2-type bound, not causal per-step omission with a running ledger and budget.
- **What would eliminate the difference:** an online variant of the bound used to schedule per-step skipping.

### 4g. Classical analogue: event-triggered control

- **Source:** Tabuada, "Event-Triggered Real-Time Scheduling of Stabilizing Control Tasks", IEEE TAC 52(9):1680–1685, 2007 (DOI 10.1109/TAC.2007.904277). Related: the self-triggered control of Anta and Tabuada, and send-on-delta and event-triggered estimation (for example arXiv:1906.03211, not inspected).
- **Access level:** abstract inspected (search snippets only).
- **Mechanism (abstract level):**
  - The control task executes only when an error measure (the gap between the last-transmitted and current state) becomes large relative to the state norm.
  - The paper proves stability and gives lower bounds on inter-execution times.
  - Self-triggered variants predict the next trigger time from a model, with no continuous monitoring.
- **Overlap with H001:** It is the same logic: skip an expensive update while a provable deviation measure stays under a threshold. Self-triggering, which computes a bound before doing the work, is conceptually close to H001's pre-write bound.
- **Remaining difference:**
  - The control literature bounds plant-state deviation for stability, not the output deviation of a learned sequence model against its dense self.
  - It has no multi-group knapsack-style allocation of a shared epsilon budget.
- **What would eliminate the difference:** event-triggered or self-triggered work on multiple channels with a shared error budget allocated greedily across channels. Distributed and multi-agent event-triggered control may contain this; it was not searched here.

---

## Strongest argument that H001 is already covered by these sources

Each ingredient of H001 is present in prior art. Taken together, they reduce H001 to a combination of known pieces applied to a linear model where the analysis is easy:

| H001 ingredient | Where it already appears |
|---|---|
| Skipping small input-projection contributions at recurrent inference | Delta Networks; event-driven Mamba (2606.01135); Delta-SSM, if the snippet is accurate |
| Stored residual or ledger so skipped error is accounted and bounded per unit against dense execution of the same frozen model | EvNets (|d| ≤ h, exact for h = 0) |
| Threshold triggering on a provable deviation measure, including pre-computed (self-triggered) bounds | Event-triggered and self-triggered control (Tabuada and successors) |
| Exact or certified attribution of output error to individual diagonal modes against dense | Exact Instrument (2607.11796); Sakamoto and Sato (2510.14542) |
| Executing a diagonal complex SSM with sparse writes and closed-form homogeneous advance | Schöne et al. (2404.18508); Pierro et al. on Loihi 2 (2502.01330) |

For a linear diagonal recurrence with |a| ≤ 1, it is a textbook triangle-inequality result that omitting a write of norm ≤ beta contributes error ≤ beta, decaying by at most rho per step. So the ledger bound itself is arguably routine. A reviewer could call the epsilon-budget knapsack "event-triggered control with a shared budget."

What I did not find in any inspected source is the specific combination of:

- a pre-projection bound |g|·wbar·||u|| that avoids computing W_i u_t;
- a causal per-step output-epsilon guarantee against dense execution;
- omission (not deferral) with rho-decay accounting.

That absence comes from a bounded search and is not evidence of novelty. The Delta-SSM thesis (unresolved) is the most likely place for direct overlap.

## Search log (all on 2026-11-04)

| # | Tool / mode | Query or URL | Outcome |
|---|---|---|---|
| 1 | WebFetch | researchgate.net/publication/409965202… | HTTP 403 |
| 2 | WebFetch | arxiv.org/abs/2607.11796; arxiv.org/html/2607.11796 | Identity confirmed (v2, Aug 2026); methods read |
| 3 | WebFetch | github.com/isrlab/selective-layer-audit | README read; file list obtained |
| 4 | WebFetch | arxiv.org/abs/2610.36617; arxiv.org/html/2610.36617 | Identity confirmed; methods read |
| 5 | WebSearch standard | "Seghairi \"Quantization and Sparsity in State Space Models for Efficient Edge Deployment\"" | No match |
| 6 | WebSearch extended | Seghairi dissertation 2026 state space models Delta-SSM sigma-delta sparse edge | Snippet-level only; unconfirmed attribution |
| 7 | WebSearch extended | "Delta-SSM" Mamba-2 spiking temporal redundancy event-based | No Delta-SSM paper; found 2606.01135, SpikMamba, Mamba-Spike |
| 8 | WebSearch standard | "\"Abderraouf Seghairi\"" | No relevant person |
| 9 | WebSearch standard | sigma-delta event-based state space model S5 neuromorphic Loihi sparse inference | 2502.01330, 2404.18508, 2505.06417, 2511.22554 |
| 10 | WebFetch | arxiv.org/html/2502.01330v1; arxiv.org/html/2404.18508; arxiv.org/abs/2606.01135 | Read |
| 11 | WebFetch | arxiv.org/pdf/2511.22554v2 | Text not extractable (fall-detection S4D on Loihi 2); not assessed |
| 12 | WebSearch standard | delta encoding Mamba state space model temporal sparsity threshold skip input projection | Nothing specific |
| 13 | WebSearch standard | event-triggered recurrent neural network inference error bound threshold skip update guaranteed deviation | Delta Networks, EvNets, EGRU, Skip RNN, 1906.03211 |
| 14 | WebFetch | ar5iv 1612.05571; arxiv.org/html/2112.00891v2 | Read; arxiv.org/pdf/1612.05571 not extractable |
| 15 | WebFetch | arxiv.org/abs/2605.10994 | Unrelated (internally triggered learning) |
| 16 | WebSearch standard | Tabuada event-triggered real-time scheduling stabilizing control tasks triggering condition error norm | Snippets only |
| 17 | WebSearch extended | certified error bound skipping input updates linear recurrent unit diagonal state space model inference budget per step | 2510.14542, Skip RNN, LRU |
| 18 | WebFetch | arxiv.org/html/2510.14542v1 | Read |
| 19 | WebSearch standard | Mamba inference skip state update tokens dynamic per-step adaptive computation SSM error guarantee 2026 | Nothing relevant (Mamba-3, DeMa) |

Not done in this batch:

- EdgeDRNN, DeltaRNN, DeltaLSTM hardware papers (Gao, Delbruck et al.)
- EGRU (2206.06178) methods
- 2505.06417 (sigma-delta conversion on Loihi 2) methods
- the full text of 2606.01135
- distributed or multi-agent event-triggered control with shared budgets
