# Prior-art check, batch A (H001: certified write-omission in a grouped damped-oscillator SSM)

Check date: 2026-11-04. Scope is bounded: 4 assigned sources plus a forward and terminology search.

**Method caveat.** Every page was read through a fetch tool that converts the page and then answers targeted questions about it. "Full-text inspected" means the full method section was available to that tool and was queried section by section. It does **not** mean a human read every line. Equation and section numbers are the ones the tool reported for the rendering named. Check any number before citing it in a paper. The PMLR and arXiv PDFs could not be parsed (raw binary). For those, the ar5iv HTML rendering of the arXiv version was used instead, and its numbering can differ from the camera-ready version.

---

## 1. Oscillatory State-Space Models (LinOSS)

- **Authors:** T. Konstantin Rusch, Daniela Rus (MIT)
- **Version/date:** arXiv:2410.03943v3, 18 Jun 2025 (ICLR 2025)
- **Primary URL:** https://arxiv.org/abs/2410.03943 (HTML: https://arxiv.org/html/2410.03943)
- **Access level:** full-text inspected (HTML, first ~100k of ~112k characters, which covers the main text and the early/mid appendices; the tail of the appendix was not read)
- **Relevant locations:** Eq. 1–2 (second-order ODE, first-order form); Sec. 2.3, Eqs. 3–6 (IM and IMEX discretizations); Sec. 2.4 (associative scan); Prop. 3.1 (IM eigenvalues); Prop. E.1 / Remark 1 (IMEX |λ|=1); Sec. 3.1 (parameterization); Thm. 3.3 (universality); App. C.2–C.3.
- **Verified mechanism:** Each layer is a bank of forced, undamped second-order oscillators y'' = −A y + B u with diagonal A ≥ 0 (A = ReLU(Â) by default) and readout C y + D u. The ODE is discretized two ways, giving 2×2-block recurrences. LinOSS-IM (implicit) has |λ|² = 1/(1+Δt²A_kk) ≤ 1, a numerical dissipation that the paper reads as a forgetting mechanism. LinOSS-IMEX is symplectic with |λ| = 1 exactly (under a Δt condition). Training and inference both use a dense parallel associative scan. The paper reports no skipped, sparse or adaptive computation and no bound on omitted inputs. Its only ε-statement is a universal-approximation theorem.
- **Overlap with H001:** The oscillator backbone with diagonal, non-expansive linear recurrence and a fixed linear readout is the same model class. The structural precondition for H001's ledger (ρ ≤ 1) holds by construction.
- **Remaining difference:** Damping is either absent or comes from the discretization. The state uses real 2×2 blocks, not complex-diagonal exp((−γ+iω)dt). There is no execution mechanism at all. A technical note for H001: in LinOSS's (z, y) block coordinates the transition matrix is generally non-normal. So max|λ| is not by itself a valid operator-norm contraction factor for ||h − h̃||. H001's ρ = max|a| is valid only because H001 works in the complex-diagonal (eigen) basis. If H001 is ever applied to LinOSS-style block states, the ledger needs ||M|| or an eigenbasis condition-number factor.
- **What would eliminate the difference:** A LinOSS appendix or follow-up that omits B u_n writes at inference with a perturbation bound. Nothing like that was found in the inspected text. The last ~12k characters of the appendix were not read.

## 2. Learning to Dissipate Energy in Oscillatory State-Space Models (D-LinOSS)

- **Authors:** Jared Boyer, T. Konstantin Rusch, Daniela Rus (MIT CSAIL)
- **Version/date:** arXiv:2505.12171v2, 30 Sep 2025
- **Primary URL:** https://arxiv.org/abs/2505.12171 (HTML: https://arxiv.org/html/2505.12171)
- **Access level:** full-text inspected (HTML)
- **Relevant locations:** Sec. 2.1 Eq. 1 (damped ODE); Sec. 2.2 Eqs. 2–6 (IMEX discretization, S = I + ΔtG); Prop. 3.1 Eq. 7 (eigenvalues); Prop. 3.2 Eqs. 8–9 and Prop. A.2 Eq. 11 (stability); Prop. 3.3 (bijection onto the unit disk); Sec. 3.3 (parameterization); Sec. 2.3 and Sec. 6 (dense scan, no overhead).
- **Verified mechanism:** The model adds a learned diagonal damping term: x'' = −A x − G x' + B u. It is discretized with IMEX. Damping is learned via G = ReLU(Ḡ). The step Δt = σ(Δt̄) is learnable per layer. A is clamped into a G- and Δt-dependent interval so that (G − ΔtA)² ≤ 4A holds. Under that condition the eigenvalues are a complex-conjugate pair with |λ| = 1/√(1+ΔtG) ≤ 1, a sufficient condition for stability. The (A, G) → eigenvalue map covers the unit disk, and this is used for initialization. Execution is a dense associative scan. The paper reports no skipping, events, adaptive computation, or bound on omitted writes.
- **Overlap with H001:** Learnable damping in an oscillator SSM with a guaranteed |λ| ≤ 1 covers H001's γ ≥ 0 damped-oscillator backbone almost entirely, as H001 itself concedes. D-LinOSS's per-mode |λ| gives exactly the ρ_i that H001's ledger needs.
- **Remaining difference:** The parameterization differs (IMEX 2×2 blocks vs complex-diagonal exponential). There are no groups, no input gates g_i(u), no write omission, no ledger, and no budget policy. The same non-normality caveat as LinOSS applies to block coordinates.
- **What would eliminate the difference:** A D-LinOSS variant or follow-up with inference-time write skipping and a perturbation certificate. None was found.

## 3. Delta Networks for Optimized Recurrent Network Computation

- **Authors:** Daniel Neil, Jun Haeng Lee, Tobi Delbruck, Shih-Chii Liu
- **Version/date:** ICML 2017, PMLR vol. 70 (preprint arXiv:1612.05571, Dec 2016)
- **Primary URL:** https://proceedings.mlr.press/v70/neil17a.html. The text was read via https://ar5iv.labs.arxiv.org/html/1612.05571 because the PMLR PDF could not be parsed.
- **Access level:** full-text inspected (arXiv/ar5iv version; the ICML camera-ready was not compared)
- **Relevant locations (ar5iv numbering):** Eqs. 25–28 (cached pre-activation memories M_t = WΔ + M_{t−1}); Sec. IV, Eqs. 33–36 (thresholded deltas with reference memory x̂, ĥ); Eq. 10 and Sec. V-D (speedup ≈ 1/occupancy); Eqs. 37–41 (rounding and Gaussian noise training); Eq. 44 (L1 on Δh); Sec. V-C (training directly as a delta net).
- **Verified mechanism:** Each input or hidden element is compared with a stored reference value: the last value that caused an above-threshold change. Only elements with |x_t − x̂| > Θ produce a nonzero Δ and update their reference. The gate pre-activations are kept as running sums M_t = W Δ_t + M_{{t−1}}, so a column of W is multiplied only for elements that fire. The skip decision is made **before** the multiply, using only |Δ| against a single scalar Θ. Weight norms are not used. The reference memory exists to stop drift. The paper argues, qualitatively, that small errors are corrected once a large enough change fires. It derives no bound on hidden-state or output error against the dense network and gives no analysis of error over time.
- **Overlap with H001:** The core execution idea is the same: a cheap per-element test before the matrix-vector product, which skips that product's contribution, applied to a recurrent net whose state keeps evolving. Because of the reference-memory construction, the pre-activation error at any step is W(x_t − x̂_t) with each |x_{t,i} − x̂_i| ≤ Θ. That gives an immediate implicit bound of ||W||·Θ·√n. The paper does not state this bound, but it follows in one line. Applied to a linear diagonal SSM, the same construction gives a ledger of H001's form: e_t ≤ ρ e_{t−1} + ||W||·||u_t − û_t||.
- **Remaining difference:** (i) The test is on input **change** against a global Θ. It is not a gate-scaled norm bound |g|·w̄·||u|| per group. (ii) Delta nets **defer** skipped contributions (they are caught up later). H001 **drops** them permanently and relies on decay ρ < 1. (iii) There is no explicit certificate, no per-step output budget ε, and no causal allocation of a budget across groups. (iv) The setting is nonlinear GRU/LSTM, not an LTI or gated linear SSM.
- **What would eliminate the difference:** Any follow-up in the delta-network line that (a) states a worst-case state or output error bound for a linear or SSM layer and (b) picks thresholds online to meet a per-step tolerance. Candidates to check: DeltaRNN (FPGA 2018), EdgeDRNN (arXiv 2012.13600), Spartus, and the Gao/Delbruck/Liu thesis work. These were seen only in search results (unresolved).

## 4. Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks

- **Authors:** Víctor Campos, Brendan Jou, Xavier Giró-i-Nieto, Jordi Torres, Shih-Fu Chang
- **Version/date:** arXiv:1708.06834v3, 5 Feb 2018 (v1 22 Aug 2017); ICLR 2018
- **Primary URL:** https://arxiv.org/abs/1708.06834. The text was read via https://ar5iv.labs.arxiv.org/html/1708.06834 because the PDF could not be parsed.
- **Access level:** full-text inspected (ar5iv)
- **Relevant locations:** Sec. 3, Eqs. 1–8 (Eq. 2 binarized gate; Eq. 3 gated state; Eqs. 4–5 incremental update probability; Eq. 6 skip count; Eq. 7 straight-through estimator; Eq. 8 budget loss); Sec. 4.1–4.3 and App. A (experiments).
- **Verified mechanism:** A binary gate u_t = round(ũ_t) chooses between a full update s_t = S(s_{t−1}, x_t) and a pure copy s_t = s_{t−1}. The update probability is emitted after each real update and accumulated during skips (Eq. 5). The decision for step t is therefore fixed before x_t is seen. Training uses a straight-through estimator and a soft budget loss λ·Σ u_t. Skipped steps ignore the input and do no state computation. The state is **frozen**, and no dynamics advance. The paper gives no error bound against an always-updating RNN. All evidence is empirical (adding task, MNIST, Charades, IMDB, UCF-101).
- **Overlap with H001:** It shares H001's high-level aim: learned or causal omission of per-step work under a computation budget, with the decision made before the expensive computation.
- **Remaining difference:** Skip RNN freezes the whole state (copy), while H001 always advances the homogeneous dynamics and omits only the input write, per group. Skip RNN's budget is a soft training penalty on compute. H001 enforces a hard per-step **error** budget ε at inference. Skip RNN has no certificate. Its decision ignores the content of x_t, while H001's bound depends on ||u_t|| and the gate g_i(u_t).
- **What would eliminate the difference:** A Skip-RNN variant with per-unit or per-group skipping where the skipped units keep their autonomous dynamics, plus a stated error bound. Not found. SkipW (Mayet et al., ICLR 2021) guarantees a strict **compute** budget, not error, and is an unresolved lead because its methods were not read.

---

## 5. Follow-up and terminology search: most relevant hits

| # | Work | Access | Mechanism actually used | Relation to H001 |
|---|---|---|---|---|
| 5a | **EventSSM**: Schöne, Mohan Sushma, Zhuge, Mayr, Subramoney, Kappel, "Scalable Event-by-event Processing of Neuromorphic Sensory Signals With Deep State-Space Models", arXiv:2404.18508v3 (9 Oct 2024), ICONS 2024. https://arxiv.org/abs/2404.18508 | full-text inspected (ar5iv) | Diagonal complex SSM discretized at irregular event times: x_k = exp(ΛδΔ_k) x_{k−1} + B̄ u_k (Eqs. 14–17), readout y_k = Re(C̄ x_k) + D̄ u_k (Eq. 16), associative scan (Eqs. 18–19). Between events the state decays and rotates with no input write. Event pooling subsamples **outputs** after each layer, while all events still write. No error bound. | **Structurally closest** to "homogeneous dynamics always advance, input written only sometimes", with the same Re(C h) + D u readout. The difference is that omission comes from the data (sensor sparsity), not from a controller, and there is no certificate or budget. |
| 5b | **EGRU**: Subramoney, Nazeer, Schöne, Mayr, Kappel, "Efficient recurrent architectures through activity sparsity and sparse back-propagation through time" (v1 title "EGRU: Event-based GRU..."), arXiv:2206.06178v3 (9 Mar 2023), ICLR 2023 | full-text inspected (ar5iv) | Internal state c evolves by GRU dynamics every step (Eq. 2, continuous-time Eq. 5). Output y = c·H(c − ϑ) with a trainable per-unit threshold ϑ_i and reset after an event. Skips **outgoing** communication and MACs for silent units, and sparse BPTT through a surrogate gradient (Sec. 3.2–3.3). Updates of silent units can be computed retroactively. No error bound against the dense GRU. | Shares "state keeps evolving while transmission or writes are skipped" and threshold-gated sparsity. Here sparsity is learned into the model (it **is** the model). There is no approximation of a frozen dense model and no certificate. |
| 5c | **Delta-SSM**: Seghairi, Neftci, Bouhadjar, "Delta-SSM: Exact parallelization of spiking State Space Models for Efficient Sequence Modeling", NeurIPS 2026 workshop (AXIOM) listing. https://neurips.cc/virtual/2026/157011 | abstract inspected | Mamba-2-based spiking SSM with a custom integer spiking neuron whose dynamics have an exact time-parallel form that reduces to activation quantization. The abstract claims fewer redundant operations across time and about 45% of the original compute. It does not say which operations are skipped or give an error bound. | Possibly a delta/event SSM in the Neil et al. sense. **Unresolved:** the methods were not read, and no arXiv version was found. High priority to read. |
| 5d | **S4D on Loihi 2**: Meyer et al., "A Diagonal Structured State Space Model on Loihi 2 for Efficient Streaming Sequence Processing", arXiv:2409.15022 (Sep 2024), NeurIPS 2024 workshop | full-text inspected (HTML) | Token-by-token S4D on neuromorphic hardware with diagonal states advanced every step, ReLU for activation sparsity, and 8/24-bit PTQ plus QAFT (Sec. 3.1–3.2). No sigma-delta coding, no input-write skipping, no error bound. | Same "diagonal SSM, dense advance" execution setting. It exploits only exact zeros. |
| 5e | **Pierro, Abreu, Timcheck, Stratmann, Wild, Shrestha**, "Accelerating Linear Recurrent Neural Networks for the Edge with Unstructured Sparsity", arXiv:2502.01330v1 (3 Feb 2025) | full-text inspected (HTML) | Iterative magnitude pruning to 90% weights. Activation sparsity via ReLU-fication, including a ReLU on the real part of the S5 hidden state (Sec. 2.4). Zero activations skip MACs on Loihi 2 (Eq. 3). Diagonal recurrence applied every step (Eq. 1). Only empirical fixed-point mismatch is reported (App. A.3.2). | Exact-zero skipping in a linear diagonal RNN. No tolerance-based omission, no certificate. |
| 5f | **Thakker**, "Skipping RNN State-updates Without Retraining the Original Model", SenSysML workshop 2019 (Arm Research). https://developer.arm.com/community/arm-research/b/articles/posts/skipping-rnn-state-updates-without-retraining-the-original-model | abstract inspected (blog summary only) | A small pre-RNN predictor (NN, tree, or linear model) looks at the current hidden state and input and predicts whether the input will change the memory significantly (L2 or cosine distance). The threshold is calibrated from training-data distributions. The frozen pretrained RNN is not retrained. Results are empirical only (for example, more than 25% fewer MACs with at most 0.3% accuracy loss on SST). | **Close in spirit** to H001's "decide before the write on a frozen model, using a cheap predictor of impact". The predictor is learned and heuristic, with no guarantee, and the handling of skipped state was not confirmed. |
| 5g | **Phased LSTM**: Neil, Pfeiffer, Liu, NeurIPS 2016, arXiv:1610.09513 | unresolved lead (search snippets only) | An oscillatory time gate (period τ, open ratio r_on, phase s) allows updates only in an open phase. In the closed phase the state is held, with a small leak. | Scheduled, not content-adaptive, omission. It has an oscillation but the state is held, not rotated. No bound. Low overlap. |
| 5h | Event-triggered estimation/control (e.g. arXiv:1906.03211, arXiv:2510.04814, TU/e "Dynamic thresholds in robust event-triggered control for discrete-time linear systems") | unresolved lead (search snippets only) | Transmit or update only when a deviation exceeds a threshold, and propagate open-loop dynamics in between. Classical results give bounded estimation error for linear systems. | **Mathematical precedent** for H001's ledger: a bounded perturbation through a contractive LTI system with threshold-triggered updates. Not a neural-SSM execution mechanism. Should be read and cited. |

Also seen but not relevant enough to list: PerfMamba (arXiv:2511.22849; state-channel pruning, static), BLASST (arXiv:2512.12087; bounded-error skipping of attention blocks, an attention analogue of a certified skip), SpikingSSMs (arXiv:2408.14909), and "Activity Sparsity Complements Weight Sparsity" (arXiv:2311.07625; EGRU-based).

---

## Strongest argument that H001 is already covered by these sources

Take Delta Networks (cheap per-element test before the multiply, cached matrix-vector products, recurrent state that keeps evolving). Apply it to a D-LinOSS-style damped oscillator SSM with |λ| ≤ 1 (or to the complex-diagonal EventSSM form). That already gives an execution scheme with cheap pre-write tests and a dense frozen model as reference. EventSSM already shows that a diagonal complex SSM can advance exp(ΛΔ) across steps with no input write and the same Re(C h) + D u readout. The error certificate itself is the textbook bound for a perturbed stable LTI system: ||h_t − h̃_t|| ≤ Σ_k ρ^{t−k} ||δb_k||, in recursive form e_t = ρ e_{t−1} + ||δb_t||. Event-triggered estimation and control theory has used this kind of bound, with thresholds chosen to keep error under a tolerance, for linear systems for decades (5h, not yet read). The bound β = |g|·w̄·||u|| is the submultiplicative inequality ||g W u|| ≤ |g|·||W||·||u||. Choosing the smallest c_i β_i items that fit a residual budget is a greedy knapsack. On this reading H001's ingredients are all known, and the contribution reduces to the composition. That composition would be: per-group omission (not deferral) with dynamics advanced, a gate-scaled pre-multiply bound in place of Neil et al.'s change threshold, and a hard per-step output certificate against the frozen dense model, allocated causally. **No inspected source states or uses that composition.** Its value as a contribution therefore depends on H001's measured benefit (compute saved at a given ε versus a delta-network or Skip-RNN baseline on the same model), not on any of the individual ideas. Not finding the composition here is not evidence of novelty. The search was bounded, and Delta-SSM (5c), SkipW, the DeltaRNN/EdgeDRNN line and the event-triggered-control literature were not read at method level.

Specific baselines that H001's evaluation should therefore include:
1. A delta-network-style input-change threshold on u_t (deferred, not dropped) on the same frozen SSM, whose implicit error bound is similar.
2. Skip-RNN-style whole-state freezing.
3. EventSSM-style dynamics-only advance with data-driven omission.

---

## Search log (all on 2026-11-04)

Direct fetches:
- https://arxiv.org/html/2410.03943 (LinOSS v3), first 100k characters
- https://arxiv.org/html/2505.12171 (D-LinOSS v2)
- https://proceedings.mlr.press/v70/neil17a/neil17a.pdf: unparseable binary. The fetch tool cached a copy automatically; it was not opened.
- https://ar5iv.labs.arxiv.org/html/1612.05571 (Delta Networks, arXiv version)
- https://arxiv.org/abs/1708.06834 (Skip RNN metadata)
- https://arxiv.org/pdf/1708.06834v3: unparseable binary. Cached automatically, not opened.
- https://ar5iv.labs.arxiv.org/html/1708.06834 (Skip RNN full text)
- https://arxiv.org/abs/2206.06178, https://ar5iv.labs.arxiv.org/html/2206.06178 (EGRU)
- https://arxiv.org/html/2409.15022 (S4D on Loihi 2)
- https://neurips.cc/virtual/2026/157011 (Delta-SSM listing)
- https://arxiv.org/html/2502.01330v1 (Pierro et al.)
- https://arxiv.org/abs/2404.18508, https://ar5iv.labs.arxiv.org/html/2404.18508 (EventSSM)
- https://developer.arm.com/community/arm-research/b/articles/posts/skipping-rnn-state-updates-without-retraining-the-original-model (Thakker)

WebSearch queries (standard mode):
1. delta state space model temporal sparsity Mamba delta updates inference
2. event-based GRU EGRU sparse activity recurrent network Subramoney
3. error-bounded skipping recurrent neural network certified adaptive computation guarantee
4. sigma-delta S4D state space model Loihi 2 temporal sparsity streaming
5. event-triggered state space model inference threshold skip input update error bound
6. "Delta-SSM" spiking state space models exact parallelization arxiv
7. state space model activation sparsity delta encoding S5 neuromorphic "temporal sparsity" quantization pruning
8. DeltaRNN Gao Neil Delbruck FPGA delta threshold GRU temporal sparsity 2018
9. skip input projection linear recurrent model dynamics continue phase rotation skipped tokens state space model inference
10. delta networks recurrent provable bound accumulated error threshold hidden state deviation dense
11. Mamba skip state update tokens inference speedup dynamic token skipping selective state space
12. sparse delta linear recurrent unit LRU event-driven inference thresholded input changes complex diagonal
13. Scalable event-by-event processing neuromorphic sensory signals deep state-space models Schöne
14. Phased LSTM Neil Pfeiffer Liu time gate oscillation update only during open phase
15. guaranteed output error budget skip computation linear time-invariant recurrent layer per-step tolerance epsilon neural network inference

Not done (gaps):
- Extended-mode search
- Forward-citation crawl via Semantic Scholar or Google Scholar
- Methods of Delta-SSM, SkipW, DeltaRNN/EdgeDRNN, Phased LSTM, and the event-triggered-control papers
- Author code for any of these works
