# Pivot review: two alternative directions after the H001 rejection

Reviewer: research sub-agent, 2026-11-04. This is a bounded review of about 25 tool calls. Sources
were read with a page-summarizing fetch tool (WebFetch) and a web search tool. Section and equation
numbers are as that tool reported them, so spot-check them before citing. No code was run and no
one was contacted. One disclosure: a WebFetch of `arxiv.org/pdf/2305.01122` returned raw PDF bytes,
and the tool cached them automatically under `~/.claude/projects/.../tool-results/`. I did not ask
for that, and I did not open or use the cached file. LAMP was re-read through ar5iv HTML instead.
**A search that finds nothing is not evidence of novelty.** Every "not found" below means only "not
found in this bounded search".

Context (from the caller): H001 was rejected. The omitted writes carried real information, the
worst-case ledger was about 1000x conservative, and the only savings came from exact input repeats
(memoization) or from uncertified delta caching. That lesson drives the feasibility judgments
below: *certificates rarely create savings on their own. Savings come from structure the input
actually has.*

---

## Direction 1: Online order changes in a passive learned memory reservoir

Claim to test: change the reservoir's state order online, and control both (i) the energy discarded
at each reset and (ii) the future port-output error under a declared forcing class.

### Sources

**1.1 ApHIN: Data-driven identification of latent port-Hamiltonian systems**
- Authors: Johannes Rettberg, Jonas Kneifl, Julius Herb, Patrick Buchfink, Jörg Fehr, Bernard Haasdonk.
- Date: arXiv:2408.08185, v1 15 Aug 2024, v2 16 Aug 2024.
- URL: https://arxiv.org/abs/2408.08185 ; https://arxiv.org/html/2408.08185v2
- Access: **full text inspected** (first ~100k of 128k characters; §2.1–2.4).
- Verified mechanism: an autoencoder maps high-dimensional states to a latent space of fixed
  dimension r ≪ N, chosen in advance (§2.1). There a linear pH system is parameterized through
  Cholesky factors, so stability and passivity hold by construction (§2.2). The energy balance
  (Eq. 8) and the storage function S = H − H_min (§2.2.1) carry the dissipation inequality back to
  physical space through the reconstructed Hamiltonian H∘Ψ_e (§2.4). Time stepping uses the
  implicit midpoint rule (§2.3.1). Fidelity to the full model comes only from training-loss terms
  (Eq. 27). The paper gives **no approximation error bound**, a priori or a posteriori.
- Overlap: a passive learned latent memory with explicit energy accounting. **No online order
  change, no reset accounting, no output-error certificate.** It supplies the "passive learned
  reservoir" ingredient, not the order-change claim.

**1.2 CIPHER: Identify then Realize: Contrastive Learning of Latent Port-Hamiltonian Dynamics from Partial Observations**
- Authors: Peilun Li, Kaiyuan Tan, Daniel Moyer, Thomas Beckers.
- Date: arXiv:2605.16682, v1 15 May 2026, v2 1 Oct 2026.
- URL: https://arxiv.org/abs/2605.16682
- Access: **abstract inspected**.
- Verified mechanism: two stages. First, a contrastive teacher trains an encoder and a neural ODE on
  observation histories. Second, a student learns an invertible coordinate map and pH dynamics by
  matching the encoded futures. The theory gives identifiability conditions (recovery up to a
  diffeomorphism), not error bounds. The abstract reports robustness to over-specified latent
  dimension.
- Overlap: learned pH memory from partial observations, with a fixed latent dimension. "Robust to
  over-specified dimension" hints that redundant latent coordinates exist, which is the precondition
  for order reduction, but there is no online change and no certificate. Low direct overlap.

**1.3 Error bounds for port-Hamiltonian model and controller reduction based on system balancing**
- Authors: Tobias Breiten, Riccardo Morandin, Philipp Schulze.
- Date: arXiv:2012.15266, v1 30 Dec 2020, v2 26 Jul 2021.
- URL: https://arxiv.org/abs/2012.15266
- Access: **abstract inspected**. Search snippets also cover the related pH-descriptor LQG-BT paper,
  arXiv:2111.05065.
- Verified mechanism: pH-preserving balanced truncation with two classical H∞ error bounds, chosen
  by picking the Hamiltonian representation that minimizes the bound (the Hamiltonian is not
  unique). It also gives an LQG-balanced reduced pH controller with an a priori gap-metric bound.
  Search snippets say effort-constraint reduction keeps pH structure but loses the classical H∞
  bound. All reduction is fixed-order and offline.
- Overlap: **direct for the "certified, structure-preserving truncation" half of the claim.** A
  learned linear pH reservoir can be balanced and truncated with a known H∞ bound and a passive
  result.

**1.4 Balanced Truncation of Linear Time-Varying Systems**
- Authors: Henrik Sandberg, Anders Rantzer.
- Date: IEEE TAC 49(2):217–229, 2004. Precursors appeared at CDC 2002 and IFAC 2002. Lall et al.
  1998 gave an earlier operator-theoretic version.
- URL: https://doi.org/10.1109/TAC.2003.822862 (also Lund records lup.lub.lu.se/record/161419)
- Access: **unresolved lead**. Search metadata and abstracts only.
- Verified mechanism (abstract level): time-varying Lyapunov inequalities give upper and lower
  bounds on the induced-2-norm error of time-varying balanced truncation. This generalizes "twice
  the sum of the tail", and every truncated realization is shown to be input-output stable.
- Overlap: **this is the natural certificate for an order that changes in time.** A reservoir whose
  truncation rank varies with t is a time-varying reduced model, and this line already bounds its
  worst-case input-output error. The remaining gap is narrow: the order schedule here is fixed in
  advance, whereas the direction wants it chosen online from the realized state.

**1.5 Model Compression Method for S4 with Diagonal State Space Layers using Balanced Truncation**
- Authors: Haruka Ezoe, Kazuhiro Sato.
- Date: arXiv:2402.15993 (v3); IEEE Access 12:116415–116427, 2024.
- URL: https://arxiv.org/abs/2402.15993
- Access: **abstract inspected** (via search summaries).
- Verified mechanism: balanced truncation on pretrained S4 DSS layers, removing states with small
  Hankel singular values and re-diagonalizing. The reduced parameters then initialize a smaller
  model. This is static and offline.
- Overlap: certified-style state reduction for learned SSMs already exists. The project's earlier
  review (review_1.md) records AIRE-Prune and LAST as static H∞ mode pruning of the same kind.

**1.6 Passivity preserving energy-aware design for multi-dimensional switched systems: Application to open multi-robot systems**
- Authors: Restrepo, Secchi, Robuffo Giordano (full first names not confirmed).
- Date: Automatica 181 (2025), 112496. DOI 10.1016/j.automatica.2025.112496.
- URL: https://iris.unimore.it/handle/11380/1400233
- Access: **unresolved lead**. Repository abstract via search only. No full text found.
- Verified mechanism (abstract level): a hybrid energy tank, modeled as an impulsive switched
  system, absorbs the energy jumps caused by **changes in state dimension at switching instants**.
  Passivity of plant plus tank then follows from an energy-based condition on the switching signal.
- Overlap: **direct for the "reset-energy accounting" half.** The bookkeeping of energy removed or
  injected when the state dimension changes, and keeping passivity across those changes, is already
  published for multi-dimensional switched systems. A related MTNS paper (search snippet only)
  characterizes state jumps at switching for switched pH systems in terms of the Hamiltonian before
  and after the switch.

**1.7 Online-adaptive MOR (representative)**
- Etter & Carlberg, arXiv:1902.10659 ("vector-space sieving"): enriches the basis online and
  compresses it, with a dual-weighted-residual / adjoint error estimate. Yano et al.,
  arXiv:2508.19027: online fidelity selection per component to meet a system-level tolerance.
- Access: **unresolved leads** (search snippets only).
- Overlap: online order changes driven by an error estimator are an established MOR idea, mostly
  for parametric or nonlinear PDE ROMs rather than passive LTI reservoirs.

### Synthesis for Direction 1

Every ingredient has a precedent:
- passive learned latent memory (ApHIN, CIPHER);
- structure-preserving truncation with an H∞ bound (1.3);
- time-varying truncation bounds (1.4);
- BT on learned SSMs (1.5, plus AIRE-Prune and LAST);
- energy-jump bookkeeping for dimension-switching passive systems (1.6);
- online adaptive order (1.7).

One more piece is textbook: for a stable LTI system, the future zero-input output energy of
discarded coordinates x_d equals x_dᵀ W_o x_d exactly, where W_o is the observability Gramian.

The only unclaimed combination I could see is a *state-dependent, online-chosen* truncation
schedule for a *learned* pH reservoir that carries both the energy-tank ledger and a per-reset
output-error certificate. That is a composition of known results, with a thin theory gap.

### Feasibility (M4 laptop CPU, 16 GB, about 4 device-hours, PyTorch only, synthetic data)

- **(a) First hypothesis (falsifiable).** Setup: a trained diagonalizable passive linear pH
  reservoir (n = 128, discrete-time via implicit midpoint), and a declared forcing class F of
  piecewise bursts (L2 energy ≤ E_b per burst, quiet gaps ≥ G steps, zero input in gaps).
  Policy: an online policy that truncates in balanced coordinates when the certified future-error
  term x_dᵀW_o x_d plus the Sandberg–Rantzer-style tail bound for the remaining horizon is ≤ τ, and
  re-expands at burst onset.
  Prediction: at matched certified worst-case output-error tolerance τ, this policy has **at least
  25% lower time-averaged order** than the best *fixed-order* pH-preserving balanced truncation.
  Two side conditions must also hold: zero certificate violations, and the bound no more than 10x
  conservative at the median.
- **(b) Simplest competing baseline.** There are two, and both must be beaten:
  - **fixed-order pH-preserving balanced truncation** at the same τ;
  - **exact closed-form zero-input propagation** (x_{t+k} = Λ^k x_t for a diagonal system). This
    costs almost nothing during quiet gaps and is exact. It is the direct analogue of H001's
    memoization result.
- **(c) Synthetic pilot task.** A random passive pH system, or one trained in PyTorch for a few
  minutes on a synthetic delayed-recall/filtering target, driven by sampled bursty inputs from F.
  Compute W_o and the Hankel singular values (HSVs) with a dense Lyapunov solve, which takes
  milliseconds at n = 128. Then sweep τ and record realized versus certified error, mean order,
  and energy-ledger closure (tank energy plus reservoir energy plus dissipation equals supplied
  energy, to round-off).
- **(d) Rejection result.** Reject if any of these hold:
  - the mean-order saving over fixed-order BT is under 25% at matched τ;
  - the saving falls to zero after counting re-expansion and basis-change cost;
  - the closed-form zero-input baseline matches or beats it in wall-clock or FLOPs per step;
  - the median bound-to-realized ratio exceeds 10x, which repeats H001.
- **(e) Biggest novelty risk.** The certificate pieces already exist (1.3, 1.4, 1.6, plus the
  textbook Gramian identity), so a positive result would read as "time-varying BT plus an energy
  tank, applied to a learned reservoir". There is also a structural risk. Order can only drop
  safely when the input is quiet, and then the exact zero-input propagation is already almost free.
  This is the same "savings come only from exact structure" trap that sank H001.
- **(f) Compute.** Under 0.5 device-hour, including a small training run and a 20–50 point τ sweep
  over about 10³ sequences of 10⁴ steps. Compute is not the constraint.

---

## Direction 2: Jointly represented conservative fluxes across changing mesh refinement hierarchies

Claim to test: one learned flux representation that is exactly conservative and consistent across
refinement levels while the hierarchy changes (regrid, refine, or coarsen) during a rollout.

### Sources

**2.1 FluxGNN: Graph Neural PDE Solvers with Conservation and Similarity-Equivariance**
- Authors: Masanobu Horie, Naoto Mitsume.
- Date: arXiv:2405.16183 v1 25 May 2024; ICML 2024, PMLR 235:18785–18814.
- URL: https://proceedings.mlr.press/v235/horie24a.html ; https://arxiv.org/html/2405.16183v1
- Access: **full text inspected** (first ~100k of 132k characters; §2–4, App. C, F.1). The mesh-type
  appendix was not read.
- Verified mechanism: cells are vertices and faces are edges. The update subtracts the sum over
  faces of area × normal × a learned flux (Eq. 9, matching the FVM divergence, Eq. 3).
  Conservation comes from antisymmetric messages (Thm 3.1, Eq. 7). That antisymmetry is obtained
  by making the learned flux symmetric in its two cell arguments (Thm 3.4). Linear encoder and
  decoder constraints keep the totals invariant (Lemma 3.2, Thm 3.3). Reported conservation error
  is about 1.75×10⁻⁸, similar to the FVM reference (Table 1).
- Mesh handling: the data are fine OpenFOAM solutions interpolated to a fixed coarse mesh (App.
  F.1). Generalization is tested on unseen domain shapes, not on resolution. The text read shows
  **no refinement hierarchy, no regrid during rollout, and no coarse–fine transfer.**
- Overlap: the conservative learned-flux building block, on one fixed mesh.

**2.2 LAMP: Learning Controllable Adaptive Simulation for Multi-resolution Physics**
- Authors: Tailin Wu, Takashi Maruyama, Qingqing Zhao, Gordon Wetzstein, Jure Leskovec.
- Date: arXiv:2305.01122 v1 1 May 2023; ICLR 2023 (spotlight).
- URL: https://arxiv.org/abs/2305.01122 ; https://ar5iv.labs.arxiv.org/html/2305.01122
- Access: **full text inspected** (ar5iv; §2–4, App. B.2–B.3).
- Verified mechanism: a GNN evolution model plus a GNN actor-critic that chooses split, flip and
  coarsen operations. The objective trades error against compute. State moves to the new mesh by
  **barycentric interpolation** (Eq. 4; App. B.2 in 1D, B.3 in 2D). Mass and momentum conservation
  are **neither enforced nor measured**. Only MSE/RMSE and node counts are reported. Tasks are a 1D
  nonlinear PDE family (Brandstetter et al. 2022) and 2D ArcSim cloth.
- Overlap: **this is the "changing hierarchy" half**, learned and dynamic, but non-conservative.
  The obvious gap is LAMP plus conservative transfer.

**2.3 ReNO: Representation Equivalent Neural Operators: a Framework for Alias-free Operator Learning**
- Authors: Francesca Bartolucci, Emmanuel de Bézenac, Bogdan Raonic, Roberto Molinaro, Siddhartha
  Mishra, Rima Alaifari.
- Date: NeurIPS 2023. DOI 10.52202/075280-3051.
- URL: https://papers.neurips.cc/paper_files/paper/2023/hash/dc35c593e61f6df62db541b976d09dcf-Abstract-Conference.html
- Access: **abstract inspected**.
- Verified mechanism: defines operator aliasing, meaning inconsistency between a neural operator and
  its discrete representations, and shows that it causes errors across discretizations and grids.
  The abstract says nothing about conservation, fluxes, or refinement.
- Overlap: supplies the "consistent representation across resolutions" framing, which would make a
  natural consistency criterion. No conservation content.

**2.4 Berger–Colella AMR with refluxing (conventional baseline)**
- Authors: Marsha J. Berger, Phillip Colella. "Local Adaptive Mesh Refinement for Shock
  Hydrodynamics", J. Comput. Phys. 82:64–84, 1989. DOI 10.1016/0021-9991(89)90035-1.
- URL: https://ui.adsabs.harvard.edu/abs/1989JCoPh..82...64B/abstract (open PDF at zenodo.org/records/1253914).
- Access: **unresolved lead** for the 1989 text itself (metadata only). The mechanism comes from
  secondary sources: a NASA NTRS report, an LLNL preprint, and a 2025 AMReX Boltzmann solver
  (arXiv:2502.04820, snippets only).
- Verified mechanism (secondary): at coarse–fine faces, a flux register accumulates the difference
  between the coarse flux and the time-summed fine subface fluxes, and the adjacent coarse cell is
  corrected by that difference. This restores the telescoping sum. Combined with conservative
  (volume-weighted) restriction and prolongation at regrid, it gives conservation **to round-off
  for any face-flux function**. The 2025 solver reports conservation error near machine precision.
- Overlap: **this decides the question.** Exact conservation across changing hierarchies is
  classical and indifferent to how the flux is computed. A FluxGNN-style learned flux inside a
  refluxed, conservatively regridded AMR loop is already exactly conservative.

**2.5 CLNO: Conservative Local Neural Operator for Compressible Flow Prediction**
- Authors: Buxuan Wang, Rongzu Li, Qingsong Zou.
- Date: arXiv:2610.16040 v1 12 Oct 2026 (math.NA, cross-listed cs.LG).
- URL: https://arxiv.org/abs/2610.16040
- Access: **abstract inspected**. Previously an unresolved lead, now resolved at abstract level. The
  full text was not read.
- Verified mechanism: fixed-size local subdomains carry subdomain-integrated conservative means. A
  shared local network predicts interface fluxes that drive finite-volume updates of those means.
  A convolutional decoder adds fine-scale structure through **zero-mean basis functions**, so local
  means are preserved. It is conditioned on Re and Mach, and reports support for varying resolution
  and domain size at 3–5x the cost of LNO.
- Overlap: **closest to "joint representation".** It splits conservative coarse means, carried by
  learned fluxes, from fine detail that is zero-mean by construction. That is the two-level form of
  a hierarchy-consistent conservative representation. The abstract does not mention AMR or
  hierarchies that change during a rollout.

**2.6 Other leads (unresolved, snippets only)**
- diffhydro, arXiv:2512.13403: an ML flux correction in a coarse FV solver trained against an AMR
  reference. Coarse–fine treatment is not visible in the snippets.
- MeshGraphNets (Pfaff et al., ICLR 2021): learned sizing-field remeshing. Not checked for
  conservative transfer.
- Physics-informed coarsening for multigrid graph surrogates, arXiv:2605.31013.
- An ML-flow patent that learns coarse–fine interface fluxes from fine data (USPTO 9187984).

### Synthesis for Direction 2

Exact conservation across a changing hierarchy is **not an open problem**. Refluxing plus
conservative regrid (2.4) achieves it for any learned face flux, including FluxGNN's (2.1). LAMP
(2.2) is learned and dynamic but leaves conservation out. CLNO (2.5) already pairs learned
conservative fluxes for the means with zero-mean fine detail. What remains is an **empirical**
question. Does a flux representation shared across levels (for example, coarse-face flux equal to
the sum of fine subface fluxes *by construction*, so that no register correction is needed) improve
accuracy or stability under dynamic regridding, compared with "any learned flux plus refluxing"?
That would be an architecture result, not a certificate result.

### Feasibility (same budget)

- **(a) First hypothesis (falsifiable).** Task: 1D inviscid Burgers with moving shocks, a 2-level
  block-AMR hierarchy, refinement ratio 2, and regrid every K steps by a fixed gradient tagger.
  Model: a learned flux with a hierarchical face parameterization. A coarse-face flux network is
  paired with a fine-subface network whose output is projected to sum (over time and subfaces) to
  the coarse-face flux, which makes the register correction zero by construction.
  Prediction: it reaches **at least 20% lower rollout L2 error at step 200** than the identical
  flux network used with Berger–Colella refluxing and conservative regrid, at equal parameters and
  training steps. Both must conserve to ≤ 1e-12 relative error (float64).
- **(b) Simplest competing baseline.** A FluxGNN-style symmetric learned flux (a 1D MLP on the
  left/right states) inside a standard refluxed AMR loop with volume-weighted
  restriction/prolongation. Report two secondary references alongside it: an exact Godunov/Rusanov
  scheme on the same hierarchy, and LAMP-style barycentric transfer without refluxing, to show how
  much the conservation defect matters.
- **(c) Synthetic pilot task.** Generate ground truth in PyTorch with a fine-grid (2048-cell)
  Godunov solver for 1D Burgers, using random smooth initial conditions that steepen into shocks.
  Run about 200 trajectories of 1000 fine steps, which takes minutes on CPU. Train on the composite
  2-level grid (base 128 cells, refined patches 256-equivalent) with fixed or random regrid
  schedules. Evaluate on held-out initial conditions and on unseen regrid frequencies K.
- **(d) Rejection result.** Reject if any of these hold:
  - the joint representation is not better than the refluxing baseline by 20% (CI from 3 seeds
    excluding 0);
  - the advantage disappears when the baseline's flux net is trained with the same register in the
    loop;
  - the "joint" parameterization turns out algebraically equivalent to refluxing, so the projection
    just reproduces the register correction.
- **(e) Biggest novelty risk.** Conservation is free from 2.4, so a paper cannot claim it. CLNO
  already has the conservative-mean plus zero-mean-detail split. "LAMP plus conservative transfer"
  is an obvious follow-up that anyone could do. The diffhydro and MeshGraphNets transfer details
  are unverified. Any positive result would be an incremental accuracy claim on a toy 1D problem.
- **(f) Compute.** Data generation takes under 10 minutes. Each model (MLP flux, about 10⁴–10⁵
  parameters, unrolled training over 10–20 steps) needs about 20–40 minutes of CPU training at
  float64. With 2 models × 3 seeds plus evaluation, the total is about 2.5–3.5 device-hours. That
  fits the budget, but tightly. 2D is out of budget.

---

## Recommendation

**Do not pursue Direction 1.** Each half of its claim has a direct precedent:
- certified truncation of passive systems (Breiten–Morandin–Schulze; Sandberg–Rantzer for
  time-varying order; Ezoe–Sato, AIRE-Prune and LAST for learned SSMs);
- energy-jump accounting for systems whose state dimension changes (Restrepo–Secchi–Robuffo
  Giordano energy tanks).

Its likely savings also come exactly where the input is quiet. There, exact closed-form zero-input
propagation is almost free, so it would repeat H001's finding that only exact structure, not
certification, produces savings.

**Direction 2 is the only one worth a pilot, and only if reframed as an empirical, not a
certificate, question.** Berger–Colella refluxing already makes any learned flux exactly
conservative across changing hierarchies. Run the 4-hour 1D Burgers pilot above with refluxing as
the mandatory baseline and pre-registered kill criteria. If the hierarchical flux representation
does not beat refluxing by the stated margin, stop the project and record a supported negative
result. That result would say: online omission (H001), online order change (Direction 1, by
literature), and hierarchy-consistent conservation (Direction 2, by pilot) do not yield a
certifiable, novel saving beyond classical exact mechanisms.

If the project's charter requires a *certified-novelty* contribution, rather than an empirical
architecture result, **stopping now is the better-supported choice**. On the evidence gathered
here, neither direction's certificate is new.

Highest-priority unresolved leads before any write-up:
- CLNO full text (does it handle hierarchy changes?);
- the Restrepo et al. Automatica 2025 full text;
- Sandberg–Rantzer 2004 full text;
- diffhydro (arXiv:2512.13403) coarse–fine treatment.
