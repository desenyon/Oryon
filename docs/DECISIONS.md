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
