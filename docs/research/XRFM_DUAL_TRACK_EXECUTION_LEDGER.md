# XRFM Dual-Track Living Execution Ledger

**Authoritative Tracker for Dual-Track Roadmap (`XRFM_IMPLEMENTATION_ROADMAP.md`)**  
**Initialized:** 2026-11-17  
**Baseline Audited Commit:** `90674742d8e6a5caf65336e4b64c949696e33bc0`  
**Companion Governance Ledger:** `docs/program/03_EXECUTION_LEDGER.md` (maintained in sync; `G1` remains honest `FAIL` until physical CUDA/second-GPU target verification runs on Kaggle).  
**Active Phase:** **Phase 0 COMPLETE (`GATE-0 = PASS`) → Ready for Phase 1 (`GATE-1`: Shared Governance, 7-Lang LID & Frozen Eval)**  
**Compute Spent to Date:** `0.0 GPU-h` · `$0.00` Hugging Face credits (`~$4.00` reserve 100% untouched).

---

## Anti-Drift Operating Rules
1. **Update Before & After Every Work Unit:** Every task transitions `READY -> IN_PROGRESS -> DONE` (with exact file paths, regression test command, and evidence file) before starting the next task.
2. **Regression-First Discipline:** Write failing regression tests reproducing each confirmed P0/P1 defect *before* modifying core source files, record the pre-fix failure output in `docs/program/evidence/phase0_dual_track/`, apply the minimal clean fix, and record the post-fix pass output + full `ruff`/`black`/`mypy`/`pytest` verification.
3. **Zero Silent Scope Drift:** Do not start Phase 1+ data downloads, tokenizer training, or GPU runs inside Phase 0. Do not alter public contracts in ways that break the existing 544 passing tests.
4. **Honest Hardware Boundaries:** CPU/dtype/device-mocked tests prove mathematical and code-path correctness; physical CUDA kernel timing and 2-GPU DDP execution remain explicitly tracked for Kaggle T4/P100 execution.

---

## Phase Gate Board (Phases 0–8)

| Gate | Phase | Status | Blocking Exit Criteria | Evidence |
|---|---|---|---|---|
| **`GATE-0`** | Phase 0: Repo Truth Alignment & Core P0/P1 Fixes | **PASS** | All P0-01..P0-08 work units `DONE`; 560/560 tests (`464` core + `96` neurotopo) + `ruff`/`black`/`mypy`/`bandit`/`ci_smoke_test` 100% green; README quickstart works out-of-the-box | `docs/program/evidence/phase0_dual_track/` |
| **`GATE-1`** | Phase 1: Shared Governance, 7-Lang LID & Frozen Eval | **READY** | Enforced SPDX registry, 7-lang + `ur-Latn` LID $\ge 95\%$ F1, frozen eval & contamination hashes | `GATE-0` satisfied |
| **`GATE-2`** | Phase 2: Track A 32K Tokenizer Tournament & Shards | **PENDING** | Lossless UTF-8 roundtrip, `<0.5%` byte fallback, $\le 2.0\times$ fertility parity on `ur/ar/hi` | Requires `GATE-1` |
| **`GATE-3`** | Phase 3: Track A (`XRFM Native`) Pretraining | **PENDING** | Monotonic `BPB` decrease across all 7 langs + code on Kaggle `2× T4` | Requires `GATE-2` |
| **`GATE-4`** | Phase 4: Track B Base Bake-Off & 40K SFT Corpus | **PENDING** | `Qwen3-1.7B` vs `Qwen2.5-Coder-1.5B` baselines logged; 0%-contaminated AST/safety-verified SFT | Requires `GATE-1` |
| **`GATE-5`** | Phase 5: Track B (`XRFM CoderSec`) QLoRA & Ablations | **PENDING** | Beats unadapted base on Defensive Sec + Multilingual/Roman-Urdu with $<1.5\%$ code regression | Requires `GATE-4` |
| **`GATE-6`** | Phase 6: Full Cross-Track Eval & Red-Teaming | **PENDING** | EvalPlus, CyberSecEval 2, Belebele, Roman-Urdu perturbation & over-refusal gates met | Requires `GATE-3`, `GATE-5` |
| **`GATE-7`** | Phase 7: Security RAG, Bounded Agent & ZeroGPU Space | **PENDING** | $\ge 70\%$ citation hallucination reduction; 0% sandbox escape; honest dual-track UI badges | Requires `GATE-6` |
| **`GATE-8`** | Phase 8: Reproducible Release & Technical Report | **PENDING** | Clean-clone reproduction script green; signed Model/Dataset Cards & SHAs | Requires `GATE-7` |

---

## Phase 0 Work Units (`GATE-0` Detailed Checklist)

| ID | Work Unit | Audit Finding(s) | Target Files | Status | Verification / Evidence |
|---|---|---|---|---|---|
| **`P0-01`** | **Regression-First Test Suite (`tests/test_phase0_defects.py`)** | `F-04..F-12`, `F-17` | `tests/test_phase0_defects.py` | **DONE** | 16/16 defect tests fail cleanly on unpatched HEAD (`docs/program/evidence/phase0_dual_track/P0-01_regressions_before.txt`) |
| **`P0-02`** | **Config Schema Extensions (Backward-Compatible)** | Track A Architecture §2.1 | `src/xrfm/config/schema.py`, `config/tiny.yaml` | **DONE** | Added `n_kv_heads`, `rope_theta`, `norm_eps`, `gradient_checkpointing`, `check_input_bounds`, `min_lr_ratio`; updated `tiny.yaml` `vocab_size: 2048`; 50/50 config/integration/quickstart tests pass |
| **`P0-03`** | **RoPE & RMSNorm Mixed-Precision & Caching Fixes** | `F-05` (`RoPE` FP16/BF16 upcast + uncached), `F-06` (`RMSNorm` FP16 overflow) | `src/xrfm/models/attention/rope.py`, `src/xrfm/models/layers/rmsnorm.py` | **DONE** | Cached non-persistent FP32 `cos/sin` with dynamic extension, FP32 internal trig/variance, exact `x.dtype` preservation (`fp16`/`bf16`/`fp32`), 8/8 unit & reference math tests pass |
| **`P0-04`** | **Attention SDPA Flash-Path, GQA, SwiGLU Toggle & Sync Fix** | `F-04` (dense mask kills FlashAttn), `F-07` (`use_swiglu=False` ignored), `F-08` (per-step `.any()` sync), GQA | `src/xrfm/models/attention/multi_head.py`, `src/xrfm/models/layers/transformer_block.py`, `src/xrfm/models/gpt.py` | **DONE** | Unmasked `cache_len=0` dispatches `mask=None` (`is_causal=True`); GQA `n_kv_heads` KV-cache parity verified; `StandardFFN` GELU wired for `use_swiglu=False`; activation checkpointing & `check_input_bounds` wired; 53/53 model/attention/audit tests pass |
| **`P0-05`** | **Training Loop, Grouped AdamW & Cosine Floor Fixes** | `F-09` (weight decay on 1D norms/biases), `F-10` (DDP before `.to(device)`), `F-11` (cosine > `max_steps`) | `src/xrfm/training/optimizer.py`, `src/xrfm/training/scheduler.py`, `src/xrfm/training/loop.py`, `src/xrfm/training/trainer.py` | **DONE** | `build_decay_param_groups` excludes 1D norms/biases; `model.to(device)` moved before DDP/FSDP; cosine schedule clamped with `min_lr_ratio`; 31/31 training & bit-exact resume tests pass |
| **`P0-06`** | **Evaluation Device Placement, `-100` Masking & `BPB` Metric** | `F-12` (missing `.to(device)` + `-100` counted wrong + no `BPB`) | `src/xrfm/evaluation/perplexity.py`, `src/xrfm/evaluation/benchmarks.py`, `src/xrfm/evaluation/__init__.py` | **DONE** | Batches auto-moved to model device, 3-tuple `(inp, tgt, mask)` supported, `-100` padding excluded in Top-1/Top-K, `compute_bits_per_byte` added; 33/33 eval & defect tests pass |
| **`P0-07`** | **Quarantine Legacy Checkpoint, Deduplicate UI & Align Docs** | `F-17` (README quickstart crash), `F-18` (duplicate `webui`/`vercel_app`), legacy `checkpoint_step_500.pt` | `README.md`, `vercel_app/`, `artifacts/legacy_unverified/`, `docs/architecture/DEAD_OR_EXPERIMENTAL.md` | **DONE** | Quarantined `checkpoint_step_500.pt` out of `checkpoints/`, removed byte-identical `vercel_app/` duplicate of `webui/`, updated `README.md` with honest Track A vs Track B scope |
| **`P0-08`** | **Full Quality, Security, Build & Smoke Gate (`GATE-0` Sign-Off)** | All Phase 0 items | `docs/program/03_EXECUTION_LEDGER.md`, `docs/research/XRFM_DUAL_TRACK_EXECUTION_LEDGER.md` | **DONE** | `ruff`, `ruff format`, `black`, `mypy` (91 files), `bandit`, `ci_smoke_test.py`, `run_agent_demo.py`, and **560/560 pytest tests** (`464` core + `96` neurotopo) all pass (`docs/program/evidence/phase0_dual_track/`) |

---

## Chronological Change Log

| Timestamp (UTC+5) | Work Unit | Action & Outcome | Evidence |
|---|---|---|---|
| `2026-11-17` | `INIT` | Created `XRFM_DUAL_TRACK_EXECUTION_LEDGER.md` anchored to `90674742d8e6a5caf65336e4b64c949696e33bc0`; verified baseline `ruff`, `black`, `mypy` (91 files), and 544 `pytest` tests green; claimed `P0-01`. | Baseline audit & lint run |
| `2026-11-17` | `P0-01` | Wrote `tests/test_phase0_defects.py` (16 focused regression probes covering `F-04..F-12`, `F-17`, GQA, `BPB`). Verified all 16 fail on unpatched HEAD; saved evidence; claimed `P0-02`. | `docs/program/evidence/phase0_dual_track/P0-01_regressions_before.txt` |
| `2026-11-17` | `P0-02` | Extended `ModelConfig` (`n_kv_heads`, `rope_theta`, `norm_eps`, `gradient_checkpointing`, `check_input_bounds`) & `TrainingConfig` (`min_lr_ratio`, `gradient_checkpointing`) with strict validation; fixed `config/tiny.yaml` `vocab_size: 2048`. Quickstart + 49 config/regression/integration tests passed; claimed `P0-03`. | `src/xrfm/config/schema.py`, `config/tiny.yaml` |
| `2026-11-17` | `P0-03` | Refactored `RoPE` (`rope.py`) to cache FP32 `cos/sin` tables dynamically and preserve input `fp16`/`bf16` dtype; refactored `RMSNorm` (`rmsnorm.py`) to compute mean-square in FP32 (`rsqrt`) preventing FP16 overflow at `|x|>256`. 8/8 regression + reference math tests pass; claimed `P0-04`. | `src/xrfm/models/attention/rope.py`, `src/xrfm/models/layers/rmsnorm.py` |
| `2026-11-17` | `P0-04` | Upgraded `MultiHeadAttention` (`multi_head.py`) with GQA (`n_kv_heads`) and native `mask=None` (`is_causal=True`) SDPA fast-path on unmasked full sequences; added `StandardFFN` (`transformer_block.py`) honoring `use_swiglu=False`; wired `gradient_checkpointing` and `check_input_bounds` in `XRFMModel` (`gpt.py`). 53/53 model/attention/audit tests pass; claimed `P0-05`. | `src/xrfm/models/attention/multi_head.py`, `src/xrfm/models/layers/transformer_block.py`, `src/xrfm/models/gpt.py` |
| `2026-11-17` | `P0-05` | Added `build_decay_param_groups` (`optimizer.py`) excluding 1-D norms/biases from weight decay; clamped cosine LR past `max_steps` with `min_lr_ratio` (`scheduler.py`); moved `model.to(device)` before DDP/FSDP wrap in `TrainingLoop` (`loop.py`). 31/31 training & bit-exact CPU resume tests pass; claimed `P0-06`. | `src/xrfm/training/optimizer.py`, `src/xrfm/training/scheduler.py`, `src/xrfm/training/loop.py` |
| `2026-11-17` | `P0-06` | Hardened `perplexity.py`, `benchmarks.py`, and `evaluation/__init__.py` to move inputs/targets/masks to model device, accept 3-tuple batches, exclude `ignore_index=-100` padding from Top-1/Top-K denominators, and expose `compute_bits_per_byte()`. All 16/16 `test_phase0_defects.py` + 17/17 `test_evaluation.py` tests pass! Claimed `P0-07`. | `src/xrfm/evaluation/perplexity.py`, `src/xrfm/evaluation/benchmarks.py`, `src/xrfm/evaluation/__init__.py` |
| `2026-11-17` | `P0-07` | Quarantined `checkpoints/checkpoint_step_500.pt` -> `artifacts/legacy_unverified/checkpoint_step_500.pt`, removed duplicate `vercel_app/` (100% identical to `webui/`), updated `DEAD_OR_EXPERIMENTAL.md` and `README.md` with honest dual-track (`XRFM Native` vs `XRFM CoderSec`) boundaries; claimed `P0-08`. | `artifacts/legacy_unverified/`, `README.md`, `docs/architecture/DEAD_OR_EXPERIMENTAL.md` |
| `2026-11-17` | `P0-08` | Fixed test ambient `checkpoints/` writes (`tmp_path` in `test_regression.py`, `test_training.py`, `test_distributed.py`); ran full quality & regression suite: `ruff`, `ruff format`, `black`, `mypy` (91 files), `bandit`, `ci_smoke_test.py`, `run_agent_demo.py`, and **560/560 pytest tests** (`464` core + `96` neurotopo) passed! Synchronized `docs/program/03_EXECUTION_LEDGER.md` and closed **`GATE-0 = PASS`**. | `docs/program/evidence/phase0_dual_track/` |
