# XRFM — Repository Blueprint

This document explains the purpose of every directory and file in the `xr-foundation-model` repository. No directory should exist without a clear responsibility.

---

## Root-Level Files

| File | Purpose | Classification |
|---|---|---|
| `README.md` | Project overview, quick start, branding (`XRFM`) | CORE |
| `LICENSE` | MIT license (open source standard) | CORE |
| `CONTRIBUTING.md` | Contribution guidelines, code standards, attribution requirements | CORE |
| `ROADMAP.md` | Semantic version targets, model scaling path (`XRFM-10M` → `XRFM-Multimodal`) | CORE |
| `CHANGELOG.md` | Version history with release notes | CORE |
| `CODE_OF_CONDUCT.md` | Community behavior standards | CORE |
| `SECURITY.md` | Security policy, vulnerability reporting, model integrity verification | CORE |
| `.gitignore` | Excludes checkpoints, environments, OS artifacts | CORE |
| `pyproject.toml` | Modern Python packaging (`xrfm` package, optional dependencies) | CORE |
| `requirements.txt` | Minimal dependencies for development | CORE |

---

## Top-Level Directories

### `xrfm/` — Main Python Package

The core package. Everything inside is importable (`from xrfm import ...`).

**Subdirectories:**
- `config/` — Configuration loader (`loader.py`) with typed presets. No hard-coded parameters.

**Future subpackages (not implemented yet, reserved):**
- `tokenizer/` — Tokenizer implementations (BPE, future algorithms)
- `model/` — Transformer architecture (attention, layers, full model)
- `training/` — Training loop, optimizer, scheduler, checkpointing
- `inference/` — Inference engine, KV cache, sampling
- `evaluation/` — Perplexity, benchmarks
- `api/` — FastAPI interfaces
- `utils/` — Logging, metrics, helpers (minimal; avoid bloat)

---

### `tokenizer/`

**Purpose:** Tokenizer design and implementation. Separate from `xrfm/` package to maintain clean module boundaries (tokenizer can be developed independently of model training).

**Files:**
- `DESIGN.md` — Stable interface specification (`TokenizerInterface`).
- `bpe.py` — Original BPE implementation (Phase 2).
- `interface.py` — Abstract base class for all tokenizer algorithms.
- `encode.py`, `decode.py` — Convenience wrappers.

---

### `model/`

**Purpose:** Model architecture components. Separated by responsibility.

**Subdirectories:**
- `attention/` — Multi-head attention implementation, future variants (GQA, sparse, sliding window).
- `layers/` — Transformer blocks (`TransformerBlock`), feed-forward networks (`SwiGLU`), normalization (`RMSNorm`).

**Reserved but not implemented yet:**
- `gpt.py` — Full decoder-only model (Phase 4).
- `moe.py` — Mixture of Experts architecture (v3.0+).
- `state_space.py` — Mamba / RWKV alternatives (v3.0+, research-only).

---

### `data/`

**Purpose:** Dataset storage, preparation, and version tracking.

**Subdirectories:**
- `datasets/` — Raw and processed dataset files (`tiny_shakespeare.txt`, future datasets).
- `manifests/` — Dataset version tracking (hash, source, filter version).

**Reserved but not implemented yet:**
- `loader.py` — Dataset loader using `TokenizerInterface`.
- `streaming.py` — Streaming dataset support for large-scale training.

---

### `training/`

**Purpose:** Training loop and associated infrastructure.

**Reserved files (not implemented yet):**
- `loop.py` — Main training and validation loops.
- `optimizer.py` — AdamW setup with weight decay.
- `scheduler.py` — Cosine decay with linear warmup.
- `checkpoint.py` — Save/load full training state.
- `resume.py` — Resume from any checkpoint with zero manual changes.
- `distributed.py` — DDP initialization, multi-node launch scripts.

---

### `inference/`

**Purpose:** Inference engine for serving and evaluation.

**Reserved files:**
- `engine.py` — Custom inference loop.
- `kv_cache.py` — Key-value cache for autoregressive generation.
- `sampling.py` — Temperature, nucleus (top-p), top-k sampling.
- `streaming.py` — Streaming token output for APIs.

---

### `evaluation/`

**Purpose:** Model evaluation and benchmarking.

**Reserved files:**
- `perplexity.py` — Perplexity calculation on validation sets.
- `benchmarks.py` — Benchmark framework for future standard tests (MMLU, etc.).

---

### `checkpoints/`

**Purpose:** Storage for saved model weights, optimizer states, and training metadata.

**Requirements:**
- Large enough for full checkpoints (model weights + optimizer states + scheduler + scaler + metadata).
- For 1B+ models: checkpoints exceed 4GB (FP32). Must support distributed storage or external object storage (S3, GCS) — future enhancement.
- Never committed to Git (`.gitignore` excludes `.pt` and `.pth` files).

---

### `tests/`

**Purpose:** Unit and integration tests for every module.

**Standard:** Every public function must have at least one test. Every module must include a test file.

---

### `scripts/`

**Purpose:** Utility scripts for dataset download, model conversion, benchmark execution.

**Reserved scripts:**
- `download_shakespeare.py` — Download Tiny Shakespeare dataset.
- `convert_to_gguf.py` — Convert trained weights to GGUF format (optional, future).
- `benchmark_throughput.py` — Performance measurement script.

---

### `docs/`

**Purpose:** Module-level documentation written as we build.

**Standard:** No documentation postponed until the end. Every completed module must have a `docs/<module>/README.md` or `GUIDE.md`.

---

### `benchmark/`

**Purpose:** Performance tracking and regression testing.

**Reserved:** `performance_tests.py`, `throughput_logs/`

---

### `security/`

**Purpose:** Security-related resources and documentation.

**Reserved:** Security audit scripts, dependency scanning configurations (`dependabot` or `safety`), model integrity verification scripts.

---

### `api/`

**Purpose:** Production API interfaces (FastAPI).

**Reserved:** `main.py` (FastAPI server), `models/` (Pydantic request/response schemas).

---

### `webui/`

**Purpose:** Web interface for interactive model use.

**Reserved:** Gradio interface or custom React frontend (post-v1.0, optional enhancement).

---

## Dependency Classification Summary

| Component | Classification | Why |
|---|---|---|
| `ConfigLoader` (YAML) | CORE | Required for all phases |
| BPE Tokenizer | CORE | Phase 2 implementation |
| `TokenizerInterface` | CORE | Future-proof design |
| Manual Multi-Head Attention | CORE | Full control for modifications |
| FlashAttention | OPTIONAL | Phase 9 optimization |
| RoPE / RMSNorm / SwiGLU | CORE | Modern architecture features |
| GQA | OPTIONAL | Post-v0.5.0 enhancement |
| MoE Architecture | RESEARCH-ONLY | v3.0 future version |
| Mamba / State Space | RESEARCH-ONLY | v3.0 alternative architecture |
| DDP Training | CORE | Phase 5 design; Phase 8 activation |
| FSDP | OPTIONAL | Scale beyond single-node memory |
| DeepSpeed ZeRO | RESEARCH-ONLY | Institutional-scale only |
| Custom Inference Engine | CORE | Phase 6 |
| vLLM Integration | OPTIONAL | Phase 9 production serving |
| TensorRT-LLM | RESEARCH-ONLY | Maximum performance optimization |
| CSV Logging | CORE | Phase 5 |
| Weights & Biases | OPTIONAL | Post-v0.5.0 tracking |
| FastAPI Server | CORE | Phase 10 production |
| Docker | CORE | Phase 10 deployment |
| Continuous Training Pipeline | CORE | Phase 10 design; full automation optional |
