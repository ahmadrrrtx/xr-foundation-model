## [1.0.1] — 2026-08-08 — Forensic Audit Remediation

### Correctness
- Explicit causal masking in MultiHeadAttention (was implicit via SDPA import; manual fallback was non-causal). Fixes F-01/F-02.
- Byte-level BPE tokenizer (UTF-8 bytes, whitespace-preserving, Unicode-safe, exact round-trip). Fixes F-11/F-12/F-16.
- Real mixed precision: bf16 autocast on GPU; honest fp32 on CPU. Fixes F-23.
- API import fixed (missing `search_routes`); API now loads the latest checkpoint. Fixes F-41/F-42.
- Scheduler state is checkpointable; resume restores the LR schedule. Fixes F-24.
- Checkpoints store config/seed/versions and load with `map_location="cpu"`. Fixes F-32/F-33.
- Loss masking: padded targets use -100 and are ignored in CE and perplexity. Fixes F-15/F-39.
- Seeded, reproducible dataloaders and training. Fixes F-25.
- Dataset splits by line boundaries with exact-line dedup; newlines preserved. Fixes F-17/F-18/F-19.
- Model vocab is built from the tokenizer's actual vocabulary (was 50304 vs 1024/408). Fixes F-13.

### Data
- Added `data/datasets/corpus.txt`: 11 public-domain books + PSF-licensed code slice (~1.8 M BPE tokens).

### Tests
- New ground-truth suite `tests/test_audit_verification.py` (causality, reference math, KV-cache
  equivalence, resume, reproducibility, tokenizer fidelity, loss masking); new API tests.

### Docs
- Added `docs/audit/BASELINE.md`, `docs/audit/FORENSIC_AUDIT.md`, `docs/audit/GAP_ANALYSIS.md`,
  `docs/research/MODEL_COMPARISON.md`, `docs/architecture/XRFM_TARGET_SPEC.md`,
  `docs/training/COMPUTE_PLAN.md`, `docs/implementation/REMEDIATION_PLAN.md`.
- README rewritten to match reality; legacy config preserved as `config/config.legacy-v1.yaml`.

## [v0.6.0] — 2026-07-25 — Inference Engine

## [v0.9.0] — 2026-07-25 — Optimization

### Added
## [v1.0.0] — 2026-07-25 — Production Deployment

### Added
- `api/main.py` — FastAPI app with ASGI lifespan, CORS, security headers, request timing
- `api/schemas.py` — Pydantic models for all endpoints
- `api/routes/health.py` — Health, readiness, model info
- `api/routes/completions.py` — Text generation (sync + streaming SSE)
- `api/routes/tokenize_endpoints.py` — Tokenize/detokenize
- `api/routes/metrics.py` — Server metrics
- `deployment/Dockerfile` — Multi-stage CPU production image (~1.5GB)
- `deployment/Dockerfile.gpu` — CUDA-based GPU image
- `deployment/docker-compose.yml` — CPU + GPU profiles
- `deployment/gunicorn_conf.py` — Production Gunicorn config (2N+1 workers, preload)
- `webui/index.html` + `style.css` + `app.js` — Static chat UI with SSE streaming
- `.dockerignore` — Exclude checkpoints, caches from Docker build
- `.github/workflows/ci.yml` — Lint (ruff) + test (pytest)
- `.github/workflows/cd.yml` — Docker build + push to GHCR
- `.github/workflows/release.yml` — Auto-release on semver tags
- `docs/deployment/DEPLOYMENT_GUIDE.md` — Deployment documentation
- `research/phase_10/RESEARCH.md` — Comprehensive research report (8+ sources)

### Design
- OpenAI-compatible API surface
- ASGI lifespan for model loading (once at startup)
- Streaming via Server-Sent Events
- Security: CORS, security headers, Pydantic validation
- Docker: multi-stage builds, mounted volumes for checkpoints
- CI/CD: lint → test → build → push → GHCR

## [v0.9.0] — 2026-07-25 — Optimization

### Added
- `optimization/flash_attention.py` — FlashAttention integration via `scaled_dot_product_attention()` (2-4× speedup, O(n) memory)
- `optimization/flash_attention.py` — `flash_attention_forward()` as drop-in for manual matmul+softmax+matmul
- `optimization/flash_attention.py` — Backend detection (`is_flash_attention_available`, `get_available_backend`)
- `optimization/flash_attention.py` — `benchmark_attention()` for performance comparison
- `optimization/quantization.py` — INT8 per-tensor, per-channel quantization (4× compression)
- `optimization/quantization.py` — INT4 group-wise quantization with 2-per-byte packing (8× compression)
- `optimization/quantization.py` — `QuantizedWeight` dataclass with `dequantize()` method
- `optimization/quantization.py` — `quantize_model_weights()` and `compute_compression_ratio()`
- `optimization/speculative_decoding.py` — `SpeculativeDecoder` class (draft-model accelerated generation, 2-3× speedup)
- `optimization/speculative_decoding.py` — Rejection sampling verification (exact output distribution)
- `optimization/speculative_decoding.py` — `estimate_speedup()` formula
- `optimization/__init__.py` — Package exports
- `tests/test_optimization.py` — 22 tests (FlashAttention 5, Quantization 11, Speculative 6)
- `docs/optimization/OPTIMIZATION_GUIDE.md` — Complete optimization guide

### Design
- FlashAttention: drop-in replacement, auto-selects fastest backend (flash > mem_efficient > math)
- INT8 quantization: per-tensor and per-channel affine (asymmetric) + symmetric variants
- INT4 quantization: group-wise with 2-per-byte packing, sign-extended unpacking
- Speculative decoding: mathematically exact via rejection sampling (no quality loss)
- All optimizations are opt-in and backward compatible


## [v0.8.0] — 2026-07-25 — Scaling & Distributed Training

### Added
- `training/distributed.py` — DDP/FSDP wrappers, gradient accumulation, distributed sampler, model unwrapping, FSDP checkpoint save/load, distributed logging (reduce_loss, barrier)
- `GradientAccumulator` class — manages micro-batch gradient accumulation with DDP no_sync support
- `create_distributed_dataloader()` — auto-creates DistributedSampler when torch.distributed is active
- `wrap_model_ddp()` / `wrap_model_fsdp()` / `get_raw_model()` — model wrapping/unwrapping
- `init_distributed()` / `cleanup_distributed()` — process group lifecycle
- `scripts/torchrun_launch.sh` — torchrun launch script with environment validation
- `tests/test_distributed.py` — 17 tests covering accumulators, distributed utils, training loop grad accum
- `docs/scaling/MULTI_GPU_GUIDE.md` — Complete distributed training guide

### Changed
- `training/loop.py` — gradient accumulation (loss/accum_steps), DDP no_sync on intermediate micro-batches, distributed-aware logging, barrier-based checkpointing, config-driven (grad_accum_steps, use_ddp, use_fsdp)
- `config/config.yaml` — added grad_accum_steps, use_ddp, use_fsdp settings

### Design
- Single-GPU backward compatible — all distributed features are opt-in
- Gradient accumulation: `effective_batch = per_dev_batch × grad_accum × world_size`
- DDP no_sync avoids unnecessary all-reduce on intermediate micro-batches
- Checkpoints only saved on rank 0 with barrier synchronization


## [v0.7.0] — 2026-07-25 — Evaluation Pipeline

### Added
- `evaluation/perplexity.py` — Token-level perplexity via exp(avg_cross_entropy) with proper next-token shifting
- `evaluation/perplexity.py` — Strided sliding-window perplexity for texts exceeding max_seq_len (no double-counting)
- `evaluation/perplexity.py` — `evaluate_checkpoint()` wrapper with timing/tokens-per-second diagnostics
- `evaluation/benchmarks.py` — `Benchmark` ABC for extensible evaluation framework
- `evaluation/benchmarks.py` — `TextCompletionAccuracy`: Top-1 next-token prediction accuracy
- `evaluation/benchmarks.py` — `TopKAccuracy`: Top-k next-token accuracy (Top-1 through Top-5 verified)
- `evaluation/benchmarks.py` — `run_evaluation_suite()`: PPL + Top-1 + Top-5 in one call
- `evaluation/__init__.py` — Package exports
- `tests/test_evaluation.py` — 22 tests (perplexity basic/strided/checkpoint, accuracy, suite)
- `docs/evaluation/EVALUATION_GUIDE.md` — Complete evaluation documentation
- `benchmark/eval_forward.py` — Evaluation performance benchmark

### Design
- Token-aggregated loss: sum NLL across all tokens before averaging (correct for variable-length batches)
- Strided overlap: tokens only counted once in overlapping windows
- Extensible: `Benchmark` ABC supports custom benchmarks without harness changes
- Perplexity = exp(cross_entropy): mathematically correct, zero external dependencies


### Added
- `inference/engine.py` — GenerationEngine with KV-cached autoregressive generation
- `inference/kv_cache.py` — KVCache class storing per-layer (K, V) tensors for incremental inference
- `inference/sampling.py` — Greedy, temperature, top-k, and nucleus (top-p) sampling strategies
- `inference/__init__.py` — Package exports
- KV cache support in `MultiHeadAttention.forward()` — accepts `past_kv`, returns `(output, present_kv)`
- KV cache pass-through in `TransformerBlock.forward()` — propagates `past_kv`/`use_cache` to attention
- `GPTModel.forward()` now returns `(logits, present_key_values)` when `use_cache=True`
- `RoPE.forward()` accepts `offset` parameter for correct position encoding with KV cache
- `benchmark/inference_forward.py` — Inference performance and numerical stability benchmarks
- `tests/test_inference.py` — 27 tests covering sampling, KV cache, and generation engine
- `docs/inference/INFERENCE_GUIDE.md` — Complete inference documentation
- Batch generation support via `GenerationEngine.generate_batch()`

### Changed
- `MultiHeadAttention.forward()` return type: `torch.Tensor` → `Tuple[Tensor, Optional[Tuple]]`
- `TransformerBlock.forward()` return type: `torch.Tensor` → `Tuple[Tensor, Optional[Tuple]]`
- `GPTModel.forward()` return type: `torch.Tensor` → `Tuple[Tensor, Optional[List[Tuple]]]`
- All existing tests and benchmarks updated for new return signatures

### Performance
- KV cache reduces per-step attention from O(n²) to O(n)
- Cached generation: ~O(seq_len) per step instead of O(seq_len²)

- **CRITICAL:** Training loop loss computation fixed — now predicts next token via shifted targets (`input[:, 1:]`), not identity mapping (`input`).
- **CRITICAL:** Version inconsistency resolved — all files now report v0.5.1.
- **CRITICAL:** Documentation recovered — DECISIONS.md, CHANGELOG.md, and phase_06/RESEARCH.md rewritten from actual implementation after LLM degeneration was discovered.
- **CRITICAL:** Training loop wired to `XRFMTextDataset` via `DataLoader` — replaces random dummy data with actual dataset batches.
- Training loop comments trimmed from ~60% of file to focused, actionable documentation.
- SECURITY.md updated to reflect current version support.
- ROADMAP.md updated: v0.6.0 marked as NEXT (not complete).

### Added
- `.gitignore` — protects against committing checkpoints, `__pycache__`, virtual environments.
- `pyproject.toml` — modern Python packaging with `[project]` metadata, optional dependency groups, tool configs (black, ruff, mypy, pytest).
- `requirements.txt` — minimal dependencies (torch, pyyaml, numpy).
- `xrfm/__init__.py` exports expanded to include `GPTModel`, `BytePairEncoder`, `TokenizerInterface`, `XRFMTextDataset`, `DatasetConfig`, `ModelConfig`, `TrainingConfig`.
- Regression tests for next-token prediction correctness and dataset integration (`tests/test_regression.py`).

### Changed
- ROADMAP: Phase 6 marked as "NEXT" (not complete). The `generate()` placeholder in `model/gpt.py` is not a Phase 6 inference engine.
- DECISIONS.md: Complete rewrite. All 7 TDRs now match actual implementation state.
- CHANGELOG.md: Complete rewrite with concise, verifiable entries.
- research/phase_06/RESEARCH.md: Marked as "PLANNED — NOT IMPLEMENTED" with design brief.

---

## [v0.5.0] — 2026-07-24

### Added
- Training engine: `training/loop.py` (AdamW + cosine + warmup + gradient_clip + mixed_precision + checkpoint + resume), `training/optimizer.py`, `training/scheduler.py`, `training/checkpoint.py`, `training/mixed_precision.py`.
- Tests: `test_training.py` (15 tests covering optimizer, scheduler, checkpoint, mixed_precision, loop).
- Docs: `docs/training/TRAINING_GUIDE.md`.
- Benchmark: `benchmark/training_forward.py` (basic framework).

### Design Decisions
- AdamW optimizer (CORE), cosine + warmup schedule (CORE), mixed precision with GradScaler/NoOpScaler (CORE), gradient clipping at 1.0 (CORE), checkpoint in .pt format with resume support (CORE).

---

## [v0.4.0] — 2026-07-24

### Added
- Transformer architecture: `model/gpt.py` (GPTModel), `model/embedding.py` (XRFMEmbedding), `model/attention/multi_head.py` (MultiHeadAttention), `model/attention/rope.py` (RoPE), `model/layers/rmsnorm.py` (RMSNorm), `model/layers/swiglu.py` (SwiGLU), `model/layers/transformer_block.py` (TransformerBlock).
- Tests: 52 tests across embedding (11), attention (17), transformer block (13), model architecture (16).
- Docs: `docs/model/ARCHITECTURE.md` (complete architecture documentation with tensor shapes, mathematical derivations, design decisions).
- Benchmark: `benchmark/model_forward.py`.

### Design Decisions
- Manual multi-head attention (CORE) for full control. RoPE over ALiBi. SwiGLU over GELU. RMSNorm over LayerNorm. Pre-norm architecture. Weight tying (default True). Xavier initialization.

---

## [v0.3.0] — 2026-07-24

### Added
- Dataset pipeline: `xrfm/data/loader.py` (XRFMTextDataset, file verification, normalization, split, chunking, manifest generation).
- Tests: `test_data_loader.py` (16 tests).
- Docs: `docs/data/DATASET_GUIDE.md`.

---

## [v0.2.0] — 2026-07-24

### Added
- BPE tokenizer: `tokenizer/bpe.py` (BytePairEncoder), `tokenizer/interface.py` (TokenizerInterface ABC), `tokenizer/encode.py`, `tokenizer/decode.py`.
- Tests: `test_tokenizer_bpe.py` (16 tests).
- Docs: `tokenizer/DESIGN.md`, `docs/tokenizer/README.md`.

---

## [v0.1.0] — 2026-07-24

### Added
- Project foundation: `xrfm/` package, `ConfigLoader` with YAML validation and dot-notation access, typed dataclasses (`ModelConfig`, `TrainingConfig`), `ConfigPresets`.
- Professional open-source standards: LICENSE (MIT), CONTRIBUTING, ROADMAP, CHANGELOG, CODE_OF_CONDUCT, SECURITY.
- Architecture review: `ARCHITECTURE_REVIEW.md`.
- Research documentation: TDR, implementation roadmap, repository blueprint, risk assessment, API design, long-term evolution plan.
