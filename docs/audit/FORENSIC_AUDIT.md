# XRFM — Complete Repository Forensic Audit (Phases 1–8)

**Audit date:** 2026-08-08 · **HEAD:** `cff2dc6` (branch `audit/forensic-v2` created from it)
**Classification legend:** OBSERVED (read/executed) · INFERRED (derived from code) · RESEARCH-BACKED (cited) · RECOMMENDATION (proposed action)
**Severity:** CRITICAL / HIGH / MEDIUM / LOW

Every subsystem below is traced as **INPUT → TRANSFORMATION → OUTPUT → CONSUMER**, then audited.

---

## 1. System Overview

XRFM is a from-scratch PyTorch decoder-only transformer (GPT-style) with: config-driven YAML, custom character-level BPE tokenizer, an in-memory text dataset with character-proportional splits, a single-file training loop with DDP/FSDP scaffolding, manual attention with implicit causal masking via SDPA, KV-cached inference, perplexity evaluation, quantization, speculative-decoding scaffolding, a FastAPI server, Docker, CI/CD, a RAG search module, and a Gradio/static web UI.

Claimed scale: "from 10M parameters to billion-parameter models without rewrites" (README). Measured shipped model: **19.19 M parameters** (vocab 50304 × d_model 256 dominates; ~6.3 M non-embedding).

---

## 2. Per-Subsystem Trace & Findings

### 2.1 Model (`model/`, `xrfm/config/`)

**Trace:** `config/config.yaml` → `ConfigLoader.model_config()` → `GPTModel.__init__` → `XRFMEmbedding` + `TransformerBlock`×6 (RMSNorm pre-norm → `MultiHeadAttention` → SwiGLU FFN) → `norm_final` → `lm_head` (weight-tied) → logits → consumer (training loop, inference, perplexity).

**F-01 · CRITICAL · OBSERVED — Causal masking is implicit and environment-dependent.**
`MultiHeadAttention.forward` never constructs a causal mask itself. With `mask=None` and `cache_len==0` it tries `from optimization.flash_attention import flash_attention_forward`; on success it uses `F.scaled_dot_product_attention(is_causal=True)`; **on any import failure it silently falls back to full bidirectional attention** (verified: manual scores contain no `-inf`). Training correctness therefore depends on an optimization module importing. Any change to `optimization/` breaks causality silently. There is **no test** asserting causality.

**F-02 · HIGH · OBSERVED — Manual (non-flash) attention path is non-causal by default.** Same mechanism as F-01; the fallback path with `mask=None` has no masking at all. The `mask` parameter defaults to `None` in `GPTModel.forward`, and the training loop never passes a mask.

**F-03 · MEDIUM · OBSERVED — RoPE recomputes frequencies every forward.** `RoPE.apply_rotary_emb` registers `inv_freq` but never uses it; it rebuilds `arange`, `inv_freq`, `freqs`, `cos_full`, `sin_full` on every call (CPU/GPU inefficiency; correctness unaffected — verified exact match vs an independent reference, offsets 0/5/128). Known to the project itself ("M-3: RoPE redundant frequency computation", `AUDIT_CLOSURE.md`).

**F-04 · LOW · OBSERVED — RoPE uses the "half-split" rotation convention, not the interleaved convention.** `rotate_half` = `cat([-x2, x1])` with duplicated `[cos, cos]`/`[sin, sin]`. Mathematically valid (verified: exact match vs reference; relative-position property holds) and internally consistent, but differs from Llama/Mistral/Qwen implementations (which interleave even/odd dims). Non-issue for correctness; relevant only for cross-framework weight compatibility.

**F-05 · MEDIUM · OBSERVED — Initialization is nonstandard for deep transformers.** Embeddings use Xavier-uniform (`limit = sqrt(6/(vocab+d_model))`); Q/K/V/O, SwiGLU also Xavier; no residual-stream scaling (e.g., GPT-2 `0.02/sqrt(2·n_layers)` output scaling), no final-layer scaling, no "small" init for FFN as used by Llama/GPT-2. Fine at 6 layers; a known stability risk at 32+ layers. (RESEARCH-BACKED: GPT-2 scaled init; Llama uses std=0.02 for most linear layers.)

**F-06 · LOW · OBSERVED — Attention projections use `bias=True`; modern LLMs (Llama, Mistral, Qwen) use `bias=False`.** Adds ~66 K params at this scale; not a bug.

**F-07 · LOW · OBSERVED — `use_swiglu` config field is dead.** `ModelConfig.use_swiglu` exists; `GPTModel` and `TransformerBlock` never read it (always SwiGLU).

**F-08 · MEDIUM · OBSERVED — `padding_idx=0` freezes embedding row 0 and token 0 is a NUL control character.** The model can never learn representations for byte 0; padding uses the same ID (see F-15), and attention is not masked to it.

**F-09 · PASS — Weight tying is real and correct.** `lm_head.weight is embedding.embedding.weight`; `logits == h_final @ E^T` verified (max diff 0.0). Parameter count 19,192,576 (embedding-dominated due to F-18).

**F-10 · PASS — RMSNorm, SwiGLU, residual pre-norm structure verified numerically** against independent manual computations (exact matches). KV-cache incremental forward == full forward (max diff ≈1e-7). Checkpoint (50304-vocab) loads into the current-config model with `strict=True`.

### 2.2 Tokenizer (`tokenizer/`)

**Trace:** raw text → `_preprocess_text_for_training` (whitespace collapse) → word-level char-list BPE → vocab/merges JSON → `encode()` (word-split, rank-map merge) → IDs → model. `decode()`: IDs → token strings → join.

**F-11 · CRITICAL · OBSERVED — `decode(encode(x)) != x` for every input: all whitespace is destroyed.** `encode()` does `" ".join(text.split())` and encodes word-by-word; the vocabulary cannot contain spaces (merges never cross whitespace); decode concatenates without spaces. A language model whose tokenizer cannot represent `\n`, `\t`, or double spaces cannot generate code, structured text, or paragraphs. Also, `\n` exists in the vocab (id 10) but is **never emitted** by encode.

**F-12 · CRITICAL · OBSERVED — Not byte-level; crashes on non-Latin-1 text.** Vocab is `bytes([i]).decode("latin-1")` for i∈[0,256) but `encode()` splits Python characters; any char > U+00FF raises `ValueError` (verified: `你`, Arabic, emoji). The docstrings claim byte-level BPE (GPT-2/tiktoken style) — false.

**F-13 · HIGH · OBSERVED — Vocabulary is far below target and empty of special tokens.** Committed `vocab.json`: `vocab_size_target=1024`, actual **408** tokens (BPE stopped: no pairs ≥2 remained); `special_tokens={}`; no BOS/EOS/PAD/UNK. Model config expects 50304. Three-way mismatch (see BASELINE §5).

**F-14 · MEDIUM · OBSERVED — train/encode merge inconsistency.** `train()` applies a merge even when `best_pair_str` already exists in vocab without recording it in `merges`, so `encode` cannot reproduce merges the trainer applied; also `_find_next_available_id` can collide with `reserved_ids`. Edge-case correctness gap; no infinite loop (merges strictly shrink words).

**F-15 · MEDIUM · OBSERVED — Lossy/awkward padding.** Dataset pads short chunks with token id 0 (NUL); the collate function pads to batch max length with 0; loss (`cross_entropy` without `ignore_index`) **trains on padded positions to predict NUL** whenever a batch mixes lengths.

**F-16 · MEDIUM · INFERRED — Token efficiency is poor for general text.** Measured ~3.3–3.8 tokens/word and ~0.7 tokens/char on general English (vs ~1.3–1.5 tokens/word for Llama-3's 128 K tiktoken BPE / GPT-2 byte BPE, RESEARCH-BACKED). For the repeated sample sentence it is 1.1 tokens/word. Fine for a char-level toy; expensive at scale.

### 2.3 Data Pipeline (`xrfm/data/loader.py`, `data/`)

**Trace:** file → `verify_text_file` → read → `normalize_text` (all `\n`/`\t`→space, collapse) → `split_dataset` (character-index split) → `chunk_text` (tokenize, slice max_seq_len) → `XRFMTextDataset.__getitem__` (shift targets) → `DataLoader` + `xrfm_collate_fn` (pad to batch max) → `TrainingLoop`.

**F-17 · CRITICAL · OBSERVED — The shipped dataset is a toy, not training data.** `data/datasets/sample.txt` is the **same sentence repeated 100×**. `config.yaml` names `tiny_shakespeare` as default dataset, which **does not exist** in the repo. The dataset is unsuitable for any pretraining claim; it exists only for pipeline smoke tests.

**F-18 · HIGH · OBSERVED — Character-index splitting causes train/val/test leakage and arbitrary mid-token cuts.** `split_dataset` splits a single string by character ranges with no document boundaries; with repeated paragraphs the same text appears in train AND val AND test (leakage); chunks are cut mid-sentence arbitrarily.

**F-19 · HIGH · OBSERVED — `normalize_text` destroys document structure** (newlines/tabs → spaces), so the model can never learn document or line boundaries — incompatible with the tokenizer's inability to emit whitespace anyway.

**F-20 · HIGH · OBSERVED — No loss masking for padding (see F-15); no packing; no document boundaries; no BOS/EOS; no dedup; no filtering; no sharding/streaming.** `manifest` helpers (`build_manifest`, `save_manifest`) exist but nothing calls them. `paths.manifest_dir` in config is unused.

**F-21 · MEDIUM · OBSERVED — Every chunk wastes its final token** (input = chunk[:-1], target = chunk[1:]; the last token of each chunk never receives a loss). Standard in nanoGPT-style setups; acceptable for toys, wasteful at scale.

**F-22 · LOW · INFERRED — Entire corpus held in RAM** and re-tokenized on every `__init__`; no caching. Fine for toy corpora; does not scale.

### 2.4 Training System (`training/`)

**Trace:** `TrainingLoop` → `train_step` (CE over shifted targets → `/grad_accum` → scaler.scale → backward → unscale → clip → step → scheduler.step) → checkpoint save → resume path.

**F-23 · CRITICAL · OBSERVED — "Mixed precision / BF16" is a fiction.** `MixedPrecisionLoader(enabled=True)` only creates a `GradScaler`; **nothing converts the model or activations to bf16/fp16 (no `autocast` anywhere)**. With the GPU absent this is a `NoOpScaler`; on GPU it would scale the loss and immediately unscale without any reduced-precision compute. Docstrings claim "bfloat16 training … numerical stability" — false. Config default `mixed_precision: true` therefore has no effect.

**F-24 · HIGH · OBSERVED — Scheduler state is not checkpointable.** `SchedulerLoader` has no `state_dict`/`load_state_dict`. `CheckpointLoader` checks `hasattr(scheduler, "state_dict")` → skips. On resume, `current_step` is restored but **the LR schedule restarts at step 0**, so LR jumps back to ~0 and re-warms — a silent reproducibility/resume bug.

**F-25 · HIGH · OBSERVED — No RNG seeding anywhere.** `torch.manual_seed`, `random.seed`, `numpy.seed` are never called by the training stack; DataLoader shuffle is non-deterministic across runs. `datasets.seed: 42` exists in config but only seeds `split_dataset`'s local `random`.

**F-26 · MEDIUM · OBSERVED — `error_if_nonfinite=True` in `clip_grad_norm_`.** A single NaN gradient raises and kills training (no skip-and-log fallback); on CPU (NoOpScaler) there is no AMP overflow protection to soften this. (Design choice; documented as risk.)

**F-27 · MEDIUM · OBSERVED — No validation loop.** `best_loss` tracks *training* loss only; `training_loop` never evaluates on the val split. "Validation" exists only as a separate evaluation module that nothing calls during training.

**F-28 · LOW · OBSERVED — Reported loss during grad-accum cycles is the last micro-batch's loss**, not the accumulated average (metric only; gradients are correct).

**F-29 · PASS — Gradient accumulation logic verified.** `zero_grad` at cycle start, `loss/grad_accum`, `_micro_counter % grad_accum == 0` gating, DDP `no_sync` on intermediate micro-batches — logic is correct for grad_accum=1 and 2 in tests we ran.

**F-30 · PASS — AdamW + cosine-with-warmup scheduler are correct** (verified formulas). `OptimizerLoader` is a thin wrapper over `torch.optim.AdamW`. Note: weight decay applies to all params incl. norms/biases/embeddings (no group exclusion) — minor deviation from some recipes (e.g., OLMo removes wd on embeddings, RESEARCH-BACKED).

### 2.5 Checkpointing (`training/checkpoint.py`)

**F-31 · HIGH · OBSERVED — Committed checkpoint has an EMPTY optimizer state** and no config/scheduler/commit info; it cannot resume and its `loss=0.0116, step=500` cannot be verified against any log in the repo. It also matches vocab 50304 while the shipped tokenizer has 408 tokens → **the "pre-trained weights" and "vocabulary" shipped together are mutually incompatible** (commit `98474c7`).

**F-32 · MEDIUM · OBSERVED — `torch.load(..., weights_only=True)` without `map_location`.** A checkpoint saved on CUDA cannot be loaded on a CPU-only machine (device mismatch); portability gap.

**F-33 · MEDIUM · OBSERVED — Docstring claims config is saved for reproducibility; the code does not save it** (comment admits it is a "RESEARCH-ONLY extension"). No config, tokenizer version, dataset version, seed, or commit hash in checkpoints → experiments are not reproducible from artifacts alone.

### 2.6 Distributed (`training/distributed.py`)

**F-34 · MEDIUM · OBSERVED — Distributed scaffolding is untested theory.** DDP/FSDP wrappers, samplers, `reduce_loss`, barriers exist and are unit-tested in isolation on CPU (gloo single-process paths); **no multi-process training run has ever been executed in this repo's CI or here** (no GPU; the torchrun script only validates the environment, it does not train). FSDP wrapper does not set `device_id`/`sync_module_states`; FSDP checkpoint helpers are best-effort. Treat as theoretical; DDP-with-gloo on CPU multi-process was not exercised.

**F-35 · LOW · OBSERVED — Dead code:** `GradientAccumulator` (in distributed.py) duplicates the loop's own accumulation and is unused; `KVCache` class (inference/kv_cache.py) is unused by `GenerationEngine` (which uses list-of-tuples). `evaluation/benchmarks.py` — verify (see §2.8).

### 2.7 Inference (`inference/`)

**Trace:** prompt → `tokenizer.encode` → `GenerationEngine.generate` (prefill with cache → loop: single-token forward with past_kv → `sample_token` (greedy/temp/top-k/top-p)) → IDs → decode → text. API consumes this.

**F-36 · PASS — Decode-step causality is correct by construction** (single query token attends to all cached past keys; no mask needed) and **KV cache matches full forward** (verified, max diff ~1e-7). Prefill causality relies on the same implicit flash path as training (F-01).

**F-37 · MEDIUM · OBSERVED — No stop-sequence handling despite `stop` in the API schema**; no repetition penalties; `finish_reason` is always `"length"`; no default EOS (tokenizer has none). Sampling (temperature/top-k/top-p) is implemented correctly (reviewed math; standard).

**F-38 · LOW · OBSERVED — `generate_batch` recomputes the full prefix every step** (no cache) — documented tradeoff, fine.

### 2.8 Evaluation (`evaluation/`)

**F-39 · HIGH · OBSERVED — Evaluation is infrastructure around intrinsic metrics only.** `compute_perplexity` (CE-sum over all tokens, **including padded positions** — no `ignore_index`), strided PPL (correct algorithm), top-1/top-5 next-token accuracy. **No standard benchmarks** (MMLU/HellaSwag/Winogrande/WikiText); docs explicitly say none are shipped. The only meaningful numbers today are val-loss/PPL on toy data.

**F-40 · MEDIUM · OBSERVED — `run_evaluation_suite`'s top-k accuracy metric was verified to exist**; but perplexity counts padding tokens and the same causal-dependence as F-01. No test compares PPL of a trained model to a random baseline.

### 2.9 API / Deployment / CI (`api/`, `deployment/`, `.github/`)

**F-41 · CRITICAL · OBSERVED — The API does not import.** `api/main.py` line 109: `from api.routes import completions, health, metrics, search_routes, tokenize_endpoints` → `ImportError: cannot import name 'search_routes' from 'api.routes'` (file does not exist). The claimed "Production API server" cannot start; all downstream (Docker CMD, gunicorn, HF Space, web UI) is broken by this.

**F-42 · HIGH · OBSERVED — The API never loads the committed checkpoint; it serves a freshly random-initialized 19.2 M model** with a 256-token (untrained) tokenizer. Generation output is unlearned noise; "pre-trained" claims in the Space/UI are false in practice.

**F-43 · HIGH · OBSERVED — CI masks type errors:** `mypy ... || true` always passes. CI runs 192 self-referential tests only; nothing trains or evaluates a real model. The `cd.yml` release flow and `release.yml` reference `xrfm` on PyPI which was never published.

**F-44 · MEDIUM · OBSERVED — GPU Docker image likely ends with CPU torch**: `requirements.txt` reinstall (`torch>=2.0.0`) after the cu121 wheel would overwrite it with the default CPU index wheel; CUDA 12.4 runtime vs cu121 wheel mismatch; `workdir`/`USER` inconsistencies; `preload_app` with per-process model load would multiply RAM.

**F-45 · LOW · OBSERVED — `metrics.py`/`health.py` read globals that may be unset; `CORS allow_origins=["*"]`; `webui` mount wrapped in bare `except: pass`.**

### 2.10 Optimization (`optimization/`)

**F-46 · MEDIUM · OBSERVED — FlashAttention wrapper is sound but is the *only* thing making training causal (F-01), and `benchmark_attention` no-ops CUDA timing on CPU.** Quantization (INT8/INT4 groupwise with packing) is implemented and unit-tested with round-trip error checks — self-contained and reasonable. Speculative decoding exists but is unused by the engine; its draft/verify equivalence test is a unit test only.

### 2.11 Search/RAG (`xrfm/search/`)

**F-47 · LOW · OBSERVED — Functional BM25 + "dense" store (tokens only, no vectors despite the name).** Works on small in-memory corpora; `agent.py` decodes generated tokens with the broken tokenizer; the underlying model is random (F-42). A side feature, not core.

### 2.12 Documentation / Claims vs. Reality

**F-48 · HIGH · OBSERVED — Systematic overclaiming:** README "Production Ready — All Phases Complete"; `UPGRADE_PLAN_A_PLUS.md` "enterprise-grade … A++++++"; "1,000× tokenizer speedup" (rank-map is standard; unverified); "FlashAttention 2-4× speedup" (never benchmarked on GPU in-repo); "Mixed precision BF16" (F-23); "pre-trained model" (F-31/F-42); README references `scripts/validate_training.py` that never existed; version string chaos (§3 of BASELINE); CHANGELOG ordering broken (v0.6.0 header above v1.0.0).

**F-49 · LOW — Missing security considerations:** no input-length cap on `/v1/tokenize`; `torch.load` of user-controlled checkpoints (weights_only mitigates code exec but device/CPU DoS remains); no auth on the API (acceptable for local dev).

---

## 3. Verified-PASS Items (important, so the audit is fair)

1. RMSNorm, SwiGLU, residual pre-norm — exact vs independent reference.
2. RoPE (half-split convention) — exact vs independent reference at offsets {0,5,128}; relative-position property holds.
3. Weight tying — object identity + logits==h·Eᵀ exact.
4. KV-cache incremental vs full forward — equal (~1e-7).
5. AdamW config, cosine+warmup schedule math — correct.
6. Gradient accumulation semantics — correct.
7. Sampling math (temp/top-k/top-p), greedy determinism — correct.
8. INT8/INT4 quantize→dequantize round-trip with bounded error — unit tests pass.
9. Checkpoint of current config loads with strict=True (but see F-31/F-42 for what that checkpoint means).
10. All 192 pre-existing tests pass — but they are self-referential (see §4).

## 4. What the 192 Tests Do NOT Verify

- Causal masking (no assertion that future positions are masked).
- Any ground-truth math (no reference implementations anywhere in `tests/`).
- Loss decreases when training on real data.
- Overfitting capability (correctness of the learning loop).
- Resume restores LR schedule / optimizer state.
- Tokenizer round-trip fidelity, whitespace, Unicode.
- Data split leakage.
- API import / server startup.
- Padding → loss masking.
- Reproducibility (seeding).

---

## 5. Severity Summary

| Severity | Count | IDs |
|---|---|---|
| CRITICAL | 5 | F-01, F-11, F-12, F-23, F-41 |
| HIGH | 9 | F-02, F-13, F-17, F-18, F-19, F-20, F-24, F-25, F-31, F-39, F-42, F-43 (12 — see note) |
| MEDIUM | 14 | F-03, F-05, F-08, F-14, F-15, F-16, F-21, F-26, F-27, F-32, F-33, F-34, F-37, F-40, F-44, F-46 |
| LOW | 12 | F-04, F-06, F-07, F-22, F-28, F-35, F-38, F-45, F-47, F-48, F-49 |

> Counts overlap between categories (e.g., F-48 has several sub-issues). The GAP_ANALYSIS.md table is the authoritative list.
