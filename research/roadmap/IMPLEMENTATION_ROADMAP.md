# XRFM — Implementation Roadmap

Every phase includes: Objectives, Deliverables, Dependencies, Risks, Success Criteria, Estimated Complexity.

---

## Phase 1: Foundation (COMPLETE — v0.1.0)

**Objectives:** Establish professional open-source structure, branded project (`XRFM`), config system, original code with attribution.
**Deliverables:** `xrfm/` package, `ConfigLoader`, professional docs (`LICENSE`, `CONTRIBUTING`, `ROADMAP`, `CHANGELOG`, `SECURITY`, `CODE_OF_CONDUCT`), architecture review.
**Dependencies:** None.
**Risks:** None (completed).
**Success Criteria:** Git tagged `v0.1.0`; architecture passes review; no hard-coded hyperparameters.
**Complexity:** Low.

---

## Phase 2: Tokenizer (READY — Pending Approval)

**Objectives:** Implement original Byte Pair Encoding (BPE) tokenizer. Design stable `TokenizerInterface` for future algorithm swaps.
**Deliverables:**
- `tokenizer/bpe.py` (original BPE training and encoding/decoding)
- `tokenizer/interface.py` (`TokenizerInterface` abstract class)
- `tokenizer/encode.py`, `tokenizer/decode.py`
- Tests: `tests/test_tokenizer_bpe.py`
- Documentation: `tokenizer/DESIGN.md` updated with implementation notes
**Dependencies:** Phase 1 complete (config loader available, directory structure ready).
**Risks:**
- BPE vocabulary size selection may need tuning for future dataset scales.
- Mitigation: Config-driven vocab size (`model.vocab_size`); vocabulary saved separately from code.
**Success Criteria:**
- Tokenizer trains on sample text and produces integer sequences.
- `encode()` and `decode()` are inverses (lossless roundtrip for complete vocabulary).
- Stable interface supports future BPE → SentencePiece swap without dataset loader changes.
**Complexity:** Medium.
**Classification of Components:**
- BPE implementation = CORE.
- `TokenizerInterface` design = CORE.
- Vocabulary persistence (`save`/`load`) = CORE.

---

## Phase 3: Dataset Pipeline

**Objectives:** Build dataset loader supporting Tiny Shakespeare, WikiText, OpenWebText, and future streaming/custom datasets.
**Deliverables:**
- `data/datasets/` — dataset download/preparation scripts
- `xrfm/data/loader.py` — dataset loader using `TokenizerInterface`
- `xrfm/data/streaming.py` — streaming dataset support (future-proof for large datasets)
- `tests/test_data_loader.py`
- `docs/data/DATASET_GUIDE.md`
**Dependencies:** Phase 2 (tokenizer interface stable).
**Risks:**
- Small datasets (Tiny Shakespeare) may cause overfitting quickly; larger datasets require storage.
- Mitigation: Config-driven dataset selection; dataset version tracking (hash or manifest file).
**Success Criteria:**
- Dataset loader produces `(input_ids, target_ids)` batches compatible with training loop.
- Dataset can be switched via config (`datasets.default`) without code changes.
- Streaming mode works (optional enhancement — basic loader required, streaming optional).
**Complexity:** Medium.

---

## Phase 4: Transformer Architecture

**Objectives:** Implement original GPT-style decoder-only transformer with modern architecture features (RoPE, RMSNorm, SwiGLU).
**Deliverables:**
- `model/attention/multi_head.py` (original multi-head attention with masking)
- `model/layers/transformer_block.py` (pre-norm residual structure)
- `model/gpt.py` (full model: embedding, positional encoding, blocks, output projection)
- `tests/test_model_forward.py` (shape validation, gradient flow check)
- `docs/model/ARCHITECTURE.md`
**Dependencies:** Phase 2 (tokenizer) and Phase 3 (dataset loader) — not strictly required for model definition but needed for integration testing.
**Risks:**
- Deep networks (32+ layers) may suffer gradient instability; shallow networks (6 layers for 10M) are stable.
- Mitigation: Gradient clipping (`config.training.gradient_clip`), layer normalization (`use_rmsnorm`), residual connections.
**Success Criteria:**
- Model accepts `(batch_size, seq_len)` input and produces `(batch_size, seq_len, vocab_size)` logits.
- Gradient flows correctly through all layers (`torch.autograd.gradcheck` or equivalent manual verification).
- Model parameters scale predictably with config changes (`d_model`, `n_layers`, `n_heads`).
**Complexity:** High (core architecture implementation).
**Classification of Components:**
- Manual multi-head attention = CORE.
- RoPE integration = CORE (configurable).
- RMSNorm = CORE (configurable).
- SwiGLU = CORE (configurable).
- FlashAttention replacement = OPTIONAL (Phase 9).
- GQA (Grouped Query Attention) = OPTIONAL (post-v0.5.0, requires attention module modifications).

---

## Phase 5: Training Engine

**Objectives:** Implement production-quality training loop with mixed precision, gradient clipping, checkpointing, resume capability, and DDP hooks.
**Deliverables:**
- `training/loop.py` — main training loop (`train_step`, `validation_step`)
- `training/optimizer.py` — AdamW setup with weight decay
- `training/scheduler.py` — cosine decay with warmup
- `training/checkpoint.py` — save/load checkpoints including optimizer, scheduler, scaler, token count, dataset version, git commit hash
- `training/resume.py` — resume from any checkpoint with zero manual changes (`config.training.resume_from`)
- `tests/test_training_loop.py`
- `docs/training/TRAINING_GUIDE.md`
**Dependencies:** Phase 3 (dataset loader) and Phase 4 (model architecture).
**Risks:**
- Training instability at larger scales (loss divergence, gradient explosion).
- Mitigation: Mixed precision (`mixed_precision: true`), gradient clipping (`gradient_clip: 1.0`), careful learning rate scheduling (`warmup_steps`), checkpoint frequency (`checkpoint_every`).
- Multi-GPU synchronization errors (DDP requires proper initialization).
- Mitigation: DDP hooks added but only activated when `torch.distributed` is initialized; single-GPU training remains default.
**Success Criteria:**
- Training loop runs without errors on Tiny Shakespeare dataset.
- Loss decreases consistently over first 1000 steps.
- Checkpoints can be saved and resumed (`resume_from` path set in config, training continues from saved step/token count).
- Checkpoint includes full state (model weights, optimizer, scheduler, gradient scaler, current step, token count, dataset version, git hash, config snapshot).
**Complexity:** High.
**Classification of Components:**
- Mixed precision (`torch.cuda.amp`) = CORE.
- Gradient clipping = CORE.
- Checkpointing/resume = CORE.
- DDP hooks = CORE (design only; activation requires multi-GPU environment).
- FSDP integration = OPTIONAL (Phase 8).
- DeepSpeed ZeRO = RESEARCH-ONLY.

---

## Phase 6: Inference Engine

**Objectives:** Implement streaming generation with KV cache, temperature/top-k/top-p sampling, batch inference.
**Deliverables:**
- `inference/engine.py` — custom inference loop
- `inference/kv_cache.py` — KV cache implementation (store previous K, V tensors)
- `inference/sampling.py` — temperature, nucleus (top-p), top-k, greedy decoding
- `generation/text.py` — high-level generation interface
- `tests/test_inference_streaming.py`
- `docs/inference/INFERENCE_GUIDE.md`
**Dependencies:** Phase 4 (model) and Phase 5 (training — for loading trained weights).
**Risks:**
- KV cache memory grows linearly with sequence length; very long sequences (2048+) may exceed GPU memory.
- Mitigation: Configurable `max_seq_len`; future PagedAttention (vLLM integration, Phase 9) addresses memory fragmentation.
**Success Criteria:**
- Streaming generation produces tokens one at a time (not full sequence after complete generation).
- Temperature/top-k/top-p sampling produces varied outputs.
- Batch inference handles multiple prompts efficiently.
**Complexity:** Medium.
**Classification of Components:**
- Custom inference loop = CORE.
- KV cache = CORE.
- Sampling strategies (temperature, top-k, top-p) = CORE.
- Beam search = OPTIONAL (can be added to sampling module later).
- Speculative decoding = RESEARCH-ONLY (requires small draft model integration).
- vLLM compatibility = OPTIONAL (Phase 9).

---

## Phase 7: Evaluation Pipeline

**Objectives:** Implement automated evaluation for perplexity and benchmark comparison.
**Deliverables:**
- `evaluation/perplexity.py` — perplexity calculation on validation set
- `evaluation/benchmarks.py` — benchmark framework (placeholder for MMLU, HellaSwag, etc.)
- `tests/test_evaluation.py`
- `docs/evaluation/EVALUATION_GUIDE.md`
**Dependencies:** Phase 5 (training loop — must load model weights for evaluation).
**Risks:**
- Benchmark datasets may not be available or licensed.
- Mitigation: Start with perplexity (requires only validation text); add external benchmarks as optional modules.
**Success Criteria:**
- Perplexity computed correctly on held-out dataset.
- Evaluation script runs independently of training (loads checkpoint, evaluates, outputs metric).
**Complexity:** Low-Medium.
**Classification of Components:**
- Perplexity evaluation = CORE.
- Benchmark framework = CORE (structure); specific benchmarks (MMLU, etc.) = OPTIONAL (add as needed).

---

## Phase 8: Scaling & Multi-GPU

**Objectives:** Enable distributed training and larger model scales (XRFM-100M, XRFM-1B).
**Deliverables:**
- `training/distributed.py` — DDP initialization and launch utilities
- `training/fsdp_wrapper.py` — optional FSDP integration (if adopted)
- `scripts/launch_multi_gpu.sh` — example launch script
- `docs/scaling/MULTI_GPU_GUIDE.md`
**Dependencies:** Phase 5 (training loop must be DDP-ready). Phase 4 (model architecture must handle sharded state if FSDP adopted).
**Risks:**
- Multi-GPU requires InfiniBand or high-speed Ethernet for efficient all-reduce.
- FSDP requires careful checkpoint reconstruction (must gather full parameters for saving).
- Mitigation: DDP only for Phase 8; FSDP only adopted if needed (optional enhancement). Checkpoint system designed to handle sharded states from the beginning.
**Success Criteria:**
- Training script launches with `torchrun` or `torch.distributed.launch` without errors.
- Model trains on multiple GPUs (single-node multi-GPU first, multi-node later).
- Checkpoint system supports distributed saves (optional but designed for).
**Complexity:** High (infrastructure and synchronization challenges).
**Classification of Components:**
- DDP launch scripts = CORE (Phase 8).
- FSDP wrapper = OPTIONAL.
- Multi-node (multi-server) = RESEARCH-ONLY (requires institutional infrastructure).

---

## Phase 9: Optimization & Production Readiness

**Objectives:** Improve performance and prepare for deployment.
**Deliverables:**
- `optimization/flash_attention.py` — optional FlashAttention integration
- `optimization/quantization.py` — quantization utilities (INT8, INT4, AWQ, GPTQ concepts)
- `optimization/speculative_decoding.py` — optional speculative decoding framework
- `deployment/fastapi_server.py` — production API with streaming
- `deployment/docker/` — Dockerfile and compose files
- `benchmark/performance_tests.py` — throughput and latency benchmarks
**Dependencies:** Phase 6 (inference engine) and Phase 7 (evaluation — for measuring optimization impact).
**Risks:**
- Optimization changes (quantization) may affect model accuracy.
- Mitigation: Benchmark accuracy before and after optimization; maintain unoptimized weights as reference.
**Success Criteria:**
- FlashAttention (optional) improves training/inference speed by measurable amount.
- Quantization produces valid GGUF or optimized format files.
- FastAPI server responds to requests with low latency.
- Docker container runs without external GPU dependencies (CPU mode supported, GPU optional).
**Complexity:** Medium-High.
**Classification of Components:**
- FlashAttention = OPTIONAL.
- Quantization (INT8/INT4) = OPTIONAL.
- Speculative decoding = RESEARCH-ONLY.
- FastAPI server = CORE (production deployment).
- Docker deployment = CORE.
- vLLM compatibility layer = OPTIONAL.
- TensorRT-LLM export = RESEARCH-ONLY.

---

## Phase 10: Deployment & Continuous Improvement

**Objectives:** Launch public-facing service and establish continuous training pipeline.
**Deliverables:**
- Hosted API endpoint (FastAPI on cloud or Hugging Face Spaces)
- Web UI (`webui/` — Gradio or custom React interface)
- Continuous data collection pipeline (with privacy safeguards)
- Retraining/update workflow
- Public documentation (`docs/` completed for all modules)
**Dependencies:** All previous phases complete. Phase 8 (multi-GPU) not strictly required but strongly recommended for large-scale updates.
**Risks:**
- Public deployment creates security risks (prompt injection, data leakage, misuse).
- Mitigation: Implement input filtering, output filtering, rate limiting, authentication (if needed), and audit logging (security measures from `SECURITY.md`).
- Continuous training may cause catastrophic forgetting.
- Mitigation: Use small learning rates for updates, mix new data with original training data, evaluate before and after updates.
**Success Criteria:**
- Public API responds correctly.
- Web UI allows interactive chat with model.
- Continuous update pipeline can retrain/fine-tune model on new data and deploy updated weights.
**Complexity:** High (production engineering, security, operations).
**Classification of Components:**
- FastAPI server + Web UI = CORE.
- Continuous training pipeline = CORE (architecture supports it; full automation optional post-v1.0).
- Public cloud hosting = OPTIONAL (can deploy locally or via Docker first).
- Security filters (red-teaming) = CORE (design phase includes security; production requires active monitoring).
