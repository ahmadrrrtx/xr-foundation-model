# Technology Decision Record (TDR) — XRFM

Each record: Context, Options, Trade-offs, Recommendation, Classification.
Classification key: **CORE** (required now), **OPTIONAL** (add later), **RESEARCH-ONLY** (investigate future).

---

## TDR-001: Configuration System — v0.1.0

**Context:** XRFM needs a single source of truth for hyperparameters supporting 10M → 1B+ without rewrites.

**Options:** Raw Python dict, YAML + custom loader, JSON, Hydra + OmegaConf, Pydantic Settings.

**Chosen:** YAML + `ConfigLoader` with typed dataclasses (`ModelConfig`, `TrainingConfig`) and preset profiles (`ConfigPresets`). Readable, supports comments, lightweight, no extra dependencies beyond `pyyaml`.

**Classification:** YAML loader = **CORE**. Hydra = OPTIONAL (post-v1.0).

**Implementation:** `xrfm/config/loader.py`. Config-driven design verified: changing model scale requires only YAML updates.

---

## TDR-002: Tokenizer Strategy — v0.2.0

**Context:** Tokenizer must support BPE now and future algorithm swaps without dataset loader rewrites.

**Options:** BPE only, BPE with stable interface, external library.

**Chosen:** Original BPE with stable `TokenizerInterface` ABC (`encode`, `decode`, `vocab_size`, `save`, `load`). Dataset loader depends only on the interface, enabling zero-rewrite algorithm swaps.

**Classification:** BPE + TokenizerInterface = **CORE**. SentencePiece/Unigram/TikToken = OPTIONAL (post-v0.5.0).

**Implementation:** `tokenizer/bpe.py` (BytePairEncoder), `tokenizer/interface.py` (ABC).

**Reference:** Sennrich et al. (2016) — BPE algorithm. Implementation is original.

---

## TDR-003: Distributed Training Framework — v0.1.0

**Context:** XRFM must eventually train 7B+ models. Free resources only support single-node/single-GPU for now.

**Options:** DDP, FSDP, DeepSpeed ZeRO, Megatron-LM, torch.compile + DDP.

**Chosen:** DDP hooks designed into training loop from Phase 5 (not activated in single-GPU mode). FSDP optional for Phase 8+. DeepSpeed/Megatron deferred to RESEARCH-ONLY.

**Classification:** DDP hooks = **CORE** (design). FSDP = OPTIONAL (Phase 8). DeepSpeed/Megatron = RESEARCH-ONLY.

---

## TDR-004: Attention Implementation — v0.4.0

**Context:** Core transformer mechanism. Performance and flexibility trade-offs matter for scaling.

**Options:** `nn.MultiheadAttention`, manual attention, FlashAttention, xFormers, Triton kernels.

**Chosen:** Original manual multi-head attention (`W_q`, `W_k`, `W_v`, `W_o`; scaled dot-product; masking; RoPE integration). Full control for future GQA, sliding window, sparse attention. FlashAttention is a drop-in replacement (Phase 9).

**Classification:** Manual attention = **CORE**. FlashAttention = OPTIONAL (Phase 9). Triton/xFormers = RESEARCH-ONLY.

**Implementation:** `model/attention/multi_head.py`.

**Reference:** Vaswani et al. (2017). Implementation is original.

---

## TDR-005: Inference Engine — v0.6.0 (PLANNED)

**Context:** Serving XRFM models requires efficient token generation with streaming, batching, and memory management.

**Options:** Custom inference loop, vLLM, TensorRT-LLM, llama.cpp, ONNX Runtime.

**Chosen (DESIGN ONLY — NOT IMPLEMENTED):** Custom inference engine (`generate()` with KV cache, temperature/top-k, batch inference) as CORE for Phase 6. vLLM compatibility as OPTIONAL (Phase 9).

**Classification:** Custom engine = **CORE** (Phase 6). vLLM = OPTIONAL (Phase 9). TensorRT-LLM/ONNX/Speculative = RESEARCH-ONLY.

---

## TDR-006: Experiment Tracking — v0.5.0

**Context:** Reproducibility requires tracking configs, metrics, and artifacts.

**Chosen:** Basic `print()` logging for v0.5.0 (no external dependencies). CSV logging and optional TensorBoard planned for Phase 7. WandB as OPTIONAL.

**Classification:** CSV + print = **CORE** (v0.5.0 current). TensorBoard/WandB = OPTIONAL (Phase 7).

---

## TDR-007: Software Architecture Principles — v0.1.0

**Principles adopted:**
- **Single Responsibility:** `tokenizer/` handles tokenization; `model/` handles architecture; `training/` handles optimization.
- **Dependency Inversion:** High-level modules depend on abstract interfaces (`TokenizerInterface`, `ConfigLoader`), not concrete implementations.
- **Open/Closed:** Modules open for extension (new tokenizer algorithms) but closed for modification (stable interfaces).
- **Config-Driven:** No hard-coded hyperparameters. Every parameter flows through `ConfigLoader`.

**Classification:** Clean architecture principles = **CORE**.

---

## Summary: Classification Matrix

| Component | Classification | Phase | Status |
|---|---|---|---|
| YAML ConfigLoader | CORE | v0.1.0 | Implemented |
| BPE Tokenizer | CORE | v0.2.0 | Implemented |
| TokenizerInterface | CORE | v0.2.0 | Implemented |
| Dataset Pipeline | CORE | v0.3.0 | Implemented |
| Manual Multi-Head Attention | CORE | v0.4.0 | Implemented |
| RoPE | CORE | v0.4.0 | Implemented |
| RMSNorm | CORE | v0.4.0 | Implemented |
| SwiGLU | CORE | v0.4.0 | Implemented |
| GPTModel | CORE | v0.4.0 | Implemented |
| Weight Tying | CORE | v0.4.0 | Implemented |
| AdamW Optimizer | CORE | v0.5.0 | Implemented |
| Cosine + Warmup Schedule | CORE | v0.5.0 | Implemented |
| Checkpoint Save/Load | CORE | v0.5.0 | Implemented |
| Mixed Precision (AMP) | CORE | v0.5.0 | Implemented |
| Gradient Clipping | CORE | v0.5.0 | Implemented |
| Training Loop | CORE | v0.5.0 | Implemented |
| Inference Engine | CORE | v0.6.0 | Not started |
| DDP Hooks | CORE (design) | v0.5.0 | Designed |
| FlashAttention | OPTIONAL | v0.9.0 | Deferred |
| FSDP | OPTIONAL | v0.8.0 | Deferred |
| GQA | OPTIONAL | post-v1.0 | Deferred |
| vLLM Integration | OPTIONAL | v0.9.0 | Deferred |
| DeepSpeed | RESEARCH-ONLY | future | Deferred |
| MoE Architecture | RESEARCH-ONLY | v3.0 | Deferred |
| Multimodal | RESEARCH-ONLY | v3.5 | Deferred |
