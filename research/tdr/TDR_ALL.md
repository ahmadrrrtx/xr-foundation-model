# Technology Decision Record (TDR) — XRFM

Format: Each record includes Context, Options, Trade-offs, Recommendation, Classification.
Classification key: **CORE** (required now), **OPTIONAL** (add later), **RESEARCH-ONLY** (investigate future).

---

## TDR-001: Configuration System

**Context:** XRFM needs a single source of truth for hyperparameters that supports 10M → 1B+ without rewrites.

**Options:**
1. Raw Python dict (simple, no dependencies, no validation)
2. YAML + custom loader (current approach, `ConfigLoader`)
3. JSON (less readable, harder to comment)
4. Hydra + OmegaConf (powerful multi-run, complex, adds dependency)
5. Pydantic Settings (strong validation, integrates well with Python)

**Trade-offs:**
- Raw dict: Fast to start, impossible to scale (no presets, no validation).
- YAML + custom: Readable, supports comments, easy presets (`ConfigPresets`), lightweight. Requires manual validation (done via `ConfigLoader`).
- Hydra: Excellent for experiment tracking (auto-saves config, multi-run overrides), but over-engineered for Phase 1. Adds `hydra` dependency.
- Pydantic Settings: Strong type validation, can read from env files. Less flexible for multi-file config composition.

**Recommendation:** Keep YAML + `ConfigLoader` (current) as **CORE**. Evaluate Hydra adoption only if multi-run tracking becomes essential (post-v1.0, **OPTIONAL**).

**Classification:** YAML loader = CORE. Hydra = OPTIONAL (post-v1.0).

---

## TDR-002: Tokenizer Strategy

**Context:** Tokenizer must support current BPE implementation and future algorithm swaps.

**Options:**
1. Implement BPE only (fast, simple, no interface abstraction)
2. Implement BPE with stable `TokenizerInterface` (current design, `tokenizer/DESIGN.md`)
3. Use external tokenizer library (`tiktoken`, `transformers`) for speed

**Trade-offs:**
- BPE only: Fastest to implement Phase 2, but requires dataset loader rewrites if switching algorithms later.
- Stable interface: Slight overhead of abstract interface, but dataset loader (`data/loader.py`) will call `tokenizer.encode()` without knowing algorithm. Enables future SentencePiece/Unigram/TikToken swap with zero dataset loader changes.
- External library: Faster development, but violates the principle of full ownership and makes future customization harder.

**Recommendation:** Implement BPE from scratch with stable `TokenizerInterface` (encode/decode/vocab_size/save/load). **CORE** for Phase 2.

**Future algorithms:** SentencePiece, WordPiece, Unigram, TikToken-style = **OPTIONAL** (post-v0.5.0, interface already supports swap).

---

## TDR-003: Distributed Training Framework

**Context:** XRFM must eventually train 7B+ models. Free resources only support single-node/single-GPU for now.

**Options:**
1. PyTorch `DistributedDataParallel` (DDP) — native, well-documented, standard
2. `FullyShardedDataParallel` (FSDP) — PyTorch native, sharding parameters/gradients/optimizer states
3. DeepSpeed ZeRO (ZeRO-1/2/3) — Microsoft library, highly optimized, complex configuration
4. Megatron-LM — NVIDIA framework, very optimized for very large models, steep learning curve
5. `torch.compile` + native DDP — modern PyTorch approach (PyTorch 2.0+)

**Trade-offs:**
- DDP: Simplest multi-GPU setup. Replicates full model on each GPU. Memory-limited (model must fit on single GPU).
- FSDP: Shards model across GPUs, allows larger models on same hardware. More complex checkpointing (must reconstruct full model from shards).
- DeepSpeed ZeRO-3: Most memory-efficient (shards parameters, gradients, optimizer states). Highest communication overhead. Requires significant integration work.
- Megatron-LM: Highly optimized for very large models (70B+). Over-engineered for 10M–1B scale. Best adopted only when necessary.

**Recommendation:**
- **CORE (Phase 5):** Design training loop with DDP hooks (`torch.nn.parallel.DistributedDataParallel`). This ensures multi-GPU compatibility without complex sharding.
- **OPTIONAL (Phase 8+):** Add FSDP support for scaling beyond single-node memory limits.
- **RESEARCH-ONLY (future):** DeepSpeed ZeRO-3 and Megatron-LM for 70B+ scale. These require institutional resources and significant engineering investment.

**Classification:** DDP hooks = CORE. FSDP = OPTIONAL. DeepSpeed/Megatron = RESEARCH-ONLY.

---

## TDR-004: Attention Implementation

**Context:** Attention mechanism is the core of transformer architecture. Performance and flexibility trade-offs matter for scaling.

**Options:**
1. Standard `nn.MultiheadAttention` (PyTorch native)
2. Manual multi-head attention (current design approach — full control, educational value, original code)
3. FlashAttention (via `torch.nn.functional.scaled_dot_product_attention` — optimized kernel, faster, less memory)
4. `xFormers` (efficient attention implementations, some approximations)
5. Triton custom kernels (maximum performance, highest complexity)

**Trade-offs:**
- Manual implementation: Full control, no hidden behavior, easy to extend (GQA, sliding window, sparse attention). Slower than optimized kernels.
- FlashAttention: 2–4x faster for long sequences, same mathematical result. Integrated in PyTorch 2.0+. Minimal code change (drop-in replacement for `scaled_dot_product_attention`).
- `xFormers`: Includes efficient implementations with some approximations (e.g., sparse patterns). Good for very long contexts but introduces approximation errors.
- Triton: Maximum customization, requires writing CUDA-level kernels. Not recommended unless custom attention patterns are needed.

**Recommendation:**
- **CORE (Phase 4):** Implement manual multi-head attention (original, full control). This ensures understanding and allows future modifications (GQA, RoPE integration).
- **OPTIONAL (Phase 9):** Replace manual attention with FlashAttention (`scaled_dot_product_attention`) for performance optimization. Zero architecture rewrites required — the interface (`MultiHeadAttention` class) remains the same.
- **RESEARCH-ONLY:** Triton custom kernels or `xFormers` approximations.

**Classification:** Manual attention = CORE. FlashAttention = OPTIONAL. Triton/xFormers = RESEARCH-ONLY.

---

## TDR-005: Inference Engine

**Context:** Serving XRFM models requires efficient token generation with streaming, batching, and memory management.

**Options:**
1. Custom inference loop (manual autoregressive generation, KV cache, sampling strategies)
2. `vLLM` (high-throughput serving with PagedAttention, continuous batching, prefix caching)
3. `TensorRT-LLM` (NVIDIA-optimized inference, fastest on NVIDIA GPUs, more complex setup)
4. `llama.cpp` / GGUF (CPU and edge GPU inference, excellent for local deployment)
5. ONNX Runtime (portable, optimized, requires model export)

**Trade-offs:**
- Custom loop: Full control, easy to extend (add speculative decoding, custom sampling), no external dependencies. Not optimized for high-throughput multi-user serving.
- `vLLM`: De-facto standard for production LLM serving. PagedAttention minimizes KV cache fragmentation. Continuous batching improves throughput. Requires GPU with CUDA.
- `TensorRT-LLM`: Fastest inference on NVIDIA GPUs. Requires model compilation and engine building. Less flexible than vLLM for rapid experimentation.
- `llama.cpp`: Excellent for CPU and low-resource GPU inference. Good for edge deployment and testing. Not suitable for high-throughput production serving.
- ONNX Runtime: Portable across devices. Requires ONNX export (additional tool chain). Not optimized specifically for LLM autoregressive generation.

**Recommendation:**
- **CORE (Phase 6):** Build custom inference engine (streaming generation, KV cache, temperature/top-k/top-p, batch inference). This ensures full understanding and allows custom modifications (e.g., reasoning chain generation, tool use integration).
- **OPTIONAL (Phase 9+):** Add `vLLM` compatibility layer (export weights to vLLM format, or serve via vLLM API). This enables production-scale serving without rewriting the model.
- **RESEARCH-ONLY:** `TensorRT-LLM` optimization, ONNX export for edge devices, speculative decoding (can be added to custom loop later).

**Classification:** Custom engine = CORE. vLLM compatibility = OPTIONAL. TensorRT-LLM/ONNX/Speculative decoding = RESEARCH-ONLY.

---

## TDR-006: Experiment Tracking

**Context:** Reproducibility requires tracking configs, metrics, random seeds, hardware info, and artifacts.

**Options:**
1. CSV logging (simple, portable, no dependencies)
2. `TensorBoard` (standard for PyTorch, good for loss curves)
3. `Weights & Biases` (WandB) — powerful experiment tracking, requires account/API key
4. `MLflow` — open-source experiment tracking, requires server setup
5. `Aim` — open-source alternative to WandB

**Trade-offs:**
- CSV: Zero dependencies, easy to parse, not interactive.
- TensorBoard: Interactive visualization, standard in PyTorch ecosystem, requires `tensorboard` package.
- WandB: Best interactive experience (automatic artifact tracking, comparison dashboards), but requires external account and network access.
- MLflow: Self-hosted, good for enterprise, complex setup.
- Aim: Self-hosted, lighter than MLflow, less mature than WandB.

**Recommendation:**
- **CORE (Phase 5):** CSV logging + optional `TensorBoard` integration. CSV ensures portability; TensorBoard provides interactive visualization without external dependencies.
- **OPTIONAL (post-v0.5.0):** `Weights & Biases` integration via modular interface (`xrfm/utils/logging.py`). This allows users to choose tracking without forcing dependencies.
- **RESEARCH-ONLY:** `MLflow`, `Aim` (evaluate only if self-hosted tracking becomes a requirement).

**Classification:** CSV + TensorBoard = CORE. WandB = OPTIONAL. MLflow/Aim = RESEARCH-ONLY.

---

## TDR-007: Software Architecture Principles

**Context:** XRFM should follow clean architecture (separation of concerns, stable interfaces, no circular dependencies).

**Principles Adopted:**
- **Single Responsibility:** `tokenizer/` handles tokenization; `model/attention/` handles attention; `training/` handles optimization loops; `inference/` handles generation.
- **Dependency Inversion:** High-level modules (`training/`) depend on abstract interfaces (`ConfigLoader`, `TokenizerInterface`), not concrete implementations.
- **Open/Closed:** Modules open for extension (new tokenizer algorithms, new attention variants) but closed for modification (stable interfaces protect existing code).
- **Interface Segregation:** Small, focused interfaces (`TokenizerInterface`, `ModelConfig`) rather than large monolithic classes.
- **Config-Driven:** No hard-coded hyperparameters. Every parameter flows through `ConfigLoader`.

**Patterns Adopted:**
- **Factory Pattern:** `ConfigPresets` acts as a factory for model configurations.
- **Strategy Pattern:** Tokenizer algorithms implement `TokenizerInterface`; dataset loaders implement a similar stable interface (planned).
- **Repository Pattern:** Data access abstracted through dataset loader interfaces.

**Patterns Not Adopted (Yet):**
- Full Hexagonal Architecture: Over-engineered for current scale. Adopt if service boundaries become complex (post-v1.0, when serving and training become separate services).
- Event Sourcing: Not needed for training pipeline.

**Classification:** Clean architecture principles = CORE. Factory/Strategy patterns = CORE. Full Hexagonal/Event Sourcing = RESEARCH-ONLY (evaluate for production deployment phase).
