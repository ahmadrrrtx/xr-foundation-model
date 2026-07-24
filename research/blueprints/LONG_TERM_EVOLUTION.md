# XRFM — Long-Term Evolution Plan

This document maps how today's architecture (`v0.1.0`) evolves toward future versions without requiring major rewrites. Every transition is designed to be incremental.

---

## Version Timeline

| Version | Model Name | Key Change | Architecture Impact | Rewrites Required? |
|---|---|---|---|---|
| v0.1.0 | Foundation | Project setup, config system | None | N/A |
| v0.2.0 | XRFM-10M | Tokenizer (BPE) + dataset loader | Tokenizer interface defined; dataset loader uses interface | None |
| v0.3.0 | XRFM-10M | Dataset pipeline (Tiny Shakespeare → WikiText) | Config-driven dataset selection (`datasets.default`) | None |
| v0.4.0 | XRFM-10M | Transformer architecture (attention, layers, model) | Manual multi-head attention with RoPE/RMSNorm/SwiGLU | None |
| v0.5.0 | XRFM-50M | Training engine (loop, optimizer, scheduler) | Config-driven hyperparameters; DDP hooks added | None |
| v0.6.0 | XRFM-50M | Inference engine + evaluation | Custom inference loop; KV cache; streaming; perplexity evaluation | None |
| v0.7.0 | XRFM-100M | Scaling + multi-GPU (DDP) | DDP activation via config; same code path for single/multi GPU | None |
| v0.8.0 | XRFM-300M | Optimization + production deployment | FlashAttention optional; quantization optional; FastAPI server; Docker | None (optional features only) |
| v0.9.0 | XRFM-1B | Full production pipeline | Continuous training; public hosting; web UI; security filters | None (addition of services, not rewrites) |
| v1.0.0 | XRFM-1B (Stable) | Stable release with full documentation | All modules complete; all interfaces stable | None |
| v1.5.0 | XRFM-7B | Scale to larger models | FSDP optional; larger dataset storage; multi-node training (optional) | None (FSDP is optional enhancement) |
| v2.0.0 | XRFM-7B (Stable) + New Architecture Branch | Potential architecture evaluation (encoder-decoder or alternative) | New model architecture file (`model/alternative.py`) without breaking `GPTModel` | None for existing users |
| v3.0.0 | XRFM-MoE | Mixture of Experts | `model/moe.py` — new architecture module; dataset loader and training loop unchanged; config preset `xrfm_moe` added to `ConfigPresets` | None for existing users (MoE is a new model option) |
| v3.5.0 | XRFM-Multimodal | Multimodal integration (vision + audio) | New data pipeline (`data/multimodal/`); model architecture extended with vision encoder; tokenizer unchanged | Minimal (multimodal is new feature branch) |
| v4.0.0 | XRFM-Reasoning | Reasoning-optimized model | Training loop extended with RL-based optimization (GRPO or DPO concepts); dataset pipeline extended with reasoning traces; inference engine unchanged | Minimal (training enhancement, not architecture rewrite) |

---

## Key Design Principles Enabling This Evolution

### 1. Config-Driven Scaling
Every version change requires only a config update (`ConfigPresets`), not code rewrites. Changing from `XRFM-10M` to `XRFM-1B` updates `d_model`, `n_layers`, `n_heads`, `max_seq_len`, and `vocab_size` via config.

### 2. Stable Interfaces
The `TokenizerInterface`, dataset loader interface, model interface, and training loop interface remain unchanged. Adding new algorithms or architectures creates new implementations of existing interfaces, not new interfaces.

### 3. Optional Enhancements
Every performance or scale improvement is an optional enhancement (FlashAttention, FSDP, vLLM, quantization, speculative decoding). The core code works without them. Users can adopt them incrementally.

### 4. Modular Architecture
No circular dependencies. Changing the attention mechanism (manual → FlashAttention) only affects `model/attention/`, not dataset loading, training, or inference. Changing the tokenizer algorithm only affects `tokenizer/`, not the dataset loader (thanks to `TokenizerInterface`).

### 5. Original Code with Attribution
Every module is original implementation. References to papers (Vaswani, Karpathy, Raschka, Meta, DeepSeek) are documented in comments and `CONTRIBUTING.md`. This ensures the project is not a tutorial fork and can be maintained independently.

---

## Transition Requirements

### v0.5.0 → v1.0.0 (Stable)
- All modules complete.
- All interfaces stable.
- Full documentation (`docs/` complete).
- Professional open-source standards met (`LICENSE`, `CONTRIBUTING`, `ROADMAP`, `CHANGELOG`, `SECURITY`, `CODE_OF_CONDUCT`).
- Testing framework operational (`tests/` covers all public modules).
- Continuous training pipeline designed (implementation optional but architecture ready).

### v1.0.0 → v1.5.0 (Scale)
- FSDP optional integration (if multi-node required).
- Dataset storage scaled to handle 1T+ tokens (external storage integration: S3, GCS, or parallel file systems).
- Checkpoint storage scaled for 7B model weights (external object storage, sharded checkpoint formats).

### v1.5.0 → v2.0.0 (Architecture Branch)
- Potential evaluation of encoder-decoder or alternative architectures.
- New architecture file (`model/alternative.py`) without breaking `GPTModel`.
- Config presets expanded (`xrfm_encoder_decoder`, etc.).

### v2.0.0 → v3.0.0 (MoE)
- `model/moe.py` — new architecture module.
- Config preset `xrfm_moe` added.
- Dataset loader unchanged (same data format, same tokenizer).
- Training loop enhanced with MoE-specific load balancing (optional — core loop remains the same).

### v3.0.0 → v3.5.0 (Multimodal)
- `data/multimodal/` — new dataset pipeline.
- `tokenizer/` unchanged (text tokenizer remains; vision data handled separately).
- `model/gpt.py` extended with vision encoder integration (optional branch).

### v3.5.0 → v4.0.0 (Reasoning)
- `training/loop.py` extended with RL-based optimization (GRPO/DPO concepts).
- Dataset pipeline extended with reasoning trace generation.
- Inference engine unchanged (same generation interface; reasoning is a training/data enhancement).

---

## Evolution Constraints

To maintain the "no major rewrites" principle:

1. **No breaking interface changes** for core components (`ConfigLoader`, `TokenizerInterface`, model interface) without a major version bump.
2. **No removal of optional dependencies** — once an optional feature is added (e.g., FlashAttention), it remains available but is never required.
3. **Backward compatibility** — users who train `XRFM-10M` with `v0.2.0` should be able to load that checkpoint in `v0.9.0` (with appropriate compatibility layer, if needed).
4. **Config compatibility** — new config keys can be added; old config keys are deprecated gradually (with warnings) rather than removed immediately.
