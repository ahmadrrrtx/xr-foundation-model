# XRFM — Decision Log

Every significant architectural or implementation decision is recorded here.
This preserves context over the long-term evolution of the project.

Format:
- Date / Version
- Problem
- Options Considered
- Chosen Solution
- Why
- Reversible? (Yes / No / Later)

---

## 2026-07-24 — v0.1.0 — Architecture Freeze
- **Problem:** Original Phase 1 was tutorial-style, not professional open-source.
- **Options:** (1) Continue as tutorial, (2) Refactor to professional standards, (3) Restart.
- **Chosen:** Refactor to professional open-source platform (`XRFM` / `xrfm`).
- **Why:** Long-term goal is a foundation model platform supporting 10M → 70B + MoE + Multimodal + Reasoning.
- **Reversible:** No (rebranding and restructuring are permanent, but architecture supports both old and new concepts).

---

## 2026-07-24 — v0.1.0 — Configuration System
- **Problem:** How to manage hyperparameters for scalable model architectures?
- **Options:** (1) Raw Python dicts, (2) JSON, (3) YAML + custom loader, (4) Hydra + OmegaConf, (5) Pydantic Settings.
- **Chosen:** YAML + `ConfigLoader` (custom) with typed `ConfigPresets`.
- **Why:** Readable, supports comments, lightweight, no external dependency for core config, supports presets (`XRFM-10M`, `XRFM-1B`), validated with `ConfigLoader`.
- **Reversible:** Later — Hydra can be adopted as optional enhancement (post-v1.0) without breaking existing YAML configs.
- **Classification:** CORE.

---

## 2026-07-24 — v0.1.0 — Tokenizer Strategy
- **Problem:** Tokenizer must work now (BPE) and support future algorithms (SentencePiece, Unigram, TikToken-style) without rewrites.
- **Options:** (1) Implement BPE only (no abstraction), (2) Implement BPE with stable `TokenizerInterface`, (3) Use `tiktoken` directly (fast, less ownership), (4) Use `transformers` tokenizer (external dependency).
- **Chosen:** Original BPE implementation with stable `TokenizerInterface` abstract base class (`encode`, `decode`, `vocab_size`, `save`, `load`).
- **Why:** Full ownership of tokenizer pipeline; dataset loader depends only on interface, not algorithm; future algorithm swaps require zero dataset loader changes; original code aligns with open-source independence goal.
- **Reversible:** Later — interface allows future algorithm adoption without breaking existing dataset/training code.
- **Classification:** CORE (BPE + Interface). SentencePiece/Unigram/TikToken = OPTIONAL (post-v0.5.0).

---

## 2026-07-24 — v0.1.0 — Distributed Training Approach
- **Problem:** How to design for multi-GPU without requiring it immediately?
- **Options:** (1) DDP only (PyTorch native), (2) FSDP (PyTorch native, sharding), (3) DeepSpeed ZeRO, (4) Megatron-LM.
- **Chosen:** DDP hooks designed into training loop (Phase 5, activated Phase 8). FSDP = OPTIONAL (post-Phase 8). DeepSpeed/Megatron = RESEARCH-ONLY.
- **Why:** DDP is standard, well-documented, requires minimal code changes. FSDP requires careful checkpoint reconstruction. DeepSpeed and Megatron are complex and only needed for very large models (70B+).
- **Reversible:** Later — DDP hooks can be activated without code rewrites; FSDP or DeepSpeed can be integrated as optional modules.
- **Classification:** DDP design = CORE. FSDP = OPTIONAL. DeepSpeed/Megatron = RESEARCH-ONLY.

---

## 2026-07-24 — v0.1.0 — Attention Implementation
- **Problem:** Attention mechanism must be original, controllable, and scalable.
- **Options:** (1) `nn.MultiheadAttention` (PyTorch native, hidden behavior), (2) Manual multi-head attention (full control, original), (3) FlashAttention (optimized kernel, drop-in), (4) `xFormers` (approximate, optimized for very long sequences), (5) Triton custom kernels.
- **Chosen:** Manual multi-head attention (original) for Phase 4. FlashAttention (`torch.nn.functional.scaled_dot_product_attention`) = OPTIONAL (Phase 9, zero interface changes). Triton/xFormers = RESEARCH-ONLY.
- **Why:** Manual implementation gives full control for future modifications (GQA, sliding window, sparse attention). FlashAttention is a performance optimization, not an architecture change — it can replace manual attention without changing `MultiHeadAttention` interface.
- **Reversible:** Later — FlashAttention can be adopted as optional enhancement; interface remains stable.
- **Classification:** Manual = CORE. FlashAttention = OPTIONAL. Triton/xFormers = RESEARCH-ONLY.

---

## 2026-07-24 — v0.1.0 — Inference Engine Strategy
- **Problem:** Inference must work for small-scale (Phase 6) and scale to production serving (Phase 10) without rewrites.
- **Options:** (1) Custom inference loop only, (2) `vLLM` only, (3) `TensorRT-LLM` only, (4) Custom loop + vLLM compatibility layer.
- **Chosen:** Custom inference loop (streaming, KV cache, sampling) for Phase 6. `vLLM` compatibility = OPTIONAL (Phase 9+). `TensorRT-LLM` = RESEARCH-ONLY.
- **Why:** Custom loop ensures full understanding and allows custom extensions (speculative decoding, reasoning chain generation). vLLM is the standard production serving engine but requires GPU infrastructure. Custom loop serves educational and small-scale needs; vLLM integration adds production scale without architecture rewrites.
- **Reversible:** Later — custom engine interface (`InferenceEngine`) can support vLLM backend without changing user-facing API.
- **Classification:** Custom = CORE. vLLM = OPTIONAL. TensorRT-LLM = RESEARCH-ONLY.

---

## 2026-07-24 — v0.3.0 — Dataset Pipeline Design (Phase 3)
- **Problem:** How to load text datasets (Tiny Shakespeare, WikiText, OpenWebText) into a format compatible with XRFM training pipeline while maintaining reproducibility and scalability.
- **Options:** (1) Raw Python file reading only, (2) Hugging Face `datasets` library (optional integration), (3) WebDataset / streaming format (future scale), (4) Original dataset loader with stable interface (`XRFMTextDataset`).
- **Chosen:** Original dataset loader (`xrfm/data/loader.py`) with `TokenizerInterface` integration, dataset manifest generation, and config-driven settings. Hugging Face `datasets` evaluated but deferred as optional enhancement (post-v0.5.0). Streaming mode (`streaming: true`) reserved for Phase 8+.
- **Why:** Full ownership of dataset pipeline ensures reproducibility (manifests), clean architecture (layered: verify -> normalize -> tokenize -> chunk), and future scalability (multilingual/code/instruction datasets can use same loader interface without rewrites). External library (`datasets`) would add dependency overhead for Phase 3 without significant benefit for small-to-medium datasets.
- **Reversible:** Later — Hugging Face `datasets` can be adopted as optional enhancement; loader interface remains stable. Streaming mode activation requires no loader rewrites.
- **Classification:** Original loader = CORE. HF `datasets` integration = OPTIONAL (post-v0.5.0). Streaming activation = OPTIONAL (Phase 8+). WebDataset format = RESEARCH-ONLY.
