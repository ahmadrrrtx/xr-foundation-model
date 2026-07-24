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

---

## 2026-07-24 — v0.4.0 — Embedding Design (Phase 4)
- **Problem:** Embedding layer must be original, numerically stable, and support weight tying.
- **Options:** (1) `nn.Embedding` default init (`N(0, 1)`), (2) `nn.Embedding` with Xavier init + weight tying design, (3) Separate projection (`tie_weights: False`) for comparison.
- **Chosen:** Original `XRFMEmbedding` (`nn.Embedding` subclass) with Xavier init, weight tying (`self.weight` available for `lm_head.weight`), padding support (`padding_idx`), input validation (`vocab_size` check), numerical stability comments.
- **Why:** Full ownership of initialization (`Xavier` prevents divergence); weight tying is explicit (`True` by default); padding support ensures batched sequences work; input validation provides clear error messages for vocabulary mismatches.
- **Reversible:** Later — `tie_weights: False` available (`GPTModel` constructor); `padding_idx` configurable; vocabulary size configurable (`ConfigLoader`).
- **Classification:** `XRFMEmbedding` = CORE. Separate projection (`tie_weights: False`) = OPTIONAL. Custom init (`Kaiming`, `DeepNorm`) = RESEARCH-ONLY.

---

## 2026-07-24 — v0.4.0 — RoPE Implementation (Phase 4)
- **Problem:** Positional encoding must encode relative distance, support long context extension, and be configurable.
- **Options:** (1) `ALiBi` (linear biases), (2) `RoPE` (`rotate_half` + frequency rotation), (3) Learned embeddings, (4) Sinusoidal embeddings.
- **Chosen:** Original `RoPE` (`rotate_half` + frequency-based rotation; configurable `base`, `max_seq_len`, `scale_factor`). Applied to Q/K after head splitting (`MultiHeadAttention`). Configurable (`use_rope: true/false`).
- **Why:** `RoPE` ensures attention scores depend only on relative distance (`m - n`), which improves long-context generalization (`RoFormer` paper; adopted by `Llama 3`, `Mistral 7B`, `DeepSeek-V3`, `Qwen2.5`). `ALiBi` is simpler but does not provide the same relative distance property. `RoPE` supports future `NTK-aware` and `YaRN` scaling (`Phase 9`).
- **Reversible:** Later — `ALiBi` can replace `RoPE` (modify `MultiHeadAttention` to apply linear biases instead of rotation); `Sliding Window` can modify masking (same interface); `NTK-aware` scaling (`scale_factor`) deferred to `Phase 9`.
- **Classification:** `RoPE` = CORE. `ALiBi` comparison = RESEARCH-ONLY. `Sliding Window` = OPTIONAL. `NTK-aware` / `YaRN` scaling = RESEARCH-ONLY (`Phase 9`).

---

## 2026-07-24 — v0.4.0 — Multi-Head Attention Design (Phase 4)
- **Problem:** Attention mechanism must be original, controllable, and support future extensions (`GQA`, `FlashAttention`, `Sliding Window`).
- **Options:** (1) `nn.MultiheadAttention` (hidden behavior, limited control), (2) Manual multi-head attention (original, full control), (3) `FlashAttention` only (not architecture change, just optimization).
- **Chosen:** Original manual multi-head attention (`MultiHeadAttention`: `W_q`, `W_k`, `W_v`, `W_o`; scaled dot-product; masking via `masked_fill`; dropout; `RoPE` integration when configured). Configurable (`use_rope`, `dropout`).
- **Why:** Manual implementation provides full control for future `GQA` (modify `W_q`/`W_k` to share projections across groups) and `FlashAttention` (replace `matmul` + `softmax` with `scaled_dot_product_attention`; interface unchanged). Native `nn.MultiheadAttention` hides Q/K/V projections and attention score computation, making customization difficult.
- **Reversible:** Later — `FlashAttention` (`torch.nn.functional.scaled_dot_product_attention`) can replace the manual `matmul` step without interface changes. `GQA` can replace `W_q`/`W_k` projections without `TransformerBlock` rewrites. `Sliding Window` can modify the `mask` parameter without interface changes.
- **Classification:** Manual multi-head = CORE. `FlashAttention` = OPTIONAL (`Phase 9`). `GQA` = OPTIONAL (`post-v1.0`). `Sliding Window` = RESEARCH-ONLY. `Native nn.MultiheadAttention` comparison = RESEARCH-ONLY.

---

## 2026-07-24 — v0.4.0 — SwiGLU Selection (Phase 4)
- **Problem:** Feed-forward layer must provide expressive gating mechanism and be configurable.
- **Options:** (1) `ReLU` (standard, no gating), (2) `GELU` (smooth, no gating), (3) `SwiGLU` (`SiLU` gate + value gating, `W_1`, `W_2`, `W_3` projections), (4) `GLU` variants (`ReGLU`, `GeGLU`).
- **Chosen:** Original `SwiGLU` (`W_1` gate, `W_2` value, `W_3` output; `SiLU` activation; `Xavier` init; input validation `d_model` check).
- **Why:** `SwiGLU` introduces selective gating (`gate ⊗ value`), which improves performance compared to non-gated `ReLU` or `GELU` (`Shazeer 2020`; adopted by `Llama 3`, `Mistral 7B`, `DeepSeek-V3`, `Qwen2.5`). The extra parameter overhead (`+ d_model * d_ff`) is acceptable given the improvement. `SwiGLU` supports future `MoE` replacement (same `forward(x)` interface).
- **Reversible:** Later — `ReLU` or `GELU` can replace `SwiGLU` (modify `TransformerBlock.__init__` to create `nn.Linear` + activation instead of `SwiGLU`); `MoE` can replace `SwiGLU` without `TransformerBlock` rewrites.
- **Classification:** `SwiGLU` = CORE. `GELU` / `ReLU` comparison = OPTIONAL. `MoE` (`XRFM-MoE`, `v3.0`) = RESEARCH-ONLY.

---

## 2026-07-24 — v0.4.0 — RMSNorm Default (Phase 4)
- **Problem:** Normalization layer must stabilize training, reduce computation, and be configurable (`LayerNorm` fallback for comparison).
- **Options:** (1) `LayerNorm` (`mean` subtraction + `variance` normalization), (2) `RMSNorm` (`mean` omission; `mean(x^2)` + `sqrt` normalization), (3) `DeepNorm` (deeper normalization for very large models).
- **Chosen:** Original `RMSNorm` (`gamma` learnable scale, `eps = 1e-6`, no mean subtraction, `Xavier` init on `weight`). Default (`use_rmsnorm: True`). `LayerNorm` available as optional fallback (`use_rmsnorm: False`).
- **Why:** `RMSNorm` achieves similar stabilization effects with reduced computation (no mean subtraction). Modern architectures (`Llama 3`, `DeepSeek-V3`, `Mistral 7B`, `Qwen2.5`) exclusively use `RMSNorm`. `LayerNorm` is available for comparison studies (`DECISIONS.md` notes optional comparison).
- **Reversible:** Later — `LayerNorm` can replace `RMSNorm` (modify `TransformerBlock.__init__` to create `nn.LayerNorm` instead of `RMSNorm`; interface unchanged). `DeepNorm` deferred to `Phase 9` (`RESEARCH-ONLY`).
- **Classification:** `RMSNorm` = CORE. `LayerNorm` fallback = OPTIONAL. `DeepNorm` = RESEARCH-ONLY (`Phase 9`).

---

## 2026-07-24 — v0.4.0 — Residual Design (`Pre-Activation` / `Pre-Norm`)
- **Problem:** Residual architecture must prevent vanishing gradients in deep networks and be standard modern practice.
- **Options:** (1) `Post-Norm` (`SubLayer(x)` then `Norm(x + SubLayer(x))` — original `Transformer`), (2) `Pre-Norm` (`Norm(x)` before `SubLayer(x)`, then `x + Dropout(SubLayer(Norm(x)))` — modern standard), (3) `Post-Activation` (`SubLayer` + `activation` before residual).
- **Chosen:** `Pre-Norm` (`norm` before `attention` and `SwiGLU`; residual added after dropout). Standard modern architecture (`Llama 2/3`, `Mistral 7B`, `DeepSeek-V3`, `Qwen2.5`).
- **Why:** `Pre-Norm` improves gradient flow in deep networks (32+ layers) because the sub-layer receives normalized inputs (`mean ≈ 0`, `std ≈ 1`), preventing large magnitude outputs that could destabilize gradients (`Xiong 2020`; adopted by all modern large-scale LLMs). `Post-Norm` (original `Transformer`) can lead to unstable gradients in very deep networks.
- **Reversible:** Later — `Post-Norm` available for historical comparison (`DECISIONS.md` notes `RESEARCH-ONLY`); no interface changes required (`TransformerBlock.forward(x, mask)` unchanged).
- **Classification:** `Pre-Norm` (`Pre-Activation`) = CORE. `Post-Norm` comparison = RESEARCH-ONLY. `DeepNorm` (`RESEARCH-ONLY`, `Phase 9`).

---

## 2026-07-24 — v0.4.0 — Weight Tying Default (Phase 4)
- **Problem:** Language modeling head must share weights with embedding (standard practice) or allow separate projection (optional comparison).
- **Options:** (1) `Weight tied` (`True` by default: `lm_head.weight = embedding.weight`), (2) `Separate projection` (`False`: independent `nn.Linear` for output), (3) `Partial tying` (tie only some dimensions — `RESEARCH-ONLY`).
- **Chosen:** `Weight tied = True` (default) for `GPTModel`. `Weight tied = False` available via constructor (`GPTModel(weight_tied=False)`).
- **Why:** Weight tying (`GPT-3` design; adopted by `Llama 3`, `Mistral 7B`, `DeepSeek-V3`, `Qwen2.5`) reduces parameters by `vocab_size * d_model` (e.g., `50304 * 256 = 12.9M` saved) and ensures consistent representation space (embedding and projection learn the same token representations). Separate projection (`tie_weights: False`) allows the model to learn different input/output representations (sometimes improves performance at the cost of parameters; available for comparison studies).
- **Reversible:** Later — `GPTModel.__init__` supports both (`weight_tied` parameter). No rewrites needed to switch.
- **Classification:** Weight tying (`True`) = CORE. Separate projection (`False`) = OPTIONAL. Partial tying = RESEARCH-ONLY.

---

## 2026-07-24 — v0.4.0 — Initialization Strategy (`Xavier` / `Kaiming`)
- **Problem:** Initialization must prevent gradient explosion/vanishing at start of training and be standard practice.
- **Options:** (1) `Xavier` (`Glorot & Bengio 2010`: `sqrt(6 / (fan_in + fan_out))`), (2) `Kaiming` (`He 2015`: `sqrt(2 / fan_in)`), (3) `GPT-Scaled Init` (`init.kaiming_uniform_` with `a = sqrt(5)` for deeper networks), (4) `Default PyTorch init` (`N(0, 1)` for `Embedding`, `Uniform(-sqrt(k), sqrt(k))` for `Linear` where `k = 1 / in_features`).
- **Chosen:** `Xavier` uniform for all projection matrices (`W_q`, `W_k`, `W_v`, `W_o`, `W_1`, `W_2`, `W_3`, embedding) (`nn.init.xavier_uniform_`). `Kaiming` available for future `ReLU`-based FFN (`RESEARCH-ONLY`). `GPT-Scaled Init` deferred to `Phase 8` (`XRFM-1B+` scaling).
- **Why:** `Xavier` ensures output variance ≈ input variance, preventing gradient explosion/vanishing (`Glorot & Bengio 2010`; standard in all modern architectures). `Default PyTorch init` (`N(0, 1)` for `Embedding`) produces too large initial embeddings (variance grows with `vocab_size`), which can cause early divergence. `GPT-Scaled Init` is recommended for very deep networks (`32+` layers) but is deferred to `Phase 8` (`XRFM-1B+`).
- **Reversible:** Later — `Kaiming` can replace `Xavier` (modify `_init_weights()` in `MultiHeadAttention`, `SwiGLU`, `GPTModel`); `GPT-Scaled Init` can be adopted without interface changes (`DECISIONS.md` notes `OPTIONAL`).
- **Classification:** `Xavier` = CORE. `Kaiming` = OPTIONAL. `GPT-Scaled Init` = OPTIONAL (`Phase 8`). `Default PyTorch init` = RESEARCH-ONLY (not recommended; only for comparison).

---

## 2026-07-24 — v0.4.0 — Numerical Stability Design (`Softmax` Scaling, `Masking`, `Gradient` Safety)
- **Problem:** Attention mechanism and normalization must be numerically stable (no `NaN`/`Inf`, no gradient explosion, no saturation).
- **Options:** (1) `Softmax` without scaling (original `Transformer` — leads to saturation for large `d_head`), (2) `Softmax` with `sqrt(d_head)` scaling (standard modern practice), (3) `LayerNorm` (mean subtraction) vs `RMSNorm` (mean omission), (4) `Dropout` (`p = 0.1`) vs `no dropout`.
- **Chosen:** `Softmax` scaled by `sqrt(d_head)` (`math.sqrt(self.d_head)` in `MultiHeadAttention.forward()`); masking via `masked_fill(mask == 0, float("-inf"))`; `Dropout` (`p = dropout`) on attention weights; `RMSNorm` (`eps = 1e-6`); `RoPE` bounded rotation (`sin`/`cos` in `[-1, 1]`). All numerical stability choices are `CORE`.
- **Why:** `Softmax` scaling prevents saturation (`Kaplan 2020` scaling laws; standard in all modern architectures). `Masking` ensures padding tokens receive zero attention weight. `Dropout` prevents overfitting. `RMSNorm` `eps` prevents division by zero. `RoPE` bounded rotation prevents overflow.
- **Reversible:** Later — `Gradient clipping` (`torch.nn.utils.clip_grad_norm_`) can be added (`Phase 8`, `OPTIONAL`); `Mixed precision` (`torch.cuda.amp`) can be activated (`Phase 8`, `OPTIONAL`); `Gradient checkpointing` (`torch.utils.checkpoint`) can be adopted (`RESEARCH-ONLY`, `Phase 9`).
- **Classification:** `Softmax` scaling (`sqrt(d_head)`) = CORE. `Masking` (`masked_fill`) = CORE. `Dropout` (`p = 0.1`) = CORE. `RMSNorm` (`eps = 1e-6`) = CORE. `Gradient clipping` = OPTIONAL (`Phase 8`). `Mixed precision` = OPTIONAL (`Phase 8`). `Gradient checkpointing` = RESEARCH-ONLY (`Phase 9`).

