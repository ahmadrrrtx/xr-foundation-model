# Changelog

## [v0.4.0] — 2026-07-24

### Added
- Full decoder-only model (`GPTModel`): embedding + stacked `TransformerBlock` + final `RMSNorm` + `lm_head` with weight tying by default (`weight_tied=True`).
- `TransformerBlock`: pre-normalization (`norm` before sub-layer), residual connections (`x + Dropout(SubLayer(Norm(x)))`), `MultiHeadAttention` (manual, with optional `RoPE`), `SwiGLU` feed-forward.
- `XRFMEmbedding`: original embedding layer (`Xavier` init, weight tying design, padding support, input validation).
- `RoPE`: original rotary positional embedding (`rotate_half` + frequency rotation; configurable `base`, `max_seq_len`, `scale_factor` for future `NTK-aware` extension).
- `MultiHeadAttention`: original manual multi-head attention (`W_q`, `W_k`, `W_v`, `W_o`; scaled dot-product; masking; dropout; `RoPE` integration when configured).
- `SwiGLU`: original gated feed-forward (`W_1` gate, `W_2` value, `W_3` output; `SiLU` activation; `Xavier` init; input validation).
- `RMSNorm`: original `RMSNorm` (`gamma` learnable scale; `eps = 1e-6`; no mean subtraction; `Xavier` init).
- `tests/test_embedding.py`: `11` passing tests (shape, init, validation, config, gradient flow).
- `tests/test_attention.py`: `17` passing tests (shape, validation, masking, gradient, numerical stability, `RoPE` integration).
- `tests/test_transformer_block.py`: `13` passing tests (shape, integration, residual, pre-norm, config, input validation).
- `tests/test_model_architecture.py`: `16` passing tests (init, parameter count, weight tying, forward, gradient, numerical stability, input validation, config integration).
- `benchmark/model_forward.py`: basic benchmark framework (parameter count verification, forward pass timing, throughput measurement, future expansion reserved for `Phase 7`).
- `docs/model/ARCHITECTURE.md`: complete architecture documentation (`tensor` shapes, mathematical derivations, component diagrams, design decisions, future extension notes, performance considerations, numerical stability verification, security notes, self-review checklist).
- `DECISIONS.md`: `Phase 4` architecture entries added (`Embedding`, `RoPE`, `Multi-Head Attention`, `SwiGLU`, `RMSNorm`, `Residual`, `Weight Tying`, `Initialization`, `Numerical Stability`).

### Design Decisions
- Original manual multi-head attention (`CORE`) — full control for future `GQA`, `FlashAttention`, `Sliding Window`.
- `RoPE` (`CORE`) over `ALiBi` — relative distance property improves long-context performance; configurable (`use_rope`).
- `SwiGLU` (`CORE`) over `GELU`/`ReLU` — selective gating improves performance (`Shazeer 2020`); `MoE` replacement supported via stable interface.
- `RMSNorm` (`CORE`) over `LayerNorm` — reduced computation (`mean` omission); modern standard (`Llama 3`, `DeepSeek-V3`).
- `Pre-Norm` (`Pre-Activation`) (`CORE`) over `Post-Norm` — improves gradient flow in deep networks (`Xiong 2020`); standard modern architecture.
- Weight tying (`True` by default) (`CORE`) — reduces parameters (`vocab_size * d_model` saved); improves stability (`GPT-3` design); separate projection (`False`) available (`OPTIONAL`).
- `Xavier` initialization (`CORE`) — prevents gradient divergence (`Glorot & Bengio 2010`); `Kaiming` / `GPT-Scaled Init` deferred (`OPTIONAL`/`RESEARCH-ONLY`).
- Numerical stability (`CORE`): `softmax` scaling (`sqrt(d_head)`), masking (`masked_fill`), dropout (`p = 0.1`), `eps = 1e-6`, bounded `RoPE` rotation.

### Performance / Benchmark
- `XRFM-10M` preset (`vocab_size=50304`, `d_model=256`, `n_layers=6`, `n_heads=8`, `d_ff=1024`) produces approximately `19.2M` parameters (`embedding`: `12.9M`, `attention`: `1.6M`, `SwiGLU`: `4.7M`, `norm`: `3K`, `output`: `0` due to weight tying). Note: The preset label (`XRFM-10M`) refers to the target scale category; the actual parameter count reflects the full architecture dimensions configured in `config/config.yaml`.
- Basic benchmark (`benchmark/model_forward.py`) confirms no `NaN`/`Inf` in forward outputs; parameter count verified; timing framework established (`avg_time_ms`, `std_time_ms`, `throughput_seqs_per_sec`). Full evaluation pipeline (`multi-GPU`, `memory profiling`, `latency`) reserved for `Phase 7` (`v0.7.0`).

### Classification Updates (`DECISIONS.md`)
- `Xavier` init (`CORE`); `Kaiming` (`OPTIONAL`); `GPT-Scaled Init` (`OPTIONAL`, `Phase 8`); `Default PyTorch init` comparison (`RESEARCH-ONLY`).
- `Softmax` scaling (`CORE`); gradient clipping (`OPTIONAL`, `Phase 8`); mixed precision (`OPTIONAL`, `Phase 8`); gradient checkpointing (`RESEARCH-ONLY`, `Phase 9`).

---

## [v0.3.0] — 2026-07-24

### Added
- Dataset pipeline (`xrfm/data/loader.py`): `XRFMTextDataset` (file verification, normalization, split, chunking, tokenizer integration via `TokenizerInterface`, manifest generation `build_manifest`/`save_manifest`).
- `tests/test_data_loader.py`: `16` passing tests.
- `docs/data/DATASET_GUIDE.md`: dataset pipeline documentation.
- `DECISIONS.md`: `Phase 3` dataset strategy entry added (`TDR-002` extended).
- `CHANGELOG.md`: `v0.3.0` entry added.

### Design Decisions
- Original dataset loader (`CORE`); Hugging Face `datasets` (`OPTIONAL`, `post-v0.5.0`); streaming (`OPTIONAL`, `Phase 8`); `WebDataset` format (`RESEARCH-ONLY`).

---

## [v0.2.0] — 2026-07-24

### Added
- Original BPE tokenizer (`tokenizer/bpe.py`, `tokenizer/encode.py`, `tokenizer/decode.py`, `tokenizer/interface.py`, `tokenizer/DESIGN.md`).
- `tests/test_tokenizer_bpe.py`: `16` passing tests.
- `docs/tokenizer/README.md`: module documentation.
- `DECISIONS.md`: `Phase 2` tokenizer strategy entry added (`TDR-001`).
- `CHANGELOG.md`: `v0.2.0` entry added.

### Design Decisions
- Original BPE (`CORE`); `TokenizerInterface` stable (`CORE`); `SentencePiece` / `Unigram` / `TikToken` (`OPTIONAL`, `post-v0.5.0`).

---

## [v0.1.0] — 2026-07-24

### Added
- Project foundation (`xr-foundation-model`)
- Professional open-source repository structure
- ConfigLoader with YAML validation and dot-notation access
- Config-driven architecture supporting 10M → 1B+ without rewrites
- Git workflow with semantic version tags
- Professional documentation standards (LICENSE, CONTRIBUTING, ROADMAP)

### Design Decisions
- Pure PyTorch (no TensorFlow, no external LLM libraries for core model)
- Config-driven hyperparameters (no hard-coded values)
- Original implementation with documented conceptual references
- Clean architecture: no circular dependencies, dependency injection preferred
