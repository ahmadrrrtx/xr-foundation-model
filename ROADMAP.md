# XR Foundation Model Roadmap

## Version Strategy: Semantic Versioning

- v0.1.0: Phase 1 — Project Foundation (complete)
- v0.2.0: Phase 2 — Tokenizer
- v0.3.0: Phase 3 — Dataset Pipeline
- v0.4.0: Phase 4 — Transformer Architecture (COMPLETE — `model/gpt.py` + component modules; all `tests/test_*.py` passing; `docs/model/ARCHITECTURE.md` complete; `benchmark/model_forward.py` reserved)
- v0.5.0: Phase 5 — Training Engine (COMPLETE — training/loop.py + optimizer + scheduler + checkpoint + mixed_precision; tests/test_training.py: 15 passing; docs/training/TRAINING_GUIDE.md complete; benchmark/training_forward.py reserved; DECISIONS.md Phase 5 entries added; self-review checklist complete)
- v0.6.0: Phase 6 — Inference Engine
- v0.7.0: Phase 7 — Evaluation Pipeline
- v0.8.0: Phase 8 — Scaling & Distributed Training
- v0.9.0: Phase 9 — Optimization (FlashAttention, quantization, speculative decoding)
- v1.0.0: Phase 10 — Production Deployment & API

## Future Models

- XRFM-10M (v0.2.0)
- XRFM-50M (v0.5.0)
- XRFM-100M (v0.7.0)
- XRFM-300M (v0.9.0)
- XRFM-1B (v1.5.0 — requires institutional compute)
- XRFM-7B (v2.0.0 — requires significant funding)
- XRFM-MoE (v3.0.0 — mixture of experts architecture)
- XRFM-Multimodal (v4.0.0 — vision and audio integration)

## Architecture Principles

Every version maintains backward compatibility with config presets. Changing `model.d_model` should not require code rewrites.
