# XR Foundation Model Roadmap

## Version Strategy: Semantic Versioning

- v0.1.0: Phase 1  — Project Foundation ✅ (COMPLETE)
- v0.2.0: Phase 2  — Tokenizer ✅ (COMPLETE)
- v0.3.0: Phase 3  — Dataset Pipeline ✅ (COMPLETE)
- v0.4.0: Phase 4  — Transformer Architecture ✅ (COMPLETE)
- v0.5.0: Phase 5  — Training Engine ✅ (COMPLETE)
- v0.5.1:          — Stabilization Release ✅
- v0.6.0: Phase 6  — Inference Engine ✅ (COMPLETE)
- v0.7.0: Phase 7  — Evaluation Pipeline ✅ (COMPLETE)
- v0.8.0: Phase 8  — Scaling & Distributed ✅ (COMPLETE)
- v0.8.0: Phase 8 — Scaling & Distributed Training
- v0.9.0: Phase 9  — Optimization ✅ (COMPLETE) — Optimization (FlashAttention, quantization, speculative decoding)
- v1.0.0: Phase 10  — Production Deployment ✅ (COMPLETE) — Production Deployment & API

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
