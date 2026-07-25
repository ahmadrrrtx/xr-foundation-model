# XR Foundation Model (XRFM)

**Version:** v0.1.0  
**License:** MIT  
**Status:** Active development — Foundation architecture complete (Phase 1)

---

## What This Is

XRFM is an open-source foundation model platform built from scratch in pure PyTorch. It is not a tutorial, not a wrapper around an API, and not a renamed fork. Every module is original, production-quality, and designed to scale from 10M parameters to billion-parameter models without rewrites.

## Project Philosophy

We think like a professional AI research lab, not a tutorial project. Every design decision supports future scaling: MoE architectures, multimodal extensions, multi-node training, and production deployment.

## Quick Start (Phase 1 — Complete)

```bash
git clone https://github.com/your-org/xr-foundation-model.git
cd xr-foundation-model
pip install -r requirements.txt
python -c "from xrfm.config.loader import ConfigLoader; c = ConfigLoader(); print('XRFM config loaded:', c.get('project.name'))"
```

## Architecture

See `ARCHITECTURE_REVIEW.md` for detailed design decisions and `docs/` for module documentation.

Key features:
- Config-driven model architecture (`xrfm/config/loader.py`)
- Original PyTorch implementation (no copied source from existing tutorials)
- Professional open-source standards (`LICENSE`, `CONTRIBUTING`, `ROADMAP`, `CHANGELOG`, `CODE_OF_CONDUCT`, `SECURITY`)
- Semantic versioning with model names: `XRFM-10M`, `XRFM-50M`, `XRFM-100M`, `XRFM-300M`, `XRFM-1B`, `XRFM-7B`, `XRFM-MoE` (future)

## References and Attribution

This project draws on core concepts from the research literature:
- Vaswani et al. (2017) — Attention mechanism
- Karpathy (2023) — `nanoGPT` (conceptual reference for training loops; no code copied)
- Raschka (2024) — `LLMs-from-Scratch` (conceptual reference for tokenizer and architecture; no code copied)
- Meta AI (2024) — Llama 3 architecture concepts (RoPE, RMSNorm, SwiGLU, GQA — implemented originally)
- DeepSeek-AI (2024) — Sparse attention and MoE architecture concepts (implemented originally)

All source code in this repository is original.

## Roadmap

See `ROADMAP.md`. Phase 2 (Tokenizer) is planned but requires approval before implementation.

## Contributing

See `CONTRIBUTING.md`. All contributions must include type hints, docstrings, and tests.
