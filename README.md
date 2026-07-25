# XR Foundation Model (XRFM)

**Version:** v1.0.0
**License:** MIT
**Status:** Active development — Scaling Training + Inference + Evaluation complete (Phases 1–7) Distributed complete (Phases 1–8)

---

## What This Is

XRFM is an open-source foundation model built from scratch in pure PyTorch. It is not a tutorial, not a wrapper around an API, and not a renamed fork. Every module is original and designed to scale from 10M parameters to billion-parameter models without rewrites.

## Architecture

- **Config-driven:** All hyperparameters flow through `ConfigLoader`. Changing model scale requires only YAML updates.
- **Modern:** Decoder-only transformer with RoPE, RMSNorm, SwiGLU, pre-norm residuals, weight tying.
- **Manual attention:** Full control for future GQA, FlashAttention, sliding window extensions.
- **Original:** No code copied from nanoGPT, litGPT, OLMo, or any tutorial.

## Module Status

| Module | Path | Status |
|---|---|---|
| Config System | `xrfm/config/loader.py` | v0.1.0 |
| BPE Tokenizer | `tokenizer/bpe.py` | v0.2.0 |
| Dataset Pipeline | `xrfm/data/loader.py` | v0.3.0 |
| Transformer Model | `model/gpt.py` | v0.4.0 |
| Training Engine | `training/loop.py` | v0.5.1 |
| Inference Engine | `inference/` | Phase 6 (next) |
| Evaluation | `evaluation/` | Phase 7 |
| Distributed Training | `training/distributed.py` | Phase 8 |

## Quick Start

```bash
git clone https://github.com/ahmadrrrtx/xr-foundation-model.git
cd xr-foundation-model
pip install -e .
python -c "from xrfm import ConfigLoader; c = ConfigLoader(); print(c.get('project.name'))"
```

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## References

- Vaswani et al. (2017) — Attention mechanism
- Su et al. (2023) — RoPE (RoFormer)
- Shazeer (2020) — SwiGLU
- Zhang & Sennrich (2019) — RMSNorm
- Loshchilov & Hutter (2019) — AdamW
- Meta AI (2024) — Llama 3 architecture concepts
- DeepSeek-AI (2024) — DeepSeek-V3 architecture concepts

All source code is original. Concepts only — no line-for-line copying.

## Roadmap

See `ROADMAP.md`. Next milestone: **Phase 6 — Inference Engine**.

## Contributing

See `CONTRIBUTING.md`. All contributions must include type hints, docstrings, and tests.
