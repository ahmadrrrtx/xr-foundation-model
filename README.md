# XR Foundation Model (XRFM)

**Version:** v1.0.0  
**License:** MIT  
**Status:** ✅ Production Ready — All Phases Complete

---

## What This Is

XRFM is an open-source foundation model built from scratch in pure PyTorch. It is not a tutorial, not a wrapper around an API, and not a renamed fork. Every module is original and designed to scale from 10M parameters to billion-parameter models without rewrites.

## Architecture

- **Config-driven:** All hyperparameters flow through `ConfigLoader`. Changing model scale requires only YAML updates.
- **Modern:** Decoder-only transformer with RoPE, RMSNorm, SwiGLU, pre-norm residuals, weight tying.
- **Manual attention:** Full control for future GQA, FlashAttention, sliding window extensions.
- **Original:** No code copied from nanoGPT, litGPT, OLMo, or any tutorial.

## Module Status (v1.0.0)

| Module | Path | Status |
|---|---|---|
| Config System | `xrfm/config/loader.py` | ✅ v1.0.0 |
| BPE Tokenizer | `tokenizer/bpe.py` | ✅ v1.0.0 |
| Dataset Pipeline | `xrfm/data/loader.py` | ✅ v1.0.0 |
| Transformer Model | `model/gpt.py` | ✅ v1.0.0 |
| Training Engine | `training/loop.py` | ✅ v1.0.0 |
| Inference Engine | `inference/` | ✅ v1.0.0 |
| Evaluation | `evaluation/` | ✅ v1.0.0 |
| Distributed Training | `training/distributed.py` | ✅ v1.0.0 |
| API Server | `api/` | ✅ v1.0.0 |
| Optimization | `optimization/` | ✅ v1.0.0 |

## Quick Start

```bash
git clone https://github.com/ahmadrrrtx/xr-foundation-model.git
cd xr-foundation-model
pip install -e .
python -c "from xrfm.config.loader import ConfigLoader; c = ConfigLoader(); print(c.get('project.name'))"
```

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Running the API

```bash
pip install fastapi uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then visit: `http://localhost:8000/docs` for the API documentation.

## Training Validation

To validate the training pipeline works:

```bash
python scripts/validate_training.py
```

This trains a tiny model on a small dataset and confirms loss convergence.

## Features

- ✅ Config-driven architecture (change `config/config.yaml` to scale model)
- ✅ Modern transformer (RoPE, RMSNorm, SwiGLU)
- ✅ KV cache for efficient inference
- ✅ FlashAttention integration (2-4× speedup)
- ✅ Quantization (INT8/INT4)
- ✅ Distributed training (DDP/FSDP)
- ✅ FastAPI server with streaming
- ✅ Docker deployment (CPU + GPU)
- ✅ GitHub Actions CI/CD

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

See `ROADMAP.md`. Current version: **v1.0.0 — Production Release**

## Contributing

See `CONTRIBUTING.md`. All contributions must include type hints, docstrings, and tests.
