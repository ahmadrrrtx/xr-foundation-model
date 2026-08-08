# XR Foundation Model (XRFM)

**Version:** 1.0.1
**License:** MIT
**Status:** Research / experimental. A from-scratch decoder-only language
model in pure PyTorch. After the 2026-08 forensic audit (`docs/audit/`), the
codebase is *trainable, reproducible, and measurable at small scale* — it is
**not** a production foundation model and makes no capability claims beyond
what its training runs demonstrate (see `docs/audit/FINAL_AUDIT.md` and
`XRFM_FINAL_REPORT.md`).

## What This Is

XRFM is an original decoder-only transformer (RoPE, RMSNorm, SwiGLU,
pre-norm residuals, weight tying) with a byte-level BPE tokenizer, a
line-boundary data pipeline, a training loop with checkpoints/resume/validation,
KV-cached inference, perplexity evaluation, and an API server.

## Architecture

- **Config-driven:** all hyperparameters flow through `ConfigLoader`
  (`config/config.yaml`; per-size presets in `config/tiny.yaml`,
  `config/medium.yaml`).
- **Modern:** decoder-only transformer with RoPE, RMSNorm, SwiGLU, pre-norm
  residuals, weight tying, explicit causal masking.
- **Honest precision:** fp32 on CPU; real bf16 autocast only when a GPU is
  present (config `training.mixed_precision`).
- **Reproducible:** every checkpoint stores config, seed, and versions;
  training is seeded end-to-end (config `training.seed`).

## Quick Start

```bash
git clone https://github.com/ahmadrrrtx/xr-foundation-model.git
cd xr-foundation-model
pip install -e .
python -c "from xrfm.config.loader import ConfigLoader; print(ConfigLoader().get('project.name'))"
```

## Training your own model

```bash
python scripts/train_custom_model.py --dataset_path data/datasets/corpus.txt --max_steps 2000
```

This trains a byte-level BPE tokenizer on the train split, builds a model whose
vocabulary matches the tokenizer exactly, trains with periodic validation, and
saves checkpoints + a JSONL metrics log. See `docs/training/TRAINING_GUIDE.md`.

## Running Tests

```bash
pip install -e ".[dev]" httpx
python -m pytest tests/ -v
```

The suite includes ground-truth verification tests (causality, reference math,
tokenizer fidelity, reproducibility, resume) beyond the original shape tests.

## Running the API

```bash
pip install fastapi uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`/health`, `/v1/completions` (+ streaming), `/v1/tokenize`, `/v1/models`,
`/v1/search` are available. The API loads the latest checkpoint from
`checkpoints/` when present.

## Documentation

- Audit: `docs/audit/BASELINE.md`, `docs/audit/FORENSIC_AUDIT.md`,
  `docs/audit/GAP_ANALYSIS.md`, `docs/audit/FINAL_AUDIT.md`
- Research: `docs/research/MODEL_COMPARISON.md`
- Architecture: `docs/architecture/XRFM_TARGET_SPEC.md`
- Training & compute: `docs/training/COMPUTE_PLAN.md`, `docs/training/TRAINING_GUIDE.md`
- Implementation plan: `docs/implementation/REMEDIATION_PLAN.md`
- Final report: `XRFM_FINAL_REPORT.md`

## Honest Limitations (as of 2026-08-08)

- Trained at most on ~2 M tokens of public-domain prose+code on a CPU-only
  sandbox. This demonstrates the pipeline, not a foundation model.
- The legacy `checkpoints/checkpoint_step_500.pt` (vocab 50304) is **not**
  compatible with the current tokenizer (2048) and is kept only as evidence;
  see `docs/audit/BASELINE.md` §5 and §7.
- `training/distributed.py` (DDP/FSDP) is scaffolding validated only in
  single-process mode; multi-GPU training has not been exercised.

## References

- Vaswani et al. (2017), Su et al. (2023) RoPE, Shazeer (2020) SwiGLU,
  Zhang & Sennrich (2019) RMSNorm, Loshchilov & Hutter (2019) AdamW.
- Conceptual references only; implementation is original.
