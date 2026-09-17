# XR Foundation Model (XRFM)

**Version:** 1.0.1 (single source: `xrfm.__version__`)
**License:** MIT
**Status:** Research / experimental. A from-scratch decoder-only language
model in pure PyTorch. After the 2026-08 forensic audit (`docs/audit/`) the
codebase is *trainable, reproducible, and measurable at small scale* — it is
**not** a production foundation model and makes no capability claims beyond
what its training runs demonstrate (see `docs/audit/FINAL_AUDIT.md` and
`XRFM_FINAL_REPORT.md`).

## What XRFM Is Today

XRFM is now **one coherent Python package** (`src/xrfm/`) rather than a
collection of sibling script folders: an original decoder-only transformer
(RoPE, RMSNorm, SwiGLU, pre-norm residuals, weight tying), a byte-level BPE
tokenizer with an explicit special-token contract, a line-boundary data
pipeline with deterministic seeded splits, a validated typed configuration
system, a `Trainer` with checkpoints/resume/validation, KV-cached inference,
and perplexity/accuracy evaluation — all reachable through one public API.

```python
import xrfm

cfg     = xrfm.load_config("config/tiny.yaml")   # packaged default: xrfm.load_config()
model   = xrfm.XRFM(cfg)                         # the transformer (alias of XRFMModel)
tok     = xrfm.BPETokenizer.pretrained()         # packaged tokenizer resource
train   = xrfm.TextDataset(cfg.data, tok, split="train")
trainer = xrfm.Trainer(model, config=cfg)
result  = trainer.train(train)

out  = xrfm.generate(model, "Once upon a time", tokenizer=tok,
                     max_new_tokens=64, temperature=0.8, top_k=50)
ppl  = xrfm.evaluate(model, val_loader)
```

## Architecture

```text
xrfm
 ├── config/        typed + validated configs (schema.py, loader.py)  — no CWD dependence
 ├── tokenization/  Tokenizer contract + byte-level BPE (packaged vocab.json)
 ├── data/          splits · packing · dataset · manifests (separate responsibilities)
 ├── models/        XRFMModel (GPTModel alias) + attention/layers
 ├── training/      Trainer facade over TrainingLoop (DDP/FSDP hooks, resume)
 ├── inference/     generate() + GenerationEngine (KV cache, top-k/top-p)
 ├── evaluation/    evaluate() → perplexity + accuracy benchmarks
 ├── optimization/  secondary: SDPA attention, INT8/INT4, speculative decoding
 ├── search/        secondary: local RAG search agent (consumes core)
 └── research/      experimental: XRFM-NeuroTopo research model (isolated)
```

Dependency direction is one-way (config → tokenization → data → models →
training → inference/evaluation; secondary layers consume the core; the API
server depends on XRFM, never the reverse). Details:
**`docs/architecture.md`**, rationale: **`docs/adr/0001-xrfm-core-architecture.md`**,
pre-refactor audit: `docs/architecture/PHASE_0_AUDIT.md`.

## Install

```bash
git clone https://github.com/ahmadrrrtx/xr-foundation-model.git
cd xr-foundation-model
pip install -e .            # library (torch, pyyaml, numpy)
pip install -e ".[dev]"     # + pytest/ruff/black/mypy
pip install -e ".[api]"     # + fastapi/uvicorn/httpx for the API server
```

The installed package works from any directory — `cd / && python -c
"import xrfm; print(xrfm.load_config().model.d_model)"` needs no repo
checkout.

## Run

```bash
# train (thin script over xrfm APIs: trains tokenizer → builds dataset → trains → checkpoints)
python scripts/train_custom_model.py --dataset_path data/datasets/corpus.txt --max_steps 2000

# CI-style end-to-end smoke train (config → tokenizer → dataset → train → resume)
python scripts/ci_smoke_test.py

# CLI shipped with the package
xrfm info
xrfm validate-config config/tiny.yaml

# API server (loads packaged tokenizer; picks up checkpoints/ when present)
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Status of Subsystems (implemented / experimental / planned)

**Implemented (tested, supported):** transformer forward/attention stack
(causality ground-truth tested), byte-level BPE (lossless Unicode
round-trip), data pipeline (seeded line-boundary splits, validated packing,
contract-based padding/masking), typed config validation, single-device
training loop with checkpoint/resume + DDP/FSDP *code paths* (see
limitations), KV-cached generation with temperature/top-k/top-p/repetition
penalty, perplexity + top-1/top-5 evaluation, packaging (wheel ships only
`xrfm`, resources included), FastAPI server (health/completions/tokenize).

**Experimental (present, not guaranteed stable):** `xrfm/research/neurotopo`
(research architecture with its own training/eval), `xrfm/search` RAG agent,
INT8/INT4 quantization, speculative decoding, `KVCache` preallocated buffer,
multi-GPU/DDP execution, `benchmark/` microbenchmarks.

**Planned (not implemented):** large-scale pretraining pipeline and
dataset-scale tooling (Phase 1 focus: data pipeline, tokenizer, distributed
training, reproducibility, evaluation at scale). Nothing in this list is
claimed to work merely because an interface exists.

## Honest Limitations (as of 2026-09-18)

- Trained at most on ~2 M tokens of public-domain prose+code on CPU-only
  sandboxes. This demonstrates the pipeline, not a foundation model.
- The legacy `checkpoints/checkpoint_step_500.pt` (vocab 50304) is **not**
  compatible with the current tokenizer and is kept only as evidence.
- `training/distributed.py` (DDP/FSDP) is validated only in single-process
  mode; multi-GPU training has not been exercised.
- The pre-Phase-0 import paths (`model.*`, `tokenizer.*`, ...) still work
  inside a repo checkout via deprecation shims; they are **not** installed
  with the wheel and will be removed no earlier than v2.0.

## Documentation

- Architecture: `docs/architecture.md` · ADR:
  `docs/adr/0001-xrfm-core-architecture.md` · Phase 0 audit:
  `docs/architecture/PHASE_0_AUDIT.md` · best-practices research:
  `docs/research/PHASE_0_BEST_PRACTICES.md`
- Training: `docs/training/TRAINING_GUIDE.md` · Data:
  `docs/data/DATASET_GUIDE.md` · Evaluation:
  `docs/evaluation/EVALUATION_GUIDE.md` · Inference:
  `docs/inference/INFERENCE_GUIDE.md` · Deployment:
  `docs/deployment/DEPLOYMENT_GUIDE.md`
- Audit trail: `docs/audit/` · Model spec: `docs/model/ARCHITECTURE.md`

## References

- Vaswani et al. (2017), Su et al. (2023) RoPE, Shazeer (2020) SwiGLU,
  Zhang & Sennrich (2019) RMSNorm, Loshchilov & Hutter (2019) AdamW.
- Conceptual references only; implementation is original.
