# XR Foundation Model (XRFM)

**Version:** 1.0.1 (single source: `xrfm.__version__`)
**License:** MIT
**Status:** Research / experimental. A from-scratch decoder-only language
model in pure PyTorch. After the 2026-09 forensic audit (`docs/audit/`) the
codebase is *trainable, reproducible, and measurable at small scale* — it is
**not** a production foundation model and makes no capability claims beyond
what its training runs demonstrate (see `docs/audit/FINAL_AUDIT.md` and
`XRFM_FINAL_REPORT.md`).

Phase 1 (2026-09-18) completed the **data foundation**: a serious, reproducible, scalable, auditable pipeline for future 125M–1B+ pretraining.

## What XRFM Is Today

XRFM is now **one coherent Python package** (`src/xrfm/`) rather than a
collection of sibling script folders: an original decoder-only transformer
(RoPE, RMSNorm, SwiGLU, pre-norm residuals, weight tying), a byte-level BPE
tokenizer with an explicit special-token contract, a **document-centric data
pipeline** with deterministic splitting, deduplication, filtering, mixing,
sharding and manifests, a validated typed configuration system, a `Trainer`
with checkpoints/resume/validation, KV-cached inference, and perplexity/accuracy
evaluation — all reachable through one public API.

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

# Phase 1 new: full data pipeline
from xrfm.data import Document, DataPipeline, PipelineConfig

docs = [Document(text="Hello world", source="web", language="en")]
pipeline = DataPipeline(config=PipelineConfig(output_dir="processed/test", sequence_length=512), tokenizer=tok)
result = pipeline.run(docs)  # → shards, manifest, report
```

## XR Intelligence Runtime (new)

XRFM now also contains an additive, model-agnostic agent runtime under
`xrfm.agent`. This is **not** a claim that the small XRFM neural checkpoint
was trained for tool use. The runtime separates the neural model from
structured decisions, permissions, execution, observations, and verification.

```python
from pathlib import Path
from xrfm.agent import AgentRuntime, PermissionPolicy, SafeFilesystemTools
from xrfm.agent import ScriptedBackend, ToolExecutor, ToolRegistry

registry = ToolRegistry()
SafeFilesystemTools(Path.cwd()).register(registry)  # read-only, root-confined
executor = ToolExecutor(registry, PermissionPolicy(allowed_permissions={"filesystem.read"}))
state = AgentRuntime(ScriptedBackend(), executor).run(
    "Find the largest file in this project and summarize what it is."
)
print(state.final_answer)
```

The demo executes a real filesystem tool; it does not use fake outputs. The optional API exposes `/v1/agent/run` and `/v1/agent/stream` (SSE). Set `XRFM_AGENT_API_KEY` to protect the endpoints and `XRFM_AGENT_ROOT` to scope filesystem access. For a capable local model, point `OpenAICompatibleBackend` at llama.cpp, Ollama, or vLLM using `XRFM_AGENT_BASE_URL` and `XRFM_AGENT_MODEL`. External weights are never bundled. The existing Transformer remains a
first-class native research backend candidate, but it is currently a small
prose LM without native tool-use training. See:
`docs/XRFM_AGENTIC_ARCHITECTURE.md`, `docs/XRFM_AGENTIC_IMPLEMENTATION_PLAN.md`,
`docs/research/XRFM_CURRENT_STATE_AUDIT.md`, and
`docs/XRFM_AGENTIC_FINAL_REPORT.md`.

Run the deterministic end-to-end demo:

```bash
python scripts/run_agent_demo.py
```

## Architecture

```text
xrfm
 ├── config/        typed + validated configs (schema.py, loader.py)  — no CWD dependence
 ├── tokenization/  Tokenizer contract + byte-level BPE (packaged vocab.json)
 ├── data/          Phase 1: document schema, normalization, language, quality, PII,
 │                  exact+near dedup, document-level split, contamination,
 │                  mixing, tokenization, packing, sharding, manifests, reporting, pipeline
 │                  Phase 0 compat: splits · packing · dataset · manifests still present
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
data pipeline: **`docs/data-pipeline.md`**, ADR: **`docs/adr/0002-xrfm-data-pipeline.md`**,
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
# Phase 1: build data pipeline on golden dataset
xrfm data build --input tests/data/golden --output-dir processed/golden-test --sequence-length 512
xrfm data stats --manifest processed/golden-test/manifests/golden-test-v0.1.0-manifest.json
xrfm data inspect --input tests/data/golden --num 10

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

## Data Pipeline (Phase 1)

**Current dataset:** `xrfm-pretrain-v0.1` development corpus (golden dataset, 33 docs, MIT-licensed, tiny but complete pipeline)

**Current token count:** ~1K tokens on golden dataset (pipeline verified), scalable to billions with same architecture

**Current sources:** Official = golden, tiny_shakespeare, python_stdlib_slice (license-clean). Candidate large sources documented in `configs/data/sources.yaml` (wikipedia, fineweb-edu, dolma, etc.) but `enabled: false` until final corpus decision.

**Current limitations:**
- Heuristic language detector (not SOTA fastText/GlotLID, but pluggable)
- PII filtering is heuristic, not guaranteed PII-free
- Near dedup O(n²) for Phase 1, not yet LSH-optimized for trillion tokens
- No cluster-scale parallelism yet (single machine, multi-process ready)
- No streaming HF download yet (file-based ingestion)

**How to build:**
```bash
xrfm data build --config configs/data/pretrain.yaml --input tests/data/golden --output-dir processed/xrfm-pretrain-v0.1
```

**How to inspect:**
```bash
xrfm data stats --manifest processed/xrfm-pretrain-v0.1/manifests/xrfm-pretrain-v0.1.0-manifest.json
xrfm data inspect --input processed/xrfm-pretrain-v0.1/documents/raw.jsonl --num 10 --mode random
```

**How to reproduce:**
```bash
# Same inputs + same config + same tokenizer + same pipeline version = same dataset_id
# Manifest contains dataset_id, checksums, git_commit, processing_config_hash
cat processed/xrfm-pretrain-v0.1/manifests/*.json | grep dataset_id
```

**What is experimental:**
- Large source candidates in `sources.yaml` with `enabled: false`
- FastText language detector (optional)
- SimHash near-dedup (interface ready)

**What will be added later:**
- HF datasets streaming ingestion
- LSH-optimized near dedup for trillion tokens
- GlotLID language ID
- Cluster-scale parallelism benchmarks
- Population of contamination registry with eval sets

Full details: `docs/data-pipeline.md`

## Status of Subsystems (implemented / experimental / planned)

**Implemented (tested, supported):** transformer forward/attention stack
(causality ground-truth tested), byte-level BPE (lossless Unicode
round-trip), **Phase 1 data pipeline** (document-centric, normalization,
language/quality/PII filtering, exact+near dedup, document-level deterministic
splitting, contamination boundary, mixing, tokenization with provenance,
packing, deterministic sharding with worker-aware partitioning, manifests with
dataset_id + checksums, reporting, inspection CLI, training integration verified
with one real training step), typed config validation, single-device
training loop with checkpoint/resume + DDP/FSDP *code paths* (see
limitations), KV-cached generation with temperature/top-k/top-p/repetition
penalty, perplexity + top-1/top-5 evaluation, packaging (wheel ships only
`xrfm`, resources included), FastAPI server (health/completions/tokenize).

**Experimental (present, not guaranteed stable):** `xrfm/research/neurotopo`
(research architecture with its own training/eval), `xrfm/search` RAG agent,
INT8/INT4 quantization, speculative decoding, `KVCache` preallocated buffer,
multi-GPU/DDP execution, `benchmark/` microbenchmarks, large source candidates.

**Planned (not implemented):** large-scale pretraining on billion-token
corpus (pipeline ready, corpus selection pending), cluster-scale benchmarks.

## Honest Limitations (as of 2026-09-18)

- Data pipeline Phase 1 completed on golden dataset (~1K tokens) and verified with training step; large-scale billion-token corpus not yet built (architecture ready, candidate sources documented)
- Trained at most on ~2 M tokens of public-domain prose+code on CPU-only
  sandboxes previously; Phase 1 adds pipeline but not yet large model training
- The legacy `checkpoints/checkpoint_step_500.pt` (vocab 50304) is **not**
  compatible with the current tokenizer and is kept only as evidence.
- `training/distributed.py` (DDP/FSDP) is validated only in single-process
  mode; multi-GPU training has not been exercised (sharding logic tested for 1,2,4,8 ranks)
- The pre-Phase-0 import paths (`model.*`, `tokenizer.*`, ...) still work
  inside a repo checkout via deprecation shims; they are **not** installed
  with the wheel and will be removed no earlier than v2.0.

## Documentation

- Architecture: `docs/architecture.md` · ADR-0001:
  `docs/adr/0001-xrfm-core-architecture.md` · ADR-0002:
  `docs/adr/0002-xrfm-data-pipeline.md` · Data pipeline:
  `docs/data-pipeline.md` · Phase 0 audit:
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
- Dolma (Ai2), FineWeb, OLMo — data pipeline best practices (see ADR-0002)
- Conceptual references only; implementation is original.
