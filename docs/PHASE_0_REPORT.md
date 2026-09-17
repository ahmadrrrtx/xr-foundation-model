# Phase 0 Final Report — XRFM Architecture Freeze

**Date:** 2026-09-18 · **Branch:** `phase-0-architecture` · **Base:** `421fac0`
Companion docs: `docs/architecture/PHASE_0_AUDIT.md` (pre-refactor evidence),
`docs/research/PHASE_0_BEST_PRACTICES.md` (design inputs),
`docs/architecture.md` (current architecture),
`docs/adr/0001-xrfm-core-architecture.md` (decisions).

## A. Initial findings (summary)

Two entangled worlds: seven generic top-level packages (`model/`,
`tokenizer/`, `training/`, `inference/`, `evaluation/`, `optimization/`,
`api/`) plus a partial `xrfm/` package (shared infra + the entire
experimental NeuroTopo stack + RAG search). Dependencies crossed in both
directions. Verified defects: CWD-dependent `ConfigLoader` (README
quick-start failed from `/tmp`), unvalidated config dataclasses, decorative
`DatasetConfig`, ignored `seed`, infinite-loop chunking at
`overlap >= max_seq_len`, hard-coded `pad_id=0` masking content tokens,
version drift (1.0.0 in API vs 1.0.1 in package), `pip install` shipping 8
top-level packages, `vocab.json` missing from the wheel, borderline-flaky
NeuroTopo overfit test. Baseline: 322/323 tests passed.

## B. Research findings

PyPA src-layout guidance; OLMo-core (typed dataclass config composition,
`src/olmo_core/`); HF Transformers (single deliberate public surface);
nanoGPT (thin scripts, boring modules); PyTorch packaging guidance;
dataclass-vs-pydantic analysis (stdlib dataclasses + explicit validation
chosen to avoid a new dependency). Full table:
`docs/research/PHASE_0_BEST_PRACTICES.md`.

## C. Final architecture

```text
src/xrfm/                     # THE package (wheel ships only this)
├── __init__.py               # public API (XRFM, XRFMConfig, Tokenizer, Trainer, generate, ...)
├── py.typed · cli.py         # PEP 561 marker · xrfm info / validate-config
├── config/                   # schema.py (validated dataclasses) + loader.py + packaged default YAML
├── tokenization/             # interface (special-token contract) + bpe.py + vocab.json
├── data/                     # splits.py · packing.py · dataset.py · manifest.py (+ loader.py compat)
├── models/                   # gpt.py (XRFMModel=GPTModel) + embedding/attention/layers
├── training/                 # trainer.py (Trainer) + loop/optimizer/scheduler/checkpoint/distributed/metrics
├── inference/                # generate.py + engine.py + sampling.py + kv_cache.py
├── evaluation/               # runner.py (evaluate) + perplexity.py + benchmarks.py
├── optimization/ search/ experiment/          # secondary consumers
└── research/neurotopo/       # experimental research model, isolated
api/ scripts/ benchmark/ config/ data/ checkpoints/ deployment/ webui/ vercel_app/ notebooks/
model/ tokenizer/ training/ inference/ evaluation/ optimization/   # deprecated shims (repo-only)
```

## D. Public API

```python
from xrfm import (
    XRFM, XRFMModel, GPTModel,            # one model class, three names
    XRFMConfig, ModelConfig, TrainingConfig, DatasetConfig,   # validated typed configs
    load_config, default_config, ConfigLoader, ConfigError,
    Tokenizer, TokenizerInterface, BPETokenizer, BytePairEncoder,
    encode_text, decode_text,
    Dataset, TextDataset, XRFMTextDataset,
    Trainer, TrainingLoop, TrainResult,
    generate, GenerationEngine, GenerationResult,
    evaluate, compute_perplexity,
)
```

## E. Changes implemented

1. `git mv` of all library code into `src/xrfm/` (history preserved);
   NeuroTopo flattened into `xrfm/research/neurotopo/`.
2. New `xrfm/__init__.py` public API; per-subsystem `__init__` facades.
3. New validated config schema + CWD-independent loader + packaged default
   preset; YAML presets cleaned (version/streaming/`default` key).
4. Data subsystem split into splits/packing/dataset/manifest with the
   Phase 0 contracts (seed consumed, position-based masking, strict
   packing validation, tokenizer-contract pad ids).
5. Tokenizer contract properties + `BPETokenizer.pretrained()`; vocab.json
   shipped as package data.
6. `XRFMModel` typed-config constructor (`pad_token_id` from config);
   `Trainer` facade; `generate()`; `evaluate()`; minimal `xrfm` CLI.
7. Root deprecation shims (top-level packages) + in-package research shims
   with DeprecationWarnings.
8. api/, scripts/, benchmark/, deployment updated to canonical imports; no
   sys.path hacks; Dockerfiles install the package.
9. pyproject: src-layout discovery (only `xrfm*`), dynamic version,
   package-data, `xrfm` console script, `[api]`/`[dev]` extras.
10. CI workflows updated (mypy/lint/test paths); requirements.txt aligned.
11. Docs: architecture.md, ADR-0001, Phase 0 audit, best-practices
    research, README, CHANGELOG, user guides re-pointed to `xrfm.*`.
12. Tests: 5 new suites + updated existing tests; flaky overfit threshold
    stabilized; loss-masking test rewritten to the new contract.

## F. Problems fixed

Config validation (D2) · package boundaries/site-packages pollution ·
dead config (D3, D13) · seed behavior (D4) · chunking infinite loop (D5) ·
tokenizer/data pad contract (D6, D7) · version drift (D8) · triple
config-default sources (D9) · path-coupled model/trainer (D10) · API
sys.path hack (D11) · flaky test (D12) · CWD dependence (D1) ·
vocab.json packaging.

## G. Tests (executed, this environment)

- `pytest tests/` → **456 passed, 0 failed** (~3 min; includes the
  config→tokenizer→dataset→model→train-step→inference integration test).
- `ruff check .` → clean · `ruff format --check .` → clean ·
  `black --check .` → clean.
- `mypy src/xrfm/<core subpackages> src/xrfm/cli.py api/` →
  "Success: no issues found in 60 source files".
- `python -m build` → sdist + wheel built; wheel contains **only `xrfm`**.
- Clean-venv install (`pip install --no-deps dist/*.whl`) then from `/`:
  `import xrfm`, `load_config()`, `BPETokenizer.pretrained()`,
  `generate(...)` — all OK; legacy packages correctly absent.
- `python scripts/ci_smoke_test.py` → PASS (60 steps, loss 6.24 → 2.21,
  checkpoint save + resume verified).

## H. Remaining / intentionally deferred

- `xrfm/research/` (NeuroTopo) is functional but not held to core
  lint/type gates; `xrfm/optimization`/`search` are secondary consumers.
- DDP/FSDP still validated only single-process (pre-existing).
- 75 MB legacy checkpoint retained as audit evidence.
- Shims must be maintained until ≥ v2.0 (removal checklist in ADR).
- Docstring-level docs in research bundle still reference old internals.

## I. Compatibility / breaking changes

- **Kept working (shimmed, warned):** `model.*`, `tokenizer.*`,
  `training.*`, `inference.*`, `evaluation.*`, `optimization.*`,
  `xrfm.core`, `xrfm.neurotopo`, `xrfm.topology`, `xrfm.memory`,
  `xrfm.dynamics`, `xrfm.neurons`, `xrfm.uncertainty`,
  `xrfm.nt_training`, `xrfm.nt_evaluation` (shims not shipped in the
  wheel for top-level names).
- **Behavior changes (documented):** last real token of each chunk is now
  masked (-100) instead of predicting the pad token; embedding
  `padding_idx` comes from `ModelConfig.pad_token_id` (default None)
  instead of hard-coded 0; `shuffle: true` in configs is now honored
  (seeded) — previously ignored; `ConfigLoader.dataset_config()` returns a
  typed `DatasetConfig` instead of a dict.
- **New canonical names:** `XRFMModel`/`XRFM` (=`GPTModel`),
  `BPETokenizer` (=`BytePairEncoder`), `TextDataset` (=`XRFMTextDataset`),
  `Tokenizer` (=`TokenizerInterface`).
- Version stays **1.0.1** (reorganization is not a release).

## J. Phase 0 verdict

1. **Is `xrfm` the clear public package?** Yes — single src-layout package,
   deliberate `__all__`, wheel contains only `xrfm`, resources included.
2. **Is the dependency direction clean?** Yes — one-way, test-guarded
   (no fastapi import in core; secondary layers are consumers only).
3. **Is configuration trustworthy?** Yes — typed, validated (incl.
   cross-field), CWD-independent, single version source, no dead fields.
4. **Is the data API coherent?** Yes — splits/packing/dataset/manifest
   separated with explicit, tested contracts.
5. **Can Phase 1 build on this without another rewrite?** Yes — data
   pipeline, tokenizer work, large-scale training, distributed training,
   and evaluation each have a stable boundary to extend.
6. **What remains before Phase 1?** Decide the pretraining data strategy
   and dataset-scale tooling design; exercise DDP on real hardware;
   graduate or remove research code; plan shim removal for v2.0.

## K. Follow-up hardening (same session, after the main report)

- CI: new `package` job — builds sdist+wheel, asserts the wheel's only
  top-level package is `xrfm`, installs it into a clean venv, and imports
  `xrfm` + runs `xrfm info` from `/`. Simulated locally end-to-end (pass).
- CLI verified both ways: `xrfm validate-config config/tiny.yaml` → VALID;
  a bad config (`d_model=100, n_heads=6`) →
  `INVALID: model.d_model: (100) must be divisible by model.n_heads (6)`,
  exit code 1.
- Doc sweep completed: notebook artifact path fixed
  (`tokenizer/vocab.json` → `src/xrfm/tokenization/vocab.json`);
  historical-path notes added to `DECISIONS.md` and
  `docs/model/ARCHITECTURE.md`; HF Space app compile-checked.
- Lint policy finalized for the exact CI commands (`ruff check .` /
  `ruff format --check .` repo-wide): deprecated root shims and the
  research bundle excluded from core format gates; `**/*.md` excluded so
  documentation snippets keep their human-formatted style (ruff ≥ 0.16
  would reformat fenced Python blocks). All repo-wide gates green.
