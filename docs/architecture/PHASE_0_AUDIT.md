# Phase 0 — Architecture Audit (pre-refactor)

**Date:** 2026-09-18 · **Scope:** full repository inspection *before* any code change.
This document is the evidence base for the Phase 0 reorganization
(`docs/adr/0001-xrfm-core-architecture.md`). It records the structure as it
was at commit `421fac0`.

---

## 1. Repository shape at audit time

Two parallel, entangled Python "worlds" existed at the repository root:

1. **The v1.x GPT stack** — live top-level packages:
   `model/`, `tokenizer/`, `training/`, `inference/`, `evaluation/`,
   `optimization/`, plus the `api/` FastAPI app.
2. **The `xrfm/` package** — containing *some* shared infrastructure
   (`config/`, `data/`, `experiment/`) and the *entire* experimental
   NeuroTopo research model (`core/`, `neurotopo/`, `topology/`, `memory/`,
   `dynamics/`, `neurons/`, `uncertainty/`, `routing/`, `knowledge/`,
   `nt_training/`, `nt_evaluation/`) and the RAG search subsystem
   (`search/`).

Supporting layers: `scripts/` (11 entry points), `benchmark/`,
`webui/` + `vercel_app/` (JS, HTTP-only), `deployment/`, `notebooks/`,
`config/` (5 YAML files), `data/` (training corpora), `checkpoints/`
(75 MB legacy checkpoint tracked in git), `tests/` (33 files, 323 tests).

## 2. Verified import dependency map

Edges below are *actual* `import`/`from` statements (not inferred from names).

```text
scripts/*            → model.gpt, tokenizer.bpe, training.{loop,optimizer,scheduler,
                       checkpoint,distributed,metrics}, inference.engine,
                       evaluation.perplexity, xrfm.data.loader, xrfm.config.loader,
                       xrfm.experiment.tracking
api/main.py          → inference.engine, model.gpt, tokenizer.bpe
api/routes/*         → api.main (globals), api.schemas, xrfm.search.indexer, xrfm.__version__
deployment/hf_space  → api.main, inference.engine, model.gpt, tokenizer.bpe
benchmark/*          → model.gpt, training.{loop,optimizer,scheduler,checkpoint}, inference.engine

model/gpt.py         → model.embedding, model.layers.*, xrfm.config.loader   ← CROSS-BOUNDARY
training/loop.py     → training.*, xrfm.config.loader                        ← CROSS-BOUNDARY
inference/engine.py  → inference.sampling, model.gpt
evaluation/*         → model.gpt
optimization/specdec → model.gpt
xrfm/data/loader.py  → tokenizer.interface                                   ← CROSS-BOUNDARY
xrfm/search/agent.py → inference.engine, tokenizer.bpe                       ← CROSS-BOUNDARY
xrfm/nt_training/param_match.py → model.gpt                                  ← CROSS-BOUNDARY
xrfm/core,model…     → xrfm-internal only (neurotopo stack is self-contained)
tests/*              → both worlds directly (33 files)
```

Consequences:

* The dependency direction between the two worlds is **bidirectional**
  (`model → xrfm.config` and `xrfm.search → inference`), i.e. there is no
  package boundary at all — `xrfm/` is not a package, it is a folder that
  shares the repo with seven sibling top-level packages.
* `pip install` of the project **installs 8 top-level packages into
  site-packages** (`model`, `training`, `tokenizer`, `inference`,
  `evaluation`, `optimization`, `api`, `xrfm` — verified by building the
  wheel and listing it). Any unrelated project or module named `model` or
  `api` on the same interpreter is shadowed. This is a packaging defect.
* `tokenizer/vocab.json` (the pretrained tokenizer) is **not shipped in the
  wheel** (no package-data config), so an installed copy of XRFM cannot load
  its own pretrained tokenizer.

## 3. Defects found (all verified by reading code or executing it)

| # | Defect | Evidence |
|---|--------|----------|
| D1 | `ConfigLoader()` default path `config/config.yaml` is **CWD-dependent**; the README quick-start (`ConfigLoader().get(...)`) fails from any other directory (`cd /tmp && python -c "import xrfm; xrfm.ConfigLoader()"` → `FileNotFoundError`). | `xrfm/config/loader.py:22`; reproduced |
| D2 | `ModelConfig`/`TrainingConfig` dataclasses have **no validation**; `d_model % n_heads != 0`, negative LR etc. are only checked inside `GPTModel.__init__` (and only some everywhere else). Config consumers other than the model get no checks. | `xrfm/config/loader.py`; `model/gpt.py:74-87` |
| D3 | `DatasetConfig` is defined and exported but **never used** by `XRFMTextDataset` (the dataset takes raw kwargs instead). Decorative configuration. | `xrfm/data/loader.py` |
| D4 | `split_dataset_lines(seed=...)` accepts `seed` and **never uses it** — splitting is always sequential + exact-line dedup, while the YAML says `shuffle: true` and the docstring implies seeded behavior. | `xrfm/data/loader.py` |
| D5 | `chunk_text` **infinite-loops** when `overlap >= max_seq_len` (stride ≤ 0; the `if start < 0` guard only catches `overlap > len-so-far`). No validation of `max_seq_len <= 0` either. | `xrfm/data/loader.py` |
| D6 | Dataset hard-codes `pad_id=0` and treats *any token equal to `pad_id`* as padding when masking targets — content tokens with id 0 silently lose their targets; the tokenizer's real `<\|pad\|>` id is ignored. The model hard-codes `padding_idx=0` for the same reason. | `xrfm/data/loader.py`, `model/gpt.py:92` |
| D7 | Last real token of every chunk is trained to predict `pad_id` ("end of chunk ≡ emit padding"), which is not a defensible LM target and contradicts the `-100` masking used for real padding. | `xrfm/data/loader.py __getitem__` |
| D8 | Version drift: `pyproject.toml` + `xrfm/__init__` + active YAMLs say `1.0.1`, but `api/main.py` (`version="1.0.0"`), `api/routes/metrics.py`, `scripts/train_custom_model.py`, `api/*` docstrings and Docker headers say `1.0.0`. Four YAML files each duplicate `project.version`. | grep across repo |
| D9 | `TrainingLoop` reads ~20 config values via `loader.get(...)` with **code-level defaults** that duplicate the YAML defaults and the `TrainingConfig` dataclass defaults — three independent sources of truth. | `training/loop.py:100-130` |
| D10 | `GPTModel(config_path=...)` and `TrainingLoop(config_path=...)` load YAML themselves — the model and trainer are coupled to file-system paths and YAML, not to typed config objects. | `model/gpt.py:38`, `training/loop.py:57` |
| D11 | `api/main.py` inserts the repo root into `sys.path` at import time and hard-codes relative paths (`tokenizer/vocab.json`, `checkpoints/`) — the API only works when launched from a repo checkout. | `api/main.py:14-16` |
| D12 | Flaky borderline test: `tests/test_neurotopo_phase11.py` overfit threshold `loss < 0.5`, measured `0.539` (passes/fails depending on environment). | pytest run 2026-09-18 |
| D13 | YAML `datasets.streaming` field is parsed by nothing (dead config). `config/medium.yaml` sets `datasets.default: medium_corpus` (no such file in repo). | grep + config files |
| D14 | `xrfm/knowledge/` and `xrfm/routing/` are empty staged placeholders (NeuroTopo research), sitting at the same level as the real config/data infrastructure, making the public package unreadable. | directory contents |

## 4. What actually works (kept as-is, by design)

The v1.x GPT stack is in good technical shape and Phase 0 deliberately does
**not** rewrite it:

* `model/` — original decoder-only transformer (RoPE, RMSNorm, SwiGLU,
  pre-norm residuals, weight tying, KV-cache hooks). Verified causal-masking
  tests exist and pass.
* `tokenizer/` — byte-level BPE v2, lossless round-trip on any Unicode,
  special-token ids resolved from the vocab (`pad/bos/eos/unk`).
* `training/` — DDP/FSDP hooks, grad accumulation, AMP, checkpoint/resume,
  JSONL metrics; CI runs a real CPU smoke train.
* `inference/` — KV-cached generation with temperature/top-k/top-p/
  repetition penalty/stop sequences.
* `evaluation/` — strided perplexity with ignore_index handling +
  benchmark ABC (top-1/top-5 accuracy).
* 322/323 tests passed at audit time (the one failure is D12).
* The `xrfm.search` RAG subsystem and `optimization/` are self-contained
  *consumers* of the core — their dependency direction is fine; only their
  import paths are wrong.

## 5. Compatibility surface that must survive Phase 0

* `xrfm.__version__`, `xrfm.ConfigLoader`, `xrfm.data.loader.XRFMTextDataset`
  (documented in README).
* All `tests/`, `scripts/`, `api/`, `benchmark/`, notebooks imports —
  mechanically updatable in-repo.
* Top-level `model.*`, `tokenizer.*`, `training.*`, `inference.*`,
  `evaluation.*`, `optimization.*` import paths — treated as
  **deprecated public paths**: kept alive via thin repo-level shim modules
  that re-export from `xrfm`, excluded from the wheel, documented for
  removal.
* `xrfm.core`, `xrfm.neurotopo`, `xrfm.topology`, `xrfm.memory`,
  `xrfm.dynamics`, `xrfm.neurons`, `xrfm.uncertainty`, `xrfm.nt_training`,
  `xrfm.nt_evaluation` (NeuroTopo research paths) — same treatment, shims
  inside the package.
* Checkpoint format: pure `state_dict` + metadata dicts — unaffected by
  module moves (no pickled classes).
