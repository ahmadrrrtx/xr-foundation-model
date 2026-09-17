# ADR 0001 — XRFM core architecture (Phase 0 freeze)

**Status:** Accepted · **Date:** 2026-09-18 · **Deciders:** XRFM maintainers
**Supersedes:** the implicit "repository of sibling top-level packages"
layout. **Amended by:** future ADRs only.

---

## Context

At Phase 0 the repository contained two entangled worlds: seven generic
top-level Python packages (`model/`, `tokenizer/`, `training/`,
`inference/`, `evaluation/`, `optimization/`, `api/`) plus a partially
populated `xrfm/` package holding shared infra (`config/`, `data/`) and the
entire experimental NeuroTopo stack. Dependencies crossed the boundary in
both directions (`model → xrfm.config`, `xrfm.search → inference`), so
there was no package boundary at all. Full evidence:
`docs/architecture/PHASE_0_AUDIT.md`.

Consequences of that layout: `pip install` shipped eight top-level
packages into site-packages (namespace pollution: any environment with
another `model`/`api` module breaks); `tokenizer/vocab.json` was not
packaged; `ConfigLoader()` depended on the process working directory;
configuration dataclasses had no validation; the dataset ignored its
`seed`; chunking infinite-looped on `overlap >= max_seq_len`; version
strings drifted (1.0.0 in the API vs 1.0.1 in the package).

## Decisions and rationale

### D1 — `xrfm` is the public package; the library lives in `src/xrfm/`

*Single importable package, src-layout.*

* Why: one installable unit can be tested, versioned, and documented; the
  wheel then *cannot* ship applications or scripts by accident (structural
  guarantee, not a convention). src-layout is the PyPA recommendation for
  libraries and matches OLMo-core, the closest open reference project.
* Alternative rejected: keeping top-level packages with a curated wheel
  include-list — one forgotten directory away from re-introducing the
  defect.

### D2 — Subpackage layout mirrors subsystem responsibilities

`config / tokenization / data / models / training / inference /
evaluation` (+ secondary `optimization`, `search`, `research`,
`experiment`). The exact Phase-0 proposal (`tokenization`, `models`)
was adopted; `data/loader.py` was split into `splits/packing/dataset/
manifest` because dataset *representation*, *splitting*, *packing*, and
*provenance* change for different reasons and had different bugs.

### D3 — Dependency direction is acyclic and one-way

`config → tokenization → data → models → training → inference/evaluation`;
secondary layers (`optimization`, `search`, `research`, `api`) consume the
core and are never imported by it. This is what makes Phase 1 (data
pipeline + large-scale training) possible without another rewrite: the
trainer can be replaced/upgraded without touching models, and evaluation
never needs the trainer.

### D4 — Configuration is typed, validated dataclasses; no path-coupled components

`ModelConfig` / `TrainingConfig` / `DatasetConfig` compose into
`XRFMConfig` with `validate()` + serialization (OLMo-core-style, stdlib
dataclasses instead of pydantic to keep the dependency set minimal).
Models and trainers accept config *objects*; YAML strings are accepted
only at the edges (loader, legacy kwargs) and immediately become typed
objects. Rationale: D2 of the audit — invalid configs previously reached
model construction, and three independent default-sources drifted.

### D5 — No CWD dependence, ever

`load_config(None)` resolves the packaged default resource; explicit paths
are the caller's choice. The API's `sys.path` insertion was removed; the
API installs the package. Rationale: README quick-start had to be run from
the repo root, which broke in any installed/real environment.

### D6 — One public API surface; internals are private by policy

`from xrfm import XRFM, XRFMConfig, Tokenizer, Trainer, Dataset, generate,
evaluate, ...` is the supported surface (`xrfm.__all__`). Deep modules stay
importable for research but are not guaranteed stable. This is the
Transformers/nanoGPT lesson scaled to XRFM's size: users should not need
the folder map.

### D7 — Backward compatibility: shims, not rewrites

All pre-Phase-0 import paths (top-level `model.*`, `tokenizer.*`, ... and
in-package `xrfm.core`, `xrfm.neurotopo`, ...) keep working via thin
deprecation shims that re-export the canonical objects. Root shims are
excluded from the wheel; in-package shims ship (they are cheap). Removal
is promised no earlier than v2.0. Canonical aliases introduced without
renames: `XRFMModel = GPTModel`, `BPETokenizer = BytePairEncoder`,
`TextDataset = XRFMTextDataset`, `Tokenizer = TokenizerInterface`.

### D8 — Secondary and experimental code is isolated, not deleted

NeuroTopo (research model), search/RAG, quantization/speculative decoding
stay — but under `xrfm/research/neurotopo/` and clearly-secondary
packages, exempt from core lint gates until they graduate. Rationale: the
repo's own audit policy (don't destroy useful work) plus the Phase 0 goal
(a readable core).

### D9 — Version: one source

`xrfm.__version__` is authoritative; `pyproject.toml` reads it via
setuptools `dynamic`; YAML presets carry no version; the API and scripts
import it. Repo version remains **1.0.1** — reorganization is not a
release.

## What must NOT bypass this architecture (future work rules)

1. No core module may import `xrfm.research.*`, `xrfm.search`,
   `xrfm.optimization`, `api`, or `scripts`. (Test-guarded.)
2. No new subsystem may parse YAML itself; add fields to
   `config/schema.py` with validation, or accept typed config objects.
3. No component may resolve paths from the process CWD implicitly.
4. No dataset/packing change may reintroduce token-identity padding
   assumptions; use the tokenizer contract.
5. Product features (API/UI/RAG) grow by *consuming* `xrfm`'s public API;
   if the public API lacks something, extend the public API deliberately.
6. Research experiments belong in `xrfm/research/`; graduating one into
   core requires an ADR.

## Consequences

* Positive: clean installability (`pip install` one package, resources
  included), validated configs everywhere, deterministic splits, safe
  chunking, single version, lint/type/test gates aligned with the layout,
  and a stable target for Phase 1 (data pipeline, tokenizer work,
  large-scale training, distributed training, evaluation) with no further
  restructuring expected.
* Negative / accepted costs: deprecated shims must be maintained until
  v2.0; `xrfm/research` imports look longer; one behavior change in the
  dataset contract (last-real-token target is now masked instead of
  teaching the model to predict the pad token) and `padding_idx` now comes
  from the tokenizer contract instead of a hard-coded 0 — both documented
  in the CHANGELOG.
