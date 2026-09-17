# Phase 0 — Best-Practices Research (design inputs)

**Date:** 2026-09-18 · Prior to any code change. This records what was
studied and which principles XRFM adopts — and, just as important, which
practices were deliberately *not* adopted.

## 1. Sources studied

| Source | What was looked at | Takeaway relevant to XRFM |
|---|---|---|
| **PyPA / setuptools packaging docs** | src-layout vs flat-layout, package discovery (`[tool.setuptools.packages.find]`) | Flat layouts with multiple top-level packages are the classic way to ship accidental packages. src-layout makes "only the library is packaged" the *default* instead of a rule to remember. |
| **OLMo / OLMo-core (AllenAI)** | Repository organization (`src/olmo_core/…`), config system | Config-driven design: typed dataclass configs (`TransformerConfig`, `TrainerConfig`, …) composed into an `ExperimentConfig`, each with `validate()` + `build()`; components are constructed *from* configs, never from YAML paths. |
| **HuggingFace Transformers** | Public API layering | A huge library still keeps one deliberate `__init__.py` surface; internal modules are importable but never advertised. Lazy/submodule imports keep hard dependencies optional. |
| **nanoGPT / minGPT** | Minimal research-repo structure | Small, boring, explicit modules (`model.py`, `train.py`, `sample.py`); training scripts are thin, reproducible entry points. Do not build abstractions a 10M-param research model cannot justify. |
| **PyTorch packaging & distribution guidance** | Recommended install/launch patterns, `torch.compile`, DDP launch | Library code stays device-agnostic; *scripts* decide device/DDP. Entry points belong to the app layer, not the library. |
| **Python community guidance on dataclass vs pydantic for config** | Validation at boundaries | Stdlib dataclasses + explicit `__post_init__`/`validate()` are sufficient when the config surface is small and dependency-minimal; pydantic buys coercion/aggregation at the cost of a new runtime dependency. |

## 2. Adopted principles (and how they map to Phase 0 changes)

1. **One package, shipped from `src/`.**
   All library code moves to `src/xrfm/`. The wheel then *cannot* contain
   `api/`, `scripts/`, or the legacy top-level packages, no matter what
   lands at the repo root later. (PyPA default; OLMo-core precedent.)

2. **Typed, validated, composable configuration — no path-coupled
   components.**
   `ModelConfig` / `TrainingConfig` / `DatasetConfig` become real validated
   dataclasses in `xrfm/config/schema.py`, composed into `XRFMConfig`
   (OLMo-style), with `validate()`, `to_dict()`, `from_dict()` and precise
   error messages. Models/trainers accept config *objects*
   (`XRFMModel(cfg)`) and only compatibility wrappers accept paths.

3. **No CWD-dependent behavior.**
   OLMo/HF/PyTorch Lightning all require an explicit config path or accept
   a packaged default; none rely on the process working directory. XRFM:
   explicit path → packaged default resource (`importlib.resources`) →
   clear error. Never `os.getcwd()`.

4. **One deliberate public API surface.**
   `from xrfm import XRFM, XRFMConfig, Tokenizer, Trainer, Dataset,
   generate, evaluate, …` (HF-style single surface). Everything else is
   internal naming that users should not need to know.

5. **Scripts are thin entry points** (nanoGPT principle): they parse CLI
   args, build typed configs, and call package APIs. No business logic
   accumulates in `scripts/`.

6. **Secondary features are consumers of core, never entangled with it**
   (`api → xrfm`, `search → inference/tokenization`,
   `optimization → models`). Enforced by moving all core into one package
   with an acyclic internal dependency direction:

   ```text
   config → tokenization → data → models → training → inference → evaluation
                                  ↑                ↑
                        optimization ──────────────┘
                        search, research/* (experimental), api (repo-level app)
   ```

7. **Stdlib dataclasses, not pydantic**, for the config schema: XRFM's
   config surface is small, the project advertises a tiny dependency set
   (torch/pyyaml/numpy), and validation logic we own is easier to keep
   stable than a framework migration later. Revisit only if config grows
   coercion-heavy features (secrets, env interpolation).

8. **Single version source.** `xrfm/__init__.__version__` is authoritative;
   `pyproject.toml` reads it via setuptools `dynamic` version; every other
   file imports it. YAML configs stop carrying a version field.

## 3. Deliberately NOT adopted (and why)

| Practice | Reason to reject for XRFM (Phase 0) |
|---|---|
| Hydra/OmegaConf or pydantic-settings config frameworks | New dependency + new paradigm for ~40 config fields; dataclasses + YAML loader is auditable and boring (Principle 6). |
| `setup.py`/`setup.cfg` hybrid packaging | Deprecated direction; pyproject-only. |
| Namespace packages / multi-package distribution | The opposite of the Phase 0 goal. |
| HF-style lazy `TYPE_CHECKING` module graphs | XRFM has 3 hard deps; import time is not a problem worth the indirection yet. |
| nanoGPT's `configurator.py` (exec-based config override) | `exec` of CLI strings is exactly the kind of cleverness Phase 0 removes. |
| OLMo's full `build()`-per-component factory layer | XRFM is one model family, not a component zoo; direct constructors with typed configs are sufficient. |
| Renaming `GPTModel`/`BytePairEncoder` classes outright | Breaking renames buy nothing in Phase 0; new canonical names are introduced as aliases (`XRFMModel = GPTModel`, `BPETokenizer = BytePairEncoder`). |
| Deleting experimental code (NeuroTopo, RAG, quantization) | Principle: do not destroy useful work. It is *isolated* under `xrfm/research/` and clearly marked secondary/experimental instead. |
