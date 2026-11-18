# XRFM Dependency Map and Canonical Source of Truth

**Status:** Evidence-based mapping completed before any structural change in this release pass.
**Baseline commit:** `dcc426610ad6e337985a43b93f3d83bad40f669d`
**Date:** 2026-11-19

## 1. Canonical implementation

| Concern | Canonical location | Evidence |
|---|---|---|
| Python package | `src/xrfm/` | `pyproject.toml` uses `[tool.setuptools.packages.find] where = ["src"], include = ["xrfm*"]`; the wheel top-level check in CI asserts only `xrfm`. |
| Public import API | `src/xrfm/__init__.py` | Defines `__version__`, `__all__`, and re-exports config, model, tokenizer, data, training, inference and evaluation symbols. |
| Transformer | `src/xrfm/models/gpt.py` (+ `models/attention`, `models/layers`) | `scripts/ci_smoke_test.py`, `api/main.py`, benchmarks and 39 Python importers reference `xrfm.models.gpt`. |
| Tokenizer | `src/xrfm/tokenization/` | 21 Python importers of `xrfm.tokenization.bpe`; packaged resource `tokenizer_32k_multilingual.json` is listed in `[tool.setuptools.package-data]`. |
| Tokenizer research / T4 tournament | `src/xrfm/tokenization/tournament.py`, `scripts/run_tokenizer_tournament.py`, `docs/experiments/tokenizer_tournament_v1.json` | `tests/test_phase2_tokenizer_tournament.py` and `scripts/verify_reproducibility.py` pin these paths. Preserved. |
| Training | `src/xrfm/training/` | `scripts/ci_smoke_test.py`, `scripts/train_native_from_hub.py`, `scripts/train_custom_model.py`, config files. |
| Inference | `src/xrfm/inference/` | 13 Python importers including `api/main.py`, `benchmark/inference_forward.py`, `deployment/huggingface_space/app.py`. |
| Evaluation | `src/xrfm/evaluation/` | `scripts/evaluate_checkpoint.py`, `scripts/run_eval_protocol.py`, `tests/test_phase8_gate.py`. |
| CoderSec | `src/xrfm/codersec/` | `xrfm.codersec.dataset` has 3 direct Python importers; train/merge/eval scripts target this package. |
| CLI | `src/xrfm/cli.py`; console script `xrfm` | `[project.scripts] xrfm = "xrfm.cli:main"`; exercised by CI clean-install job. |
| API service | `api/` | Separate application, type-checked and tested in CI; not part of the library wheel. |

A static AST scan of all tracked `.py` files found **151 modules** under `src/xrfm/`; **116** have at least one direct Python importer outside their own file. Modules with no direct importers are mostly reached through package `__init__.py` re-exports, compatibility tests, or documentation-preserved research (see §4).

## 2. Previously suspected duplicate trees

The paths `model/`, `tokenizer/`, `training/`, `inference/`, `evaluation/` and `optimization/` were **root-level import shims**, already removed in commit `4147d8f` on `release/first-public-batch`. They were thin re-export modules, not independent implementations. The removal was recorded in `docs/release/ARCHIVE_INDEX.md` and every removed file remains recoverable from Git history. The current tree contains no root-level `model/`, `tokenizer/`, `training/`, `inference/`, `evaluation/` or `optimization/` directory.

No active duplicate implementation of the transformer, tokenizer, trainer, inference engine or configuration loader remains in the current checkout.

## 3. Active versus reference-only directories

| Directory | Classification | Reason |
|---|---|---|
| `config/` | Active | `config/tiny.yaml`, `native-25m.yaml`, `native-85m.yaml` are referenced by training scripts and model telemetry. |
| `configs/` | Active | `configs/data/*.yaml`, `configs/eval/*.json*` contain dataset sources and frozen evaluation manifests. |
| `data/` | Active reference data | `data/datasets/corpus.txt` (5.3 MB) and manifests are used by local reproduction scripts; not packaged in the wheel. |
| `artifacts/legacy_unverified/` | Quarantined evidence | 76.8 MB checkpoint explicitly documented as never loaded by runtime or API. Retained per no-delete policy. |
| `research/` | Historical research documents | Phase reports and blueprints; retained for reproducibility and provenance. |
| `notebooks/` | Research artifact | Single Colab notebook; not imported by CI. |
| `docs/` | Documentation + release evidence | 167 files including experiment records and release reports. |
| `deployment/` | Active deployment | Dockerfile, gunicorn config, HF Space app. |
| `benchmark/` | Active scripts | Three benchmark entry scripts. |
| `scripts/` | Active tooling | 21 scripts including reproducibility verification and CI smoke test. |

## 4. Modules with no direct importers — disposition

The following groups were checked individually rather than assumed dead:

- `xrfm.agent.*` submodules (`runtime`, `backends`, `safety`, `evaluation`, ...) are re-exported by `src/xrfm/agent/__init__.py` and exercised through `from xrfm.agent import ...` in `tests/test_phase0_agent.py`; they are active.
- `xrfm.core.*`, `xrfm.topology.*`, `xrfm.memory.*`, `xrfm.neurons.*`, `xrfm.dynamics.*`, `xrfm.uncertainty.*`, `xrfm.nt_training.*`, `xrfm.nt_evaluation.*` and `xrfm.neurotopo.*` implement the NeuroTopo research line. `tests/test_compat_shims.py` and `tests/test_neurotopo_phase1.py` import these package paths, and `docs/implementation/` records the corresponding experiments. They are research evidence and must not be deleted.
- `xrfm.knowledge` and `xrfm.routing` are staged placeholder packages documented in `docs/architecture/PHASE_0_AUDIT.md` and imported by `tests/test_neurotopo_phase1.py`. Removing them would break a current test; they are retained.
- `xrfm.experiment.cards` is a generator (`python -m xrfm.experiment.cards`) referenced by three published card files. Zero static importers is expected for a module invoked as a script.
- `xrfm.codersec.kaggle_phase6_eval_kernel` is referenced by `docs/experiments/FINAL_BENCHMARK_MATRIX.md` as the source of the red-team evaluation kernel; retained as experimental evidence.

**Conclusion:** No further source deletion is justified on the current evidence. The only removals in repository history were the six verified root shim trees; they are archived via Git and indexed in `docs/release/ARCHIVE_INDEX.md`.

## 5. Packaging and PyPI naming

- Distribution metadata name is currently `xrfm`, version `1.0.1`, with import package `xrfm`.
- PyPI `xrfm` 0.4.5 is an unrelated project ("xRFM: Scalable and interpretable kernel methods for tabular data", by Daniel Beaglehole and David Holzmüller). It is **not** this project. No new PyPI publication has been attempted.
- `xrfm-foundational-model`, `xrfm-research` and `xr-foundation-model` returned HTTP 404 on PyPI at audit time, so a distinct name is available if publication is ever approved. The import interface should remain `xrfm`.
- Existing release workflow documentation already warns users to install from source, not from PyPI.

## 6. Reproducing this map

```bash
# AST scan used for importer counts (summary written above)
python3 - <<'PY'
import ast
from pathlib import Path
root = Path('.')
for p in root.rglob('*.py'):
    if '.git' in p.parts: continue
    ast.parse(p.read_text(encoding='utf-8', errors='ignore'))
PY

# Confirm current top-level package layout
git ls-files 'src/xrfm/**/*.py' | cut -d/ -f2 | sort -u
```
