# Dead Code and Reorganisation Audit

**Baseline commit:** `dcc426610ad6e337985a43b93f3d83bad40f669d`
**Date:** 2026-11-19
**Decision rule:** A file is removed only when it is not imported by active modules, not dynamically loaded, not an entry point, not required by tests, not required by published reproduction and not essential historical evidence. If it has historical value, it is archived rather than deleted.

## 1. What was already removed (before this pass)

Commit `4147d8f` removed six root-level shim directories:

`model/`, `tokenizer/`, `training/`, `inference/`, `evaluation/`, `optimization/`

Evidence that removal was safe:

1. Each file was a thin re-export (roughly 17 lines) pointing into `src/xrfm/`, not an independent implementation.
2. `pyproject.toml` packages only `xrfm*` from `src/`, so the shims were never shipped in the wheel.
3. A repository-wide search of `*.py`, `*.md`, `*.yaml`, `*.yml` and `*.ipynb` found no active imports of the root-level paths.
4. `tests/test_compat_shims.py` was updated to test in-package shims only.
5. The CI clean-install job passed before and after the removal, proving the public API was unaffected.
6. All 39 removed files remain recoverable via Git history and are indexed in `docs/release/ARCHIVE_INDEX.md`.

No files were deleted in the current release-engineering pass.

## 2. Candidate directories examined in this pass

| Candidate | Verdict | Evidence |
|---|---|---|
| `src/xrfm/core/`, `topology/`, `memory/`, `neurons/`, `dynamics/`, `uncertainty/`, `nt_training/`, `nt_evaluation/`, `neurotopo/` | **Preserve — research implementation** | Imported by `tests/test_compat_shims.py` and `tests/test_neurotopo_phase1.py`; documented in `docs/implementation/NEUROTOPO_IMPLEMENTATION_LOG.md` and phase result files. Deleting these would erase published experimental evidence. |
| `src/xrfm/knowledge/`, `routing/` | **Preserve — staged placeholders under active tests** | Imported by `tests/test_neurotopo_phase1.py`; documented as staged placeholders in `docs/architecture/PHASE_0_AUDIT.md`. |
| `src/xrfm/research/neurotopo/` | **Preserve — research bundle** | Explicitly excluded from core lint/type gates by `pyproject.toml`; preserved by design. |
| `src/xrfm/tokenization/tournament.py` | **Preserve — research identity** | Imported by `scripts/run_tokenizer_tournament.py` and `tests/test_phase2_tokenizer_tournament.py`; required to reproduce the T4 tokenizer tournament. |
| `src/xrfm/tokenization/vocab.json`, `tokenizer_32k_multilingual.json` | **Preserve — artefacts / serialization** | Listed in `[tool.setuptools.package-data]`; `vocab.json` is the legacy control in the tokenizer tournament. |
| `artifacts/legacy_unverified/checkpoint_step_500.pt` | **Quarantine, do not delete** | 76.8 MB pre-audit checkpoint. Its README states it is never loaded by runtime or API; retained as baseline evidence per the no-deletion rule. |
| `data/datasets/corpus.txt` | **Preserve — training data reference** | 5.3 MB local corpus used by reproduction scripts; excluded from wheel and lint. |
| `config/config.legacy-v1.yaml`, `config/v1.1-medium.yaml` | **Preserve — historical configs** | Pinned legacy configurations; referenced in version history. |
| `scripts/xrfm-codersec-phase-*.log` (historical untracked logs referenced in older reports) | **Not present in current checkout** | No tracked files match; no action needed. |

## 3. Dynamic and non-Python usage checked

- **Entry points:** only `xrfm = xrfm.cli:main`; intentionally preserved.
- **Module-as-script usage:** `python -m xrfm.experiment.cards` is invoked from generated model/dataset cards; preserved despite zero static importers.
- **Package data:** tokenizer JSON files and `config/*.yaml` are packaged; preserved.
- **Config strings:** `config/`, `configs/`, `docs/experiments/` and `data/` paths were searched across markdown, YAML, JSON and notebooks before drawing conclusions.
- **Notebook references:** the single notebook is research-only and does not define an active runtime path.

## 4. Branch and archive state

- No force-push, history rewrite, tag replacement or destructive operation was performed.
- Tag `v2.0.0` remains at `b5d2120ba7`, untouched.
- Recoverable copy of all previously removed files: Git history; see `docs/release/ARCHIVE_INDEX.md` for exact restoration commands.

## 5. Change decision

The professional structure already matches the proposed target closely: `.github/`, `src/xrfm/` with model/tokenization/training/inference/evaluation/codersec/cli subpackages, `configs/`, `scripts/`, `tests/`, `docs/`, `notebooks/`, `api/`, `deployment/`, `artifacts/`, website source and release documentation. Reorganising further would risk breaking pinned research paths without evidence of duplication. Therefore this pass makes **no additional deletions** and records the proof above.
