# Phase Six — Professional README and Repo Metadata

**Status:** README reviewed and badge updated.

## README.md

The existing README is professional and comprehensive. Sections:

| Section | Status |
|---|---|
| Trained models table (dual-track) | Complete |
| Model cards links | Complete |
| Live website link | Complete |
| Installation (`pip install -e ".[dev]"`) | Complete |
| Quick Start (Python API + CLI) | Complete |
| Architecture (Track A + Track B) | Complete |
| Evaluation & Red-Teaming table | Complete |
| Reproducibility (`verify_reproducibility.py`) | Complete |
| Documentation links | Complete |
| Safety & Dual-Use Boundaries | Complete |
| Contributing / Code of Conduct | Complete |
| License (MIT) | Complete |

## Change Made

Updated CI badge branch from `main` to `release/first-public-batch` to reflect the active release branch.

## Repo Metadata

| File | Status |
|---|---|
| `LICENSE` | MIT — present |
| `CODE_OF_CONDUCT.md` | Present |
| `CONTRIBUTING.md` | Present |
| `SECURITY.md` | Present |
| `CHANGELOG.md` | Present |
| `pyproject.toml` | Fixed (removed black, added transformers/peft) |
| `README.md` | Professional, complete |

## No PyPI Publish

Per constraint: Do not publish new PyPI package until `xrfm` naming conflict independently verified. The public PyPI name `xrfm` belongs to an unrelated distribution — the README documents this.
