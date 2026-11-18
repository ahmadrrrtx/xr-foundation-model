# Archive Index — Repository Cleanup (Phase Two)

**Purpose:** Documents items removed during repository reorganization so they remain recoverable via Git history.

**Rule:** No deletion occurs without an entry here. Every removed path remains in Git history (commit `1f7f233` and earlier on `release/first-public-batch`), so it can be restored with `git checkout <commit> -- <path>`.

---

## Removed: Top-level deprecated shim directories

These seven directories were import shims for pre-Phase-0 paths. ADR 0001 (Section D7) promised removal "no earlier than v2.0." Since the repo is now at v2.0.0, they are removed. All canonical code lives in `src/xrfm/`.

| Removed path | Canonical replacement | Git history reference |
|---|---|---|
| `model/` | `src/xrfm/models/` | `git checkout 1f7f233 -- model/` |
| `tokenizer/` | `src/xrfm/tokenization/` | `git checkout 1f7f233 -- tokenizer/` |
| `training/` | `src/xrfm/training/` | `git checkout 1f7f233 -- training/` |
| `inference/` | `src/xrfm/inference/` | `git checkout 1f7f233 -- inference/` |
| `evaluation/` | `src/xrfm/evaluation/` | `git checkout 1f7f233 -- evaluation/` |
| `optimization/` | `src/xrfm/optimization/` | `git checkout 1f7f233 -- optimization/` |

**Verification:** Before deletion, confirmed no references to these root-level import paths exist anywhere outside `src/xrfm/` (searched `*.py`, `*.ipynb`, `*.md`, `*.yaml`, `*.yml`). Scripts use `from xrfm.models import ...` etc., not root-level imports.

## Preserved (not removed)

| Path | Reason |
|---|---|
| `src/xrfm/` | Canonical package source |
| `research/` | NeuroTopo experimental research docs — constraint: preserve tokenizer research |
| `src/xrfm/research/neurotopo/` | NeuroTopo research implementation — preserve |
| `data/` | Actively referenced by scripts (`data/datasets/corpus.txt`, `data/manifests/`) |
| `config/` | Actively referenced by scripts (`config/tiny.yaml`, `config/config.yaml`, etc.) |
| `api/` | FastAPI application — separate from `src/xrfm/`, referenced by CI type-check |
| `deployment/` | Docker/space deployment configs |
| `benchmark/` | Benchmark scripts |
| `notebooks/` | Colab notebook |
| `artifacts/` | Legacy unverified checkpoints + README |
| `scripts/` | Training/eval/benchmark scripts |

## Untracked root-level files (not deleted, not tracked)

| Path | Status |
|---|---|
| `checkpoints/checkpoint_step_2.pt` | Gitignored (`checkpoints/` in `.gitignore`) |
| `xrfm-codersec-phase-6-red-teaming-eval.log` | Untracked, large log |
| `xrfm-codersec-qlora-ablations-b1-b2-b3.log` | Untracked, large log |

## Restoration

To restore any removed path:

```bash
git checkout 1f7f233 -- <path>
# or from any earlier commit:
git log --oneline -- <path>
```
