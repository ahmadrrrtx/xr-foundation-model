=== Phase Zero Baseline ===

## Phase Zero — Repository Protection and Baseline Inspection

| Field | Value |
|-------|-------|
| Repository | https://github.com/ahmadrrrtx/xr-foundation-model |
| Baseline commit | eb2e60089b3bd4d44e5028c69999c4c797cd2807 |
| Branch | release/first-public-batch |
| Preserved tag | v2.0.0 (6c611662d70362aea3cc66c874da206cb93f9f91) |
| Remote | origin (https://github.com/ahmadrrrtx/xr-foundation-model.git) |
| Working tree | Clean |
| Date | 2026-11-17 (local timezone Asia/Karachi) |

### Inspection Steps Completed
1. ✅ Found existing XRFM repository checkout
2. ✅ Verified Git remote (origin → github.com/ahmadrrrtx/xr-foundation-model)
3. ✅ Inspected current branch (main → release/first-public-batch)
4. ✅ Recorded current commit SHA (eb2e600)
5. ✅ Fetched latest remote changes
6. ✅ Inspected working-tree status (clean)
7. ✅ Identified uncommitted changes (none at baseline)
8. ✅ Preserved existing release tag v2.0.0
9. 🔄 Inspecting model and dataset artifacts
10. 🔄 Checking GitHub credentials for push/PR

### Phase One — CI/CD Fixes Applied

**Root causes identified from GitHub Actions run #54:**
1. Black formatting check failed (exit code 1) — black 26.10.1 and ruff 0.17.5 disagree on formatting
2. Pytest exit code 2 — test collection errors due to missing dependencies (fastapi, pyarrow, huggingface_hub, transformers, peft)

**Fixes applied:**
1. Removed `black --check .` from CI — ruff format is the canonical formatter (black and ruff disagree due to version differences; ruff format passes locally on all files)
2. Changed `pip install -e ".[dev]"` to `pip install -e ".[dev,huggingface_hub]"` in all CI jobs (lint, type-check, test, training-smoke)
3. Added `transformers>=4.51.0,<4.53.0` and `peft>=0.11.0` to `[dev]` extras in pyproject.toml (needed for test collection of Phase 4/5 codersec tests)
4. Removed `black>=23.0` from `[dev]` extras (replaced by ruff format)

**Local verification:**
- ruff check: ALL CHECKS PASSED
- ruff format --check: 218 files already formatted
- mypy on security/experiment/cli: SUCCESS
- pytest: 678/678 passed
- pytest collection: 678 tests collected, 0 errors


### Phase One — CI/CD Fixes (v2)

**CI trigger expanded:** Added `release/*` to push/pull_request triggers in `.github/workflows/ci.yml` so CI runs on `release/first-public-batch` without requiring a PR to main/develop first.

**Status:** Fixes committed and pushed to `release/first-public-batch` at commit `76560ae`. CI should trigger on the `release/*` branch pattern.

**Local verification (all passing):**
- ruff check: ALL CHECKS PASSED
- ruff format --check: 218 files already formatted
- pytest: 645/645 passed
- mypy on new code: clean

**Phase Two — Repository Cleanup (DONE):**
- Removed 6 root-level deprecated shim directories (model/, tokenizer/, training/, inference/, evaluation/, optimization/)
- All canonical code in src/xrfm/ unchanged
- Updated tests/test_compat_shims.py (dropped root-level shim tests, kept in-package)
- Added docs/release/ARCHIVE_INDEX.md with restoration instructions
- 39 files deleted, recoverable via Git history

**Phase Three — CD/Release pipelines:** Already configured correctly (trigger on tag pushes v*, minimal permissions). No changes needed.

**Next:** Wait for CI to turn green on release/first-public-batch, then proceed to Phase Four (model cards + dataset cards audit).

### Phase One — CI Fix (v3): mypy import-untyped error

CI run #55 on release/first-public-batch failed:
- mypy exit code 1: `src/xrfm/tokenization/hf_bridge.py:144` — `tokenizers` module missing library stubs (import-untyped)
- Test (3.10) exit code 1: Unable to see details (logs truncated, requires login)

**Fix:** Added `# type: ignore[import-untyped]` to `from tokenizers import Tokenizer` in hf_bridge.py.

**Still unresolved:** Test (3.10) failure — details unavailable from GitHub web UI (requires login). May be a Python 3.10 dependency compatibility issue on ubuntu-latest (Ubuntu 26). Need to investigate if CI run #57 (for commit f5d093c) passes.

**Commits pushed to release/first-public-batch:**
- 76560ae: CI trigger on release/* branches
- 44bd165: Log updates
- f5d093c: Phase Five website
- 913bedc: mypy fix

### Phase Eight — Public Link Validation (DONE)

All external links verified:

| Link | Status | Notes |
|---|---|---|
| huggingface.co/ahmadrrrtx/xrfm-native-25m | VALID | 25.2M params, BPB curves confirmed |
| huggingface.co/ahmadrrrtx/xrfm-native-85m | VALID | 86.4M params, BPB curves confirmed |
| huggingface.co/datasets/ahmadrrrtx/xrfm-codersec-sft-40k | VALID | 24.1k rows, train split |
| huggingface.co/datasets/ahmadrrrtx/xrfm-native-shards-v1 | VALID | Shards dataset exists |

All model cards, dataset cards, README, and website links point to live resources.

### Remaining Blocker

**CI test (3.10) exit code 1** — Details unavailable from GitHub web UI (requires login). All 645 tests pass locally on Python 3.13. Dependencies (transformers, peft) both support Python 3.10. Need to investigate on next CI run or find alternate way to view error details.
