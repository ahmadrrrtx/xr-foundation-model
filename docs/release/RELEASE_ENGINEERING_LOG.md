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

