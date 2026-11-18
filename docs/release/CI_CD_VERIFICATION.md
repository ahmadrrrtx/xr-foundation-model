# CI/CD Verification Report

**Date:** 2026-11-19
**Release branch:** `release/first-public-batch`
**Pull request:** https://github.com/ahmadrrrtx/xr-foundation-model/pull/5

## 1. Failures diagnosed from downloaded logs

| Run | Head | Observed failure | Root cause |
|---|---|---|---|
| CI #53 `44374167245` | `466a61d9` | Lint job exit 1 | `ruff format --check` would reformat 4 files |
| CI #54 `44375949675` | `eb2e6008` | Lint exit 1; tests exit 2 | `black --check` disagreed with Ruff on the same 4 files; pytest collection failed with `ModuleNotFoundError: pyarrow` |
| CI #62 `44390630623` | `dcc42661` | Python 3.10 test failure | `test_agent_http_stream_does_not_buffer_backend_work` asserted `< 0.2 s` and took 1.58 s on a loaded runner; 3.11/3.12 cancelled by fail-fast |
| Pages #5 `44390630530` | `dcc42661` | Configure Pages failed | GitHub Pages was never enabled; API returned HTTP 404 |
| CD #2 `44405070698` | `b2206be4` | Docker build failed | `docker/metadata-action` produced no tag for a non-default branch push; buildx refused `--push` |

Logs were downloaded from the Actions API and inspected directly; no conclusion above was inferred from annotations alone.

## 2. Changes made

- Rewrote the four ambiguous statements so **Ruff and Black produce identical output**; restored `black --check .` as a real CI gate instead of dropping it.
- Pinned `ruff==0.17.5` and `black==26.10.1` in `[dev]` so formatter agreement is reproducible.
- Added `pyarrow>=14.0.0` to `[dev]`; test collection no longer depends on the `hub` extra being present.
- Replaced wall-clock streaming assertions with explicit process-safe gates. The tests still fail if the first event is buffered behind backend work.
- Set `fail-fast: false` on the Python matrix.
- Updated first-party actions to Node-24-compatible majors: `checkout@v7`, `setup-python@v7`, `upload-artifact@v7`, `configure-pages@v6`, `upload-pages-artifact@v5`, `deploy-pages@v5`; Docker actions to `login@v4`, `metadata@v6`, `build-push@v7`; release action to `softprops@v3`.
- Added minimal explicit `permissions` blocks to CI, CD, release and Pages workflows.
- Added `type=ref,event=branch` to Docker metadata tags so branch verification builds have a publishable tag; restored tag-only CD trigger after verification.

No quality gate was removed, relaxed or converted to a warning.

## 3. Final CI evidence

| Run | URL | Result |
|---|---|---|
| #63 | https://github.com/ahmadrrrtx/xr-foundation-model/actions/runs/44401826878 | 8/8 jobs success — first green run after formatter/test repairs |
| #64 | https://github.com/ahmadrrrtx/xr-foundation-model/actions/runs/44404199116 | 8/8 jobs success after website and documentation commit |
| #67 | https://github.com/ahmadrrrtx/xr-foundation-model/actions/runs/44406117302 | 8/8 jobs success on final verification commit `d86be03` |

Run #67 job results: Package & Clean Install ✅, Type Check (mypy) ✅, Dependency & Agent Security ✅, Lint & Format ✅, Training Pipeline Smoke Test ✅, Run Tests 3.10/3.11/3.12 ✅.

Exact test counts from downloaded logs:

- Python 3.10: `645 passed, 6 warnings in 151.54s`
- Python 3.11: `645 passed, 6 warnings in 135.26s`
- Python 3.12: `645 passed, 6 warnings in 91.02s`
- CPU training smoke: `CI TRAINING SMOKE: PASS (steps=60, loss 6.24 -> 2.197, ckpt=checkpoint_step_60.pt, resume step=60)`

The smoke test constructs a tiny model, runs forward/loss/backward/optimizer step, serializes and restores a checkpoint with exact resume behaviour, and runs a generation smoke step. No published checkpoint is loaded.

## 4. CD evidence

- Historical v2.0.0 CD run: https://github.com/ahmadrrrtx/xr-foundation-model/actions/runs/44374094332 (success).
- Historical v2.0.0 release run: https://github.com/ahmadrrrtx/xr-foundation-model/actions/runs/44374094367 (success).
- Temporary branch verification #2: https://github.com/ahmadrrrtx/xr-foundation-model/actions/runs/44405070698 — failed, revealing the missing-tag bug.
- Corrected branch verification #3: https://github.com/ahmadrrrtx/xr-foundation-model/actions/runs/44405181518 — **success**, Docker image built and pushed to GHCR.
- The temporary branch trigger was then removed; the committed CD workflow again triggers only on `v*` tags and manual dispatch.

The fine-grained PAT cannot call the workflow-dispatch API (HTTP 403), so the branch-trigger verification was used instead. No new release tag was created.

## 5. Pages status

Website build and link validation pass locally and in the Pages workflow build job. Deployment remains blocked by a repository setting: `actions/configure-pages@v6` cannot create a Pages site with the available token (`Resource not accessible by integration`). This requires a one-time manual enablement in repository settings: **Settings → Pages → Source → GitHub Actions**. See `WEBSITE_DEPLOYMENT.md`.

## 6. Credential handling

Tokens were stored outside the repository in a mode-0600 file and used only for authenticated reads, pushes, Hugging Face card uploads and GitHub API calls. No token appears in source, workflow files, logs, documentation or Git history.
