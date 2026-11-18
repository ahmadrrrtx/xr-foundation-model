# Phase Three — CD and Release Pipeline Reliability

**Status:** Reviewed and verified — no changes needed.

## CD Pipeline (`.github/workflows/cd.yml`)

**Trigger:** Push to tags matching `v*`

**Jobs:**
- `build` — Builds Docker image from `deployment/Dockerfile`, pushes to `ghcr.io/ahmadrrrtx/xr-foundation-model`

**Permissions:** `contents: read`, `packages: write` (minimal)

**Tags:** semver (full version, major.minor, edge)

**Assessment:** Reliable. Triggered only on tag pushes, uses GITHUB_TOKEN for registry auth, builds from a pinned Dockerfile. Successfully ran for v2.0.0 tag (run #1).

## Release Pipeline (`.github/workflows/release.yml`)

**Trigger:** Push to tags matching `v*`

**Jobs:**
- `release` — Creates GitHub Release using `softprops/action-gh-release@v1`

**Permissions:** `contents: write` (minimal)

**Assessment:** Reliable. Creates release with tag name, auto-generates release notes from CHANGELOG.md, includes installation instructions (source + Docker). Marks prereleases for rc/beta tags. Successfully ran for v2.0.0 tag (run #1).

## GitHub Pages Pipeline (`.github/workflows/pages.yml`)

**Trigger:** Push to `main` and `release/*`, workflow_dispatch

**Jobs:**
- `deploy` — Deploys `./docs` to GitHub Pages

**Permissions:** `contents: read`, `pages: write`, `id-token: write` (minimal)

**Assessment:** Newly created for Phase Five. Deploys from `/docs` directory. Environment: `github-pages`.

## CI Pipeline (`.github/workflows/ci.yml`)

**Trigger:** Push/PR to `main`, `develop`, `release/*`

**Jobs:**
- `lint` — ruff check + ruff format --check
- `type-check` — mypy on core packages + api/
- `test` — pytest on Python 3.10/3.11/3.12 matrix
- `package` — Build sdist + wheel, clean install test
- `security` — bandit + pip-audit
- `training-smoke` — ci_smoke_test.py

**Assessment:** Fixes applied (removed black, added dev deps). mypy fix committed. Test (3.10) failure unresolved — see RELEASE_ENGINEERING_LOG.md for details.

## No `|| true` shortcuts
All quality gates remain intact. No weakened checks.

## No force-push / no tag replacement
v2.0.0 tag preserved at `6c61166`. All work on `release/first-public-batch` branch.
