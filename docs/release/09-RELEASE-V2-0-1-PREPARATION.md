# Phase Nine — Prepare v2.0.1 Release (PATCH)

**Status:** BLOCKED — waiting for CI green on `release/first-public-batch`.

## Release Plan

**Tag:** `v2.0.1` (patch, does NOT overwrite `v2.0.0` at `6c61166`)

**Scope:** Docs/site/metadata only — NO weight changes

**Prerequisites (all must pass before tagging):**
1. [ ] CI green on `release/first-public-batch` — BLOCKED (test 3.10 failure)
2. [ ] Website live at https://ahmadrrrtx.github.io/xr-foundation-model/
3. [ ] Model cards accurate — VERIFIED
4. [ ] Links verified — VERIFIED
5. [ ] Open PR with summary, changes, tests, risk assessment — PENDING

## Changes for v2.0.1

- Repository cleanup (removed 6 deprecated root-level shim directories)
- CI/CD fixes (removed black, added transformers/peft to dev extras)
- GitHub Pages website (static HTML/CSS/JS)
- ARCHIVE_INDEX.md with restoration instructions
- Updated RELEASE_ENGINEERING_LOG.md

## Release Tag Constraints

- Tag must be unique — `v2.0.1` does not exist yet
- No force-push
- No tag replacement
- `v2.0.0` at `6c61166` preserved
- No credentials in repo

## Release Workflow

Once CI is green:
1. Open PR from `release/first-public-batch` to `main` with full summary
2. Do NOT merge without explicit user approval
3. After merge to `main`, tag `v2.0.1` on the merge commit
4. GitHub Actions `release.yml` and `cd.yml` trigger automatically on tag push
