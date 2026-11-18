# Release Readiness Report

**Date:** 2026-11-19
**Project:** XR Foundation Model (XRFM)
**Starting commit:** `dcc426610ad6e337985a43b93f3d83bad40f669d`
**Release branch:** `release/first-public-batch`
**Verification commit:** `d86be03`
**Pull request:** https://github.com/ahmadrrrtx/xr-foundation-model/pull/5
**Preserved tag:** `v2.0.0` → `6c611662d70362aea3cc66c874da206cb93f9f91`, unchanged

## 1. Completed work

### Repository protection
- Created a fresh checkout, verified `origin`, clean tree, branch and baseline SHA.
- Created a dedicated release branch; no force-push, history rewrite or tag replacement.
- Stored supplied credentials outside the repository with mode 0600; never committed or logged them.
- Confirmed GitHub push permission and Hugging Face write access.

### CI/CD
- Downloaded and diagnosed actual logs for runs #53, #54, #62 and Pages #5.
- Repaired formatter conflict rather than removing a gate; both Ruff and Black now pass with pinned versions.
- Fixed pytest collection by declaring `pyarrow` in dev dependencies.
- Replaced flaky wall-clock streaming assertions with deterministic process-gated ordering tests.
- Updated all workflows to current Node-24-compatible action majors and minimal permissions.
- Verified CI run #67: 8/8 jobs green; 645 tests passed on Python 3.10, 3.11 and 3.12.
- Verified CD run #3 after fixing branch tagging; Docker image built and pushed to GHCR. Restored tag-only trigger.

### Repository structure and research preservation
- Produced `DEPENDENCY_MAP.md` and `DEAD_CODE_AUDIT.md` with AST import evidence and path-level disposition.
- Confirmed `src/xrfm/` is canonical and wheel contains only `xrfm`.
- Preserved T4 tokenizer, `vocab.json`, tokenizer tournament harness, NeuroTopo research and experiment records.
- No files deleted or archived in this pass; prior root-shim removals remain recoverable and indexed.

### Website
- Implemented a 12-page static research site under `website/`, with original SVG diagrams, accessible responsive CSS, SEO metadata, sitemap, robots and 404 page.
- Build and link validation pass locally and in the Pages workflow build job.
- Public deployment is blocked by a one-time Pages source setting; see `WEBSITE_DEPLOYMENT.md`.

### Documentation and repository presentation
- Rewrote README with dual-track identity, real badges, model/dataset tables, licensing, limitations and reproduction steps.
- Added `CITATION.cff` without inventing a DOI or publication.
- Published release engineering, CI/CD, dependency, dead-code, model/dataset and license reports.
- Attempted repository metadata update via API; fine-grained PAT returned HTTP 403. Description, homepage and topics must be set manually or with a token permitted to administer repository settings.

### Hugging Face artefacts
- Updated native model cards with exact revisions, architecture, telemetry and BPB explanation.
- Added an accurate CoderSec SFT dataset card and published `QUALITY_REPORT.json`.
- Verified card metadata: native licenses remain Apache-2.0; CoderSec dataset now declares `other`.
- No model weights or dataset files were overwritten.

## 2. Change inventory (baseline → verification commit)

- **Files added:** 44 (site source and generated pages, release reports, HF card sources, CITATION.cff, filter script, quality report).
- **Files modified:** 12 (4 workflows, README, release log, pyproject, two source formatting fixes, three tests).
- **Files removed:** 0.
- **Files archived in this pass:** 0.
- **Hugging Face files updated:** 4 (three README.md cards and one QUALITY_REPORT.json).

Significant deletions in earlier history were six root-level shim directories (39 files), proven unused and recoverable via Git; see `docs/release/ARCHIVE_INDEX.md`.

## 3. Evidence index

| Area | Evidence |
|---|---|
| CI failures and fixes | `docs/release/CI_CD_VERIFICATION.md` |
| CI green | https://github.com/ahmadrrrtx/xr-foundation-model/actions/runs/44406117302 |
| CD green | https://github.com/ahmadrrrtx/xr-foundation-model/actions/runs/44405181518 |
| Dependency map | `docs/release/DEPENDENCY_MAP.md` |
| Dead-code proof | `docs/release/DEAD_CODE_AUDIT.md` |
| Artifact audit | `docs/release/MODEL_AND_DATASET_AUDIT.md` |
| Licensing | `docs/release/LICENSE_AND_ATTRIBUTION_MATRIX.md` |
| Website status | `docs/release/WEBSITE_DEPLOYMENT.md` |
| Dataset measurement | `docs/release/codersec_dataset_quality_v1.json` |

## 4. Remaining concerns

1. **GitHub Pages not enabled.** Requires manual repository setting before the site can be publicly verified.
2. **Repository metadata not applied.** PAT lacks permission for repository settings; description, homepage and topics need manual configuration or an authorised token.
3. **CoderSec checkpoints absent.** No adapter or merged weights exist publicly; the site says so.
4. **CoderSec SFT defects.** Malformed ChatML, placeholder code and duplicates are measured but the published revision is preserved. A corrected v1.1 must be separately versioned.
5. **License review items.** AYA upstream revision, OWASP/MITRE/NIST document-level terms and tokenizer artefact classification need maintainer/legal review.
6. **No new release tag.** v2.0.1 should be created only after the above blockers and final approval.

## 5. Recommendation

Approve the engineering changes in PR #5. Do **not** merge or create a release tag until:

- GitHub Pages is enabled and the public URL returns HTTP 200.
- Repository description, homepage and topics are applied.
- A final review accepts the disclosed CoderSec dataset quality issues and license review items.

The recommended next release is a patch release **v2.0.1** (documentation, engineering and metadata only; no model-weight changes).
