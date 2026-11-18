# Artifact-Specific License and Attribution Matrix

**Date:** 2026-11-19
**Policy:** A source-code license does not determine model-weight or dataset licensing. No artifact has been relicensed merely for visual consistency. Metadata is changed only where the actual basis is established; uncertain cases are marked for maintainer review.

| Artifact | Current declaration | Verified basis | Status / action |
|---|---|---|---|
| GitHub repository source code | `LICENSE` = MIT | Repository root `LICENSE`; `pyproject.toml` `license = "MIT"` | Consistent. Keep. |
| Python distribution | name `xrfm`, version 1.0.1, license MIT | `pyproject.toml`; PyPI `xrfm` 0.4.5 is an unrelated tabular-kernel project | Do not publish to PyPI without renaming the distribution. Import name remains `xrfm`. |
| `ahmadrrrtx/xrfm-native-25m` weights | Hugging Face `license: apache-2.0` | Model card frontmatter and repo tag | Consistent with developer's declaration; keep. Card must state random initialization. |
| `ahmadrrrtx/xrfm-native-85m` weights | Hugging Face `license: apache-2.0` | Model card frontmatter and repo tag | Consistent; keep. |
| `ahmadrrrtx/xrfm-native-shards-v1` | Hugging Face `license: odc-by` | Card frontmatter and `manifest.json` | Aggregate dataset declaration ODC-By-1.0. Underlying sources are mixed: FineWeb-Edu/FineWeb-2 ODC-By-1.0, OpenCoder Stage-2 MIT, AI4Bharat Sangraha CC-BY-4.0. Card must list all source licenses. |
| Track A tokenizer artefacts | No separate SPDX file | `tokenizer_32k_multilingual.json`, `vocab.json`; distributed inside MIT-licensed source and inside Apache-2.0 model repos | Documented as project artefacts. Legal classification of tokenizer data remains a **maintainer-review item**; no new license asserted. |
| CoderSec base model | `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, Hugging Face `license: apache-2.0` | Verified through Hugging Face API | Track B lineage must be described as Qwen3-1.7B base + QLoRA, not as original/from-scratch. |
| CoderSec SFT dataset | **No license tag on HF repo** | Row-level: Apache-2.0 18,000; CC0-1.0 4,000; MIT 2,100 | Metadata gap. Set card frontmatter to `license: other` with explicit row-level breakdown; do not claim a single permissive license. Original revision retained. |
| CyberNative security rows | Row license Apache-2.0; upstream `CyberNative/Code_Vulnerability_Security_DPO` tagged apache-2.0 | Hugging Face API + dataset manifest | Attribution retained in provenance field and dataset card. |
| AYA multilingual rows | Row license Apache-2.0 | Dataset manifest / row fields | Attribute to AYA source; verify the exact upstream revision before any redistribution beyond this dataset. **Maintainer review.** |
| OpenCoder Stage-2 code rows | MIT | Upstream repo tagged `license: mit`, revision `7d28f40d…` | Consistent. However all 2,100 derived rows are placeholder code and are proposed for removal in v1.1. |
| `xrfm_security_kb` RAG rows | CC0-1.0 | Row-level license field | Project-authored knowledge snippets; keep attribution in card. |
| OWASP knowledge resources | OWASP documents are generally CC BY-SA 4.0 (some Top 10 pages CC BY 3.0) | OWASP published license notices | Do not absorb into Apache/MIT weights. Cite the specific OWASP document used. Exact document-level mapping is a **maintainer-review item**. |
| MITRE CWE / ATLAS references | MITRE terms differ by resource and version | External resource notices | Attribute by identifier and revision; avoid blanket SPDX claim. **Maintainer review.** |
| NIST SP 800-series references | Usually U.S. Government works, but individual documents may carry different notices | NIST publication notices | Cite document number and revision; confirm per-document reuse terms. **Maintainer review.** |
| `ahmadrrrtx/scriptshield-tiny-aya-fire-steering` | `cc-by-nc-4.0` | Hugging Face API | Unrelated prior experiment. Must not be presented as part of XRFM Native or CoderSec. |

## Metadata changes made in this pass

- No Hugging Face license tag was changed.
- No weight or dataset file was overwritten.
- The repository code license remains MIT.
- The CoderSec dataset card will disclose its mixed row-level licensing rather than asserting a false single license.
- Open questions are listed above and remain visible to maintainers and users.
