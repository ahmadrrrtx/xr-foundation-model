---
license: other
language:
  - en
  - ur
  - ar
  - es
pretty_name: XRFM CoderSec SFT Dataset (v1)
task_categories:
  - question-answering
  - text-generation
tags:
  - xrfm
  - codersec
  - cybersecurity
  - multilingual
  - sft
---

# XRFM CoderSec SFT Dataset (published v1)

**Published revision:** `808d7505152eea93e6cd4af5f2269f43712afc2d`
**Actual size:** 24,100 records. The `40k` in the repository name is a target, not the published row count.

This file is the exact dataset used by the documented Track B QLoRA experiments. It is preserved unchanged for reproducibility. A quality audit found material defects; see **Known issues** before training on it.

## Schema

Each JSONL row contains:

| Field | Type | Description |
|---|---|---|
| `record_id` | string | Stable record identifier |
| `source` | string | `cybernative`, `aya`, `xrfm_security_kb` or `opencoder_stage2` |
| `language` | ISO 639-3-ish code | `en`, `urd`, `arb` or `spa` |
| `chatml` | string | Three-role system/user/assistant conversation |
| `domain` | string | `defensive_security`, `multilingual_qa`, `rag_citation` or `code` |
| `license` | SPDX string | Row-level source license |
| `license_tier` | string | `tier_a` or `tier_b` provenance tier |
| `code_blocks` | list | Extracted code blocks with language and code text |
| `provenance` | object | `source_uri` pointing to the upstream dataset |

## Distribution (measured from the published file)

| Dimension | Count |
|---|---:|
| Total rows | 24,100 |
| CyberNative / defensive security | 12,000 |
| AYA / multilingual QA | 6,000 |
| Project security KB / RAG citation | 4,000 |
| OpenCoder Stage-2 / code | 2,100 |
| Unique ChatML records | 12,107 |
| Exact duplicate rows | 11,993 |
| Rows with code blocks | 2,100 |
| Empty assistant responses | 0 |
| Provenance gaps | 0 |

Row licenses: Apache-2.0 (18,000), CC0-1.0 (4,000), MIT (2,100). The dataset-level license metadata is `other` because no single license covers all rows.

## Sources and provenance

- `CyberNative/Code_Vulnerability_Security_DPO` — Apache-2.0, revision `1788f0a6…`.
- AYA multilingual QA rows — Apache-2.0 (row-level declaration; upstream revision verification is a maintainer-review item).
- Project-authored `xrfm_security_kb` defensive-security snippets — CC0-1.0.
- `OpenCoder-LLM/opc-sft-stage2` — MIT, revision `7d28f40d…`.

Security knowledge references cite OWASP ASVS, MITRE CWE and NIST SP 800-53 identifiers. Reuse terms for those external knowledge resources must be checked per document; no blanket SPDX claim is made here. See the license matrix in the source repository.

## Filtering and validation performed by the publisher

The generating script `scripts/build_and_publish_codersec_sft.py` applied source-tier filtering, contamination checks against `configs/eval/frozen_benchmarks_v1.json`, a weaponization screen and a 40,000-example target. The published result was 24,100 validated records; strict validation rejected the remainder.

## Known issues (measured, not inferred from a single sample)

1. **Malformed ChatML in 24,100 / 24,100 rows.** Canonical `<|im_start|>` and `<|im_end|>` markers are absent. The file uses corrupted `licsystem`, `licuser`, `licassistant` and `irit` markers instead.
2. **Placeholder implementations in 2,100 / 2,100 code rows.** Every code block is a `def func_<id>(x, y)` stub returning `result = x + y  # placeholder`, even for tasks such as sorting a list or reading a CSV file. The snippets are syntactically valid but do not solve the stated task.
3. **High duplication.** 11,993 exact duplicate rows; the 12,000 CyberNative rows collapse to 7 unique responses. The prompt "Write secure Python code for user authentication." appears 3,000 times.
4. **No evaluation of downstream utility.** Presence in this dataset does not prove that a model learned secure coding behaviour.

A bounded regex scan found no obvious emails, AWS keys, private-key headers, bearer tokens or Hugging Face token-like strings. This is not a formal privacy audit.

## Reproducible correction

The source repository provides `scripts/filter_codersec_sft.py`. It canonicalizes ChatML, drops placeholder-only code rows and deduplicates. On the published revision it produces 10,007 records (6,000 AYA, 4,000 security KB, 7 unique CyberNative records) and writes a manifest. The corrected output has **not** been published under this repository; it must be released as a separately versioned dataset so the checkpoint-training data remains unchanged.

## Usage example

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="https://huggingface.co/datasets/ahmadrrrtx/xrfm-codersec-sft-40k/resolve/main/data/data.jsonl")
print(ds["train"][0])
```

## Citation and limits

Cite the XRFM source repository's `CITATION.cff`. No DOI or peer-reviewed publication is claimed. Do not represent this dataset as 40,000 records and do not use it as evidence of production-grade cybersecurity capability.
