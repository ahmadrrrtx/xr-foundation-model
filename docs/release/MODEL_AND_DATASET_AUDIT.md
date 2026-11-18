# Model and Dataset Audit

**Audit method:** Hugging Face REST metadata, file inventories, `datasets-server` size/splits endpoints, bounded JSONL download of the 14.5 MB SFT file, and small config/telemetry files. No full model weights or large corpora were downloaded locally.
**Date:** 2026-11-19

## 1. Published artefacts discovered

### Models owned by `ahmadrrrtx`

| Repository | Revision (main) | Files | License metadata |
|---|---|---|---|
| `ahmadrrrtx/xrfm-native-25m` | `3487b9c80612e735d9766d557e7e4e34d672efa0` | `model.safetensors` (100.8 MB), `model_config.json`, `tokenizer_32k_multilingual.json`, `training_telemetry.json`, `README.md` | `apache-2.0` |
| `ahmadrrrtx/xrfm-native-85m` | `dbe5f102e01c4e2ea4d07e92615ce4cb40b8644c` | `model.safetensors` (345.8 MB), config/tokenizer/telemetry/card | `apache-2.0` |
| `ahmadrrrtx/scriptshield-tiny-aya-fire-steering` | present | unrelated prior experiment | `cc-by-nc-4.0` |

No CoderSec model repository exists on the account. The following identifiers were checked and returned HTTP 404: `ahmadrrrtx/xrfm-codersec-qlora`, `ahmadrrrtx/xrfm-codersec-flagship-merged`, `ahmadrrrtx/xrfm-codersec-1.7b`, `ahmadrrrtx/xrfm-codersec`. Repository records explicitly say publication was **deferred** because the Kaggle kernel had no `HF_TOKEN` secret. Current status: **Track B weights are not publicly available**; the website and README must not imply otherwise.

### Datasets owned by `ahmadrrrtx`

| Repository | Revision | Actual public data | License metadata |
|---|---|---|---|
| `ahmadrrrtx/xrfm-native-shards-v1` | `1652affcb1e95455a6af27b72ab424116bb9884f` | 28 files, 45.27 MB: 8 `.npy` train shards, 8 language validation `.npy` + 8 `.jsonl` files, `manifest.json`, tokenizer, card | `odc-by` |
| `ahmadrrrtx/xrfm-codersec-sft-40k` | `808d7505152eea93e6cd4af5f2269f43712afc2d` | 2 files: `data/data.jsonl` (14.48 MB) and `.gitattributes` | **no license tag** |

## 2. Native shards — verification

The dataset viewer reports 623 rows because it indexes only the language validation JSONL files. The training files are nevertheless present in the repository and listed in the manifest:

- 8 train shards: `tokens/train_shard-00000-of-00008.npy` … `00007`.
- 9,577 training sequences of 2,049 `uint16` values (2,048 tokens + next-token label).
- 19,613,696 train tokens; 467,279 validation tokens; 20,080,975 total.
- 29,491 source documents; 28,868 train / 623 validation, split by SHA-256 hash with seed 42.
- Deduplication: SHA-256 + MinHash LSH 8×4. Decontamination manifest: `configs/eval/frozen_benchmarks_v1.json`.
- Sources: FineWeb-Edu and FineWeb-2 (ODC-By-1.0), OpenCoder Stage-2 (MIT), AI4Bharat Sangraha (CC-BY-4.0).
- Per-language validation files: `en`, `ur`, `ur-Latn`, `ar`, `hi`, `es`, `fr`, `code`.

The name "40k" does not apply to this dataset; that discrepancy belongs to the CoderSec SFT repository.

## 3. Native model cards — verified figures

Both cards disclose random initialization and explicitly state that no third-party pretrained weights were loaded.

| Field | Native 25M | Native 85M |
|---|---|---|
| Tied parameters | 25,200,000 | 86,446,464 |
| Architecture | `XRFMModel`, GQA, RoPE, RMSNorm, SwiGLU | same |
| Vocabulary | 32,768 (T4 Byte-BPE) | 32,768 (T4 Byte-BPE) |
| Steps | 900 | 160 |
| Processed tokens | 58,982,400 | 10,485,760 |
| Hardware | Tesla T4, 1,572.03 s, peak 3,865.93 MB | Tesla T4, 994.72 s, peak 2,580.41 MB |
| Exact-resume probe | max weight diff 0.0 | max weight diff 0.0 |
| Final train loss | 5.3325 | 6.8362 |

BPB is **bits per byte** of held-out validation text/code: lower is better and is a compression/likelihood measure, not a coding or cybersecurity capability score. Measurements are per-domain, not a single weighted aggregate. Step 0 is the randomly initialized baseline; final is after the listed steps. The 20.08M tokens in the dataset are **unique prepared dataset tokens**, while 58.98M/10.49M are **tokens processed during training** across repeated passes. Scaling-probe observations elsewhere in the repository are separate experiments and must not be presented as these checkpoints' results.

## 4. CoderSec SFT dataset — measured quality

Full-file measurement of revision `808d7505` is recorded in `docs/release/codersec_dataset_quality_v1.json`.

| Measure | Result |
|---|---|
| Actual rows | 24,100 (not 40,000) |
| Sources | CyberNative 12,000; AYA 6,000; `xrfm_security_kb` 4,000; OpenCoder Stage-2 2,100 |
| Domains | defensive_security 12,000; multilingual_qa 6,000; rag_citation 4,000; code 2,100 |
| Languages | en 18,100; ur 2,000; arb 2,000; spa 2,000 |
| Row licenses | Apache-2.0 18,000; CC0-1.0 4,000; MIT 2,100 |
| Provenance gaps | 0 |
| Empty assistant responses | 0 |
| Python syntax errors in code blocks | 0 of 2,100 parsed blocks |

Material defects found:

1. **Malformed ChatML in all 24,100 rows.** Canonical `<|im_start|>` / `<|im_end|>` markers occur 0 times. Instead every row uses corrupted markers `licsystem`/`licuser`/`licassistant` and `irit` (24,100 rows). Three-role structure is otherwise present.
2. **Placeholder code in all 2,100 OpenCoder rows.** Every code block is `def func_<id>(x, y)` returning `result = x + y  # placeholder`, regardless of the task ("sort a list", "implement a stack", "read a CSV file"). That is 100% of code rows and 8.7% of the dataset. The snippets are syntactically valid but semantically wrong.
3. **Heavy duplication.** 11,993 duplicate rows by exact ChatML hash; only 12,107 unique records. All duplicates originate from CyberNative (12,000 rows collapse to 7 unique responses); top prompt "Write secure Python code for user authentication." appears 3,000 times.
4. **No HF dataset card.** The repository contains only `.gitattributes` and `data/data.jsonl`; no README/frontmatter, so license and schema are undiscoverable through the dataset page.

Heuristic PII/secret scans found 0 email addresses, 0 AWS access keys, 0 private-key headers, 0 bearer tokens and 0 Hugging Face token-like strings. This is a bounded regex check, not a formal privacy audit.

### Required remediation

- Preserve revision `808d7505` unchanged: it is the data used by the documented QLoRA experiments.
- Publish a card for the existing revision that reports the defects honestly.
- Prepare a separate corrected revision/dataset (v1.1) with canonical ChatML and the 2,100 placeholder-only code rows removed (22,000 records remaining). Do not overwrite the original data file.
- Do not claim Track B fine-tuning produced a usable model until checkpoints are published.

## 5. Link status

All model and dataset links used by the website/README resolve over HTTPS. CoderSec checkpoint links are intentionally omitted; a "status: not published" statement is used instead.
