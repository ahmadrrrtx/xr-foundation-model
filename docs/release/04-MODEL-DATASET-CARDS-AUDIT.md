# Phase Four — Model Cards + Dataset Cards Audit

**Status:** Complete. All cards audited and verified accurate.

## Audited Files

| File | Status | Notes |
|---|---|---|
| `MODEL_CARD_NATIVE.md` | VERIFIED | Track A: Native 25M/85M, from scratch, BPB curves, contamination checks |
| `MODEL_CARD_CODERSEC.md` | VERIFIED | Track B: CoderSec QLoRA on Qwen3-1.7B, red-teaming results, bake-off comparison |
| `DATASET_CARD.md` | VERIFIED | Native shards (20.08M tokens) + CoderSec SFT (24,100 validated examples) |

## Model Card Details

### MODEL_CARD_NATIVE.md

- **Developer:** XRFM project
- **Version:** 1.0.0 (matches package versioning)
- **Variants:** XRFM Native 25M (~25.2M params, commit `3487b9c8`), XRFM Native 85M (~86.4M params, commit `dbe5f102`)
- **Training Data:** 20.08M decontaminated tokens, 8 domains, L+1 causal uint16 shards
- **HF Repos:** `ahmadrrrtx/xrfm-native-25m`, `ahmadrrrtx/xrfm-native-85m`
- **Evaluation:** Strict monotonic BPB decrease across all 8 domains
- **Safety:** Base models, may generate harmful content, downstream safety layering required

### MODEL_CARD_CODERSEC.md

- **Developer:** XRFM project
- **Version:** 1.0.0
- **Base Model:** Qwen/Qwen3-1.7B (Apache-2.0)
- **Method:** QLoRA (NF4 + FP16, r=16, α=32)
- **SFT Dataset:** `ahmadrrrtx/xrfm-codersec-sft-40k` (24,100 validated examples)
- **Bake-off:** Qwen3-1.7B won (0.8611 avg vs 0.8333 for Qwen2.5-Coder-1.5B-Instruct)
- **Red-Teaming (GATE-6):** 0.0% weaponization, 0.0% false-refusal, 100% Roman-Urdu stability
- **Intended Use:** Defensive-security coding assistant only

### DATASET_CARD.md

- **Native Shards:** `ahmadrrrtx/xrfm-native-shards-v1`, 20,080,975 tokens, 9,577 sequences, 8 shards
- **CoderSec SFT:** `ahmadrrrtx/xrfm-codersec-sft-40k`, 24,100 validated examples, 0% contamination, 0% weaponization, 100% tier-A/B provenance
- **Validation Pipeline:** ChatML schema, assistant-only loss masking, AST verification, PII scan, 13-gram contamination, anti-weaponization

## Links Validated (Phase Eight)

| Link | Status |
|---|---|
| huggingface.co/ahmadrrrtx/xrfm-native-25m | VALID |
| huggingface.co/ahmadrrrtx/xrfm-native-85m | VALID |
| huggingface.co/datasets/ahmadrrrtx/xrfm-codersec-sft-40k | VALID |
| huggingface.co/datasets/ahmadrrrtx/xrfm-native-shards-v1 | VALID |

## Improvements Made

None required — cards were already complete and accurate. All links point to live HF resources. Version numbers consistent with package versioning.

## Dual-Track Lineage Preserved

Track A (XRFM Native, from random init) and Track B (CoderSec, QLoRA on Qwen3-1.7B) are clearly separated and never combined.
