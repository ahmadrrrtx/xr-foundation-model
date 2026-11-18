---
license: apache-2.0
language:
  - en
  - ur
  - ar
  - hi
  - es
  - fr
pipeline_tag: text-generation
library_name: pytorch
tags:
  - xrfm
  - from-scratch
  - multilingual
  - roman-urdu
  - code
  - decoder-only
---

# XRFM Native 25M

**Track A — XRFM Native.** An original decoder-only transformer trained from random initialization for open, reproducible multilingual representation research.

> **Lineage disclosure:** This checkpoint did **not** load any third-party pretrained weights. All parameters were randomly initialized and trained on `ahmadrrrtx/xrfm-native-shards-v1`. It is a 25.2M-parameter research model, not a production coding assistant.

## Model details

| Field | Value |
|---|---|
| Developer | M. Ahmad (RRRTX Labs), handle `ahmadrrrtx` |
| Model name | XRFM Native 25M |
| Revision documented | `3487b9c80612e735d9766d557e7e4e34d672efa0` |
| Architecture | `XRFMModel`: decoder-only transformer with grouped-query attention, RoPE, RMSNorm, SwiGLU |
| Tied parameters | 25,200,000 |
| Configuration | `d_model=384`, `n_layers=8`, `n_heads=6`, `n_kv_heads=2`, `d_ff=1024`, `max_seq_len=2048`, `rope_theta=100000` |
| Vocabulary | 32,768 |
| Tokenizer | T4, Script-Protected Byte-Level BPE (`tokenizer_32k_multilingual.json`) |
| Training dataset | `ahmadrrrtx/xrfm-native-shards-v1`, revision `1652affcb1e95455a6af27b72ab424116bb9884f` |
| Training tokens processed | 58,982,400 over 900 steps |
| Hardware | 1–2 NVIDIA Tesla T4; 1,572.03 s; peak VRAM 3,865.93 MB; PyTorch 2.11.0+cu130 |
| Checkpoint behaviour | Exact-resume probe: uninterrupted final loss 10.359375, resumed 10.359375, max weight difference 0.0 |
| Languages / scripts | English, Urdu (Perso-Arabic and Roman), Arabic, Hindi, Spanish, French, plus code |
| License | Apache-2.0 for these weights, as declared by the developer |

## Training methodology

Random initialization followed by causal language modelling on 2,048-token sequences with next-token labels. Training used the committed `config/native-25m.yaml` configuration. The dataset contains 20,080,975 unique prepared tokens (19,613,696 train, 467,279 validation); the larger training-token figure above counts tokens processed across repeated passes.

## Evaluation

Bits-per-byte (BPB) is the average number of bits used to encode each byte of the held-out validation domain under the model. Lower is better. It is a compression/likelihood measure of fit to the evaluation data; it does **not** demonstrate programming ability, instruction following, factual accuracy or defensive-security competence. Measurements are per-domain; no aggregate score is weighted or reported.

| Domain | Step 0 (random init) | Step 900 | Δ BPB |
|---|---:|---:|---:|
| en | 3.4651 | 1.8152 | -1.6499 |
| code | 4.0011 | 0.9966 | -3.0045 |
| ur | 2.6101 | 1.3674 | -1.2427 |
| ur-Latn | 3.6980 | 1.8102 | -1.8878 |
| ar | 2.7293 | 1.5708 | -1.1585 |
| hi | 1.8358 | 0.9211 | -0.9147 |
| es | 4.0700 | 2.1618 | -1.9082 |
| fr | 4.0781 | 2.1611 | -1.9170 |

Final training loss: 5.3325. Scaling-probe results published elsewhere in the repository are separate experiments and are not represented by this checkpoint.

## Intended use

- Architectural and tokenizer research.
- Small-scale multilingual language-modelling experiments.
- Reproducibility studies on checkpointing, resume behaviour and tokenizer fertility.

## Out of scope

- Production chat assistants or coding agents.
- Defensive-security decision support.
- High-stakes domains, medical/legal/financial advice, or unmoderated public deployment.
- Claims of parity with large instruction-tuned models.

## Limitations, safety and bias

The model is small, trained on a limited web/code corpus and not instruction-tuned. It can produce fluent but incorrect text, reproduce stereotypes or source bias, and has no built-in safety alignment. It does not perform reliable code review or vulnerability remediation. Users must evaluate outputs independently.

## Reproduction

```bash
git clone https://github.com/ahmadrrrtx/xrf-foundation-model.git
cd xr-foundation-model
python -m pip install -e '.[dev,hub]'
python scripts/train_native_from_hub.py --config config/native-25m.yaml
python scripts/ci_smoke_test.py
```

Dependencies: Python 3.10–3.12, PyTorch 2.0+, `tokenizers`, `safetensors`, PyYAML, NumPy and the T4 tokenizer file. Exact environment details are in `training_telemetry.json`.

## References

- [Project repository](https://github.com/ahmadrrrtx/xrf-foundation-model)
- [Official website](https://ahmadrrrtx.github.io/xr-foundation-model/)
- [Tokenizer tournament record](https://github.com/ahmadrrrtx/xrf-foundation-model/blob/main/docs/experiments/tokenizer_tournament_v1.json)
- [Dataset audit](https://github.com/ahmadrrrtx/xrf-foundation-model/blob/main/docs/release/MODEL_AND_DATASET_AUDIT.md)

**Citation:** see `CITATION.cff` in the source repository. Do not invent a DOI or scholarly publication.
