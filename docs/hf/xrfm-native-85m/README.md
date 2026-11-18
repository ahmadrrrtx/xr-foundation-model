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

# XRFM Native 85M

**Track A — XRFM Native.** An original decoder-only transformer trained from random initialization for open, reproducible multilingual representation research.

> **Lineage disclosure:** No third-party pretrained weights were loaded. This checkpoint was initialized randomly and trained on `ahmadrrrtx/xrfm-native-shards-v1`. It is an 86.4M-parameter research model, not a production coding assistant.

## Model details

| Field | Value |
|---|---|
| Developer | M. Ahmad (RRRTX Labs), handle `ahmadrrrtx` |
| Model name | XRFM Native 85M |
| Revision documented | `dbe5f102e01c4e2ea4d07e92615ce4cb40b8644c` |
| Architecture | `XRFMModel`: decoder-only transformer with grouped-query attention, RoPE, RMSNorm, SwiGLU |
| Tied parameters | 86,446,464 |
| Configuration | `d_model=640`, `n_layers=14`, `n_heads=10`, `n_kv_heads=2`, `d_ff=1920`, `max_seq_len=2048`, `rope_theta=100000` |
| Vocabulary | 32,768 |
| Tokenizer | T4, Script-Protected Byte-Level BPE (`tokenizer_32k_multilingual.json`) |
| Training dataset | `ahmadrrrtx/xrfm-native-shards-v1`, revision `1652affcb1e95455a6af27b72ab424116bb9884f` |
| Training tokens processed | 10,485,760 over 160 steps |
| Hardware | 1–2 NVIDIA Tesla T4; 994.72 s; peak VRAM 2,580.41 MB; PyTorch 2.11.0+cu130 |
| Checkpoint behaviour | Exact-resume probe: max weight difference 0.0 |
| Languages / scripts | English, Urdu (Perso-Arabic and Roman), Arabic, Hindi, Spanish, French, plus code |
| License | Apache-2.0 for these weights, as declared by the developer |

The 85M run was a shorter scaling probe than the 25M run: it processed fewer tokens and achieved higher BPB on every domain. It should not be described as a stronger general model solely because it has more parameters.

## Evaluation

BPB is bits per byte of held-out text/code under the model; lower is better. These are per-domain measurements with no aggregate weighting. BPB measures compression/likelihood fit, not programming or cybersecurity capability.

| Domain | Step 0 (random init) | Step 160 | Δ BPB |
|---|---:|---:|---:|
| en | 3.4799 | 2.2930 | -1.1869 |
| code | 4.0077 | 2.1978 | -1.8099 |
| ur | 2.6492 | 1.8611 | -0.7881 |
| ur-Latn | 3.7126 | 2.3555 | -1.3571 |
| ar | 2.6961 | 1.9244 | -0.7717 |
| hi | 1.8626 | 1.2443 | -0.6183 |
| es | 4.0712 | 2.7739 | -1.2973 |
| fr | 4.0820 | 2.8644 | -1.2176 |

Final training loss: 6.8362. The 20,080,975 dataset tokens are unique prepared tokens; 10,485,760 is the number processed in this run.

## Intended use, limits and safety

Use for architecture research, scaling comparisons, multilingual tokenization studies and checkpoint/reproducibility experiments. Out of scope: production assistants, security advice, high-stakes decision making and unmoderated deployment. The model is not instruction-tuned and may generate incorrect, biased or unsafe text. Evaluate outputs independently.

## Reproduction

```bash
git clone https://github.com/ahmadrrrtx/xrf-foundation-model.git
cd xr-foundation-model
python -m pip install -e '.[dev,hub]'
python scripts/train_native_from_hub.py --config config/native-85m.yaml
python scripts/ci_smoke_test.py
```

Dependencies: Python 3.10–3.12, PyTorch 2.0+, `tokenizers`, `safetensors`, PyYAML, NumPy and the T4 tokenizer file.

## References and citation

- [Project repository](https://github.com/ahmadrrrtx/xrf-foundation-model)
- [Official website](https://ahmadrrrtx.github.io/xr-foundation-model/)
- [Dataset audit](https://github.com/ahmadrrrtx/xrf-foundation-model/blob/main/docs/release/MODEL_AND_DATASET_AUDIT.md)

Cite the source repository's `CITATION.cff`; no DOI or peer-reviewed publication is claimed.
