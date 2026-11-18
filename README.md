# XR Foundation Model (XRFM)

[![CI](https://github.com/ahmadrrrtx/xr-foundation-model/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmadrrrtx/xr-foundation-model/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Release](https://img.shields.io/github/v/release/ahmadrrrtx/xr-foundation-model)](https://github.com/ahmadrrrtx/xr-foundation-model/releases)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fahmadrrrtx.github.io%2Fxr-foundation-model%2F)](https://ahmadrrrtx.github.io/xr-foundation-model/)

**Open-source multilingual foundation-model research with an original transformer architecture and a specialized coding and defensive cybersecurity track.**

Official website: <https://ahmadrrrtx.github.io/xr-foundation-model/>

XRFM is a research project, not a production assistant. It contains two independent model lineages that share tooling and evaluation practices but never share weights or training history.

## Technical overview

- **Track A — XRFM Native:** decoder-only transformers developed by this project and trained from random initialization on a multilingual/code corpus.
- **Track B — XRFM CoderSec:** a defensive-coding track based on the explicitly attributed `Qwen/Qwen3-1.7B` base model and QLoRA fine-tuning. No CoderSec checkpoint is publicly available at present.

The canonical Python package is `src/xrfm/`. The wheel ships only the `xrfm` package; API, deployment and training scripts remain repository tools.

## Published models

| Model | Lineage | Parameters | Tokenizer | Artifact |
|---|---|---:|---|---|
| XRFM Native 25M | Original, random initialization | 25,200,000 | T4 32K Byte-BPE | [Hugging Face](https://huggingface.co/ahmadrrrtx/xrfm-native-25m) |
| XRFM Native 85M | Original, random initialization | 86,446,464 | T4 32K Byte-BPE | [Hugging Face](https://huggingface.co/ahmadrrrtx/xrfm-native-85m) |
| XRFM CoderSec | Qwen3-1.7B base + QLoRA | ~1.7B base | Qwen tokenizer | **Not published** |

Native 25M was trained for 900 steps over 58,982,400 processed tokens; Native 85M for 160 steps over 10,485,760 tokens. Both used NVIDIA Tesla T4 hardware and reported a zero-difference exact-resume probe. The 85M run is a shorter scaling probe and has higher final BPB than 25M across all reported domains.

## Datasets

| Dataset | Revision | Actual content | License |
|---|---|---|---|
| [xrfm-native-shards-v1](https://huggingface.co/datasets/ahmadrrrtx/xrfm-native-shards-v1) | `1652affc` | 8 train shards, 9,577 sequences, 20,080,975 prepared tokens, 623 validation docs | ODC-By, mixed source licenses |
| [xrfm-codersec-sft-40k](https://huggingface.co/datasets/ahmadrrrtx/xrfm-codersec-sft-40k) | `808d7505` | 24,100 rows (the name is a target, not a count) | `other`; row licenses Apache-2.0 / CC0-1.0 / MIT |

The CoderSec dataset card and [quality report](https://huggingface.co/datasets/ahmadrrrtx/xrfm-codersec-sft-40k/blob/main/QUALITY_REPORT.json) disclose measured defects: malformed ChatML in all rows, placeholder code in all 2,100 code rows and 11,993 exact duplicates. The original revision is preserved because documented QLoRA experiments used it.

## Architecture summary

Native models use the original `XRFMModel`:

- Decoder-only transformer with grouped-query attention
- Rotary position embeddings (RoPE)
- RMSNorm and SwiGLU feed-forward blocks
- Causal masking and KV-cached generation
- 2,048-token context in the published configurations
- T4 Script-Protected Byte-Level BPE with 32,768 vocabulary entries

CoderSec uses Qwen3-1.7B with NF4 + FP16 QLoRA, adapter training and merge scripts in `src/xrfm/codersec/`. It is not an XRFM Native architecture.

## Languages and scripts

Native pretraining and evaluation cover English, code, Urdu (Perso-Arabic), Roman Urdu, Arabic, Hindi, Spanish and French. The CoderSec SFT rows are labelled `en`, `urd`, `arb` and `spa`.

## Tokenizer

The T4 tokenizer was selected in a four-way multilingual and code tournament. The winning candidate used 32,768 Script-Protected Byte-BPE tokens with atomic special tokens, 0% byte fallback and lossless round-trip in the recorded evaluation. The legacy 2K vocabulary and full tournament harness are preserved for reproducibility.

## Training methodology

Native training used causal language modelling on `uint16` shards with 2,048 tokens and a next-token label. Deduplication used SHA-256 and MinHash-LSH 8×4; contamination was checked against a frozen benchmark manifest. CoderSec experiments used QLoRA on a Qwen3-1.7B base with SFT data revision `808d7505`, followed by internal evaluation and red-team notebooks.

## Evaluation methodology

Native evaluation reports per-domain bits per byte (BPB), lower is better. BPB measures likelihood/compression fit to held-out data; it does **not** prove programming or cybersecurity capability. There is no weighted aggregate score. CoderSec reports small internal red-team suites (8 weaponization prompts, 7 false-refusal prompts, 4 Roman-Urdu perturbation pairs); these are preliminary and cannot establish population-level safety.

## Installation

```bash
git clone https://github.com/ahmadrrrtx/xr-foundation-model.git
cd xr-foundation-model
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,hub]'
```

Python 3.10, 3.11 and 3.12 are tested in CI. The PyPI name `xrfm` belongs to an unrelated tabular-kernel project; install from source, not from PyPI.

## Inference example

```python
import xrfm

cfg = xrfm.load_config("config/tiny.yaml")
model = xrfm.XRFM(cfg)
tok = xrfm.BPETokenizer.pretrained()
result = xrfm.generate(model, "Once upon a time", tokenizer=tok, max_new_tokens=64)
print(result)
```

Published Native checkpoints are research models, not instruction-tuned chat assistants.

## Reproduction

```bash
python scripts/ci_smoke_test.py
python scripts/verify_reproducibility.py
python scripts/train_native_from_hub.py --config config/native-25m.yaml
```

See [documentation](https://ahmadrrrtx.github.io/xr-foundation-model/docs/) and the [experiment registry](docs/experiments/) for pinned revisions and hardware notes.

## Data provenance

Native shards derive from FineWeb-Edu, FineWeb-2, OpenCoder Stage-2 and AI4Bharat Sangraha with pinned revisions and source licenses. CoderSec SFT rows derive from CyberNative, AYA, a project-authored security knowledge pack and OpenCoder Stage-2. OWASP, MITRE and NIST identifiers are cited with document-level attribution requirements; no blanket SPDX claim is made for those external resources.

## Licensing

Repository source is MIT. Native model weights are Apache-2.0. The Native dataset is ODC-By with mixed upstream licenses. The CoderSec base model is Apache-2.0; its SFT dataset uses mixed row-level licenses and is marked `other`. See the [artifact-specific license matrix](docs/release/LICENSE_AND_ATTRIBUTION_MATRIX.md).

## Limitations

- Small experimental models with limited training data and no instruction tuning.
- BPB improvements are not evidence of coding, reasoning or security capability.
- CoderSec weights are not public; its red-team results use very small samples.
- The published CoderSec SFT dataset contains known serialization and content defects.
- No formal privacy or safety audit has been published.

## Security policy

Report vulnerabilities through the process in [SECURITY.md](SECURITY.md). Do not open public issues for sensitive findings.

## Citation

A machine-readable [`CITATION.cff`](CITATION.cff) is maintained in the repository. Cite the exact model and dataset revisions used. No DOI, journal publication or scholarly affiliation is claimed.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Preserve the Native/CoderSec distinction, tokenizer research and artifact provenance in all changes.

## Documentation

- [Official website](https://ahmadrrrtx.github.io/xr-foundation-model/)
- [Architecture](docs/architecture.md)
- [Model and dataset audit](docs/release/MODEL_AND_DATASET_AUDIT.md)
- [Dependency map](docs/release/DEPENDENCY_MAP.md)
- [Release reports](docs/release/)
