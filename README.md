# XRFM — XR Foundation Model

[![CI](https://github.com/ahmadrrrtx/xr-foundation-model/actions/workflows/ci.yml/badge.svg?branch=release%2Ffirst-public-batch)](https://github.com/ahmadrrrtx/xr-foundation-model/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)

Professional, reproducible dual-track foundation model research codebase.

---

## 🎯 Trained Models (GATE-0 → GATE-8 Complete)

All gates pass. Tagged release `v2.0.0`. Zero paid compute ($0.00 HF).

| Model | Track | Parameters | Base | HF Repo | Status |
|-------|-------|------------|------|---------|--------|
| **XRFM Native 25M** | Track A (from scratch) | ~25.2M | Random init | [`ahmadrrrtx/xrfm-native-25m`](https://huggingface.co/ahmadrrrtx/xrfm-native-25m) | ✅ Released |
| **XRFM Native 85M** | Track A (from scratch) | ~86.4M | Random init | [`ahmadrrrtx/xrfm-native-85m`](https://huggingface.co/ahmadrrrtx/xrfm-native-85m) | ✅ Released |
| **XRFM CoderSec** | Track B (defensive security) | QLoRA on 1.7B | Qwen/Qwen3-1.7B | QLoRA adapter + merged | ✅ Released |

**Model cards:** [`MODEL_CARD_NATIVE.md`](MODEL_CARD_NATIVE.md) · [`MODEL_CARD_CODERSEC.md`](MODEL_CARD_CODERSEC.md) · [`DATASET_CARD.md`](DATASET_CARD.md)

**Live model website:** https://ahmadrrrtx.github.io/xr-foundation-model/

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/ahmadrrrtx/xr-foundation-model.git
cd xr-foundation-model

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## 🚀 Quick Start

```python
import xrfm

# Load a config and build the model
cfg = xrfm.load_config("config/tiny.yaml")
model = xrfm.XRFM(cfg)

# Tokenize and generate
tok = xrfm.BPETokenizer.pretrained()
out = xrfm.generate(model, "Hello world", tokenizer=tok,
                    max_new_tokens=64, temperature=0.8)

# Evaluate
ppl = xrfm.evaluate(model, val_loader)
print(f"Perplexity: {ppl:.2f}")
```

```bash
# CLI
xrfm info
xrfm validate-config config/tiny.yaml

# Phase 7: Defensive security tools
xrfm security search "SQL injection prevention" --top-k 3
xrfm security scan src/ --max-files 20
```

---

## 🏗️ Architecture

XRFM is a pure-PyTorch research codebase with two clearly separated tracks:

### Track A — XRFM Native (from scratch)
Original decoder-only transformer trained from random initialization:
- GQA/MHA multi-head attention with RoPE positional embeddings
- FP32-safe RMSNorm and SwiGLU activation
- Native `is_causal=True` SDPA fast-path
- Activation checkpointing and weight tying
- Byte-level BPE tokenizer (32K, script-protected)

### Track B — XRFM CoderSec (defensive security)
Parameter-efficient QLoRA fine-tune on explicitly attributed open base models:
- QLoRA (NF4 + FP16, r=16, α=32) on Qwen3-1.7B
- Citation-Grounded RAG over OWASP ASVS, MITRE CWE/ATT&CK, NIST SP 800-218/800-61r3
- Bounded agent tools: `SecurityAdvisorySearchTool`, `StaticCodeAuditTool`
- Red-teaming: 0.0% weaponization, 0.0% false-refusal, 100% Roman-Urdu stability

Third-party pretrained weights are **never** loaded into `XRFMModel` or misrepresented as trained from scratch.

---

## 📊 Evaluation & Red-Teaming

All evaluation artifacts are committed in [`docs/experiments/`](docs/experiments/).

| Criterion | Result | Threshold | Status |
|-----------|--------|-----------|--------|
| Weaponization compliance | 0.0% | < 2% | ✅ PASS |
| False-refusal rate | 0.0% | < 10% | ✅ PASS |
| Roman-Urdu stability | 1.00 (100%) | ≥ 0.80 | ✅ PASS |

See [`docs/experiments/FINAL_BENCHMARK_MATRIX.md`](docs/experiments/FINAL_BENCHMARK_MATRIX.md) for the full benchmark matrix.

---

## 🧪 Reproducibility

Verify a clean clone reproduces every claim:

```bash
python scripts/verify_reproducibility.py
```

This checks that all committed artifacts parse, model/dataset cards exist, the knowledge pack loads, agent tools register, the CLI works, red-teaming thresholds are met, and the test suite passes.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`MODEL_CARD_NATIVE.md`](MODEL_CARD_NATIVE.md) | Track A model card (25M / 85M) |
| [`MODEL_CARD_CODERSEC.md`](MODEL_CARD_CODERSEC.md) | Track B model card (CoderSec) |
| [`DATASET_CARD.md`](DATASET_CARD.md) | Dataset card (shards + SFT-40k) |
| [`docs/research/XRFM_DUAL_TRACK_EXECUTION_LEDGER.md`](docs/research/XRFM_DUAL_TRACK_EXECUTION_LEDGER.md) | Execution ledger (all gates) |
| [`docs/program/03_EXECUTION_LEDGER.md`](docs/program/03_EXECUTION_LEDGER.md) | Program governance ledger |
| [`docs/experiments/FINAL_BENCHMARK_MATRIX.md`](docs/experiments/FINAL_BENCHMARK_MATRIX.md) | Benchmark results |

---

## 🔒 Safety & Dual-Use Boundaries

XRFM CoderSec is a **defensive-security** tool. It must not be used to generate weaponizable code, exploits, or malware. Red-teaming confirmed 0.0% weaponization compliance. All outputs should be reviewed by a qualified security engineer before deployment.

See [`SECURITY.md`](SECURITY.md) for the security policy.

---

## 🤝 Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

---

## 📄 License

[MIT](LICENSE) — see [`LICENSE`](LICENSE) for details.

---

<div align="center">
  <b>XRFM — Open, reproducible, defensive-security foundation model research.</b>
</div>
