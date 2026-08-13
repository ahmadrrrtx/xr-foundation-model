# XRFM Frontier — Open-Source / Free Tooling Stack

**Date:** 2026-08-12
**Constraint:** Everything must be buildable with free/open tooling (Python/PyTorch/JAX/
Rust/C++/CUDA/Triton), legally usable, GPU-ready but CPU-developable. Verify license and
maintenance at integration time.

## 1. Core

| Need | Choice | License | Notes |
|---|---|---|---|
| Autograd framework | **PyTorch ≥ 2.2** | BSD-3 | Already used; `torch.compile`, SDPA, sparse tensors |
| Config | YAML + existing `ConfigLoader` | MIT (repo) | extend, don't replace |
| Tokenizer | existing byte-level BPE; optional **HF Tokenizers** (Rust) | Apache-2.0 | HF tokenizers for speed at scale |
| Dataset loading | **HF Datasets** + **datatrove** | Apache-2.0 | streaming, dedup, filtering at TB scale |
| Experiment tracking | JSONL (existing) + optional TensorBoard/W&B (offline) | Apache/MIT | start dependency-free |

## 2. Sequence mixers / SSM / linear attention

| Library | Provides | License | GPU | CPU fallback |
|---|---|---|---|---|
| `state-spaces/mamba` / `mamba2` | Mamba/Mamba-2 (SSD) | Apache-2.0 | CUDA kernels | slow pure-torch |
| `causal-conv1d` | fast causal conv used by Mamba | Apache-2.0 | CUDA | yes |
| **`fla` (flash-linear-attention)** | GLA, DeltaNet, Gated DeltaNet, RetNet, RWKV, Linear Attention, Mamba2 in Triton | MIT | NVIDIA (Triton) | limited |
| `BlinkDL/RWKV-LM` | RWKV v6/v7 | Apache-2.0 | CUDA | **good** (advantage) |
| HF `transformers` | Mamba2/GatedDeltaNet model defs | Apache-2.0 | various | yes |

**Recommendation:** use `fla` for Triton-accelerated linear-recurrent layers at scale; keep
pure-PyTorch reference implementations for CPU correctness tests and ablation.

## 3. Graph / sparse / hypergraph / TDA

| Library | Provides | License | GPU | Notes |
|---|---|---|---|---|
| **PyTorch Geometric (PyG)** | GNN message passing, sparse scatter, NeighborLoader | MIT | yes | mature; `torch-sparse`/`torch-scatter` |
| **DGL** | GNNs, graph batching, kernels | Apache-2.0 | yes | alternative to PyG |
| `torch.sparse` (native) | COO/CSR sparse tensors | BSD | partial | use for reference impls |
| **Triton** | custom fused kernels (top-k, gather-scatter, gated memory) | MIT | NVIDIA | write custom ops later |
| NetworkX / igraph | graph analysis (CPU, diagnostics) | BSD/GPL | no | diagnostics only, not training |
| **Gudhi** | persistent homology, Rips, simplex trees | MIT/GPLv3 (check) | no | TDA diagnostics |
| **giotto-tda** | scikit-tda, persistence | AGPL-3.0 (!) | no | **AGPL — avoid for core; only isolated analysis** |
| `scikit-tda` / `ripser.py` | PH, Ripser | MIT/licenses vary | no | diagnostic |

**Caution:** giotto-tda is AGPL — keep any TDA use in a separate, optional, non-redistributed
analysis tool, or prefer Gudhi/MIT options. Do not let AGPL touch the core model package.

## 4. Memory / retrieval

| Need | Choice | License |
|---|---|---|
| Associative/neural memory | custom (Gated DeltaNet-style) on PyTorch | repo MIT |
| Optional fast nearest neighbors | `faiss-cpu`/`faiss-gpu` | MIT |
| Dense retrieval (diagnostic) | `sentence-transformers` (Apache-2.0) | Apache-2.0 |

No external vector DB is part of the *architecture* (no RAG bolt-on). FAISS is only for
optional analysis/eval.

## 5. Training / distributed / optimization

| Need | Choice | License |
|---|---|---|
| Distributed | existing DDP/FSDP hooks; validate on NCCL | BSD |
| ZeRO/offload (later) | DeepSpeed or FSDP2 | Apache-2.0 / BSD |
| Triton kernels | custom + `fla` | MIT |
| Mixed precision | existing autocast (bf16 on GPU) | BSD |
| Scheduling | existing cosine+warmup | repo |

## 6. Evaluation / uncertainty

| Need | Choice | License |
|---|---|---|
| Intrinsic PPL/top-k | existing `evaluation/` | repo |
| External benchmarks | `lm-evaluation-harness` | MIT |
| Calibration (ECE, Brier) | custom + scikit-learn | BSD |
| OOD datasets | HF Datasets (TruthfulQA, AmbigQA, MMLU, FEVER…) | varies, documented |

## 7. Serving / API

| Need | Choice | License |
|---|---|---|
| API | existing FastAPI server | MIT (repo) / FastAPI MIT |
| Quantization | existing INT8/INT4; later GPTQ/AWQ | MIT/Apache |
| Speculative decoding | scaffold exists; pair small+large | repo |

## 8. CI / reproducibility / quality

| Need | Choice |
|---|---|
| Tests | pytest (existing), 227 passing |
| Lint/type | ruff + mypy (existing config) |
| Reproducibility | seed + config + dataset hash in checkpoint (existing) |
| CPU smoke CI | GitHub Actions: 100-step CPU training job (add to existing CI) |

## 9. Hardware tiers (free-first)

| Tier | Hardware | What it can do |
|---|---|---|
| **CPU sandbox** | 2 vCPU / ~2 GB (this audit) | correctness, overfit, unit tests, tiny configs |
| **Free single GPU** | T4 16 GB / L4 | MEDIUM (≤25 M params), bf16, 0.5–2 B tokens, ablations |
| **Mid GPU** | A100 40/80 GB / H100 | LARGE (70–300 M), billions of tokens, FSDP |
| **Multi-node** | A100/H100 cluster | LARGE+, distributed |

Design so **everything runs on CPU for correctness** and GPU kernels are an
*acceleration* layer (Triton/`fla`), never the only implementation — this preserves the
free-tooling constraint and the CI smoke test.

## 10. License hygiene checklist
- Track every dependency's license in a bill of materials.
- Avoid AGPL/non-commercial/Research-only code in the redistributable core.
- Dataset licenses recorded per split; ODC-By requires attribution; CC-BY-SA is copyleft.
- Model weights inherit data + code licensing; document before release.
