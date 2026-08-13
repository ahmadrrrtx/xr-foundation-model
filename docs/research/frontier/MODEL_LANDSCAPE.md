# XRFM Frontier — Model Landscape

**Date:** 2026-08-12
**Purpose:** A consolidated map of architecture families relevant to XRFM, with
"what problem it solves / cost / maturity / OSS / license" for build/buy/borrow decisions.

## 1. Family tree

```
Sequence mixers
├── Attention (softmax)                 ← CONTROL (existing XRFM Transformer)
│   ├── Sparse / sliding-window / SWA
│   └── Linear attention (kernel feature maps)
├── State-space models
│   ├── LTI: S4, S5, Hyena              ← good non-text; weak selectivity
│   └── Selective: Mamba(S6), Mamba-2(SSD), Falcon-H1
├── Linear-attention-as-RNN
│   ├── RetNet (multi-scale decay)
│   ├── GLA (data-dependent gate)
│   ├── RWKV (WKV time/channel mixing)
│   └── DeltaNet / Gated DeltaNet(-2)   ← error-correcting associative memory
├── Memory-centric
│   ├── Titans LMM/MAC (deep neural memory, test-time learning)
│   ├── Modern Hopfield / associative memory
│   └── NTM/DNC (external memory; legacy)
├── Graph / hypergraph
│   ├── Message-passing GNN (static/dynamic adjacency)
│   ├── Hypergraph (HGraphormer, EHNN)
│   └── Graph-SSM hybrids (STG-Mamba, DG-Mamba)
├── Sparse / conditional compute
│   ├── Dynamic sparse training (RigL, SRigL, SET)
│   └── MoE (Switch, Mixtral, DeepSeek-V3)
├── Neuro-inspired
│   ├── Liquid / neural-ODE / CfC
│   └── Predictive coding / STDP (research)
└── Hybrids
    └── Jamba/Samba/Zamba/Qwen3-Next (SSM/linear + attention ± MoE)
```

## 2. Comparison across mission dimensions

| Family | Long context | Inference memory | Train parallelism | Stability | LM quality at scale | Maturity | Best at |
|---|---|---|---|---|---|---|---|
| Transformer | costly but exact | KV grows n | excellent | excellent | best/sota | production | retrieval/reasoning |
| Mamba-2/SSD | excellent | constant | good (matmul) | good | competitive | growing | long seq, efficiency |
| RWKV | good | constant | good | good | competitive | mature (community) | edge/CPU, efficiency |
| Gated DeltaNet-2 | excellent | constant | good (chunked) | good | SOTA linear-recurrent | emerging | associative recall |
| RetNet/GLA | good | constant | good | good | good | mature | efficient mixing |
| Titans/MIRAS | excellent (>2M) | constant + persistent | good (parallel mem update) | newer | beats Mamba-2/DeltaNet/Transformer++ | emerging (Dec 2025) | long memory/test-time learning |
| Modern Hopfield | depends | matrix | good | good | component, not standalone | research | content-addressed retrieval |
| GNN/dynamic graph | n/a (relational) | graph state | moderate | variable | mostly non-LM | research | structure/relations |
| DST (RigL) | n/a | sparse | moderate | good | matches dense at sparsity | research | sparse training |
| MoE | n/a | active experts | good (with comms) | routing risks | SOTA (DeepSeek) | production | conditional compute/scale |
| Liquid/neural-ODE | good | state | poor (ODE solver) | stiffness | non-LM mostly | research | continuous dynamics |
| Hybrid SSM+Attn | excellent | constant + small KV | good | good | SOTA-ish (Jamba etc.) | growing | **practical LM backbone** |

## 3. Build-vs-borrow decisions

| Component | Borrow (OSS) | Build custom | Why |
|---|---|---|---|
| Tokenizer | ✅ reuse XRFM byte-level BPE | — | already correct, lossless |
| Dense Transformer control | ✅ existing XRFM | — | keep as control |
| SSM mixer | ✅ Mamba-2 / `fla` (Apache/MIT) | thin wrapper | no reason to re-derive SSD kernels |
| Associative memory | ⚠️ reference impl | ✅ **custom gated memory** | core IP; integrate with topology/confidence |
| Dynamic topology | — | ✅ **custom** | the novel part; no off-the-shelf LM graph mixer |
| Concept graph | ⚠️ use PyG for ops | ✅ custom latent graph | GNN libs help with sparse ops, not the idea |
| Uncertainty head | — | ✅ custom | architecture-specific signals |
| MoE routing | ✅ borrow ideas (DeepSeek balance) | config option | proven; keep optional |
| Sparse ops | ✅ PyTorch sparse / Triton | fused kernels later | reference first |

## 4. Licensing snapshot (to verify before use)

- **PyTorch**: BSD-3.
- **mamba-ssm / mamba2 / causal-conv1d**: Apache-2.0 (NVIDIA GPU + CUDA).
- **RWKV**: Apache-2.0.
- **flash-linear-attention (fla)**: MIT (research library implementing GLA/DeltaNet/etc.).
- **Hugging Face tokenizers/datasets/datatrove**: Apache-2.0.
- **PyTorch Geometric**: MIT; DGL: Apache-2.0.
- **Gudhi / giotto-tda** (TDA): MIT / Apache-2.0 / GPL-v3 variants — check per package.
- **Triton**: MIT.
- **DeepSpeed**: Apache-2.0 (not needed initially).
- **Titans official code**: confirm license at clone (expected Apache-2.0).

> Always re-verify license and maintenance status at integration time; this snapshot is
> from search results, not a legal audit.

## 5. The central architectural lesson of 2023–2026

The field has converged on **hybrid, memory-centric, sparsely-activated** models rather
than a single "Transformer killer":
- Attention survives because it does exact content retrieval that compressed state cannot.
- SSM/linear-attention provide constant-memory long-horizon mixing.
- MoE provides conditional compute.
- Titans/MIRAS add a *learned, updatable memory* as a first-class object.

XRFM-NT's novelty is **not** to replace any one of these, but to make *connectivity and
memory themselves dynamic, sparse, graph-structured, and uncertainty-aware* — a
composition not present as a dominant architecture today.
