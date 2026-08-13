# XRFM Frontier — Post-Transformer Sequence Model Landscape

**Date:** 2026-08-12
**Purpose:** Detailed comparison of candidate non-Transformer sequence mixers, with the
dimensions the mission requires: problem solved, complexity, memory, stability,
long-context, parallelism, inference, OSS implementation, license, GPU needs.

## 1. The unifying view: sequence models as associative memory

MIRAS (Google, 2025) unifies modern sequence models as **online optimization over an
associative memory**:
- *Memory architecture*: vector (RNN/SSM state), matrix (linear attention / Hopfield),
  or deep MLP (Titans).
- *Attentional bias / objective*: dot-product (Hebbian), MSE/delta-rule, nonlinear MSE.
- *Retention gate*: forgetting/decay regularizer.
- *Memory algorithm*: the optimizer updating memory (SGD/momentum/etc.).

This view is the conceptual backbone for XRFM: a token mixer is a *memory update +
retrieval* rule. The question is which rule(s) to use and how to combine them with a
sparse dynamic graph.

## 2. Model-by-model

### 2.1 Transformer (CONTROL)
- **Problem solved:** exact, content-based all-to-all retrieval; dominant quality.
- **Complexity:** O(n²·d) train and prefill; KV cache O(n) memory, decode O(n) per token.
- **Memory behavior:** unbounded KV cache (exact but expensive).
- **Stability:** extremely well understood; LayerNorm/RMSNorm, residual, warmup.
- **Long context:** strong recall but quadratic cost; needs sliding-window/GQA/attention sinks.
- **Parallelism:** fully parallel training; tensor/pipeline/sequence parallel mature.
- **Inference:** KV-cache decode slows with length; paged attention helps.
- **OSS:** ubiquitous; Apache/MIT. License: varies.
- **GPU:** works on any CUDA; flash-attn for speed.
- **XRFM role:** the **CONTROL MODEL** (already implemented).

### 2.2 Mamba (S6 selective SSM) — [paper](https://arxiv.org/pdf/2312.00752)
- **Problem solved:** linear-time long-sequence modeling with *content-dependent* state.
- **Mechanism:** discretized SSM `h_t = Ā_t h_{t-1} + B̄_t x_t` where Δ,B,C are
  input-dependent (selectivity); fused selective scan.
- **Complexity:** O(n) train and infer; constant state (N×D).
- **Memory:** fixed-size recurrent state; no KV cache.
- **Stability:** good with RMSNorm + gated block; custom kernels.
- **Long context:** excellent scaling; weaker fine-grained recall vs attention.
- **Parallelism:** parallel scan; tensor parallel possible; original kernels single-GPU-centric.
- **Inference:** ~5× higher throughput than Transformers; constant memory.
- **OSS:** `state-spaces/mamba` (Apache-2.0); depends on `causal-conv1d` + CUDA.
- **GPU:** NVIDIA + CUDA required for fast kernels; CPU fallback slow.
- **Relevance:** **HIGH** — a selective SSM is a strong candidate *working-memory/state*
  component for XRFM. The input-dependent gating is the cleanest example of
  "context-dependent connectivity/update."

### 2.3 Mamba-2 (SSD / Structured State-Space Duality) — [paper](https://arxiv.org/abs/2405.21060)
- **Problem solved:** makes SSMs matmul/tensor-core friendly and connects SSM↔attention.
- **Mechanism:** scalar-identity selective SSM = 1-semiseparable masked attention;
  chunkwise algorithm 2–8× faster than Mamba; adds grouped-value heads.
- **Complexity/memory:** O(n), fixed state.
- **Stability/long context:** improved; tensor-parallel friendly.
- **OSS:** `state-spaces/mamba2` (Apache-2.0); Triton/CUDA kernels; HF `transformers`
  supports Mamba2 (pure-torch fallback exists).
- **GPU:** NVIDIA CUDA; CPU fallback usable for testing only.
- **Relevance:** **HIGH** — preferred SSM over Mamba-1 if an SSM path is adopted
  (faster, more parallel, attention-duality gives a route to hybrid layers).

### 2.4 RWKV (→ v7 "Goose") — [paper](https://arxiv.org/html/2305.13048v2), [evolution survey](https://arxiv.org/html/2411.02795v1)
- **Problem solved:** 100% RNN that trains in parallel (linear attention) and decodes O(1).
- **Mechanism:** time-mixing (WKV linear attention with learned decay) + channel-mixing
  (gated FFN); v7 adds dynamic state evolution.
- **Complexity:** O(n) train (linear-attention form); O(1) per-token decode.
- **Memory:** fixed state; no KV cache.
- **Stability:** custom init; time-dependent softmax for numerics; large community.
- **Long context:** good but single-vector state limits very-fine recall.
- **Parallelism:** parallel train formulation; CUDA/Triton kernels.
- **Inference:** constant-time decode; very small memory; runs on CPU/edge well.
- **OSS:** `BlinkDL/RWKV-LM` (Apache-2.0); HF support; fla/RWKV kernels.
- **GPU:** NVIDIA; strong CPU support (advantage for XRFM's CPU sandbox).
- **Relevance:** **MEDIUM-HIGH** — the WKV gating/decay is a clean, proven local update
  rule; CPU friendliness fits the free-tooling constraint.

### 2.5 Gated DeltaNet / Gated DeltaNet-2 — [analysis](https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15), [paper](https://arxiv.org/abs/2605.22791)
- **Problem solved:** fixed-matrix associative memory with error-correcting writes.
- **Mechanism:** `S_t = g_t S_{t-1} + k_t ⊗ β_t(v_t − S_{t-1}ᵀ k_t)`; DeltaNet-2
  decouples channel-wise *erase* (key axis) and *write* (value axis) gates,
  recovering KDA/Gated DeltaNet as special cases; chunkwise WY parallel form.
- **Complexity:** O(n) train (chunked), O(1) decode; fixed matrix state S∈R^{d_k×d_v}.
- **Memory:** bounded; better interference resistance than scalar-gate models.
- **Stability:** L2-normalized Q/K, gated; Triton kernels.
- **Long context:** strong RULER/needle retrieval at fixed state; SOTA among linear
  recurrent at 1.3B/100B FineWeb-Edu tokens.
- **OSS:** `NVIDIA/GatedDeltaNet-2` (research code, license TBC); Qwen3-Next uses
  Gated DeltaNet layers interleaved with full attention.
- **GPU:** NVIDIA (Triton); pure-PyTorch reference possible.
- **Relevance:** **HIGHEST for XRFM's memory component** — the delta rule is a
  differentiable, gated, content-addressable write/erase; it directly implements a
  *learned, updatable associative memory* with explicit connectivity (S acts as a
  fast-weight/adjacency-like structure).

### 2.6 RetNet / linear attention / GLA
- **Problem solved:** parallel train + recurrent infer with exponential decay.
- **Mechanism:** `S = γS + k⊗v` (RetNet); data-dependent matrix gate (GLA).
- **Complexity/memory:** O(n) train, O(1) infer, fixed state.
- **OSS:** various (MIT/Apache); `fla` (flash-linear-attention) library.
- **Relevance:** **MEDIUM** — subsumed by Gated DeltaNet as a design point; useful
  ablations (fixed decay vs gated vs delta).

### 2.7 Titans (LMM / MAC) + MIRAS — [Google blog](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)
- **Problem solved:** *test-time learning* into a deep neural long-term memory beyond a
  fixed vector/matrix state; three-tier memory.
- **Mechanism (MAC):** core attention branch (short-term) + contextual-memory branch
  that reads/writes a deep MLP memory + persistent-memory branch (frozen weights).
  Memory updated by a momentum + adaptive forgetting (weight decay) rule gated by
  "surprise"; associative-memory loss; parallelizable.
- **Complexity:** near-linear train, linear infer; memory module adds depth/cost.
- **Long context:** scales beyond 2,000,000 tokens; strong BABILong.
- **OSS:** official code released by Google (Apache-2.0, check repo); PoC
  implementations exist (e.g., `kolejnyy/titans-lmm`, MIT).
- **GPU:** standard CUDA; CPU for smoke tests.
- **Relevance:** **HIGH** — direct blueprint for XRFM's working/episodic/persistent
  memory separation and surprise-gated writes. Risk: newest (Dec 2025), least battle-tested.

### 2.8 Hybrids (Jamba, Samba, Zamba, Qwen3-Next)
- **Problem solved:** SSM/linear-attn memory + attention for exact retrieval.
- **Pattern:** ~80–90% recurrent/linear blocks + sparse/sliding-window/full attention
  every N layers; sometimes MoE.
- **Relevance:** **HIGH (architectural template)** — XRFM should likely be a *hybrid*:
  sparse dynamic graph/SSM for bulk mixing + occasional attention for exact local/global
  retrieval. This is the empirical winner in 2024–2025.

## 3. Summary matrix

| Model | Train | Infer decode | State | OSS maturity | License | CPU-friendliness | XRFM fit |
|---|---|---|---|---|---|---|---|
| Transformer | O(n²) | O(n)/token | KV grows | excellent | varies | good (small) | CONTROL |
| Mamba-1 | O(n) | O(1) | vector | mature | Apache-2.0 | poor (kernels) | SSM candidate |
| Mamba-2/SSD | O(n) | O(1) | vector | good | Apache-2.0 | poor (kernels) | preferred SSM |
| RWKV | O(n) | O(1) | vector | mature | Apache-2.0 | **excellent** | local mixer candidate |
| Gated DeltaNet-2 | O(n) | O(1) | matrix | emerging | research | medium (ref impl) | **memory core** |
| RetNet/GLA | O(n) | O(1) | matrix | good | varies | medium | ablation |
| Titans/MIRAS | ~O(n) | O(n-ish) | deep MLP | emerging | Apache-2.0 | medium | memory blueprint |
| Hybrid SSM+Attn | mixed | mixed | mixed | growing | varies | medium | **recommended shape** |

## 4. Recommendation for XRFM's mixer design

1. **Use a fixed-state associative memory as the bulk sequence mixer** (Gated DeltaNet /
   Mamba-2-class) rather than dense attention — this gives linear scaling and an
   explicit, updatable connectivity object (the memory matrix / state).
2. **Interleave a small amount of exact attention** (sliding-window + periodic global,
   as in hybrids) for precise retrieval — do not bet entirely on compressed state.
3. **Make the memory gates context/topology dependent** — this is where XRFM's novelty
   lies: instead of gates derived only from the token, derive them from the *active
   neural subgraph* and persistent concept memory.
4. **Keep the Transformer as an apples-to-apples control** (same params, tokens,
   tokenizer, optimizer, compute).
