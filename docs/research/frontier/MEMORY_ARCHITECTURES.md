# XRFM Frontier — Memory Architectures

**Date:** 2026-08-12
**Purpose:** Investigate working/episodic/semantic/procedural memory as *computational
components*, not RAG. Decide what forms memory should take in XRFM.

## 1. The four-memory taxonomy (cognitive, used as an engineering metaphor)

| Memory type | Function in humans | Candidate computational substrate in XRFM |
|---|---|---|
| **Working** | Holds current tokens/active thought | Recurrent state (SSM/linear-attn matrix) + sliding-window attention over the current segment |
| **Episodic** | Remember specific events/sequences | Updatable associative/neural memory written during the forward pass (Titans-style) |
| **Semantic** | Stable facts/concepts | Persistent learned embeddings + a structured concept graph; frozen-after-pretraining weights |
| **Procedural** | Skills/routines ("how") | The model weights themselves (routing policies, update rules); learned optimizer/memory-update rule |

The first three are explicitly modeled; procedural memory is the trained parameters.

## 2. Mechanisms surveyed

### 2.1 Fixed-state associative memory (linear attention / Hopfield)
- Linear attention `S += φ(k)⊗φ(v)` is a Hebbian association matrix; query `o = S φ(q)`.
- Modern Hopfield networks with log-sum-exp energy retrieve patterns via a softmax
  update **mathematically equivalent to one attention step**, with exponential storage
  capacity. Capacity depends on *pattern separation*; encoding networks (HEN) reduce
  meta-stable states [1](https://arxiv.org/abs/2409.16408).
- **Fit:** natural working/episodic store; bounded memory; parallel training.

### 2.2 Delta-rule / gated memory (Gated DeltaNet)
- `S_t = g_t S_{t-1} + k_t ⊗ β_t(v_t − S_{t-1}ᵀk_t)`: error-correcting write.
- DeltaNet-2 decouples channel-wise erase and write gates, reducing interference and
  improving long-context retrieval [2](https://arxiv.org/abs/2605.22791).
- **Fit:** **strongest candidate for episodic memory** — explicit read/write/erase with
  content addressing and bounded capacity.

### 2.3 Deep neural long-term memory (Titans)
- A small MLP maps key→value and is itself *updated online* (momentum + surprise-gated
  weight decay) as data streams; attention = short-term, neural module = long-term,
  persistent weights = task knowledge [3](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/).
- Outperforms Mamba-2/Gated DeltaNet/Transformer++ at comparable size and scales >2M tokens.
- **Fit:** blueprint for episodic memory when matrix state is insufficient; use a
  *small* memory MLP to keep cost bounded.

### 2.4 External differentiable memory (NTM/DNC lineage)
- Read/write heads with content+location addressing over an external matrix.
- Largely superseded by associative-memory/long-context models; unstable to train at scale.
- **Fit:** *reference only*; the read/write interface informs XRFM's memory API, not
  the implementation.

### 2.5 Structured / graph memory (concept graph)
- Nodes = concepts/entities; edge types = supports/contradicts/related/temporal/uncertain.
- Interacts directly with neural state (read edge features into hidden state; write
  discovered relations out). Differentiable edge weights + sparse discrete structure.
- **Fit:** the **semantic memory** representation; high-risk, staged (see
  NEUROTOPOLOGY.md). This is *not* an external RAG KG — it is internal and learned.

## 3. Design proposal for XRFM

```
token x_t
   │
   ▼
[WORKING]  recurrent state h_t (Mamba-2/Gated-DeltaNet)  ──► local precision
   │            + sliding-window attention (bounded KV)
   ▼
[EPISODIC] gated associative memory S_t (delta rule)     ──► sequence-long recall
   │            (optionally a small neural memory MLP a la Titans for long context)
   ▼
[SEMANTIC] persistent concept embeddings C + concept graph G (read-only at infer)
   │
   ▼
output + confidence
```

**Properties enforced:**
- All memory is *differentiable and trained end-to-end* (no frozen retrieval bolt-on).
- Capacity is **configurable** (state dims, memory slots, graph size) — no hard-coded
  tiny fixed memory.
- Updates are **gated by surprise/uncertainty** (Titans insight): write only when the
  input is not already predicted, with momentum and forgetting.
- **Persistent memory** is separate from sequence memory, mirroring short- vs long-term.
- Memory state is part of the **compute graph** (it gates routing and confidence), not
  just a cache.

## 4. Risks / failure modes
- **Interference:** bounded associative memory saturates/collides; mitigate with
  channel-wise erase/write (DeltaNet-2), normalization, and surprise gating.
- **Staleness/forgetting:** adaptive decay must not drop critical facts; monitor
  retrieval recall over distance.
- **Cost:** a deep neural memory MLP per step can be slower than linear recurrent
  models; keep it small and/or update periodically/chunkwise.
- **Graph memory grounding:** inducing a correct concept graph from text is unsolved;
  treat as a *learned latent* object with edge-prediction auxiliary loss, not a
  hand-authored ontology.

## 5. What XRFM should NOT do
- Do not equate "memory" with RAG over a vector DB (external, non-differentiable,
  not part of the architecture's computation).
- Do not use a single unbounded state vector for all memory (RWKV's known limitation).
- Do not hard-code a fixed tiny memory; expose capacity in config.
