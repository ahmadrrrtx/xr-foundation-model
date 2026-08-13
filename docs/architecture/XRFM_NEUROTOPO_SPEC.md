# XRFM-NeuroTopo (XRFM-NT) — Architecture Specification

**Status:** PROPOSAL (pre-implementation; awaits approval per mission Section 25)
**Date:** 2026-08-12
**Control model:** existing XRFM Transformer (`model/gpt.py`) — **must not be deleted.**
**Hypothesis ID:** H1 (see NEUROTOPOLOGY.md)

> This is a formal, implementable spec for the **selected candidate C**. It deliberately
> starts minimal (v1) and stages advanced features behind config flags. The graph is
> part of computation: it routes messages, gates memory, and produces uncertainty — it
> is not a visualization around a Transformer.

---

## 1. Design principles

1. **Graph-as-computation.** Neural units communicate only through learned, sparse,
   context-dependent edges; no dense token-to-token self-attention in the NT path
   (attention may remain as an optional, bounded baseline/hybrid block).
2. **Memory as a learned object.** Working/episodic/semantic memory are explicit,
   differentiable, configurable-capacity components — not RAG.
3. **Uncertainty as an output, not a vibe.** A dedicated head produces calibrated
   answer/qualify/abstain decisions from graph+memory evidence.
4. **Everything ablatable and matched to the control.** Same tokenizer, data, optimizer,
   scheduler, param count, token budget, compute.
5. **CPU-correct, GPU-fast.** Pure-PyTorch reference first; Triton/`fla` kernels later.
6. **Strongest falsifiable hypothesis over maximal complexity.**

## 2. High-level dataflow (v1)

```
 tokens x_t
   │
   ▼
[Embedding]  E (weight-tied, reused from XRFM)
   │
   ▼
[Input projection + optional causal conv]  ──► inject into N module states
   │
   ▼
 for layer l = 1..L:
   ┌──────────────────────────────────────────────────────────────┐
   │ 1. LOCAL DYNAMICS     h ← LocalSSM/GRU(h, x)   (per module)   │
   │ 2. TOPOLOGY GEN       A_l,t = TopologyGen(h, M, c_t)          │
   │                       (local edges + top-k long-range, gated) │
   │ 3. SPARSE MSG PASS    h ← h + Σ_{j∈N(i)} g_ij · Msg(h_j,e_ij)│
   │ 4. EPISODIC MEMORY    M ← GatedDeltaWrite(M, k, v, erase,write)│
   │ 5. READ MEMORY        c ← M q ;  h ← h + c                    │
   │ 6. FFN (SwiGLU)       h ← h + FFN(RMSNorm(h))                 │
   └──────────────────────────────────────────────────────────────┘
   │
   ▼
[Pool/Read selected modules] ──► z_t
   │
   ├──► [LM head]  ──────────► logits p(x_{t+1})        (weight-tied to E)
   │
   └──► [Confidence head] ──► u_t ∈ {ANSWER, QUALIFY, ABSTAIN,
                                     REQUEST_CONTEXT, REQUEST_EVIDENCE}
                              + p_known/ambiguous/unknown/ood/conflict
```

## 3. Component specifications

### 3.1 Embedding & input
- Reuse `XRFMEmbedding` (weight-tied byte-level BPE).
- Project embeddings `x_t ∈ R^d` to initialize/update module states.
- Optional short causal Conv1d (kernel 3–4) for local token structure (Mamba-style).

### 3.2 Neural units / modules
- **N modules** per layer, each with state dimension `d` (so `H ∈ R^{N×d}`).
  N is small-to-moderate (64–512); modules are *groups of neurons*, not scalar units.
- At each token, input is broadcast/injected into modules (e.g., via a learned
  input-to-module assignment or uniform injection with learned gains).
- **Local dynamics:** each module runs a cheap recurrent update — choose one:
  - (v1) gated recurrence / mini-GRU, or
  - (v1.1) a diagonal/selective SSM (Mamba-2-class) for content-dependent gating.
  This provides per-module, multi-timescale state.

### 3.3 Topology generator `A_t = f(H_t, M, c_t)`
- **Local prior edges:** each module connects to its k_local nearest modules in a fixed
  or slowly-learned 1D/2D lattice (dense local, cheap).
- **Long-range candidates:** low-rank bilinear score
  `s_ij = (U h_i)^T (V h_j)/√d + b + (P m_i)^T(Q m_j)` (memory term),
  where `m_i` is module i's slice of episodic memory.
- **Sparsification:** keep top-k_long outgoing edges per module via `torch.topk`;
  apply sigmoid gate `g_ij ∈ (0,1)`; add always-on local edges.
- **Plasticity/decay:** `A_t = σ(decay ⊙ A_{t-1} + new_scores)` with per-edge
  erase/write gates (DeltaNet-2 analogy); reset between sequences.
- **Load balancing:** entropy/auxiliary loss so no module/edge hub collapses (MoE-style).
- **Connectivity regularizer:** penalize fragmentation (spectral: drive Laplacian toward
  one connected component) — from STBAM evidence.

### 3.4 Sparse message passing
- Message: `m_ij = Msg(h_j, e_ij)` where edge feature `e_ij` is a small learned
  embedding of edge *type/score* (support/conflict signals live here).
- Aggregation: `h_i' = h_i + Σ_{j∈N_t(i)} g_ij · m_ij` (scatter/gather).
- Update: a GRU/SSM cell combines aggregated messages.
- Pure-PyTorch reference with `topk` + `index_add_`; fused Triton gather-scatter later.

### 3.5 Episodic / associative memory (gated delta rule)
- Memory matrix `M ∈ R^{d_k × d_v}` (configurable; per layer or shared).
- Keys/queries/values from module states: `k,q,v = projections(H)`.
- **Write (Gated Delta Rule-2 style):**
  `M ← α ⊙ M + write ⊙ (k ⊗ β(v − Mᵀk) ⊙ erase)` with channel-wise erase/write
  gates and decay `α` derived from state (surprise-gated).
- **Read:** `c = q M` (optionally normalized/L2 as in Gated DeltaNet).
- Surprise gating: write magnitude ∝ prediction error `‖v − Mᵀk‖` (predictive-coding /
  Titans insight) so only novel information is stored.
- Pure-PyTorch chunkwise reference; optional Triton/`fla` acceleration.

### 3.6 Semantic / persistent memory (v1 minimal; v1.1+)
- **v1:** a frozen-after-pretraining set of persistent vectors `C ∈ R^{K×d}` (like
  Titans persistent memory) attended/read every step.
- **v1.1 (staged):** a latent **concept graph** `G_c = (V_c, E_c)` with typed edges
  (supports/contradicts/related/temporal/uncertain). Neural state reads edge features;
  an edge-prediction auxiliary loss discovers relations. This is *internal, latent,
  differentiable* — not an external RAG KG.

### 3.7 Routing / modularity (optional, staged)
- v1: top-k long-range edges already give conditional subgraph activation.
- v2: MoE-style module routing (DeepSeek auxiliary-loss-free bias balancing) so only a
  subset of modules computes per token; a shared expert for stability.

### 3.8 Output
- **Readout:** select/pool module states (e.g., a learned query attends over modules,
  or use designated output modules) → `z_t ∈ R^d`.
- **LM head:** reused weight-tied `Eᵀ` → logits.
- **Confidence head:** MLP on `[z_t; memory read; graph evidence vector]` producing
  logits over 5 states plus scalar calibration.

### 3.9 Graph-evidence vector (feeds uncertainty)
- Aggregate per-step edge statistics: support-weight, contradict-weight, retrieval
  margin (best − second-best memory match), mean surprise, effective rank of H,
  module-activation entropy. These become inputs to the confidence head — this is the
  architectural coupling that makes abstention evidence-grounded rather than token-prob-based.

## 4. Configuration (proposed additions)

```yaml
neurotopo:
  enabled: true
  n_modules: 128          # N neural modules per layer
  module_dim: 64          # d (must satisfy N*d roughly matching d_model)
  n_layers: 4
  local_degree: 8         # always-on local edges per module
  longrange_topk: 16      # learned long-range edges per module
  edge_dim: 16            # edge feature dim
  topology_update_every: 1
  decay_init: 0.9
  # memory
  memory:
    type: gated_delta     # gated_delta | ssd | none
    d_k: 128
    d_v: 128
    surprise_gate: true
    persistent_slots: 64  # K persistent vectors
  # uncertainty
  uncertainty:
    enabled: true
    n_states: 5
  # staged / off by default
  concept_graph: {enabled: false}
  moe_routing: {enabled: false}
  tda_reg: {enabled: false}
  hybrid_attention_every: 0   # 0 = no attention; >0 = insert bounded attention block
```

## 5. Parameter-count matching to control
- NT block params ≈ projections + edge scorer + message MLP + memory + FFN. Tune
  `n_modules`, `module_dim`, `d_k/d_v`, and FFN ratio so total params match the
  Transformer control at each size (TINY/SMALL/MEDIUM) within ±2%. A parameter-budget
  script is required before training.

## 6. Computational complexity
- Per layer per token:
  - Local dynamics: O(N·d).
  - Topology scoring: O(N²·d) for candidate scores, but **N ≪ sequence length** and
    scoring is low-rank (O(N·r·d)); top-k selection O(N² log k) over modules only.
  - Message passing: O(N·k·d).
  - Memory: O(d_k·d_v) per step (fixed, sequence-independent); chunkwise O(n·d_k·d_v).
  - Total: **O(L·(N·k·d + d_k·d_v)) per token, independent of sequence length** — the
    core efficiency claim vs attention's O(n²·d).
- Memory at inference: fixed (module states + memory matrix), no growing KV cache.

## 7. Package layout (mission Section 18)

```
xrfm/
  core/           # NTModel, block orchestration, config dataclasses
  topology/       # topology generator, local prior, top-k, edge gates, connectivity loss
  neurons/        # module state, local dynamics (GRU/SSM), message/update functions
  dynamics/       # SSM/diagonal-selective updates, causal conv, multi-timescale
  memory/         # gated delta memory, persistent memory, (later) concept graph
  routing/        # top-k routing, load balancing, (later) MoE
  uncertainty/    # confidence head, evidence vector, abstention decision, calibration
  knowledge/      # concept/entity nodes, typed edges, edge prediction (staged)
  tokenizer/      # (reuse existing)
  training/       # NT loop, losses, aux losses, metrics (extend existing)
  evaluation/     # PPL + ECE/Brier/selective/OOD/abstention
  inference/      # recurrent-state generation (no KV cache), confidence-gated decoding
```

The existing `model/` (Transformer) stays untouched as the control; NT lives in `xrfm/`.

## 8. Interfaces (so it plugs into the existing harness)
- `NTModel.forward(input_ids, state=None, mask=None) -> (logits, state, uncertainty)`
  where `state` bundles module states, adjacency, and memory (analogous to KV cache).
- Drop-in compatible with `training/loop.py`-style loops (a new `train_step_nt`),
  `evaluation/perplexity.py`, and `inference/engine.py` (recurrent generation).

## 9. What is explicitly OUT of v1
- Full concept-graph writes, hyperedges, TDA runtime, MoE, Triton kernels,
  multi-node FSDP tuning, attention hybrid (configurable but default off). All are
  staged and behind flags.

## 10. Success / failure gates (link to training plan)
- **Gate 1 (correctness):** overfit a tiny corpus to <0.5 loss; gradients finite;
  deterministic seed reproducibility; save/resume equivalence.
- **Gate 2 (graph works):** static-graph vs dynamic-graph ablation shows dynamic ≥ static;
  connectivity regularizer keeps one component; no routing collapse.
- **Gate 3 (memory works):** needle-in-haystack retrieval improves with episodic memory.
- **Gate 4 (control comparison):** matched SMALL run vs Transformer on val PPL, long-context
  retrieval, throughput, VRAM.
- **Gate 5 (uncertainty):** confidence head improves ECE/selective accuracy vs token-prob
  baseline; measured abstention/hallucination on unanswerable/conflicting/OOD sets.
- If any gate fails decisively, document why and change direction (mission rule).
