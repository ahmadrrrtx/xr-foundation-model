# XRFM Frontier — Architectural Candidates & Selection

**Date:** 2026-08-12
**Purpose:** Generate ≥5 candidate architectures, score them on the mission's criteria,
and select the one with the **strongest falsifiable hypothesis** (not the most complex).

All candidates are evaluated against the **CONTROL**: the existing XRFM Transformer
(same params/tokens/tokenizer/optimizer/compute). All reuse the existing training/eval
harness and byte-level BPE tokenizer.

---

## 1. Candidate architectures

### Candidate A — Sparse Dynamic Neural-Graph LM ("NeuroGraph")
- **Idea:** Replace attention with sparse message passing over N learned neural
  *modules* (not tokens). Each token is injected into module states; a context-generated
  adjacency `A_t = f(H_t, memory)` (local edges + top-k long-range) routes messages.
- **Token rep:** embedding → projected into module states.
- **Neuron state:** H_t ∈ R^{N×d}, recurrent across the sequence.
- **Edge fn:** gated low-rank scorer + memory term; top-k + local.
- **Node update:** GRU/SSM-style + message aggregation.
- **Memory:** none intrinsic (relies on recurrent state).
- **Output:** pool module states → LM head.
- **Complexity:** O(L·N·k·d), sequence-length-independent mixing.
- **Risk:** high — unproven at LM scale; graph could collapse.

### Candidate B — Graph-Coupled State-Space LM ("GraphSSM")
- **Idea:** Each neural module is an Mamba-2/Gated-DeltaNet **state**; a sparse graph
  couples states between modules (inter-module message passing between SSMs).
- **Neuron state:** per-module SSM h_t; edges gate inter-module state mixing.
- **Edge fn:** learned sparse coupling (local + top-k), slowly varying.
- **Node update:** SSD/delta-rule within module; graph message between modules.
- **Memory:** fixed-state SSM/associative matrix.
- **Complexity:** O(L·(n·d + N·k·d)), linear in sequence.
- **Risk:** medium — SSMs proven; graph coupling is the new part; most grounded.

### Candidate C — NeuroTopological Recurrent LM ("NeuroTopo") ← SELECTED basis
- **Idea:** Combine (1) sparse dynamic graph over neural modules, (2) gated associative
  memory (delta rule) as episodic store, (3) persistent concept embeddings, (4) a
  graph-grounded uncertainty/abstention head. The graph is part of compute: routes
  messages, gates memory writes, and supplies evidence/conflict signals.
- **Token rep:** embedding + optional short causal conv.
- **Neuron state:** H_t ∈ R^{N×d} recurrent.
- **Graph:** A_t from local prior + top-k long-range (state+memory), gated/decayed.
- **Edge fn:** low-rank bilinear + memory; edge *types* include support/conflict signals.
- **Node update:** message passing + gated recurrent update.
- **Memory:** working (recurrent state) + episodic (gated associative matrix) + semantic
  (persistent concept embeddings; concept graph staged later).
- **Routing:** top-k sparse; MoE-style load balancing optional.
- **Uncertainty:** confidence head over hidden+memory+graph-evidence → {ANSWER, QUALIFY,
  ABSTAIN, REQUEST_CONTEXT, REQUEST_EVIDENCE}.
- **Output:** read selected module states → logits.
- **Objective:** LM CE + λ_sparse + λ_connect + λ_confidence (+ optional edge/memory
  prediction), determined experimentally.
- **Complexity:** O(L·(N·k·d + d_m²)) where d_m is memory dim; linear in sequence.
- **Risk:** medium–high; more components but each is ablatable.

### Candidate D — Sparse Plastic Neural LM ("PlasticNet")
- **Idea:** Pure fast-weight / Hebbian plasticity: weights of a recurrent layer are
  updated online by a gated delta rule; no explicit graph. Essentially a deep Gated
  DeltaNet/Titans variant.
- **Risk:** low–medium; very trainable; but **least novel** (≈ DeltaNet/Titans) and
  doesn't test the graph hypothesis.

### Candidate E — Memory-Graph Cognitive LM ("CogGraph")
- **Idea:** Candidate C plus a full **learned latent concept graph** (entity/concept
  nodes, typed edges) that is *written to* during the sequence and read for generation,
  plus hyperedges for multi-concept binding and TDA regularization.
- **Risk:** **highest**; graph induction from text is unsolved; too many novel parts at
  once. Better as a **v2 extension** of C, not the first experiment.

### Candidate F (control variant) — Hybrid SSM + Sparse Attention
- **Idea:** 80% Mamba-2/DeltaNet blocks + periodic sliding-window/global attention, no
  neural-graph. The 2024–25 empirical winner.
- **Risk:** lowest; strongest baseline; but tests no new graph principle. Useful as a
  **second baseline** ("hybrid recurrent control") to isolate the graph's contribution.

---

## 2. Scoring matrix (1–5; 5 best)

| Criterion | A NeuroGraph | B GraphSSM | **C NeuroTopo** | D Plastic | E CogGraph | F Hybrid (baseline) |
|---|---|---|---|---|---|---|
| Novelty | 4 | 4 | **4** | 2 | 5 | 2 |
| Trainability | 2 | 4 | **3** | 5 | 2 | 5 |
| GPU efficiency | 4 | 4 | **4** | 5 | 2 | 5 |
| Memory efficiency | 4 | 5 | **4** | 5 | 2 | 4 |
| Long context | 4 | 5 | **5** | 4 | 4 | 5 |
| Language modeling potential | 2 | 4 | **4** | 4 | 2 | 5 |
| Reasoning potential | 3 | 3 | **4** | 3 | 4 | 3 |
| Hallucination control | 1 | 2 | **5** | 2 | 4 | 1 |
| Interpretability | 4 | 3 | **5** | 2 | 5 | 2 |
| Scalability | 3 | 4 | **4** | 5 | 2 | 5 |
| Implementation complexity (5=simple) | 2 | 3 | **2** | 5 | 1 | 4 |
| **Falsifiable hypothesis strength** | 3 | 4 | **5** | 2 | 3 | 2 |

## 3. Selection: Candidate C (NeuroTopo / "XRFM-NT")

**Rationale:**
- It has the **strongest, most specific falsifiable hypothesis**: a graph that *routes,
  gates memory, and drives abstention* should beat (i) the Transformer control and
  (ii) a memory-only recurrent control (D/F) on long-context retrieval *and* calibrated
  abstention, at matched compute.
- Every novel component is **independently ablatable** (mission Section 20): static vs
  dynamic graph, memory on/off, uncertainty head on/off, topology regularization.
- It is the minimal design that still tests the **full mission thesis** (dynamic
  topology + structured memory + uncertainty + evidence). Candidate E is C plus too much;
  Candidates A/B/D test only a subset; F is a baseline.
- It reuses proven, licensed components (gated delta memory, SSM-style updates, MoE load
  balancing) so risk is concentrated in the graph/uncertainty coupling, not in
  re-deriving sequence mixing.

**Guarding against over-complexity (mission rule 23):** the **v1 implementation turns
off by default**: concept-graph writes, hyperedges, TDA regularization, neural-memory
MLP, and MoE. The first experiment is the smallest version that tests H1:
sparse dynamic graph + gated associative memory + confidence head + LM loss + sparsity/
connectivity/confidence auxiliary losses.

## 4. Controls and ablations (preview)
- **CONTROL-1:** existing XRFM Transformer.
- **CONTROL-2:** hybrid recurrent (Gated DeltaNet/Mamba-2 + periodic attention), no
  neural graph — isolates "memory" from "graph."
- **ABLATIONS:** static graph; dynamic graph; +memory; +uncertainty; +topology reg;
  full. Each at matched params/tokens/compute.
