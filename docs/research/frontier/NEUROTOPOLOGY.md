# XRFM Frontier — NeuroTopology Concept Synthesis

**Date:** 2026-08-12
**Purpose:** Extract the *computational principles* from neuroscience, graph theory,
sparsity, and memory research that justify XRFM-NT, without fake neuroscience.

## 1. Principles worth borrowing (and their computational form)

| Brain principle | Computational manifestation in XRFM-NT | Evidence |
|---|---|---|
| **Local computation** | Dense message passing within neural modules/neighborhoods | Cortical microcircuits are locally dense; practical O(N·k·d) |
| **Sparse long-range links** | A few learned hub/long-range edges per unit | Small-world topology; MoE/top-k routing |
| **Modularity** | Clusters/modules with stronger intra- vs inter-connect | Cortical columns; MoE experts; spectral modularity |
| **Context-dependent connectivity** | `A_t = f(H_t, memory, context)`, gated edges | Selective SSM (Mamba); dynamic adjacency |
| **Plasticity** | Online gated write/erase of associative memory | Hebb/delta rule; Gated DeltaNet-2; Titans |
| **Multi-timescale dynamics** | Fast working state + slow persistent memory; per-unit time constants | LTC/CfC; SSMs; Titans memory tiers |
| **Separate memory systems** | Working / episodic / semantic / procedural | Cognitive neuroscience; Titans 3-branch MAC |
| **Recurrent state** | Sequence propagated through state, not all-to-all attention | SSM/RWKV; O(1) inference |
| **Predictive surprise** | Memory writes gated by prediction error/surprise | Predictive coding; Titans surprise metric |
| **Hierarchical organization** | Stacked blocks with pooling/routing across scales | Cortical hierarchy; hypergraph composition |

This is **brain-inspired**, not a brain model. No claim of biological fidelity is made.

## 2. Why a graph is the natural unifying abstraction

- A Transformer's attention is a **dense, complete graph** over tokens (soft, fully connected).
- An SSM/linear-attention model is a **single compressed state** (no explicit edges).
- A neuro-topological model sits between: an **explicit, sparse, dynamic graph over neural
  units/modules**, where edges are learned, gated, and plastic.
- Memory becomes a graph of associations; routing becomes graph edge selection;
  uncertainty becomes graph-level signals (agreement/conflict of evidence paths).

So the graph is not decoration: **routing, memory, and uncertainty are all graph
operations.**

## 3. Mapping mission requirements to mechanisms

| Mission requirement | Mechanism |
|---|---|
| Neuron-level computation | Module/unit states with local update functions |
| Dynamic neural topology | Context-generated sparse adjacency + gating |
| Learned connectivity | Trainable edge scorer + plasticity update |
| Sparse message passing | top-k edges, local neighborhoods, gather/scatter |
| Structured memory | Gated associative matrix + concept graph |
| Uncertainty estimation | Evidence/confidence head fed by graph signals |
| Knowledge verification | supports/contradicts edges + conflict detection |
| Persistent concepts | Frozen persistent memory + concept embeddings |
| State-space dynamics | Optional SSM-style local recurrence within units |
| Graph/hypergraph | Pairwise edges + optional hyperedges for multi-concept binding |
| Modular routing | Module/experts with load balancing |
| Continual learning | Plastic memory + persistent memory separation |
| Evidence-grounded generation | Confidence-gated output states |

## 4. The research hypothesis (falsifiable)

> **H1:** A language model whose primary sequence-mixing computation is *sparse
> message passing over a context-dependent, plastic neural graph* — augmented by a
> fixed-state associative memory and an explicit uncertainty head — can match or
> exceed a parameter-matched, token-matched Transformer control on language modeling
> and long-context retrieval, while providing calibrated abstention and constant
> inference memory, at a comparable compute budget.

**Falsification criteria (pre-registered intent):**
- If the graph mixer does not beat a linear-recurrent (Gated DeltaNet/Mamba-2)
  baseline at matched params/tokens on val PPL, the *dynamic graph* adds no value over
  plain associative memory.
- If it does not beat the Transformer control on long-context needle retrieval, the
  "structured memory" claim fails.
- If the uncertainty head does not improve selective accuracy / ECE over token-prob
  baselines, the architectural uncertainty claim fails.
- If it cannot train stably to at least the overfit + small-corpus milestones, the
  architecture is not viable regardless of conceptual appeal.

## 5. Principle: strongest falsifiable hypothesis, not maximal complexity

Per mission rule 23, XRFM-NT is designed to be **as simple as possible while still
testing H1**. The first implementation uses:
- one sparse dynamic graph mixer (local + top-k long-range),
- one associative memory (gated delta rule),
- one uncertainty head,
- standard next-token loss + small set of auxiliary losses (sparsity, connectivity,
  confidence),
- the existing tokenizer/data/optimizer/scheduler/checkpoint harness.

Everything else (hypergraphs, neural-memory MLP, MoE, TDA regularization, concept
graph writes) is staged behind an ablation flag.
