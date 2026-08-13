# Phase 6 — Sparse Message Passing (dynamic topology integration)

**Date:** 2026-08-12
**Commit:** `feat(neurotopo): phase 6`

## Objective
Integrate neural modules + local dynamics + dynamic topology + sparse message
passing into an end-to-end block at O(N*k*d), with no dense NxN.

## Implementation
- `xrfm/core/message_passing.py::DynamicMessagePassing`: gather W_msg h_j over
  sparse edge_index, scale by per-batch dynamic edge_weight, scatter_add into
  targets, normalize by in-degree. Handles (B,E) and (E,) weights.
- `NeuroTopoBlock`: RMSNorm -> DynamicTopology -> sparse aggregation -> GRU local
  dynamics; persists detached topology state for plasticity.
- Local edges made bidirectional (ring successors+predecessors) so information
  can propagate to immediate neighbors in either direction.

## Tests
`tests/test_neurotopo_phase6.py` (8): shapes/sparsity, 1d weight broadcast,
gradient flow, zero-weight isolation, block forward, stateful plasticity,
block backward, and GATE.

## GATE result
End-to-end synthetic one-step propagation task (source signal must appear at
its local neighbor after one block): trained from MSE ~1.03 to < 0.2 in 500
steps (<20% of initial). Confirms modules+dynamics+dynamic graph+sparse MP
compose and train.

## Results
8 passed. Full suite **271 passed**.

## Decision
GATE E/F (message passing + topology/memory-ready) PASS for the MP+dynamics+
topology composition. Next Phase 7 adds gated associative memory (no block
rewiring yet; memory integration is Phase 8).
