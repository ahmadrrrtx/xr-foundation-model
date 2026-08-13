# Phase 4 — Static Neural Graph

**Date:** 2026-08-12
**Commit:** `feat(neurotopo): phase 4`

## Objective
Implement static graph (local ring edges + fixed small-world long-range edges),
sparse edge representation (no dense NxN), and gather/scatter message passing.
Critical for the static-vs-dynamic ablation.

## Implementation
- `xrfm/topology/static_graph.py::StaticNeuralGraph`: fixed edge_index (2,E),
  ring local edges, deterministic seeded long-range links, degree(), neighbors_of().
- `StaticMessagePassing`: m = W_msg h_j; mean aggregation by in-degree; scatter_add
  over (B,N,d); optional edge features. O(N*k*d), no NxN materialization.

## Tests
`tests/test_neurotopo_phase4.py` (8): exact local connectivity, sparsity bound,
no self-loops, isolated-node zero output, gradient flow, batch support,
per-batch permutation sanity, and GATE: a single MP layer + head learns
"predict mean of neighbors" to MSE < 0.05 in 200 steps.

## Results
8 passed. GATE: neighbor-aggregation task trained (final MSE < 0.05, <20% initial).
Full suite **253 passed**.

## Decision
GATE C (static graph trains) PASS. Next: Phase 5 dynamic topology generator;
must empirically show dynamic != static on a controlled synthetic task.
