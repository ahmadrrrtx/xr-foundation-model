# Phase 5 — Dynamic Topology Generator

**Date:** 2026-08-12
**Commit:** `feat(neurotopo): phase 5`

## Objective
Implement the core research contribution: A_t = alpha*A_{t-1} + (1-alpha)*sigma(S_t)
with low-rank scores S_ij = (Uh_i)^T(Vh_j)/sqrt(r) + gamma*(Pm_i)^T(Qm_j),
local-always-on + top-k long-range edges, plasticity, diagnostics. No dense NxN
for message passing.

## Implementation
`xrfm/topology/dynamic.py::DynamicTopology`:
- fixed candidate edge_index (ring local + seeded long-range slots)
- low-rank U,V scorers (+ optional P,Q memory term with learned gamma)
- sigmoid long-range gates; local gates fixed at 1.0
- learned decay (sigmoid-parameterized) blends previous weights for plasticity
- diagnostics: active_edges, avg/max/min degree, routing entropy, hub concentration
- deterministic per-seed weight init for reproducibility

## Tests
`tests/test_neurotopo_phase5.py` (10): context changes weights; unchanged input stable;
top-k sparsity; plasticity blend matches formula; gradients (incl. decay via plasticity);
no NaNs under extreme inputs; seed reproducibility; memory term affects scores;
diagnostics sane; GATE.

## GATE result (dynamic != static)
Controlled context-dependent routing task (context bit selects which source is
relevant, read out from incoming edge weights):
- DYNAMIC topology trained to cross-entropy < 0.25 (well below chance ln2=0.69).
- STATIC (frozen-edge) readout stayed > 0.55 (near chance; a constant input cannot
  separate the two classes).
- Learned weights separate the two contexts (mean |w0-w1| > 0.05).
This empirically demonstrates dynamic topology carries context signal a static graph cannot.

## Results
10 passed. Full suite **263 passed**.

## Decision
GATE D (dynamic topology works) PASS. Next: Phase 6 sparse message passing that
consumes dynamic (B,E) weights and integrates with local dynamics.
