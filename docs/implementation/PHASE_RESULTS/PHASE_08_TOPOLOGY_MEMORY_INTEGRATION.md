# Phase 8 — Integrate Topology + Memory

**Date:** 2026-08-12
**Commit:** `feat(neurotopo): phase 8`

## Objective
Connect modules + dynamic topology + message passing + associative memory into
one recurrent step with state (H, A, M, C); run the A/B/C ablation
(static graph / dynamic graph / dynamic+memory).

## Implementation
- `xrfm/core/state.py::NTState`: bundles per-layer H, A, M, and model-level C.
- `NeuroTopoBlock` upgraded: after sparse aggregation, pools module states,
  writes/reads GatedDeltaMemory, adds memory read (and optional persistent
  memory) to the aggregate, then GRU dynamics. Topology can score using M when
  memory_dim is configured. Returns (H, A, M, diagnostics).
- Reused Phase-4 StaticNeuralGraph/StaticMessagePassing for the static ablation.

## Tests
`tests/test_neurotopo_phase8.py` (7): forward shapes with/without memory, state
detach, memory term influences topology, backward through memory, 20-step
stability, and the GATE.

## GATE result (A/B/C)
Delayed-copy task: a signal written to memory must be retrieved after 4 distractor
steps. Results after 300 steps:
- DYNAMIC+MEMORY: MSE < 0.3 (learns to retain+retrieve).
- DYNAMIC no-memory: MSE > 0.6 (no persistent store; washed out).
- STATIC graph no-memory: MSE > 0.5.
- memory loss < 50% of no-memory loss.
The three systems produce measurably different outcomes and memory adds the
predicted value.

## Results
7 passed. Full suite **287 passed**.

## Decision
GATE F (topology + memory integrate and differentiate) PASS. Next: Phase 9
language input path (existing BPE tokenizer -> embedding -> causal conv ->
module injection), no tokenizer changes.
