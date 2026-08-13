# Phase 3 — Local Neural Dynamics (GRU)

**Date:** 2026-08-12
**Commit:** `feat(neurotopo): phase 3`

## Objective
Implement local dynamics `h_i(t+1) = f(h_i(t), input_i)` behind the `Dynamics`
interface. Start with GRU; diagonal SSM later. Gate: tiny system overfits a
deterministic synthetic sequence.

## Implementation
`xrfm/dynamics/gru.py::GRUDynamics` — reset/update/candidate gates applied to
(B,N,d); orthogonal-ish xavier init (gain 0.5), update-gate bias = 1.0 for
initial persistence. Same parameters across modules.

## Tests
`tests/test_neurotopo_phase3.py` (6): shapes/bounds, zero input, repeated input
fixed-point, backward, 200-step gradient stability, and the GATE: NeuralModuleState
+ GRU + linear readout overfit a deterministic sine sequence in 300 Adam steps
(final MSE < 0.05, <20% of initial).

## Results
6 passed (no warnings). GATE: initial loss -> final loss well below 0.05,
demonstrating the recurrent dynamics can fit a deterministic sequence.
Full suite **245 passed**.

## Decision
GATE for local dynamics PASS. Next: Phase 4 static neural graph (local edges,
top-k long-range, sparse adjacency, message passing); gate = synthetic MP task trains.
