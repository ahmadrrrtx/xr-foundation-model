# Phase 10 — Language Model Head + Causal Loss

**Date:** 2026-08-12
**Commit:** `feat(neurotopo): phase 10`

## Objective
Module readout -> hidden -> weight-tied logits; causal CE with padding mask;
validation PPL; one training step.

## Implementation
`xrfm/core/model.py::NeuroTopoModel`:
- InputLayer -> stacked NeuroTopoBlocks with per-token carried state (H,A,M).
- Final RMSNorm + mean-pool over modules + readout Linear.
- Weight-tied logits via dot product with the (shared) embedding matrix.
- Causal cross-entropy with ignore_index=-100 for padding.
- init_state/forward for incremental decode.

## Tests
`tests/test_neurotopo_phase10.py` (6): forward shapes, loss decreases over 5 steps,
padding positions ignored, weight tying (shared storage), state carries/updates,
PPL computable.

## Results
6 passed. One training step reduces loss. Full suite **299 passed**.

## Decision
GATE (one text training step) PASS. Next Phase 11 first language overfit.
