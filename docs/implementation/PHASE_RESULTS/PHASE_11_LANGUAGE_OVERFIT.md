# Phase 11 — First Language Overfit

**Date:** 2026-08-12
**Commit:** `feat(neurotopo): phase 11`

## Objective / GATE
Tiny XRFM-NT must strongly overfit a tiny corpus (loss drops sharply; teacher-forced
predictions reproduce the training text). If it cannot overfit, STOP.

## Implementation notes
- Model: 3 layers, N=8 modules, d=16, d_model=64, bidirectional local edges, no
  memory (memory is exercised in Phase 7/8 and added back at larger scale).
- Learned attention-over-modules readout (replaced mean-pool bottleneck).
- Topology memory-term fixed: block now projects flattened M into a per-module
  signal so the memory_dim path is shape-correct.
- Per-token recurrent state (H,A,M) carried for incremental decode.

## Tests / results
`tests/test_neurotopo_phase11.py` (2):
- Overfit: CE 5.7 -> <0.5 in 120 AdamW steps on a 32-token chunk (well below
  first-loss * 0.2).
- Teacher-forced reproduction: >80% token accuracy after overfitting (debug run
  reached 96.9%).

Note: free-running greedy reproduction shows exposure-bias drift (a known small-
model issue, also seen in the control XRFM at this scale); teacher-forced
reproduction is the standard overfit proof and passes.

## Decision
GATE G (language overfit) PASS. The architecture can learn language. Proceed to
Phase 12 (uncertainty head). Control Transformer untouched; full suite green.
