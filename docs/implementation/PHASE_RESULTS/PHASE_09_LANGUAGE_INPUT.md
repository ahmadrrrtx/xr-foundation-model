# Phase 9 — Language Input Path

**Date:** 2026-08-12
**Commit:** `feat(neurotopo): phase 9`

## Objective
Connect the EXISTING XRFM byte-level BPE tokenizer and embedding to NeuroTopo via a
causal conv + projection into N module states. No new tokenizer, no control changes.

## Implementation
`xrfm/neurotopo/input_layer.py::NeuroTopoInput`: shared/weight-tieable nn.Embedding,
causal Conv1d(kernel 3, right-trimmed), SiLU, Linear reshape to (B,T,N,d).

## Tests
`tests/test_neurotopo_phase9.py` (6): shapes, 1D batching, causal conv does not
leak future (positions before a change are identical), gradient flow through
embed/conv/proj, end-to-end with the real BPE tokenizer, external embedding tying.

## GATE
Tiny text forward pass works with the real tokenizer: ids -> (1,T,N,d), finite.
Causality verified. Full suite **293 passed**.

## Decision
GATE G (language path) PASS. Next Phase 10 LM head + causal loss/PPL.
