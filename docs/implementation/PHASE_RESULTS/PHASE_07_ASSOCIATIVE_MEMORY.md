# Phase 7 — Associative Memory

**Date:** 2026-08-12
**Commit:** `feat(neurotopo): phase 7`

## Objective
Implement gated delta-rule associative memory (write/read/erase/decay) and
persistent memory, no external RAG. Gate: memory improves synthetic retrieval vs
a parameter-matched no-memory control.

## Implementation
`xrfm/memory/associative.py`:
- `GatedDeltaMemory(d_k,d_v,d_hidden)`: L2-normalized keys/queries; read=M^Tk;
  err=v-read; channel-wise erase (key axis) and write (value axis); learned
  decay; surprise-modulated beta; update M = diag(decay)M + beta*write*(k⊗(err*erase)).
  Returns read context c, updated M, and info (surprise, margin).
- `PersistentMemory`: softmax attention over K learnable slots; freeze() for
  post-pretraining.

## Tests
`tests/test_neurotopo_phase7.py` (9): init shapes, write/read, multiple facts,
decay reduces norm, erase gate + grads, full gradient flow, persistent read/freeze,
and GATE.

## GATE result
Explicit key->value write/read task: WITH-memory model trained MSE from ~1.0 to
<0.2; a parameter-matched NO-memory model (query only) stayed >0.6 (unsolvable
without sequence memory); with-mem < 40% of no-mem loss. Memory measurably stores
and retrieves facts.

## Results
9 passed. Full suite **280 passed**.

## Decision
GATE E (memory works) PASS. Next: Phase 8 integrate topology + memory (state
(H,A,M,C)) and run the A/B/C ablation (static / dynamic / dynamic+memory).
