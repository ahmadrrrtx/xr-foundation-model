# Phase 15 — Parameter-Matched Controls

**Date:** 2026-08-12
**Commit:** `feat(neurotopo): phase 15`

## Implementation
`xrfm/nt_training/param_match.py`: builders for CONTROL-1 (existing GPTModel via
temporary YAML config), EXPERIMENT (NeuroTopoModel), and a no-graph NT ablation
(memory-only CONTROL-2 proxy); plus a width/depth search to match parameter counts.

## Tests/GATE
4 tests: transformer builds, NT builds/counts, NT matches Transformer params within
20% on the small grid, all three forward the same ids. (Tight ±2% matching is a
training-time exercise using the provided search.)

## Decision
PASS. Next Phase 16 tiny real-corpus training/metrics.
