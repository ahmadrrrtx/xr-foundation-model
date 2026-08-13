# Phase 12 — Uncertainty Head

**Date:** 2026-08-12
**Commit:** `feat(neurotopo): phase 12`

## Objective
5-state confidence system (ANSWER/QUALIFY/ABSTAIN/REQUEST_CONTEXT/REQUEST_EVIDENCE)
driven by graph/memory evidence, not token probability alone.

## Implementation
`xrfm/uncertainty/head.py`:
- Evidence features: edge mean/min/variance/support, module concentration, effective
  rank (SVD entropy), routing entropy, memory surprise/margin.
- MLP -> 5 logits + scalar calibration; decide() with confidence threshold.

## Tests/GATE
6 tests including synthetic 5-way classification: head separates the states at
>60% accuracy (chance 20%).

## Decision
GATE H (uncertainty works on synthetic) PASS. Next: Phase 13 calibration metrics
(ECE, Brier, selective accuracy) and comparison vs token-prob confidence.
