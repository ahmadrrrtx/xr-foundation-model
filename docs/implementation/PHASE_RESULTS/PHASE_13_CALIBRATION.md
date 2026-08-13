# Phase 13 — Confidence Calibration

**Date:** 2026-08-12
**Commit:** `feat(neurotopo): phase 13`

## Objective
ECE, Brier, selective accuracy/coverage, abstention, false-confidence; compare
graph/memory confidence vs token-prob baseline.

## Implementation
`xrfm/nt_evaluation/calibration.py`: equal-mass (quantile) ECE, multi-class
Brier, coverage-accuracy curve (sorted for AUC integration), abstention/FCR,
token-prob baseline, evaluate_calibration bundle.

## Tests/GATE
7 tests. Gate: on a synthetic known/unknown set, graph-evidence confidence has a
higher selective-AUC than an uninformative token-probability baseline.

## Decision
PASS. Next Phase 14 complete NT block/model (configs, integration).
