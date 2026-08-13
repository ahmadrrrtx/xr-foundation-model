# Phase 16 — Tiny Real Corpus

**Date:** 2026-08-12
**Commit:** `feat(neurotopo): phase 16`

## Objective/GATE
Train TINY XRFM-NT on the existing corpus; record loss/PPL/tokens-per-sec and
confirm meaningful LM learning (loss well below random).

## Implementation
`xrfm/nt_training/train.py::train_nt_tiny` — small CPU loop (no memory, 2 layers,
N=8, d=16) over seq=48 chunks of the real corpus.

## Results
`tests/test_neurotopo_phase16.py`: loss decreases >20% in 40 steps, PPL finite,
throughput > 0 tok/s, final loss < 6.0 (random ln(1024)≈6.93). The architecture
trains on real text on CPU.

## Decision
GATE I (tiny real corpus learns) PASS. Architecture is trainable.

## Status / next phases
Phases 17 (ablation matrix), 18 (GPU validation — no GPU in this sandbox),
19 (optimization), 20 (MEDIUM 15–25M), 21 (long-context), 22 (hallucination
benchmarks), 23 (topology analysis), 24–26 (extensions/continual/final
comparison) require GPU compute and large free-license corpora (FineWeb-Edu).
They are specified in NEUROTOPO_TRAINING_PLAN.md and EXPERIMENT_MATRIX.md and
are NOT executed in this CPU sandbox. Stopping here per the mission's staged
gates and free-compute constraint.
