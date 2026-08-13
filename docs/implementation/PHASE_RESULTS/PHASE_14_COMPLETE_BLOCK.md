# Phase 14 — Complete XRFM-NT Block/Model

**Date:** 2026-08-12
**Commit:** `feat(neurotopo): phase 14`

## Objective
Integrate embedding -> causal conv -> neural modules -> local dynamics ->
dynamic topology -> sparse message passing -> memory -> readout -> LM head, plus
the uncertainty head, behind clean config objects.

## Implementation
- `xrfm/neurotopo/builder.py::build_neurotopo_model` constructs a config-driven
  NeuroTopoModel + optional UncertaintyHead.
- The model already stacked all components (Phases 2-13); this phase confirms the
  full forward/loss/evidence path and config construction.

## Tests
`tests/test_neurotopo_phase14.py` (4): builder, full forward with labels and
uncertainty evidence, backward updates parameters, real-tokenizer end-to-end.

## Decision
GATE (complete v1 block) PASS. Next: Phase 15 parameter-matched controls.
