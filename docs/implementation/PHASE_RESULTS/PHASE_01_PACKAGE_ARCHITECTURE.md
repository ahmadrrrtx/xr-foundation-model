# Phase 1 — Package Architecture

**Date:** 2026-08-12
**Branch:** `feat/xrfm-neurotopo-v1`
**Commit:** (see repo `feat(neurotopo): phase 1`)

## Objective
Create the modular package layout and clean interfaces for XRFM-NeuroTopo without
implementing complex behavior; keep the Transformer control untouched.

## Implementation
New packages under `xrfm/`:
`core, topology, neurons, dynamics, memory, routing, uncertainty, knowledge,
neurotopo, nt_training, nt_evaluation` (each with `__init__.py`).

- `xrfm/neurotopo/interfaces.py`: abstract `NeuralModule, Dynamics,
  TopologyGenerator, MessageFunction, MessagePassingEngine, MemorySystem,
  UncertaintyHead, NeuroTopoBlock, NeuroTopoModel` and `NeuroTopoState` dataclass.
- `xrfm/neurotopo/config.py`: `NeuroTopoConfig, TopologyConfig, MemoryConfig,
  UncertaintyConfig`; v1 staged extensions (`concept_graph`, `moe_routing`,
  `tda_reg`, `hybrid_attention_every`) default OFF per Rule 5.

## Tests
`tests/test_neurotopo_phase1.py` (5 tests): all packages import, all public symbols
present, v1 defaults off, state dataclass holds tensors, **control Transformer still
builds and forwards**.

## Results
- New tests: 5 passed.
- Full suite: **232 passed** (227 control + 5 new) in ~6.4s.
- No files under `model/` modified (RULE 1/2 honored).

## Decision
GATE B (package architecture) **PASS**. Proceed to Phase 2 (neural module state).

## Next gate
Phase 2: implement `H_t in R^{N x d}` module state with init/inject/reset, batch
handling, gradient flow; synthetic forward/backward succeeds.
