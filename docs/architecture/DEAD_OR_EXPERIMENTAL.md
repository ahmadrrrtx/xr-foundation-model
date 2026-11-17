# XRFM — Dead / Experimental Code Classification (Phase 30)

**Date:** 2026-08-08 · Decision per item: **Integrate** · **Remove** · **Mark experimental** · **Document as intentionally unused**.
Rule applied: historical research code is not blindly deleted; each item gets an explicit status.

---

## 1. Classification Table

| Item | Location | Production usage (measured) | Decision | Rationale |
|---|---|---|---|---|
| `GradientAccumulator` | `training/distributed.py` | None — used only by `tests/test_distributed.py` and its own docstring | **Mark experimental** | The `TrainingLoop` implements accumulation natively (correct, tested); the standalone class is a reusable utility for non-loop callers. Kept + docstring marked. |
| `KVCache` | `inference/kv_cache.py` | None — `GenerationEngine` uses list-of-tuples (`past_key_values`); only tests import `KVCache` | **Mark experimental** | A candidate for preallocated contiguous-buffer caching later; not integrated today. Kept + docstring marked. |
| `Benchmark` ABC + `TextCompletionAccuracy` + `TopKAccuracy` + `run_evaluation_suite` | `evaluation/benchmarks.py` | Used: exported via `evaluation/__init__`, covered by tests, and is the evaluation framework for Phase 39 | **Keep (integrated)** | Active component of the evaluation protocol. |
| `build_manifest` / `save_manifest` | `xrfm/data/loader.py` | None in production (tests only) | **Integrate in Phase 33** | These are exactly what the dataset-provenance system needs; Phase 33 builds on them rather than duplicating. |
| `SpeculativeDecoder` | `optimization/speculative_decoding.py` | None in the inference engine (exported, unit-tested only) | **Mark experimental** | Research-grade acceleration; not wired into `GenerationEngine`. Documented as experimental. |
| INT8/INT4 quantization utilities | `optimization/quantization.py` | Standalone utilities (exported, tested, round-trip verified) | **Keep (documented utility)** | Not used by default paths; a valid opt-in utility. |
| `scripts/torchrun_launch.sh` | `scripts/` | Environment-validation launcher | **Keep (documented)** | Reference entrypoint for future multi-GPU runs; validates env without training. |
| `config/config.legacy-v1.yaml` | `config/` | Intentional historical artifact | **Keep (documented)** | Preserves the pre-audit config as evidence (audit rule: don't destroy historical work). |
| `artifacts/legacy_unverified/checkpoint_step_500.pt` | `artifacts/legacy_unverified/` *(quarantined out of `checkpoints/` in Phase 0)* | Legacy pre-audit checkpoint (vocab 50304, empty optimizer state) | **Quarantined (documented evidence)** | Moved out of active `checkpoints/` glob discovery (`api/main.py`) while retaining SHA-identical audit evidence. |

## 2. Experimental Markers Added

Docstrings of `GradientAccumulator`, `KVCache`, and `SpeculativeDecoder` now carry:

> **EXPERIMENTAL (v1.1):** not used by the production training/inference paths.
> See `docs/architecture/DEAD_OR_EXPERIMENTAL.md`.

## 3. Non-Changes & Phase 0 Deduplication

- No production library code was deleted.
- Duplicate `vercel_app/` (100% byte-for-byte identical to `webui/`) was consolidated into `webui/` (`api/main.py` mounts `webui/`) as the single UI asset source of truth.
- `webui/` and `deployment/huggingface_space/` remain as shipped.
