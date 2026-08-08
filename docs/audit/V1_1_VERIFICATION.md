# XRFM v1.1 — Verification of Previous Mission Claims (Phase 24)

**Date:** 2026-08-08 · **Branch:** `feat/xrfm-v1.1-gpu-readiness`
**Rule:** claims are independently reproduced, not trusted from the report.
Statuses: **VERIFIED** (reproduced here) · **FAILED** (could not reproduce) · **ENVIRONMENT BLOCKED** (cannot be tested in this environment).

---

## 1. Claim-by-Claim Verification

| # | Claim (from XRFM_FINAL_REPORT / FINAL_AUDIT) | Status | Evidence (fresh run, 2026-08-08) |
|---|---|---|---|
| 1 | 217/217 tests passing | **VERIFIED** | `python -m pytest tests/ -q` → `217 passed, 2 warnings in 5.93s` |
| 2 | Causal masking in flash (SDPA) path | **VERIFIED** | Position probe: pos-2 output unaffected by changing pos-4 (diff 0.00e+00); pos-5 affected (1.63e+02) |
| 3 | Causal masking in manual fallback path | **VERIFIED** | Same probe with `optimization.flash_attention` import blocked: pos-2 diff 0.00e+00, pos-5 diff 1.47e+02 |
| 4 | Tokenizer Unicode/whitespace round-trip | **VERIFIED** | `decode(encode(s)) == s` for English, `café, naïve, 你好世界, مرحبا, 🚀`, newlines/tabs, code; vocab = 2048 |
| 5 | Tokenizer/model vocabulary coherence | **VERIFIED** | `GPTModel(vocab_size=tok.vocab_size())` → model 2048 == tokenizer 2048 |
| 6 | Loss masking (`ignore_index=-100`) | **VERIFIED** | `cross_entropy(ignore_index)` == manual masked CE, diff 0.00e+00 |
| 7 | Scheduler checkpoint restoration | **VERIFIED** | `state_dict` round-trip: `current_step=7`, LR 0.099513 restored exactly |
| 8 | Checkpoint loading (5000-step baseline) | **VERIFIED** | `checkpoint_step_5000.pt` loads with `strict=True`; step=5000, loss=4.7760, seed=42 in metadata |
| 9 | Training reproducibility (same seed) | **VERIFIED** | Seed 42 twice → identical final loss 5.892368 (diff < 1e-9) |
| 10 | Overfit test (Phase 17) | **VERIFIED** | 600 steps on tiny 6-doc corpus → train loss **0.0967** (< 0.15); greedy generation reproduces training text verbatim |
| 11 | Inference from final checkpoint | **VERIFIED** | `checkpoint_step_5000.pt` + `tokenizer/vocab.json` → greedy and T=0.8 generation produce coherent English from 5 prompts |
| 12 | Baseline run metrics (val PPL 301.55 etc.) | **VERIFIED** (archive) | Re-evaluated: val loss 5.7089, **val PPL 301.55** on 92,072 held-out tokens (same as report) |
| 13 | GPU forward/backward, BF16/FP16, VRAM | **ENVIRONMENT BLOCKED** | No GPU in this sandbox (`torch.cuda.is_available() == False`, no `nvidia-smi`) |
| 14 | Multi-GPU DDP/FSDP training | **ENVIRONMENT BLOCKED** | No GPU; scaffolding unit-tested in single-process mode only (as documented) |

## 2. Classification Summary

- **VERIFIED:** 12/14
- **FAILED:** 0/14
- **ENVIRONMENT BLOCKED:** 2/14 (GPU-dependent items — not failures, not passes)

## 3. Notes

- The overfit script reports 198,592 params because it trains the tokenizer at `vocab_size_target=1024` on the tiny corpus (actual vocab < 1024); this matches the Phase-17 methodology, not the 264 K figure in the FINAL_AUDIT table (which used a 1024-complete vocab). Both are TINY-class; the correctness claim (loss < threshold, verbatim generation) is what is verified.
- The 217-test count includes 20 ground-truth verification tests and 5 API tests added by the previous mission.
- No claim in the previous report was found to be false. The two blocked items are exactly the GPU items the previous report itself flagged as untested.

## 4. Implication for This Mission

The v1.0 system is confirmed stable and reproducible. The remainder of this mission (Phases 25–43) hardens it for GPU execution, adds CI training coverage, inference controls, KV-cache benchmarking, dataset provenance, the 15–20 M config, memory/throughput/budget planning, evaluation protocol, experiment tracking, and scaling-experiment design — all without launching expensive training.
