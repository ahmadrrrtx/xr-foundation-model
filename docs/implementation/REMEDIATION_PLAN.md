# XRFM — Remediation & Implementation Plan (Phase 13)

**Principle:** sequenced by dependency. Correctness → tests → tokenizer → data → training → checkpoint/eval → verification → training runs. Each batch ends with tests. No rewrites of working components (F-09, F-10, F-29, F-36 remain untouched).

---

## Batch 0 — Repository hygiene & evidence (before code)
- Create `docs/audit/BASELINE.md`, `docs/audit/FORENSIC_AUDIT.md`, `docs/research/MODEL_COMPARISON.md`, `docs/audit/GAP_ANALYSIS.md`, `docs/architecture/XRFM_TARGET_SPEC.md`, `docs/training/COMPUTE_PLAN.md`, `docs/implementation/REMEDIATION_PLAN.md` — DONE in this audit.
- Branch `audit/forensic-v2`; `main` untouched.

## Batch 1 — Correctness fixes (dependency root)
| # | File(s) | Change | Reason | Expected result | Test | Risk |
|---|---|---|---|---|---|---|
| 1.1 | `model/attention/multi_head.py` | Build explicit causal mask when `mask=None`; pass to SDPA as additive mask; keep flash path but make it opt-in flag | F-01/F-02: causality must not depend on import success | Manual fallback is causal; flash path unchanged | New causal-mask test (probe future dependence) | Low |
| 1.2 | `model/gpt.py` | Default `mask=None` still works (MHA now self-masks); expose `build_causal_mask` helper | API stability | No consumer changes needed | Existing shape tests | Low |
| 1.3 | `api/main.py` | Fix broken import: register search routes from `xrfm.search` (new `api/routes/search.py`) or drop line | F-41: API must import | `import api.main` works; `/health` responds | New API import/health test | Low |
| 1.4 | `training/scheduler.py` | Add `state_dict`/`load_state_dict`; ensure `current_step` restored | F-24 | Resume keeps LR schedule | New resume-schedule test | Low |
| 1.5 | `training/checkpoint.py` | Save config + seed + tokenizer/dataset version + commit; `map_location="cpu"` on load; keep legacy keys | F-33/F-32 | Checkpoints self-describing; CPU-loadable | Round-trip test | Low |
| 1.6 | `training/loop.py` | Set seed (config `training.seed`) before dataloader; pass `ignore_index` PAD; report mean micro-loss; `error_if_nonfinite` config flag | F-25/F-15/F-28/F-26 | Reproducible, pad-correct, robust | New seed & pad tests | Medium |
| 1.7 | `training/mixed_precision.py` | Implement real autocast(bfloat16/fp16) when GPU + supported; NoOp on CPU; docstrings corrected | F-23 | Honest mixed precision | Test that CPU path is NoOp; GPU path guarded | Low (no GPU here) |

## Batch 2 — Tokenizer remediation
| # | File(s) | Change | Reason | Expected result | Test | Risk |
|---|---|---|---|---|---|---|
| 2.1 | `tokenizer/bpe.py` | Rewrite as **byte-level BPE** (UTF-8 bytes; merges on byte lists; whitespace preserved via space-prefixed tokens; `\n`, `\t` normal tokens; decode = bytes→utf-8); keep `TokenizerInterface` | F-11/F-12/F-16 | `decode(encode(x)) ≈ x`; Unicode OK; tokens/word ↓ | Round-trip (English/Unicode/whitespace), determinism | Medium (core change; keep old code in git history) |
| 2.2 | `tokenizer/` | Add PAD/BOS/EOS/UNK special tokens reserved at top of vocab (configurable) | F-13 | PAD id stable; loss masking uses it | Special-token test | Low |
| 2.3 | `scripts/train_custom_model.py` | Tokenizer trained on train-split only; model vocab built from tokenizer; save tokenizer alongside checkpoint | F-13/contamination | Coherent pipeline | End-to-end script test | Low |
| 2.4 | `tokenizer/vocab.json` | Regenerate for a real corpus (see Batch 3); old artifact preserved in git history | F-17/F-13 | Working vocab | Load test | Low |

## Batch 3 — Data pipeline
| # | File(s) | Change | Reason | Expected result | Test | Risk |
|---|---|---|---|---|---|---|
| 3.1 | `xrfm/data/loader.py` | Split at **line/document boundaries**; stop collapsing newlines (keep raw text); hash-based dedup of lines; seedable shuffle | F-18/F-19 | No train/val leakage; structure preserved | Leakage test (no overlap of unique n-grams between splits) | Medium |
| 3.2 | `xrfm/data/loader.py` + dataset | PAD to max_seq_len with PAD id; return `attention_mask`/targets with `-100` (or pad id) for loss masking | F-15/F-20 | No loss on padding | Padding-loss test | Low |
| 3.3 | corpus | Add a small **real** public-domain corpus to `data/datasets/` (e.g., concatenated public-domain texts: Aesop, Poe, etc. — assembled locally, ~1–2 MB) + keep `sample.txt` for smoke tests, clearly labeled | F-17 | Trainable non-toy data | Token stats test | Low (choose permissive texts) |

## Batch 4 — Training loop hardening
| # | File(s) | Change | Reason | Expected result | Test | Risk |
|---|---|---|---|---|---|---|
| 4.1 | `training/loop.py` | Add validation: `eval_steps` (val loss + PPL via `evaluation.perplexity` with masking) | F-27 | Real val curve | Val-in-loop test | Low |
| 4.2 | `training/loop.py` | Metrics to CSV/JSONL (`logs/`), include tokens/sec | F-27/F-43 | Trackable history | Writer test | Low |
| 4.3 | `config/config.yaml` | Per-size configs under `config/` (tiny/small/medium) + coherent vocab | TARGET_SPEC | One-codebase scaling | Config tests | Low |

## Batch 5 — Evaluation
| # | File(s) | Change | Reason | Expected result | Test | Risk |
|---|---|---|---|---|---|---|
| 5.1 | `evaluation/perplexity.py` | `ignore_index` for pad targets; accept batch (input, target) | F-39 | Correct PPL | PPL-vs-manual test | Low |
| 5.2 | `tests/` | Ground-truth tests: causal mask, RoPE vs ref, RMSNorm vs ref, SwiGLU vs ref, weight-tied logits, KV-cache equivalence, resume-LR, seeding, overfit | RULE 1/6 | Audit-grade suite | — | Low |

## Batch 6 — API/deployment/CI
| # | File(s) | Change | Reason | Expected result | Test | Risk |
|---|---|---|---|---|---|---|
| 6.1 | `api/main.py` | Load latest checkpoint if present; honest `/health`; remove stale version claims | F-42 | Real weights served | API test | Low |
| 6.2 | `ci.yml` | Remove `|| true` on mypy; add tiny CPU training smoke job | F-43 | CI catches regressions | — | Low |
| 6.3 | `deployment/Dockerfile.gpu` | Pin torch cu-version AFTER requirements | F-44 | GPU image has CUDA torch | — | Low |
| 6.4 | docs | Fix version strings (single source: `xrfm/__init__.py`), README claims vs reality | F-48 | Honest docs | — | Low |

## Batch 7 — Verification runs (Phases 15–20) — see mission phases
7.1 Full test suite (old 192 + new). 7.2 Tiny overfit (Phase 17). 7.3 Training smoke (Phase 16). 7.4 Baseline pretraining, resumable (Phase 18). 7.5 Analysis + scaling mini-experiment TINY vs SMALL (Phase 19/20). 7.6 Final audit + report (Phase 21/22).

## Out of scope (documented, not implemented — Rule 4)
GQA/MoE/SWA/speculative-decode-in-engine/torch.compile/FSDP tuning/real multi-GPU: no measured benefit at this scale/hardware (MODEL_COMPARISON §3). Existing code kept, marked research-only in docs.
