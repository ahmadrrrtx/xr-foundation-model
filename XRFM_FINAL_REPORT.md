# XRFM — Final Report

**Date:** 2026-08-08
**Audit branch:** `audit/forensic-v2` (base `main` @ `cff2dc6` untouched)
**Environment:** CPU-only sandbox — Debian 13, 2 vCPU (Xeon 2.6 GHz), 1.9 GB RAM, 20 GB disk, Python 3.13.14, PyTorch 2.13.0+cpu, no GPU.

---

## 1. Executive Summary

XRFM is now a **genuinely trainable, reproducible, measurable decoder-only language-model system** at small scale. Starting from a repository that claimed "Production Ready — All Phases Complete" but had a broken API, an incoherent tokenizer/model split-brain (50304 vs 408 vs 1024 tokens), silently non-functional "mixed precision", causality that depended on an import succeeding, a non-resumable committed checkpoint, and 192 self-referential tests, the remediation produced:

- A mathematically verified architecture (RoPE, RMSNorm, SwiGLU, weight tying, KV cache — all checked against independent references).
- A byte-level BPE tokenizer with exact round-trips on arbitrary Unicode text.
- A line-boundary, deduped, padding-safe data pipeline on a real 1.77 M-token public-domain corpus.
- A seeded, resumable training loop with real validation and perplexity.
- **217/217 tests passing**, including 20 ground-truth verification tests.
- Real training evidence: an overfit test (loss 0.097, verbatim reproduction), a 5,000-step baseline run (10.24 M tokens; val loss 7.24→5.71, **val PPL 301.55**), and a 2-size scaling experiment showing the expected trend.

XRFM is **not** a foundation model in any production sense, and this report does not claim it is. It is a scientifically reproducible small-scale language-model system with every claim backed by executed runs.

## 2. Initial State (what was broken/missing)

See `docs/audit/FORENSIC_AUDIT.md` for the full 49-finding audit. Highlights:
- CRITICAL: causal masking implicit via SDPA import (manual path bidirectional); tokenizer destroyed whitespace and crashed on non-Latin-1; "mixed precision" did nothing; API failed to import (`search_routes` missing).
- HIGH: 3-way vocab mismatch; toy dataset (same sentence ×100); char-index splits with train/val leakage; padding trained into the loss; scheduler not checkpointable; no seeding; committed "pre-trained" checkpoint had empty optimizer state and incompatible vocab; CI masked mypy errors.
- The README referenced `scripts/validate_training.py` which never existed in any commit.

## 3. Research Findings

`docs/research/MODEL_COMPARISON.md` compares GPT-2, Llama 2, Llama 3, Mistral/Mixtral, Qwen2.5, Gemma 2, OLMo 2, and Pythia from primary sources. Key takeaways applied:
- Explicit causal masking is universal and must be built-in (Llama 3 even masks cross-document attention).
- Byte-level BPE with whitespace preservation is the industry default (GPT-2 → Llama 3 → Qwen2.5 → OLMo).
- Hyperparameters converge: AdamW β=(0.9,0.95), wd 0.1, clip 1.0, cosine + warmup, peak LR 3–6e-4 at small scale.
- Data quality dominates at fixed compute (Llama 3; OLMo's DCLM experiments); the old repeated-sentence "dataset" was indefensible.
- Reproducibility as a first-class artifact (OLMo/Pythia) requires config/seed/data-version in checkpoints.
- Techniques classified C/D for XRFM's scale: GQA, MoE, SWA, speculative decoding, long-context RoPE scaling.

## 4. Changes Implemented (all in commit `1876a33`)

| Area | Change |
|---|---|
| Attention | Explicit causal additive mask built into `MultiHeadAttention`; manual fallback now causal; flash path kept as accelerator |
| Tokenizer | `tokenizer/bpe.py` rewritten as byte-level BPE (UTF-8, whitespace-preserving, Unicode-safe); PAD/BOS/EOS/UNK reserved; old file preserved in git history |
| Data | `xrfm/data/loader.py`: line-boundary splits with exact-line dedup; newlines preserved; `pad_id` padding; targets use `-100` for padded positions; new real corpus `data/datasets/corpus.txt` |
| Training loop | Seeding (`training.seed`) + generator-seeded dataloader; `ignore_index` loss masking; accumulated-loss reporting; configurable `error_if_nonfinite`; validation hook; JSONL metrics writer (`training/metrics.py`) |
| Scheduler | `state_dict`/`load_state_dict` added (resume restores LR) |
| Checkpoint | Saves config/seed/versions (`extra`); `map_location="cpu"`; weights_only-safe (fixed TorchVersion pickle bug) |
| Mixed precision | Real `autocast(bf16)` on GPU; honest fp32/NoOp on CPU; docstrings corrected |
| Model | `GPTModel(vocab_size=…)` override so model/tokenizer are always coherent |
| API | Fixed missing `search_routes` import (new `api/routes/search.py`); lifespan loads tokenizer + latest checkpoint; honest health |
| Config | `config.yaml` (XRFM-SMALL), `tiny.yaml`, `medium.yaml`; legacy config preserved as `config.legacy-v1.yaml` |
| CI/Docker | mypy no longer masked (`|| true` removed); GPU Dockerfile pins CUDA torch after deps |
| Docs | `docs/audit/{BASELINE,FORENSIC_AUDIT,GAP_ANALYSIS,FINAL_AUDIT}.md`, `docs/research/MODEL_COMPARISON.md`, `docs/architecture/XRFM_TARGET_SPEC.md`, `docs/training/COMPUTE_PLAN.md`, `docs/implementation/REMEDIATION_PLAN.md`; README/CHANGELOG rewritten honestly; remediation banners on legacy docs |
| Tests | +20 ground-truth tests (`test_audit_verification.py`), +5 API tests (`test_api.py`); tests updated where behavior legitimately changed |

## 5. Tests (exact results)

```
$ python -m pytest tests/ -q
217 passed, 2 warnings in ~6s
```
Coverage: original 192 (shapes/API, updated where semantics changed) + causality (3), reference math (7), scheduler resume (1), reproducibility (2), tokenizer fidelity (4), loss masking (2), mixed-precision honesty (1), API (5). Every critical mathematical component has an independent-reference test. No BLOCKED items.

## 6. Training (exact configuration & results)

**Baseline run (Phase 18)** — config `config/config.yaml` (XRFM-SMALL), overridden vocab = tokenizer's 2048:
- Model: 1,318,528 params; d_model 128, 4 layers, 4 heads, d_ff 512, seq 256, vocab 2048; weight-tied; fp32 (CPU).
- Optimizer: AdamW β=(0.9,0.95), lr 3e-4 peak, wd 0.1, grad clip 1.0, warmup 200, cosine to 0.
- Data: corpus.txt (11 public-domain books + PSF code), 6,015 train chunks / 360 val chunks, ~1.77 M tokens; batch 8; **10.24 M tokens** seen; seed 42.
- Throughput: ~2,300 tok/s (~1.1 s/step).
- **Train loss: 7.69 → 4.78** (best 4.30). **Val loss: 7.24 → 5.7089. Val PPL: 1400 → 301.55** (random baseline ≈ 2048).
- 27 checkpoints saved during the run (~498 MB); post-run pruning kept the final checkpoint (`checkpoints/checkpoint_step_5000.pt`, ~16 MB incl. optimizer state) plus the legacy committed artifact for the workspace storage budget. Metrics in `logs/training_metrics.jsonl`; config/seed embedded in checkpoint `extra`. The run is fully resumable/reproducible (~73 min).

**Overfit test (Phase 17):** TINY (264 K), 600 steps on a 6-doc mini-corpus → train loss **0.097**; greedy generation reproduces the training text verbatim.

**Scaling (Phase 20):** TINY (264 K) vs SMALL (1.32 M), same corpus/tokenizer/1.64 M tokens → val PPL **777.1 vs 586.0** (larger model wins at equal data; expected scaling direction).

## 7. Evaluation (metrics and limitations)

- **Intrinsic:** token-level val PPL 301.55; top-1 accuracy not reported (padding-corrected PPL is the primary signal).
- **Generation (greedy, T=0) from checkpoint:** grammatical English sequences with real vocabulary ("The quick brown fox, and the birds of the bird, and the birds of the bird, …"). Clear lexical/syntactic learning; classic small-model repetition degeneration. No repetition penalty exists yet.
- **Fair-comparison rule honored:** no comparison against models of different data/compute/tokenizer/params is made anywhere without the protocol row. The only model-to-model comparison (TINY vs SMALL) holds everything else fixed.

## 8. Compute

| Item | Value |
|---|---|
| Hardware | 2 vCPU Xeon 2.6 GHz, 1.9 GB RAM, no GPU |
| Baseline run cost | 5,000 steps ≈ **~73 min** wall (measured log cadence), ~10.24 M tokens, ~2,300 tok/s |
| Overfit test | 600 steps ≈ 8 s (TINY) |
| Scaling pair | 1,600 steps ≈ ~10 min |
| Memory | peak RSS < 1 GB (model 5 MB + Adam 10 MB + activations) |
| Checkpoint storage | ~16 MB/ckpt × 27 = ~498 MB (gitignored) |
| Larger-scale guidance | `docs/training/COMPUTE_PLAN.md` (T4/GPU budgets, MEDIUM/LARGE specs) |

## 9. Remaining Problems (brutally honest)

1. **Not a capable language model.** 1.3 M params and 10 M tokens produce degenerate, repetitive text. It demonstrates the *system* works, not that XRFM is useful.
2. **Distributed training unverified on real hardware** (F-34). DDP/FSDP scaffolding is single-process-tested only; no GPU was available.
3. **Corpus is small and English-only** (~1.77 M tokens, 19th-century prose + code slice). No multilingual coverage, no modern web-scale text.
4. **Tokenizer quality is modest**: 2048 vocab at 0.325 tokens/char (≈3.1 chars/token) is far from Llama-3-grade token efficiency; fine for this scale only.
5. **Evaluation is intrinsic-only** (PPL). No MMLU/HellaSwag-class benchmarks — meaningless at this scale, but absent nonetheless.
6. **Legacy checkpoint incoherence**: `checkpoints/checkpoint_step_500.pt` (vocab 50304, empty optimizer state) remains in the repo, incompatible with the new tokenizer; preserved as evidence, cannot resume.
7. **Dead code remains**: `GradientAccumulator`, standalone `KVCache`, `Benchmark` scaffolding — harmless, unintegrated.
8. **CI does not yet train anything**; mypy has not been made green (fast-fails now, would need cleanup).
9. **No repetition penalty / stop-sequence support** in generation; no beam search.
10. **Versioning**: repository version bumped to 1.0.1 across indicators; `UPGRADE_PLAN_A_PLUS.md` and older research docs still contain inflated historical claims (banner-added, not rewritten).

## 10. Next Version (XRFM vNext)

**v1.1 — "learn at 100 M" (requires a GPU, e.g., free T4):**
1. Train XRFM-MEDIUM (~15–20 M) on 1–3 B tokens of a deduped, filtered slice of a public corpus (FineWeb-Edu subset, Wikipedia, or The Pile slice) with bf16 autocast, batch 32–64, seq 1024.
2. Report val PPL on a held-out WikiText-2-style split with full protocol documentation; add top-1 accuracy and a held-out-code split.
3. Add a repetition penalty and stop-sequence support to inference; benchmark KV-cache latency.
4. Validate DDP on 2 GPUs (gloo/CPU test first, then NCCL) and add FSDP sharded checkpointing tests.
5. Wire CI to run a 100-step CPU training smoke job so regressions in the pipeline fail the build.
6. Introduce config-driven loss masking for packed documents (cross-document attention masking, Llama-3 style) and evaluate packing efficiency.
7. Publish checkpoints + JSONL logs + full provenance for every run (OLMo-style openness).

**v2.0 (institutional compute):** XRFM-LARGE (~300 M) at 100 B+ tokens, GQA, larger byte-level BPE (32–64 K), multi-node DDP/FSDP, external benchmark harness.

---

*Every number in this report comes from an executed run in this workspace; none are extrapolated or fabricated. Full evidence: `docs/audit/*`, `docs/training/COMPUTE_PLAN.md`, `logs/training_metrics.jsonl`, checkpoints on disk.*
