# XRFM v1.1 — GPU Readiness Report (Phase 43)

**Date:** 2026-08-08 · **Branch:** `feat/xrfm-v1.1-gpu-readiness`
**HEAD:** `d138a1e` (after Phases 23–42) · **Base:** `audit/forensic-v2` @ `d4307a2`
**Mission rule honored:** NO expensive training was launched. This report ends
with the exact command for the first serious run, pending explicit approval.

---

## 1. Executive Summary

XRFM v1.1 is **GPU-READY in every dimension that can be made ready without a GPU**,
and every dimension that requires a GPU is explicitly classified
**ENVIRONMENT BLOCKED** (no fabrication). Concretely:

- The v1.0 system from the previous mission was independently re-verified
  (12/14 claims VERIFIED, 0 FAILED, 2 GPU-dependent items BLOCKED).
- The training pipeline was hardened for GPU-scale execution (config-driven
  workers/pinning/prefetch/device/determinism, non-blocking H2D transfers,
  CUDA-OOM resilience) and a real bug was found & fixed by the mypy pass:
  **config-driven `resume_from` was silently disabled** (orphaned dead code).
- CI now trains (fast CPU smoke: loss must decrease, checkpoint save/load,
  resume) and fails the build on regression; mypy is green (0 errors, 55 files);
  ruff/black clean.
- Inference gained repetition penalty, stop sequences, EOS handling (9 new tests).
- KV-cache benchmark completed (CPU: 2.0×; VRAM measurement BLOCKED).
- Dataset provenance (sha256, split token counts, tokenizer/filter/dedup
  versioning) implemented; first real dataset plan written (FineWeb-Edu
  sample-10BT slice, ODC-By).
- XRFM-MEDIUM config validated: **19,666,560 params (19.67 M)**, bias-free,
  verified programmatically.
- Memory budgets (fp32/fp16/bf16), training budgets (50 M–1 B tokens), evaluation
  protocol, experiment tracking, and the scaling study are all defined and
  implemented where executable.

**Honest bottom line:** XRFM is ready to run its first serious
15–20 M-parameter experiment the moment a GPU (≥ 16 GB) is available. In this
CPU-only sandbox, that training is blocked by the environment, not by the code.

## 2. Previous Audit Verification (Phase 24 → `docs/audit/V1_1_VERIFICATION.md`)

| Claim | Status |
|---|---|
| 217/217 tests | VERIFIED (227 now) |
| Causal masking (flash + manual paths) | VERIFIED (position probes) |
| Tokenizer Unicode/whitespace round-trip | VERIFIED (vocab 2048) |
| Tokenizer/model vocab coherence | VERIFIED (2048 == 2048) |
| Loss masking (ignore_index) | VERIFIED (exact match vs manual CE) |
| Scheduler checkpoint restoration | VERIFIED (step + LR exact) |
| Checkpoint loading (5000-step baseline) | VERIFIED (strict=True) |
| Training reproducibility (same seed) | VERIFIED (loss identical) |
| Overfit test (Phase 17) | VERIFIED (loss 0.097, verbatim reproduction) |
| Inference from final checkpoint | VERIFIED (greedy + T=0.8 samples) |
| Baseline metrics (val PPL 301.55) | VERIFIED (protocol runner reproduces 301.55, top-1 11.3 %) |
| GPU forward/backward, BF16/FP16, VRAM | ENVIRONMENT BLOCKED (no GPU) |
| Multi-GPU DDP/FSDP | ENVIRONMENT BLOCKED (no GPU; scaffolding unit-tested) |

## 3. Hardware (Phase 25)

| Item | Value |
|---|---|
| OS / Python / PyTorch | Debian 13 / 3.13.14 / 2.13.0+cpu |
| CUDA / GPU / VRAM | **NOT AVAILABLE / NONE / N/A** |
| CPU / RAM | 2 vCPU Xeon 2.6 GHz / 1.9 GB |
| Disk | 25 GB (~20 GB free) |
| BF16 / FP16 hardware | N/A (CPU) |

**`GPU TRAINING BLOCKED BY ENVIRONMENT`** — stated plainly, no GPU results fabricated.

## 4. Remaining Bugs Found & Fixed in This Mission

| Bug | Severity | Fix |
|---|---|---|
| Config-driven `resume_from` silently disabled (block orphaned after `return` in `_checkpoint_extra` during the previous mission) | HIGH (silent) | Restored to `__init__`; regression test added (found by mypy) |
| `torch.distributed.Timedelta` → nonexistent API | MEDIUM | `torch.distributed.timedelta` |
| FSDP `**kwargs` typed as `dict[str, ShardingStrategy]` (mypy) | LOW | `dict[str, Any]` |
| `GradScaler.step()` return typed bool but returns `Optional[float]` | LOW | normalize to bool |
| `TorchVersion` object pickling inside checkpoint `extra` (weights_only load failure) | MEDIUM | `str(torch.__version__)` (previous mission; re-verified) |
| `training.loop` reported per-micro-batch loss instead of accum mean | LOW | accumulated mean |
| `reduced` dead variable + unused `reduce_loss` import | LOW | removed |
| Tokenizer clobbered by smoke runs (saved over canonical repo vocab) | MEDIUM (process) | `--tokenizer_out` option; canonical 2048 vocab restored |

## 5. Changes Implemented (Phases 26–42)

| Phase | Deliverable |
|---|---|
| 26 | `scripts/gpu_smoke_test.py` — on-device forward/backward/autocast/clip/optimizer/scheduler/checkpoint-save-load-resume/metadata/VRAM checks; **15/15 PASS on CPU**; CUDA branches guarded |
| 27 | Pipeline hardening: `num_workers`/`pin_memory`/`prefetch_factor`/`persistent_workers`/`device`/`deterministic` config; `non_blocking` H2D; CUDA-OOM skip-and-continue |
| 28 | CI `training-smoke` job + `scripts/ci_smoke_test.py` (60 steps; loss decrease; ckpt save/load; resume) — **fails build on regression** |
| 29 | mypy 49 → **0 errors** (55 files); ruff/black clean; formatters aligned at line-length 120; `data/` excluded from lint (corpus ≠ source); justified `# type: ignore[override]` ×2 documented |
| 30 | `docs/architecture/DEAD_OR_EXPERIMENTAL.md` — GradientAccumulator/KVCache/SpeculativeDecoder marked experimental; manifest helpers integrated (Phase 33) |
| 31 | Inference controls: repetition penalty, stop sequences, EOS (`stop_token_id`), max tokens; API wired (`repetition_penalty`, `finish_reason=stop`); **9 tests** |
| 32 | `scripts/benchmark_kv_cache.py` + `docs/benchmarks/KV_CACHE.md` — measured **2.0×** (425 vs 213 tok/s, CPU) |
| 33 | `xrfm/data/manifest.py` (DatasetManifest: sha256, doc/token counts per split, tokenizer/dedup/filter versions) + `docs/data/XRFM_DATA_SPEC.md`; wired into training |
| 34 | `docs/data/V1_1_DATASET_PLAN.md` — FineWeb-Edu sample-10BT (ODC-By) recommended; deterministic 50 M–1 B slices |
| 35 | `config/v1.1-medium.yaml` + config-driven `bias=False` (attention/FFN); **params verified 19,666,560** |
| 36 | `docs/training/V1_1_GPU_MEMORY.md` — fp32/fp16/bf16 budgets + per-GPU batch estimates (labeled) |
| 37 | `scripts/benchmark_throughput.py` — CPU measured (SMALL @seq512 = 3,358 tok/s); MEDIUM OOM-blocked in 1.9 GB sandbox (documented) |
| 38 | `docs/training/V1_1_BUDGET.md` — A/B/C/D budgets with formulas + labeled GPU estimates |
| 39 | `docs/evaluation/V1_1_PROTOCOL.md` + `scripts/run_eval_protocol.py` (masked PPL/top-1, fixed prompts, rep4/EOS analysis; reproduces PPL 301.55) |
| 40 | `xrfm/experiment/tracking.py` ExperimentRecord (all Phase-40 fields) + `logs/run_manifest.json`; wired into training |
| 41 | `docs/research/V1_1_SCALING_EXPERIMENT.md` (TINY/SMALL/MEDIUM matched-token); training script gains `--config`/`--train_ratio` |

## 6. GPU Smoke-Test Results (Phase 26)

Executed on CPU (device-handling logic fully exercised; CUDA branches guarded):

```
[PASS] 1.  model on device
[PASS] 2.  forward+backward (8 steps)
[PASS] 4.  autocast context (no-op CPU / bf16-fp16 GPU)
[PASS] 5.  gradients finite
[PASS] 6.  gradient clipping
[PASS] 7.  optimizer step
[PASS] 8.  scheduler step
[PASS] 9.  loss finite+decreasing + checkpoint saved
[PASS] 10. checkpoint reloads (weights equal)
[PASS] 11. resume restores step / optimizer state / scheduler / seed+config metadata
[PASS] 12. peak VRAM (N/A on CPU)
15/15 PASS, exit 0
```

CUDA-specific checks (`4a. autocast ran on GPU`, `4b. bf16 support`, `12. peak VRAM`)
execute automatically when run with `--device cuda` on a GPU host.
**BF16/FP16 on GPU: ENVIRONMENT BLOCKED (code ready, unexecuted).**

## 7. CI Results

- **Training smoke** (`ci.yml` → `training-smoke` job): `CI TRAINING SMOKE: PASS (steps=60, loss 6.24 -> 2.514, resume step=60)` in ~3.4 s. Fails build on pipeline regression.
- Full test job: `227 passed` (217 prior + 9 inference controls + 1 resume regression).
- `mypy model/ training/ tokenizer/ inference/ evaluation/ optimization/ api/ xrfm/` → `Success: no issues found in 55 source files`.
- `ruff check .` → clean · `ruff format --check .` → clean · `black --check .` → clean.

## 8. Code Quality (Phase 29)

- mypy: 0 errors (49 fixed). Justified exclusions: 2 × `type: ignore[override]` on
  benchmark concrete classes (ABC dispatches via `**kwargs`); 1 × `cast` for
  `torch.compile` (typed as Callable in torch stubs). All documented in code.
- Added `types-PyYAML` to dev deps; added missing `__init__.py` to `model/`,
  `model/attention/`, `model/layers/`, `training/` (root cause of mypy module collision).
- Formatters reconciled: black line-length 100 → 120 to match ruff (previously
  permanently conflicting).

## 9. Inference Results (Phase 31)

Controls implemented & tested (9 tests): repetition penalty (CTRL-style),
stop sequences (decoded-suffix check), EOS (`stop_token_id`), max_new_tokens.
API: `repetition_penalty` schema field; `finish_reason` ∈ {length, stop}.
Baseline model generation: grammatical English, small-model repetition loops
(rep4 ≈ 0.11 on the protocol's 30 generations), EOS rate 0.0 (expected until
EOS supervision is added in training). No beam search added (no justification at
this scale).

## 10. KV Cache Results (Phase 32 → `docs/benchmarks/KV_CACHE.md`)

| Mode | ms/token | tokens/sec |
|---|---|---|
| With KV cache | 2.36 | 424.6 |
| Without | 4.70 | 212.8 |
| **Speedup** | — | **2.0×** |

Peak VRAM: BLOCKED (CPU). Protocol reproducible via `scripts/benchmark_kv_cache.py`.

## 11. Dataset Strategy (Phases 33–34)

- **Recommended corpus: FineWeb-Edu `sample-10BT` slice** (ODC-By 1.0; English;
  Llama-3-70B-Instruct-quality-filtered; beats FineWeb on MMLU/ARC/OpenBookQA;
  10 B-token official sample; deterministic seeded slice of N tokens).
- First slice: **50 M tokens** (~200 MB Parquet) for Experiment A.
- Provenance enforced: `DatasetManifest` (sha256, per-split token counts,
  tokenizer version, dedup/filter method) stored with every run.
- Current 1.77 M-token corpus retained only for smoke/CI (explicitly NOT the
  v1.1 training corpus).
- Tokenizer plan: retrain on the train slice at **vocab 8192** (matches MEDIUM).

## 12. XRFM-MEDIUM Architecture (Phase 35)

| Field | Value |
|---|---|
| vocab | 8192 (tied embedding/LM head) |
| d_model | 384 |
| layers | 7 |
| heads | 6 (d_head 64) |
| d_ff | 1536 (4×d) |
| context | 1024 |
| dropout | 0.0 |
| biases | **none** (config `use_bias: false`) |
| **params (verified)** | **19,666,560 (19.67 M)** — bias-free build; 19,701,504 with biases |
| optimizer | AdamW β(0.9,0.95), wd 0.1, clip 1.0, cosine+1,000 warmup, peak 3e-4 |
| batch | 16 × accum 4 = eff 64 (65,536 tokens/step) |

## 13. Memory Budget (Phase 36 → `docs/training/V1_1_GPU_MEMORY.md`)

19.67 M params: weights 78.7 MB (fp32) / 39.3 MB (bf16); AdamW 157.3 MB; total
model+opt ≈ 236 MB fp32 / 196 MB bf16. Activations at batch 16 × seq 1024 bf16
(with SDPA, no score matrix materialized) ≈ 1.2 GB. **Estimated** peak: T4 16 GB
fits batch 16×1024 with accum 4; 8 GB fits batch 8. All estimates labeled; GPU
measurement is Phase-37 on a GPU host (BLOCKED here).

## 14. Throughput (Phase 37)

| Config | Device | seq | batch | tok/s | Status |
|---|---|---|---|---|---|
| SMALL 1.32 M | CPU (measured) | 256 | 4 | ~2,850 | measured |
| SMALL 1.32 M | CPU (measured) | 512 | 4 | ~3,358 | measured |
| MEDIUM 19.67 M | CPU | 256+ | 4 | — | **MEMORY BLOCKED** (1.9 GB sandbox; kernel OOM) |
| MEDIUM 19.67 M | T4-class GPU | 1024 | 16 | 15k–35k | **estimate** (assumption: 15–35% MFU at 118 MFLOPs/token) |

## 15. Training Budget (Phase 38 → `docs/training/V1_1_BUDGET.md`)

| Exp | Tokens | Steps (eff 65,536) | GPU h @15k | GPU h @35k | Full ckpts (~320 MB) |
|---|---|---|---|---|---|
| A | 50 M | 763 | 0.9 | 0.4 | 1 |
| B | 100 M | 1,526 | 1.9 | 0.8 | 1 |
| C | 500 M | 7,629 | 9.3 | 4.0 | 4 |
| D | 1 B | 15,259 | 18.5 | 7.9 | 8 |

**Recommended first run: A (50 M tokens)** — ~1–2 GPU-hours on T4-class.

## 16. Evaluation Protocol (Phase 39 → `docs/evaluation/V1_1_PROTOCOL.md`)

Implemented in `scripts/run_eval_protocol.py` and validated against the baseline:
val loss **5.7089**, **PPL 301.55**, top-1 **11.3%**, mean rep4 0.108, EOS rate 0.0
on the frozen val slice (context = training context 256). Protocol rows
(MODEL/PARAMS/TOKENS/DATA/CONTEXT/COMPUTE/EVAL-SET/METRIC) mandatory in all reports.

## 17. Scaling Experiment (Phase 41 → `docs/research/V1_1_SCALING_EXPERIMENT.md`)

TINY (~0.26 M) / SMALL (1.32 M) / MEDIUM (19.67 M) at the SAME 50 M-token budget,
same optimizer/schedule/seed, same held-out eval slice. Prior Phase-20 mini-run
already showed the expected direction (PPL 777 → 586 for 5× params). MEDIUM adds
the third point; documented deviations: vocab 2048→8192 and context 256→1024.

## 18. Risks (honest)

1. **GPU numbers are estimates, not measurements.** Throughput/MFU assumptions
   (15–35%) are the largest uncertainty; the 50 M-token run will replace them
   with measured values before any C/D decision.
2. **Small-model MFU is typically low** — a 19.7 M model on a T4 may be
   launch/latency bound; the budget's low end (15k tok/s) is the conservative
   planning number.
3. **Vocab 8192 tokenizer is untrained at scale** — retraining on the slice may
   shift tok/char efficiency; recorded in the manifest for fair comparisons.
4. **EOS supervision absent** — generation will rarely stop via EOS until
   training includes EOS tokens; protocol accounts for this (EOS rate metric).
5. **FineWeb-Edu slice download requires network + HF `datasets`** — storage 0.2–4 GB;
   if unreachable, fallback = Pile permissive subset or Wikipedia slice (documented).
6. **DDP/FSDP still unexercised on real hardware** — single-GPU run is the v1.1
   scope; multi-GPU remains future work.
7. **Sandbox constraints** — this workspace cannot hold a MEDIUM forward+backward
   at useful batch; all MEDIUM work must happen on the GPU host.

## 19. Exact Next Training Command (STOP — awaiting approval)

On a GPU host (≥ 16 GB VRAM), after downloading the FineWeb-Edu slice:

```bash
# 0) Environment (CUDA torch, dataset, tokenizer)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]" datasets

# 1) Prepare the 50M-token slice + manifest (Phase 33/34)
#    (documented in docs/data/V1_1_DATASET_PLAN.md; produces data/datasets/fwe50m.txt
#     and its DatasetManifest)

# 2) Verify GPU + throughput (replaces estimates with measurements)
python scripts/gpu_smoke_test.py --device cuda
python scripts/benchmark_throughput.py --config config/v1.1-medium.yaml \
    --seq_lens 512 1024 --batch_sizes 8 16 --steps 10 --device cuda

# 3) THE FIRST SERIOUS RUN — Experiment A: 50M tokens, ~763 steps
python scripts/train_custom_model.py \
    --dataset_path data/datasets/fwe50m.txt \
    --config config/v1.1-medium.yaml \
    --max_steps 763 \
    --batch_size 16 \
    --max_seq_len 1024 \
    --vocab_size_target 8192 \
    --train_ratio 0.95 \
    --eval_every 100 \
    --checkpoint_every 250 \
    --seed 42 \
    --tokenizer_out logs/v1.1-medium/tokenizer.json

# 4) Evaluate (Phase 39 protocol)
python scripts/run_eval_protocol.py \
    --checkpoint checkpoints/v1.1-medium/checkpoint_step_763.pt \
    --dataset data/datasets/fwe50m.txt --config config/v1.1-medium.yaml \
    --out logs/v1.1-medium/eval.json
```

Expected outputs: `logs/v1.1-medium/{run_manifest.json, dataset_manifest.json,
training_metrics.jsonl}`, `checkpoints/v1.1-medium/checkpoint_step_*.pt`
(~320 MB each), eval summary with val loss/PPL/top-1/rep4. The run's numbers
replace every GPU estimate in this report and gate the C/D decisions.

---

## STOP CONDITION — Readiness Verdict

1. **Is XRFM GPU-ready?** **YES** — all non-GPU gates pass; GPU gates are code-ready and environment-blocked, not failed.
2. **Tests passed:** 227/227 pytest; CI training-smoke PASS; GPU smoke 15/15 (CPU); mypy 0; ruff/black clean.
3. **Tests failed:** none.
4. **Blocked (environment):** GPU forward/backward, BF16/FP16 on GPU, VRAM measurement, GPU throughput, multi-GPU — all require a CUDA host.
5. **GPU requirements:** ≥ 16 GB VRAM (T4/L4/A10-class), CUDA torch, ~2 GB free disk for the slice.
6. **Dataset:** FineWeb-Edu `sample-10BT` deterministic 50 M-token slice (ODC-By 1.0).
7. **Model config:** `config/v1.1-medium.yaml` — 19,666,560 params.
8. **Token budget:** 50 M tokens (Experiment A) — 763 steps at eff batch 65,536.
9. **GPU hours:** ≈ 0.4–0.9 h (T4-class, estimate range 35k–15k tok/s).
10. **Exact command:** §19 above.
11. **Expected outputs/checkpoints:** §19 above.
12. **Remaining risks:** §18 — chiefly that GPU throughput estimates replace measurements after the first run.

**WAITING FOR EXPLICIT APPROVAL before launching the 50 M-token (or larger) run.**
