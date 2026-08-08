# XRFM v1.1 — Training Budget (Phase 38)

**Date:** 2026-08-08
**Rule:** budgets are derived from MEASURED throughput where available (CPU, this
sandbox) and from EXPLICITLY LABELED estimates elsewhere (GPU — no GPU in this
environment; every GPU number is an estimate with its assumption stated).

---

## 1. Measured Throughput (this sandbox, CPU)

| Model | Params | seq | batch | tok/s | Note |
|---|---|---|---|---|---|
| XRFM-SMALL (vocab 2048) | 1,318,528 | 256 | 4 | ~2,850 | measured (benchmark_throughput.py) |
| XRFM-SMALL (vocab 2048) | 1,318,528 | 512 | 4 | ~3,358 | measured |
| XRFM-MEDIUM (19.7 M) | 19,666,560 | ≥256/batch≥4 | — | **MEMORY BLOCKED** | kernel-OOM in 1.9 GB sandbox; MEDIUM requires a GPU host |

CPU conclusion: SMALL-class trains at ~3k tok/s; MEDIUM-class is not trainable
in this sandbox at any useful size.

## 2. GPU Throughput Estimates (not measured — assumptions stated)

For XRFM-MEDIUM (19.67 M params):
- Training FLOPs/token ≈ 6·N = **118 MFLOPs/token** (Kaplan/Hoffmann convention).
- T4 (fp16/bf16 ≈ 65 TFLOPS dense): at an assumed **15–35% MFU** (small models
  are latency/launch-bound, so MFU is lower than for large models) →
  **15k–35k tok/s** (estimate range, to be replaced by Phase-37 measurements on a GPU host).
- A100/H100: assume 60–100k tok/s (estimate).

> These are the ONLY budget numbers that gate the decision to start expensive
> training. The mission explicitly does not start that training until a GPU host
> provides measured throughput (Phase 37).

## 3. Step Accounting (MEDIUM config: eff batch = 16 × 4 accum = 64 × seq 1024 = 65,536 tokens/step)

| Experiment | Tokens | Steps (eff 65,536 tok/step) | ~GPU h @15k tok/s | ~GPU h @35k tok/s | Full checkpoints (every 2,000 steps) | Storage (full ckpt ≈ 320 MB) |
|---|---|---|---|---|---|---|
| **A** | 50 M | 763 | 0.9 h | 0.4 h | 1 | 0.3 GB |
| **B** | 100 M | 1,526 | 1.9 h | 0.8 h | 1 | 0.3 GB |
| **C** | 500 M | 7,629 | 9.3 h | 4.0 h | 4 | 1.3 GB |
| **D** | 1 B | 15,259 | 18.5 h | 7.9 h | 8 | 2.6 GB |

(Checkpoint: 19.67 M params → weights 78.7 MB fp32 / 39.3 MB bf16 + AdamW 157.3 MB
fp32 + overhead ≈ 240–320 MB full checkpoint.)

## 4. Validation Frequency (recommended)

- `eval_every = 1,000` steps → for B (1,526 steps): 2 validation runs; for C (7,629): 8; for D (15,259): 15.
- Val cost: one forward over ~2.5% held-out slice ≈ 2.5 M tokens at the same
  throughput → minutes per eval, negligible vs training.

## 5. Decision Framework (rationale, not a promise to run all)

| Experiment | Purpose | Decision trigger |
|---|---|---|
| A (50 M) | first serious run; establish protocol + baseline PPL | always run once GPU throughput is measured |
| B (100 M) | learning-curve comparison vs A | run if A's val PPL curve is still improving at the end |
| C (500 M) | real pretraining signal (≈25 tokens/param, near compute-optimal for 20 M) | run if B shows expected scaling (see Phase 41) |
| D (1 B) | stretch: validates the system at scale | run only if C's results justify it and storage/compute allow |

**Recommended first run: Experiment A (50 M tokens)** — ~1–2 GPU-hours on a
T4-class host, produces a full evaluation-protocol result, and directly feeds
the Phase 41 scaling study (A = the MEDIUM point).

## 6. Recompute Instructions

When a GPU becomes available:
```bash
# replace the GPU estimates with measured numbers
python scripts/benchmark_throughput.py --config config/v1.1-medium.yaml \
    --seq_lens 512 1024 --batch_sizes 8 16 --steps 10 --device cuda
python scripts/gpu_smoke_test.py --device cuda
```
Then rerun the budget arithmetic above with the measured tok/s.
