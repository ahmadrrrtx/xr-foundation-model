# XRFM — KV Cache Benchmark (Phase 32)

**Date:** 2026-08-08 · **Machine:** CPU-only sandbox (2 vCPU Xeon 2.6 GHz, 1.9 GB RAM, no GPU)
**Reproducible protocol:** `scripts/benchmark_kv_cache.py` — fixed model config
(`config/config.yaml`), fixed tokenizer (`tokenizer/vocab.json`), fixed prompt
(fresh random prompt per run, seed 0), fixed repetitions, greedy decoding.

## 1. Protocol

- Model: XRFM-SMALL (1,318,528 params; d_model 128, 4 layers, 4 heads, seq 256, vocab 2048)
- Weights: `checkpoints/checkpoint_step_5000.pt` (trained baseline)
- Prompt length: 32 tokens · Generated: 32 tokens · Repetitions: 3 (averaged)
- With cache: `GenerationEngine.generate` (prefill + KV-cached single-token steps)
- Without cache: recompute the full growing prefix every step
- Device: CPU (no GPU in this environment — VRAM is **ENVIRONMENT BLOCKED**)

## 2. Results (measured)

| Mode | avg wall (s) | ms/token | tokens/sec |
|---|---|---|---|
| **With KV cache** | 0.0754 | 2.36 | **424.6** |
| Without KV cache | 0.1504 | 4.70 | 212.8 |
| **Speedup** | — | — | **2.0×** |

Peak VRAM: not measurable on CPU (would be measured with `--device cuda` on a GPU host).

## 3. Interpretation

- The measured 2.0× matches the theoretical expectation for short sequences on CPU:
  attention is O(seq²) per step, and cache reuse eliminates the quadratic
  recomputation, but at seq ≤ 64 the constant overheads (Python loop, tensor ops)
  dominate, so the speedup is below the asymptotic O(seq) gain.
- **Scaling expectation (documented, not measured here):** at longer contexts
  (seq 512–2048) and on GPU, the advantage grows toward O(seq) — typically
  10–50× at 1–2 K context — because matmul throughput dominates and the
  recompute path grows quadratically. This will be verified on GPU
  (ENVIRONMENT BLOCKED here).
- Generation remains ~2.4 ms/token on CPU for this small model; the inference
  bottleneck on CPU is per-step Python/op overhead, not raw matmuls.

## 4. How to Reproduce

```bash
# CPU (this environment)
python scripts/benchmark_kv_cache.py --checkpoint checkpoints/checkpoint_step_5000.pt \
    --prompt_len 32 --new_tokens 32 --reps 3

# GPU host (records peak VRAM too)
python scripts/benchmark_kv_cache.py --device cuda --checkpoint <ckpt> \
    --prompt_len 128 --new_tokens 128 --reps 5
```

Raw result archive: `logs/kv_benchmark.json` (written by the script with `--out`).
