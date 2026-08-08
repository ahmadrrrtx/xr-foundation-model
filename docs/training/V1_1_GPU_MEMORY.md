# XRFM v1.1 — GPU Memory Estimation (Phase 36)

**Date:** 2026-08-08 · **Model:** XRFM-MEDIUM (19,666,560 params; verified programmatically)
**Method:** analytical memory budgeting with documented assumptions. Not measured
on GPU (no GPU in this environment — GPU numbers are ESTIMATES, not measurements).

---

## 1. Model (XRFM-MEDIUM)

| Component | Params |
|---|---|
| embedding (tied with lm_head) | 3,145,728 (vocab 8192 × d 384) |
| per layer (attn 589,824 + norms 768 + SwiGLU 1,769,472) | 2,360,064 |
| 7 layers | 16,520,448 |
| final norm | 384 |
| **total** | **19,666,560 (19.67 M)** |

## 2. Memory per Parameter (bytes)

| Precision | weights | gradients | AdamW states (m, v) | optimizer total |
|---|---|---|---|---|
| FP32 | 4 | 4 | 8 | 16 |
| FP16 | 2 | 2 | 8 (Adam states kept fp32) | 12 |
| BF16 | 2 | 2 | 8 (Adam states kept fp32) | 12 |

So for 19.67 M params:

| Precision | weights | grads | optimizer | model+opt (no grads) | with grads |
|---|---|---|---|---|---|
| FP32 | 78.7 MB | 78.7 MB | 157.3 MB | 236 MB | 314.7 MB |
| FP16/BF16 | 39.3 MB | 39.3 MB | 157.3 MB | 196.7 MB | 235.9 MB |

## 3. Activations (per micro-batch, fp32 in the fwd pass unless autocast)

Dominant term for a decoder transformer: attention and FFN activations per token.

Per token, per layer, in dims (no batch): ~
- attention: `2 × seq × d_model × n_heads` for scores kept only with SDPA-math; with
  `scaled_dot_product_attention` (memory-efficient / flash) score matrices are not
  materialized → attention activation ≈ `seq × d_model` × constant ≈ small.
- FFN: `seq × d_ff` activations × a few tensors ≈ `3 × seq × d_ff`.

**Assumption (documented):** with SDPA backend (flash/mem-efficient on GPU), the
attention score matrix is NOT stored, so activations ≈ `seq × (d_model + 3·d_ff) × layers × batch × 4 B`.

For seq=1024, batch=16, fp32:
`16 × 1024 × (384 + 3×1536) × 7 × 4 B = 16 × 1024 × 4992 × 7 × 4 = 2.29 GB` — too big for 16 GB with
optimizer+weights. With **bf16 autocast**, activations halve (~1.15 GB) and the
memory-efficient SDPA path avoids the O(seq²) score matrix (else seq²×heads×batch×4 B =
16×1024²×6×4 B = 402 MB per layer × 7 = 2.8 GB — avoided by SDPA).

**Recommendation (estimate):** batch 16 × seq 1024 × grad_accum 4 (effective 64) with
bf16 autocast fits a **16 GB GPU** comfortably (see §5); batch 8 × seq 1024 fits an 8 GB GPU.

## 4. KV Cache (inference, per sequence)

Per token: `2 × d_model × layers` values (K+V). For MEDIUM:
`2 × 384 × 7 = 5,376 B/token` → seq 1024 ≈ 5.5 MB/sequence (bf16 → 2.75 MB).
Negligible at this scale (not a bottleneck; measured KV-cache behavior in docs/benchmarks/KV_CACHE.md).

## 5. Estimated Peak VRAM Budgets (bf16 autocast, SDPA)

| GPU | FP32 batch/seq | BF16 batch/seq | Notes |
|---|---|---|---|
| T4 16 GB | 8 × 512 | **16 × 1024** (accum 4) | est. 9–11 GB peak |
| L4 / A10 24 GB | 16 × 512 | **32 × 1024** (accum 2) | est. 13–16 GB peak |
| A100 40 GB | — | 64 × 1024 (accum 1) | est. 20–24 GB peak |
| A100 80 GB | — | 128 × 1024 | est. 35–40 GB peak |

Rule of thumb applied: peak ≈ model+opt (236 MB fp32 / 196 MB bf16) + activations +
1–2 GB runtime/allocator overhead + dataloader pinned buffers (≤ 1 GB).

**Max batch estimate on the detected-GPU-absent environment:** not measurable
(ENVIRONMENT BLOCKED). On a 16 GB T4, estimated max micro-batch at seq 1024 is
16–32 with bf16 + SDPA; to be confirmed by the Phase 37 throughput benchmark on a GPU host.

## 6. Documentation of Assumptions

1. SDPA memory-efficient/flash backend (no materialized attention matrix). If the
   math backend were forced, add `seq²×heads×batch×4 B` per layer.
2. AdamW states always fp32 (standard); bf16 halves only weights+activations+grads.
3. Dataloader pinned buffers ≤ 1 GB; `pin_memory=true` trades host RAM (plenty on
   GPU hosts) for faster H2D.
4. No activation checkpointing assumed. If a GPU comes up short, enabling
   `activation_checkpointing` would trade compute for ~60–70% activation savings.
5. All numbers are estimates from the model's exact parameter count; measured VRAM
   will be recorded by `scripts/gpu_smoke_test.py` (check 12) and
   `scripts/benchmark_kv_cache.py` (`--device cuda`) on a GPU host.
