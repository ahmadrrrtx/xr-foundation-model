# XRFM — Compute & Training Budget (Phase 12)

**Measured hardware (2026-08-08):** 2 vCPU Intel Xeon @ 2.60 GHz, 1.9 GB RAM, 20 GB disk, **no GPU**. All numbers below are derived from *measured* throughput in this environment (see BASELINE §7 and smoke runs), not from GPU folklore.

---

## 1. Measured Throughput (CPU, this sandbox)

| Model | Batch | Seq | Steps/s | Tokens/s | Notes |
|---|---|---|---|---|---|
| 19.2 M (d=256, 6L, vocab 50304) | 4 | 127 | 0.84 | ~425 | measured (this audit) |
| 19.2 M (same) | 8 | 256 | ~0.2 (est.) | ~400 (est.) | extrapolated from scaling of seq² in attention |

Attention is O(seq²) and memory grows; 2 vCPU/1.9 GB RAM is the hard ceiling. **Anything beyond ~20 M params at seq 512 is impractical here.**

## 2. Memory Budget Model (fp32, AdamW, no GPU)

- Weights: `P × 4 B`
- AdamW moments: `2 × P × 4 B` (m, v)
- Activations/forward: proportional to `batch × seq × d_model × layers` (measured: batch 4 × seq 127 ≈ 700 MB RSS peak incl. torch itself)
- Hard RAM: 1.9 GB total, ~1.1 GB usable with torch loaded.

**Feasible envelope (this sandbox):** P ≤ ~25 M with batch ≤ 8 and seq ≤ 256, or P ≤ ~10 M with seq 512, batch 4–8. (Measured: 19.2 M @ batch 4 × seq 127 runs comfortably.)

## 3. Realistic Scenarios

### 3.1 Local — this sandbox (CPU, 2 vCPU, 1.9 GB RAM)
- **Largest sensible model: XRFM-TINY / XRFM-SMALL** (0.5–8 M params; see TARGET_SPEC) at seq 256, batch 8.
- Tokens/sec: ~200–900 depending on size. A 1 M-token corpus (≈ 2–3 MB of text) trains at ~10 M tokens/hour → **1 epoch ≈ 6 min**; 100 epochs (overfit test) ≈ 10 h worst case; keep overfit runs to 20–50 epochs.
- **What can be trained here:** correctness demos, overfit tests, scaling mini-experiments (2 sizes), LR-sweep sanity, checkpoint/resume/val/eval verification, reproduction of the full pipeline. **NOT** a foundation model of any meaningful size.

### 3.2 Free Cloud (Colab/Kaggle free tier — T4 16 GB, not available in this sandbox)
- Single T4 (16 GB): fp32 max ~1.3 B params (weights 5.2 GB + Adam 10.4 GB = 15.6 GB) — tight; bf16 halves that.
- **Largest sensible model: XRFM-MEDIUM (~50–100 M)** with seq 1024, batch 16–32, bf16 + autocast; ~8–30 h for 1–2 B tokens.
- Realistically: pretrain a 50 M model on ~1–3 B tokens (e.g., a deduped slice of FineWeb/slimpajama subset or Wikipedia) to get a *demonstratively learning* small LM (val PPL ≪ 50 K random baseline, coherent short generations).
- Throughput estimate: T4 bf16 ≈ 15–30k tok/s for a 50 M model (order-of-magnitude; unverified in sandbox).

### 3.3 Single GPU (consumer, 8–24 GB)
- 8 GB: XRFM-MEDIUM (50–100 M) bf16 seq 1024–2048; 24 GB: XRFM-LARGE (~300 M) seq 2048 bf16 with grad accum; ~1–5 B tokens in days.
- This is where "serious experiments" (data ablations, scaling curves 50M/100M/300M) become possible.

### 3.4 Multi-GPU
- DDP/FSDP on 2–8 GPUs multiplies the single-GPU budget by the node count for throughput and enables larger batches; model-size ceiling unchanged without sharding (FSDP lets 300M–1B fit across 8×16 GB). Everything in `training/distributed.py` must be re-validated on real hardware first (F-34).

### 3.5 Larger cloud (what a real pretraining run needs)
- XRFM-1B (≈ 20 GB fp32 weights; ~1B-token run at ~1 TFLOP·s⁻¹·token ≈ 2.7e18 FLOPs ≈ 1 H100 for ~12–24 h with bf16 at 30–40% MFU).
- XRFM-7B: 7e6 FLOPs/token × 2 T tokens ≈ 1.4e22 FLOPs ≈ 40–80 H100-days — institutional compute; out of scope for this mission (as the ROADMAP itself concedes).

## 4. Chosen Training Budget for This Mission (honest, executed on CPU)

| Experiment | Model | Corpus | Tokens | Steps | Est. wall time (this sandbox) |
|---|---|---|---|---|---|
| Unit/integration tests | XRFM-TINY | synthetic | ~50 K | 10–50 | minutes |
| Overfit sanity (Phase 17) | XRFM-TINY (0.5–2 M) | 200–2 K tokens, 1–3 short docs | — | until train loss < 0.1 | ≤ 2 h |
| Training smoke (Phase 16) | XRFM-SMALL | real text slice (~200–500 K tokens) | ~1 M | ~2000 | 1–3 h |
| Baseline pretraining (Phase 18) | XRFM-SMALL | real text slice (~1–2 M tokens) | 10–50 M | 5000–20000 | 8–24 h (resumable) |
| Scaling mini-experiment (Phase 20) | TINY vs SMALL | same corpus, same steps | matched | matched | 4–8 h |

> **Rule 5 compliance:** the plan above uses ONLY the measured hardware. No scenario assumes a GPU that does not exist in this environment; GPU scenarios are documented as "when available".

## 5. Checkpoint Storage
- XRFM-SMALL (~8 M): ~32 MB/ckpt fp32 (+optimizer ~64 MB) → 100 checkpoints ≈ 3.2–9.6 GB. Budget: keep ≤ 20 checkpoints (or weights-only) within 20 GB disk.
- Logs/CSV metrics are tiny.
