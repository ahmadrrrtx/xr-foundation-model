# XRFM v1.1 — Scaling Experiment Design (Phase 41)

**Date:** 2026-08-08 · **Goal:** determine whether XRFM continues to exhibit
sensible scaling (larger model → lower val loss at equal data) across
**TINY / SMALL / MEDIUM**, using a controlled protocol.

---

## 1. Design Principles

1. **Keep every variable constant except model size.** Same tokenizer, same
   dataset slice, same token budget, same optimizer/schedule, same batch/seq,
   same evaluation slice.
2. **Match the token budget per size** (each size trains the SAME number of
   tokens), not per-parameter (a per-parameter budget would confound scale and
   data). The Phase-20 mini-run (TINY vs SMALL at 1.64 M tokens) showed the
   expected direction (PPL 777 → 586); v1.1 formalizes it with MEDIUM.
3. **Report the full protocol row** for every point (mission rule).

## 2. Experiment Matrix

| Point | Config | Params (verified) | Tokens | Context | Eval slice |
|---|---|---|---|---|---|
| TINY | `config/tiny.yaml` (vocab 2048) | ~0.26 M | 50 M (A-budget) | 256 | val slice (frozen) |
| SMALL | `config/config.yaml` (vocab 2048) | 1.32 M | 50 M | 256 | val slice (frozen) |
| MEDIUM | `config/v1.1-medium.yaml` (vocab 8192) | 19.67 M | 50 M | 1024 | val slice (frozen) |

> **Vocab caveat (documented):** TINY/SMALL use vocab 2048; MEDIUM uses 8192
> (matching the v1.1 tokenizer upgrade). This is a deliberate deviation —
> keeping vocab at 2048 for a 19.7 M model would leave 16 % of params in a
> too-small embedding (0.79 M of 19.7 M). The comparison remains meaningful
> because the tokenizer is byte-level for both (same text → same byte stream;
> only the merge table width differs). The token-efficiency difference
> (tok/char) between 2048 and 8192 vocab is recorded and reported.

## 3. Controlled Variables

- Tokenizer: byte-level BPE; **trained on the same train-slice text**, at the
  size-appropriate vocab target (2048 for TINY/SMALL, 8192 for MEDIUM).
- Optimizer: AdamW β=(0.9,0.95), wd 0.1, clip 1.0, cosine + warmup (1000 steps),
  peak LR 3e-4 (same schedule across sizes; per-size LR tuning is a follow-up).
- Batch/seq: TINY/SMALL eff batch 64 × seq 256; MEDIUM eff batch 64 × seq 1024
  (same tokens/step = 65,536). Context differs because MEDIUM is built for 1024;
  documented as a controlled deviation.
- Seed 42 everywhere. Deterministic dataloader ordering.
- Eval: same held-out val slice, evaluated at each size's training context
  (protocol rule), masked PPL + top-1.

## 4. Metrics Captured per Point

- parameter count (programmatic)
- training tokens (exact)
- validation loss / perplexity / top-1 accuracy (protocol runner)
- throughput (tok/s) and GPU-hours (if GPU) or CPU-hours
- compute: `6·N·T` FLOPs (reported for fair comparison)
- full generation samples + rep4/EOS stats (protocol)

## 5. Success Criteria / Interpretation

| Outcome | Interpretation |
|---|---|
| val PPL(MEDIUM) < val PPL(SMALL) < val PPL(TINY) at equal tokens | XRFM scales sensibly; proceed to C/D budgets |
| PPL gains flatten or invert | investigate: LR mis-scaled for size, vocab mismatch, or data-limited regime |
| MEDIUM diverges/unstable | check init, LR, warmup; see hardening findings |

## 6. Execution (once GPU is available)

```bash
# per size, same 50M-token budget (A in V1_1_BUDGET)
python scripts/train_custom_model.py --dataset_path <slice> \
    --max_steps <763> --batch_size 16 --max_seq_len <256|1024> \
    --vocab_size_target <2048|8192> --config config/<tiny|config|v1.1-medium>.yaml
python scripts/run_eval_protocol.py --checkpoint <ckpt> --dataset <slice> ...
```
(Add a `--config` flag to `train_custom_model.py` if not present — see follow-up note.)
