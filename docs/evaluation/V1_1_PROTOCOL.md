# XRFM v1.1 — Evaluation Protocol (Phase 39)

**Date:** 2026-08-08 · Applies to all v1.1 experiments (A–D in `docs/training/V1_1_BUDGET.md`).
Every evaluation is reported with the full protocol row:
MODEL / PARAMETERS / TOKENS / DATA / CONTEXT / COMPUTE / EVAL-SET / METRIC.

---

## 1. Held-Out Data (never trained on)

- The dataset (Phase 34) is split at **document boundaries**: train 95%,
  **val 2.5%, test 2.5%**.
- val/test slices are **frozen** at dataset creation time (files + sha256
  recorded in the Phase-33 manifest) and are never mixed into training data.
- The tokenizer is trained **on the train split only** (no contamination).
- All numbers below are reported on the **val slice** (model selection) and the
  **test slice** (final report, evaluated exactly once at the end).

## 2. Intrinsic Metrics

| Metric | Definition | Padding handling |
|---|---|---|
| validation loss | mean cross-entropy over non-padded val tokens | `ignore_index=-100` (implemented) |
| perplexity | `exp(val_loss)` | same |
| top-1 accuracy | fraction of positions where argmax logit == target | padded positions excluded |

Implementation: `evaluation/perplexity.py` (masked), `evaluation/benchmarks.py`
(TextCompletionAccuracy / TopKAccuracy). Run via `scripts/run_eval_protocol.py`.

**Baseline for sanity:** random-init model on the same val slice ≈ `ln(vocab)`
loss (e.g., vocab 8192 → loss ≈ 9.01, PPL ≈ 8192). Every v1.1 report must show
the random baseline alongside the trained number.

## 3. Generation Protocol (fixed, reproducible)

- **Fixed prompts** (the same 10 prompts for every experiment; stored in
  `scripts/run_eval_protocol.py`): prose, code, question, list-style, dialogue.
- **Fixed decoding parameters:** greedy (T=0) and temperature 0.8 with top-p 0.9;
  max_new_tokens = 64; repetition_penalty ∈ {1.0, 1.2} (two columns).
- **Outputs recorded verbatim** (no cherry-picking) in `logs/eval_<run_id>/`.

## 4. Repetition & Degeneration Analysis

For each generation, compute:
- `rep4` = fraction of 4-grams that repeat within the generated span (n-gram
  diversity, standard small-model degeneration metric);
- mean output length before EOS/stop (EOS behavior);
- fraction of runs hitting max_new_tokens vs stopping via EOS/stop-sequence.

## 5. EOS / Stop Behavior

- With `eos_id` from the tokenizer passed as `stop_token_id`, and optional
  `stop_sequences` (Phase 31), the protocol records: how often generation
  terminates by (a) EOS, (b) stop sequence, (c) length cap. For the v1.1
  byte-level tokenizer, EOS is `<|eos|>` (reserved id); its emission rate is
  expected to be low until the model is trained with EOS supervision.

## 6. Metric Reporting Template

```markdown
| field | value |
|---|---|
| model | XRFM-MEDIUM |
| params | 19,666,560 |
| training tokens | 50,000,000 |
| data | fineweb-edu-sample-10BT slice v1 (sha256 …) |
| context | 1024 |
| compute | <GPU>, bf16, <tokens/sec> |
| eval set | val slice (2.5%), N tokens |
| val loss | … |
| val PPL | … |
| top-1 acc | … |
| random baseline PPL | 8192 |
```

## 7. Execution

```bash
python scripts/run_eval_protocol.py --checkpoint checkpoints/v1.1-medium/<ckpt> \
    --dataset data/datasets/<slice> --out logs/eval_v1_1_run1
```
