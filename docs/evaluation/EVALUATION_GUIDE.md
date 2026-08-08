# XRFM Evaluation Pipeline — v0.7.0

> **AUDIT REMEDIATION NOTE (2026-08-08):** this document describes the original
> design. A forensic audit found and fixed several issues (implicit causal
> masking, character-level tokenizer, padding-loss, resume/scheduler state,
> API import, version chaos). See `docs/audit/FORENSIC_AUDIT.md`,
> `docs/audit/GAP_ANALYSIS.md`, and `docs/implementation/REMEDIATION_PLAN.md`
> for the authoritative current state. Historical claims below are preserved
> as evidence, not as current truth.


## Overview

The evaluation pipeline provides intrinsic metrics for language model quality:

| Metric | What It Measures | Formula |
|---|---|---|
| **Perplexity (PPL)** | Model's surprise at held-out text | `exp(avg_cross_entropy)` |
| **Top-1 Accuracy** | How often model picks the right next token | `correct / total_tokens` |
| **Top-5 Accuracy** | How often the right token is in top-5 predictions | `correct_top5 / total_tokens` |

Lower perplexity = better. Higher accuracy = better.

## Quick Start

```python
from torch.utils.data import DataLoader
from model.gpt import GPTModel
from evaluation import compute_perplexity, run_evaluation_suite

model = GPTModel()
val_dataset = XRFMTextDataset("data/datasets/val.txt", tokenizer, split="val")
val_loader = DataLoader(val_dataset, batch_size=8)

# Perplexity only
ppl = compute_perplexity(model, val_loader)
print(f"Perplexity: {ppl['perplexity']:.2f}")

# Full suite (PPL + Top-1 + Top-5)
results = run_evaluation_suite(model, val_loader)
```

## Perplexity

Perplexity is the standard intrinsic metric for language models. It measures how well the model predicts a held-out test set.

### Mathematical Definition

```
PPL = exp( -1/N * sum_i log P(token_i | context_i) )
```

Where:
- N = total number of predicted tokens
- P(token_i | context_i) = model's probability for the correct token

### Interpretation

| PPL | Interpretation |
|---|---|
| ~vocab_size (e.g., 50K) | Random guessing |
| 100–500 | Weak model |
| 30–100 | Moderate |
| 10–30 | Strong |
| < 10 | Very strong (requires training) |

### Strided Evaluation

For texts longer than `max_seq_len`, use strided evaluation:

```python
token_ids = tokenizer.encode(long_text)
ppl = compute_perplexity_strided(model, token_ids, stride=512, max_seq_len=1024)
```

The sliding window with overlap avoids double-counting while providing full context.

## Benchmarks

### TextCompletionAccuracy

Top-1 next-token prediction accuracy on held-out text:

```python
from evaluation.benchmarks import TextCompletionAccuracy

bench = TextCompletionAccuracy()
result = bench.compute(model, val_loader)
print(f"Accuracy: {result['accuracy']:.4f}")
```

### TopKAccuracy

Top-k accuracy (is the correct token in the top-k predictions?):

```python
from evaluation.benchmarks import TopKAccuracy

top5 = TopKAccuracy(k=5)
result = top5.compute(model, val_loader)
print(f"Top-5 Accuracy: {result['top5_accuracy']:.4f}")
```

### Custom Benchmarks

Extend the `Benchmark` base class:

```python
from evaluation.benchmarks import Benchmark


class MyBenchmark(Benchmark):
    def __init__(self):
        super().__init__(name="my_benchmark", description="...")

    def compute(self, model, **kwargs):
        # Run evaluation
        return {"metric": value}
```

## Architecture

```
evaluation/
├── __init__.py          # Package exports
├── perplexity.py        # PPL computation (basic + strided)
└── benchmarks.py        # Benchmark ABC + built-in benchmarks
```

## Design Decisions

1. **Token-level aggregation:** Loss is summed across all tokens, then averaged — correct for variable-length batches.
2. **Strided overlap handling:** Tokens in overlapping windows are only counted once (first occurrence).
3. **No external benchmark datasets:** Phase 7 ships with intrinsic metrics only. Standard benchmarks (MMLU, HellaSwag) require external data files — the framework supports them as plugins.
4. **Extensible base class:** `Benchmark` ABC allows adding new benchmarks without modifying the evaluation harness.

## References

- Hoffmann et al. (2022) — Chinchilla: perplexity-based evaluation
- Brown et al. (2020) — GPT-3: top-1 and top-5 accuracy reporting
- HuggingFace evaluate — perplexity API design
- EleutherAI lm-evaluation-harness — benchmark framework design
