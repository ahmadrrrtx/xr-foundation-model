"""
XRFM Evaluation Pipeline (v0.7.0).

Provides intrinsic evaluation metrics for language models:
- Perplexity (PPL) — exponential of average cross-entropy
- Text completion accuracy (top-1 and top-k)

Usage:
    from evaluation import compute_perplexity, run_evaluation_suite
    from evaluation.benchmarks import TextCompletionAccuracy, TopKAccuracy

    ppl = compute_perplexity(model, val_dataloader)
    results = run_evaluation_suite(model, val_dataloader)
"""

from evaluation.benchmarks import (
    Benchmark,
    TextCompletionAccuracy,
    TopKAccuracy,
    run_evaluation_suite,
)
from evaluation.perplexity import (
    compute_perplexity,
    compute_perplexity_strided,
    evaluate_checkpoint,
)

__all__ = [
    "compute_perplexity",
    "compute_perplexity_strided",
    "evaluate_checkpoint",
    "Benchmark",
    "TextCompletionAccuracy",
    "TopKAccuracy",
    "run_evaluation_suite",
]
