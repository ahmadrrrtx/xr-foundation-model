"""XRFM evaluation subsystem.

Public surface:

    from xrfm.evaluation import evaluate, compute_perplexity, run_evaluation_suite

``evaluate`` (from :mod:`xrfm.evaluation.runner`) is the package-level
entry point; perplexity and the benchmark ABC (``Benchmark``,
``TextCompletionAccuracy``, ``TopKAccuracy``) are importable for building
custom benchmarks. Evaluation depends only on models + data batches —
never on the trainer.
"""

from xrfm.evaluation.benchmarks import (
    Benchmark,
    TextCompletionAccuracy,
    TopKAccuracy,
    run_evaluation_suite,
)
from xrfm.evaluation.perplexity import (
    compute_perplexity,
    compute_perplexity_strided,
    evaluate_checkpoint,
)
from xrfm.evaluation.runner import EvaluationRunner, evaluate

__all__ = [
    "Benchmark",
    "EvaluationRunner",
    "TextCompletionAccuracy",
    "TopKAccuracy",
    "compute_perplexity",
    "compute_perplexity_strided",
    "evaluate",
    "evaluate_checkpoint",
    "run_evaluation_suite",
]
