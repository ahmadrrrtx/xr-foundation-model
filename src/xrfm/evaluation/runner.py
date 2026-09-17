"""
Evaluation runner for XRFM.

Stable package-level evaluation boundary:

    from xrfm.evaluation import evaluate

    results = evaluate(model, val_dataloader)
    # {"perplexity": {...}, "top1_accuracy": {...}, "top5_accuracy": {...}}

Evaluation consumes only a model and batches — never a trainer — so future
benchmark systems can call ``evaluate`` independently of the training loop.
Result serialization is JSON-ready (plain dicts of floats).
"""

from __future__ import annotations

import logging
from typing import Any

from torch.utils.data import DataLoader

from xrfm.evaluation.benchmarks import run_evaluation_suite
from xrfm.evaluation.perplexity import compute_perplexity
from xrfm.models.gpt import XRFMModel

logger = logging.getLogger("xrfm.evaluation")

__all__ = ["EvaluationRunner", "evaluate"]


def evaluate(
    model: XRFMModel,
    dataloader: DataLoader,
    max_batches: int | None = None,
    benchmarks: bool = True,
) -> dict[str, Any]:
    """Evaluate a model: perplexity + (optionally) top-1/top-5 accuracy.

    Args:
        model: Model to evaluate (switched to ``eval()`` internally).
        dataloader: Loader of ``(input_ids, target_ids)`` batches with
            next-token-shifted targets (e.g. from ``xrfm.data.TextDataset``).
        max_batches: Cap on batches (quick checks).
        benchmarks: Include top-1/top-5 accuracy benchmarks.

    Returns:
        JSON-serializable dict of metrics.
    """
    if not isinstance(model, XRFMModel):
        logger.warning(
            "evaluate() expects an XRFMModel; got %s — proceeding if forward() is compatible", type(model).__name__
        )
    return (
        run_evaluation_suite(model, dataloader, max_batches=max_batches)
        if benchmarks
        else compute_perplexity(model, dataloader, max_batches=max_batches)
    )


class EvaluationRunner:
    """Thin stateful wrapper for repeated evaluations with fixed settings."""

    def __init__(self, max_batches: int | None = None, benchmarks: bool = True) -> None:
        self.max_batches = max_batches
        self.benchmarks = benchmarks

    def run(self, model: XRFMModel, dataloader: DataLoader) -> dict[str, Any]:
        return evaluate(model, dataloader, max_batches=self.max_batches, benchmarks=self.benchmarks)
