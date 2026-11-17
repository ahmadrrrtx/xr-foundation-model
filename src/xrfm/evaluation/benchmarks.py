"""
Benchmark framework for XRFM (v0.7.0+).

Provides an extensible abstract base class for running standardized
evaluation benchmarks on GPTModel instances, with automatic device placement,
3-tuple ``(input_ids, target_ids, mask)`` batch support, and ``ignore_index``
padding exclusion.

Conceptual references (not copied):
- EleutherAI lm-evaluation-harness — benchmark framework design
- HuggingFace evaluate — benchmark abstraction pattern
- OpenAI GPT-3 paper — evaluation methodology

Implementation is original.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from xrfm.evaluation.perplexity import _unpack_eval_batch
from xrfm.models.gpt import GPTModel

logger = logging.getLogger("xrfm.evaluation")


def _get_model_device(model: nn.Module) -> torch.device:
    """Return the device hosting ``model``'s parameters (defaults to CPU)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


class Benchmark(ABC):
    """Abstract base class for evaluation benchmarks."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def compute(self, model: GPTModel, **kwargs) -> dict[str, float]:
        """Run the benchmark and return metrics."""
        ...

    def __repr__(self) -> str:
        return f"Benchmark(name='{self.name}')"


class TextCompletionAccuracy(Benchmark):
    """Next-token prediction accuracy on held-out text (excluding ``ignore_index`` padding)."""

    def __init__(self) -> None:
        super().__init__(
            name="text_completion_accuracy",
            description="Top-1 next-token prediction accuracy on held-out text",
        )

    def compute(  # type: ignore[override]  # justified: Benchmark ABC dispatches via **kwargs; concrete classes narrow to explicit params (Phase 29)
        self,
        model: GPTModel,
        dataloader: DataLoader,
        max_batches: int | None = None,
        ignore_index: int = -100,
        **kwargs,
    ) -> dict[str, float]:
        """Compute next-token prediction accuracy excluding ``ignore_index`` targets."""
        model.eval()
        device = _get_model_device(model)
        total_correct: int = 0
        total_tokens: int = 0

        with torch.no_grad():
            for batch_idx, raw_batch in enumerate(dataloader):
                batch_input_ids, batch_target_ids, batch_mask = _unpack_eval_batch(raw_batch)
                valid_mask = batch_target_ids != ignore_index
                num_tokens = int(valid_mask.sum().item())
                if num_tokens == 0:
                    if max_batches is not None and batch_idx + 1 >= max_batches:
                        break
                    continue

                if batch_input_ids.device != device:
                    batch_input_ids = batch_input_ids.to(device)
                if batch_mask is not None and batch_mask.device != device:
                    batch_mask = batch_mask.to(device)

                logits, _ = model(batch_input_ids, mask=batch_mask)
                predictions = logits.argmax(dim=-1)
                if batch_target_ids.device != predictions.device:
                    batch_target_ids = batch_target_ids.to(predictions.device)
                    valid_mask = valid_mask.to(predictions.device)

                correct = int(((predictions == batch_target_ids) & valid_mask).sum().item())

                total_correct += correct
                total_tokens += num_tokens

                if max_batches is not None and batch_idx + 1 >= max_batches:
                    break

        if total_tokens == 0:
            return {"accuracy": 0.0, "total_correct": 0, "total_tokens": 0}

        accuracy = total_correct / total_tokens
        logger.info(
            "Text completion accuracy: %.4f (%d/%d tokens)",
            accuracy,
            total_correct,
            total_tokens,
        )

        return {
            "accuracy": accuracy,
            "total_correct": total_correct,
            "total_tokens": total_tokens,
        }


class TopKAccuracy(Benchmark):
    """Top-k next-token prediction accuracy (excluding ``ignore_index`` padding)."""

    def __init__(self, k: int = 5) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        super().__init__(
            name=f"top{k}_accuracy",
            description=f"Top-{k} next-token prediction accuracy",
        )
        self.k = k

    def compute(  # type: ignore[override]  # justified: Benchmark ABC dispatches via **kwargs; concrete classes narrow to explicit params (Phase 29)
        self,
        model: GPTModel,
        dataloader: DataLoader,
        max_batches: int | None = None,
        ignore_index: int = -100,
        **kwargs,
    ) -> dict[str, float]:
        """Compute top-k accuracy excluding ``ignore_index`` targets."""
        model.eval()
        device = _get_model_device(model)
        total_correct: int = 0
        total_tokens: int = 0

        with torch.no_grad():
            for batch_idx, raw_batch in enumerate(dataloader):
                batch_input_ids, batch_target_ids, batch_mask = _unpack_eval_batch(raw_batch)
                valid_mask = batch_target_ids != ignore_index
                num_tokens = int(valid_mask.sum().item())
                if num_tokens == 0:
                    if max_batches is not None and batch_idx + 1 >= max_batches:
                        break
                    continue

                if batch_input_ids.device != device:
                    batch_input_ids = batch_input_ids.to(device)
                if batch_mask is not None and batch_mask.device != device:
                    batch_mask = batch_mask.to(device)

                logits, _ = model(batch_input_ids, mask=batch_mask)
                eff_k = min(self.k, logits.shape[-1])
                _, topk_indices = torch.topk(logits, k=eff_k, dim=-1)

                if batch_target_ids.device != topk_indices.device:
                    batch_target_ids = batch_target_ids.to(topk_indices.device)
                    valid_mask = valid_mask.to(topk_indices.device)

                targets_expanded = batch_target_ids.unsqueeze(-1)
                in_topk = (topk_indices == targets_expanded).any(dim=-1)
                correct = int((in_topk & valid_mask).sum().item())

                total_correct += correct
                total_tokens += num_tokens

                if max_batches is not None and batch_idx + 1 >= max_batches:
                    break

        if total_tokens == 0:
            return {f"top{self.k}_accuracy": 0.0, "total_correct": 0, "total_tokens": 0}

        accuracy = total_correct / total_tokens
        logger.info(
            "Top-%d accuracy: %.4f (%d/%d tokens)",
            self.k,
            accuracy,
            total_correct,
            total_tokens,
        )

        return {
            f"top{self.k}_accuracy": accuracy,
            "total_correct": total_correct,
            "total_tokens": total_tokens,
        }


def run_evaluation_suite(
    model: GPTModel,
    dataloader: DataLoader,
    max_batches: int | None = None,
    ignore_index: int = -100,
) -> dict[str, Any]:
    """Run the full evaluation suite: perplexity + benchmarks."""
    from xrfm.evaluation.perplexity import compute_perplexity

    logger.info("Starting evaluation suite...")
    results: dict[str, Any] = {}

    ppl = compute_perplexity(model, dataloader, max_batches=max_batches, ignore_index=ignore_index)
    results["perplexity"] = ppl

    acc1 = TextCompletionAccuracy()
    results["top1_accuracy"] = acc1.compute(model, dataloader, max_batches=max_batches, ignore_index=ignore_index)

    acc5 = TopKAccuracy(k=5)
    results["top5_accuracy"] = acc5.compute(model, dataloader, max_batches=max_batches, ignore_index=ignore_index)

    logger.info(
        "Evaluation complete: PPL=%.2f, Top-1=%.4f, Top-5=%.4f",
        ppl["perplexity"],
        results["top1_accuracy"]["accuracy"],
        results["top5_accuracy"]["top5_accuracy"],
    )

    return results
