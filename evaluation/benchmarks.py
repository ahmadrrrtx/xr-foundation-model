"""
Benchmark framework for XRFM (v0.7.0).

Provides an extensible abstract base class for running standardized
evaluation benchmarks on GPTModel instances.

Built-in benchmarks:
- TextCompletionAccuracy: Measures next-token prediction accuracy
  on held-out text (how often the model picks the correct next token).

The framework is designed to be extended with standard benchmarks
(MMLU, HellaSwag, etc.) without rewriting the evaluation harness.

Conceptual references (not copied):
- EleutherAI lm-evaluation-harness — benchmark framework design
- HuggingFace evaluate — benchmark abstraction pattern
- OpenAI GPT-3 paper — evaluation methodology

Implementation is original.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import DataLoader

from model.gpt import GPTModel

logger = logging.getLogger("xrfm.evaluation")


class Benchmark(ABC):
    """Abstract base class for evaluation benchmarks.

    Each benchmark defines:
    - A name and description
    - A compute() method that runs the benchmark on a model
    - A set of metrics returned as a dict

    Subclasses implement compute() to perform task-specific evaluation.
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def compute(self, model: GPTModel, **kwargs) -> dict[str, float]:
        """Run the benchmark and return metrics.

        Args:
            model: GPTModel to evaluate (should be in eval mode).
            **kwargs: Benchmark-specific parameters.

        Returns:
            Dict mapping metric names to float values.
        """
        ...

    def __repr__(self) -> str:
        return f"Benchmark(name='{self.name}')"


class TextCompletionAccuracy(Benchmark):
    """Next-token prediction accuracy on held-out text.

    Measures how often the model's most-probable next token matches
    the actual next token in the dataset. Higher accuracy means the
    model better predicts the held-out text.

    This is a simple intrinsic benchmark — it measures the model's
    ability to continue text it hasn't seen during training.

    Accuracy = (# correct top-1 predictions) / (total tokens evaluated)

    Reference range for small models (10M–100M params):
    - Random: 1/vocab_size ≈ 0.002%
    - Weak: 1–5%
    - Moderate: 5–15%
    - Strong: 15–30%
    - Very strong: 30%+ (unlikely at XRFM scale without training)
    """

    def __init__(self) -> None:
        super().__init__(
            name="text_completion_accuracy",
            description="Top-1 next-token prediction accuracy on held-out text",
        )

    def compute(
        self,
        model: GPTModel,
        dataloader: DataLoader,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """Compute next-token prediction accuracy.

        Args:
            model: GPTModel to evaluate.
            dataloader: DataLoader yielding (input_ids, target_ids) batches.
            max_batches: Optional batch limit.

        Returns:
            Dict with "accuracy", "total_correct", "total_tokens".
        """
        model.eval()
        total_correct: int = 0
        total_tokens: int = 0

        with torch.no_grad():
            for batch_idx, (batch_input_ids, batch_target_ids) in enumerate(dataloader):
                # Forward pass
                logits, _ = model(batch_input_ids)

                # Predicted token = argmax over vocab at each position
                predictions = logits.argmax(dim=-1)  # (batch, seq)

                # Compare with targets
                correct = (predictions == batch_target_ids).sum().item()
                num_tokens = batch_target_ids.numel()

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
    """Top-k next-token prediction accuracy.

    Measures whether the correct next token appears in the model's
    top-k predictions. Top-5 accuracy is commonly reported alongside
    top-1 for language model evaluation.

    Reference: Brown et al. (2020) — GPT-3 reports Top-1 and Top-5.
    """

    def __init__(self, k: int = 5) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        super().__init__(
            name=f"top{k}_accuracy",
            description=f"Top-{k} next-token prediction accuracy",
        )
        self.k = k

    def compute(
        self,
        model: GPTModel,
        dataloader: DataLoader,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """Compute top-k accuracy.

        Args:
            model: GPTModel to evaluate.
            dataloader: DataLoader yielding (input_ids, target_ids) batches.
            max_batches: Optional batch limit.

        Returns:
            Dict with "top{k}_accuracy", "total_correct", "total_tokens".
        """
        model.eval()
        total_correct: int = 0
        total_tokens: int = 0

        with torch.no_grad():
            for batch_idx, (batch_input_ids, batch_target_ids) in enumerate(dataloader):
                logits, _ = model(batch_input_ids)

                # Get top-k indices at each position
                _, topk_indices = torch.topk(logits, k=self.k, dim=-1)
                # topk_indices: (batch, seq, k)

                # Check if target is in top-k
                targets_expanded = batch_target_ids.unsqueeze(-1)  # (batch, seq, 1)
                correct = (topk_indices == targets_expanded).any(dim=-1).sum().item()

                total_correct += correct
                total_tokens += targets_expanded.numel()

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
) -> dict[str, Any]:
    """Run the full evaluation suite: perplexity + benchmarks.

    Args:
        model: GPTModel to evaluate.
        dataloader: Validation/test DataLoader.
        max_batches: Optional batch limit for quick evaluation.

    Returns:
        Dict with all metric results keyed by benchmark name.
    """
    from evaluation.perplexity import compute_perplexity

    logger.info("Starting evaluation suite...")
    results: dict[str, Any] = {}

    # 1. Perplexity
    ppl = compute_perplexity(model, dataloader, max_batches=max_batches)
    results["perplexity"] = ppl

    # 2. Top-1 accuracy
    acc1 = TextCompletionAccuracy()
    results["top1_accuracy"] = acc1.compute(model, dataloader, max_batches=max_batches)

    # 3. Top-5 accuracy
    acc5 = TopKAccuracy(k=5)
    results["top5_accuracy"] = acc5.compute(model, dataloader, max_batches=max_batches)

    logger.info(
        "Evaluation complete: PPL=%.2f, Top-1=%.4f, Top-5=%.4f",
        ppl["perplexity"],
        results["top1_accuracy"]["accuracy"],
        results["top5_accuracy"]["top5_accuracy"],
    )

    return results
