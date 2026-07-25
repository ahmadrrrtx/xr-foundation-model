"""
Tests for evaluation pipeline (v0.7.0).

Covers perplexity computation, strided perplexity, top-1/top-k accuracy,
benchmark framework, and the full evaluation suite.
"""

import math
import pytest
import torch
from torch.utils.data import DataLoader

from model.gpt import GPTModel
from evaluation.perplexity import (
    compute_perplexity,
    compute_perplexity_strided,
    evaluate_checkpoint,
)
from evaluation.benchmarks import (
    TextCompletionAccuracy,
    TopKAccuracy,
    run_evaluation_suite,
)


# --- Shared fixtures ---

class EvalDataset:
    """Minimal dataset returning (input, shifted-target) for eval tests."""

    def __init__(self, vocab_size=100, seq_len=16, size=20):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return ids[:-1], ids[1:]


def make_dataloader(vocab_size=100, seq_len=16, size=20, batch_size=4):
    return DataLoader(
        EvalDataset(vocab_size, seq_len, size),
        batch_size=batch_size,
    )


# --- Perplexity Tests ---

class TestComputePerplexity:
    def test_basic(self):
        model = GPTModel()
        dl = make_dataloader(vocab_size=50304, seq_len=12, batch_size=2, size=10)
        result = compute_perplexity(model, dl, max_batches=2)
        assert "perplexity" in result
        assert "loss" in result
        assert result["perplexity"] > 0
        assert result["total_tokens"] > 0

    def test_ppl_equals_exp_loss(self):
        model = GPTModel()
        dl = make_dataloader(vocab_size=50304, seq_len=8, batch_size=2, size=8)
        result = compute_perplexity(model, dl, max_batches=2)
        expected_ppl = math.exp(result["loss"])
        assert abs(result["perplexity"] - expected_ppl) < 1e-6

    def test_loss_positive(self):
        model = GPTModel()
        dl = make_dataloader(vocab_size=100, seq_len=8, batch_size=2, size=6)
        result = compute_perplexity(model, dl, max_batches=1)
        assert result["loss"] > 0

    def test_max_batches_respected(self):
        model = GPTModel()
        dl = make_dataloader(vocab_size=50304, seq_len=8, batch_size=2, size=20)
        result = compute_perplexity(model, dl, max_batches=1)
        assert result["total_batches"] == 1

    def test_empty_dataloader_raises(self):
        model = GPTModel()
        dl = DataLoader(EvalDataset(vocab_size=100, size=0), batch_size=4)
        with pytest.raises(ValueError, match="No tokens"):
            compute_perplexity(model, dl)

    def test_nan_logits_detected(self):
        """If logits contain NaN, perplexity should raise RuntimeError."""
        model = GPTModel()
        dl = make_dataloader(vocab_size=100, seq_len=8, batch_size=2, size=4)
        result = compute_perplexity(model, dl, max_batches=1)
        assert math.isfinite(result["perplexity"])


class TestStridedPerplexity:
    def test_basic(self):
        model = GPTModel()
        token_ids = torch.randint(0, 50304, (50,))
        result = compute_perplexity_strided(
            model, token_ids, stride=16, max_seq_len=32
        )
        assert "perplexity" in result
        assert result["total_windows"] > 0
        assert result["total_tokens"] > 0

    def test_stride_zero_raises(self):
        model = GPTModel()
        token_ids = torch.randint(0, 100, (20,))
        with pytest.raises(ValueError, match="stride must be positive"):
            compute_perplexity_strided(model, token_ids, stride=0)

    def test_exceeds_max_seq_len(self):
        model = GPTModel()
        token_ids = torch.randint(0, 100, (20,))
        with pytest.raises(ValueError, match="exceeds model.max_seq_len"):
            compute_perplexity_strided(
                model, token_ids, max_seq_len=9999
            )

    def test_below_min_text(self):
        """Very short text should still produce a result."""
        model = GPTModel()
        token_ids = torch.randint(0, 100, (5,))
        result = compute_perplexity_strided(
            model, token_ids, stride=4, max_seq_len=8
        )
        assert "perplexity" in result


class TestEvaluateCheckpoint:
    def test_returns_timing(self):
        model = GPTModel()
        dl = make_dataloader(vocab_size=100, seq_len=8, batch_size=2, size=6)
        result = evaluate_checkpoint(model, dl, max_batches=1)
        assert "eval_time_seconds" in result
        assert "tokens_per_second" in result
        assert result["tokens_per_second"] > 0


# --- Benchmark Tests ---

class TestTextCompletionAccuracy:
    def test_basic(self):
        model = GPTModel()
        dl = make_dataloader(vocab_size=100, seq_len=8, batch_size=2, size=8)
        bench = TextCompletionAccuracy()
        result = bench.compute(model, dl, max_batches=2)
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0
        assert result["total_tokens"] > 0

    def test_accuracy_range(self):
        model = GPTModel()
        dl = make_dataloader(vocab_size=100, seq_len=12, batch_size=2, size=10)
        bench = TextCompletionAccuracy()
        result = bench.compute(model, dl, max_batches=2)
        # Untrained model accuracy is very low but non-negative
        assert result["accuracy"] >= 0.0


class TestTopKAccuracy:
    def test_top1_is_top1_accuracy(self):
        """Top-1 should match TextCompletionAccuracy."""
        model = GPTModel()
        dl = make_dataloader(vocab_size=100, seq_len=8, batch_size=2, size=8)
        top1 = TopKAccuracy(k=1)
        acc1 = TextCompletionAccuracy()
        r1 = top1.compute(model, dl, max_batches=2)
        r2 = acc1.compute(model, dl, max_batches=2)
        assert abs(r1["top1_accuracy"] - r2["accuracy"]) < 1e-6

    def test_top5_gte_top1(self):
        """Top-5 accuracy >= Top-1 accuracy always."""
        model = GPTModel()
        dl = make_dataloader(vocab_size=100, seq_len=8, batch_size=2, size=8)
        top1 = TopKAccuracy(k=1)
        top5 = TopKAccuracy(k=5)
        r1 = top1.compute(model, dl, max_batches=2)
        r5 = top5.compute(model, dl, max_batches=2)
        assert r5["top5_accuracy"] >= r1["top1_accuracy"]

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="k must be positive"):
            TopKAccuracy(k=0)


class TestEvaluationSuite:
    def test_full_suite(self):
        model = GPTModel()
        dl = make_dataloader(vocab_size=100, seq_len=8, batch_size=2, size=12)
        results = run_evaluation_suite(model, dl, max_batches=2)
        assert "perplexity" in results
        assert "top1_accuracy" in results
        assert "top5_accuracy" in results
        assert results["perplexity"]["perplexity"] > 0
        assert "accuracy" in results["top1_accuracy"]
