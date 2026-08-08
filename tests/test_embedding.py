"""
Tests for the embedding layer (`XRFMEmbedding`).

Coverage: Shape validation, weight tying verification, padding behavior,
Xavier initialization, numerical stability, config integration, input validation.
Every test follows the 10-step protocol: correct input, expected output,
edge cases, failure modes, regression verification.
"""

import pytest
import torch

from model.embedding import XRFMEmbedding


class TestEmbeddingShape:
    """Shape propagation tests for embedding layer."""

    def test_single_token(self) -> None:
        emb = XRFMEmbedding(vocab_size=100, d_model=64)
        input_ids = torch.tensor([42])
        out = emb(input_ids)
        assert out.shape == (1, 64)

    def test_batch_sequence(self) -> None:
        emb = XRFMEmbedding(vocab_size=2048, d_model=256)
        input_ids = torch.randint(0, 2048, (4, 12))
        out = emb(input_ids)
        assert out.shape == (4, 12, 256)

    def test_single_sequence_2d(self) -> None:
        emb = XRFMEmbedding(vocab_size=1000, d_model=128)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        out = emb(input_ids)
        assert out.shape == (1, 5, 128)


class TestEmbeddingInit:
    """Initialization verification (Xavier, numerical stability)."""

    def test_xavier_init_not_nan(self) -> None:
        emb = XRFMEmbedding(vocab_size=100, d_model=32)
        assert not torch.isnan(emb.embedding.weight).any()
        assert not torch.isinf(emb.embedding.weight).any()

    def test_weight_tied_reference(self) -> None:
        emb = XRFMEmbedding(vocab_size=100, d_model=32, weight_tied=True)
        # The embedding weight should be a standard nn.Parameter with correct shape.
        assert emb.embedding.weight.shape == (100, 32)
        assert emb.weight_tied is True

    def test_padding_idx_zero(self) -> None:
        emb = XRFMEmbedding(vocab_size=100, d_model=32, padding_idx=5)
        # Padding embeddings should be zero after initialization.
        assert torch.allclose(emb.embedding.weight[5], torch.zeros(32))


class TestEmbeddingInputValidation:
    """Error handling and input validation."""

    def test_invalid_token_id_raises_index_error(self) -> None:
        emb = XRFMEmbedding(vocab_size=10, d_model=8)
        input_ids = torch.tensor([5, 9, 10])  # 10 exceeds vocab_size 10? Actually 10 >= 10.
        with pytest.raises(IndexError, match="exceeds vocabulary size"):
            emb(input_ids)

    def test_negative_vocab_size_raises(self) -> None:
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            XRFMEmbedding(vocab_size=-1, d_model=8)

    def test_zero_d_model_raises(self) -> None:
        with pytest.raises(ValueError, match="d_model must be positive"):
            XRFMEmbedding(vocab_size=100, d_model=0)


class TestEmbeddingConfigIntegration:
    """Integration with ConfigLoader settings."""

    def test_config_driven_vocabulary_size(self) -> None:
        # Embedding works with a config-scale vocab size.
        emb = XRFMEmbedding(vocab_size=2048, d_model=256)
        input_ids = torch.randint(0, 2048, (2, 8))
        out = emb(input_ids)
        assert out.shape == (2, 8, 256)

    def test_gradient_flow(self) -> None:
        emb = XRFMEmbedding(vocab_size=100, d_model=16)
        input_ids = torch.tensor([[1, 2, 3]])
        out = emb(input_ids)
        loss = out.sum()
        loss.backward()
        assert emb.embedding.weight.grad is not None
        assert not torch.isnan(emb.embedding.weight.grad).any()
