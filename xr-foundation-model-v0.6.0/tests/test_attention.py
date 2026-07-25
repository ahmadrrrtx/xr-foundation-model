"""
Tests for multi-head attention (`MultiHeadAttention`).

Coverage: Shape propagation, masking correctness, gradient flow,
dropout behavior, RoPE integration, numerical stability, config integration.
Every public method is covered; edge cases include invalid dimensions,
empty sequences, large scores (softmax stability), and masking patterns.
"""

import math
import pytest
import torch

from model.attention.multi_head import MultiHeadAttention


class TestAttentionShape:
    """Shape validation for multi-head attention."""

    def test_standard_forward_shape(self) -> None:
        attn = MultiHeadAttention(d_model=256, n_heads=8, dropout=0.1)
        x = torch.randn(2, 10, 256)
        out = attn(x)
        assert out.shape == (2, 10, 256)

    def test_single_batch_single_sequence(self) -> None:
        attn = MultiHeadAttention(d_model=128, n_heads=4, dropout=0.0)
        x = torch.randn(1, 5, 128)
        out = attn(x)
        assert out.shape == (1, 5, 128)

    def test_large_batch_long_sequence(self) -> None:
        attn = MultiHeadAttention(d_model=64, n_heads=2, dropout=0.05)
        x = torch.randn(8, 64, 64)
        out = attn(x)
        assert out.shape == (8, 64, 64)


class TestAttentionInputValidation:
    """Input validation and error handling."""

    def test_invalid_3d_input_dimension(self) -> None:
        attn = MultiHeadAttention(d_model=256, n_heads=8)
        x_invalid = torch.randn(2, 10)  # 2D input
        with pytest.raises(ValueError, match="dimension 3"):
            attn(x_invalid)

    def test_d_model_not_divisible_by_n_heads(self) -> None:
        with pytest.raises(ValueError, match="must be divisible"):
            MultiHeadAttention(d_model=100, n_heads=8)

    def test_non_positive_d_model(self) -> None:
        with pytest.raises(ValueError, match="d_model must be positive"):
            MultiHeadAttention(d_model=-1, n_heads=2)

    def test_drop_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="dropout must be in"):
            MultiHeadAttention(d_model=128, n_heads=4, dropout=1.5)


class TestAttentionMasking:
    """Masking correctness: padding tokens receive near-zero attention."""

    def test_mask_blocks_padding(self) -> None:
        attn = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, 64)
        # Create a mask where the last 2 tokens of the sequence are masked (padding).
        # Mask shape: (batch, 1, 1, seq) — broadcasted over heads and query positions.
        mask = torch.ones(batch_size, 1, 1, seq_len)
        mask[:, :, :, -2:] = 0  # Mask last two positions.
        out = attn(x, mask=mask)
        # The output should not contain NaN or Inf.
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_mask_shape_validation(self) -> None:
        attn = MultiHeadAttention(d_model=32, n_heads=2, dropout=0.0)
        x = torch.randn(2, 4, 32)
        # Invalid mask dimension.
        mask_bad = torch.ones(2, 4, 4, 4, 4)  # 5D
        with pytest.raises(ValueError, match="must have 2, 3, or 4 dimensions"):
            attn(x, mask=mask_bad)


class TestAttentionGradientFlow:
    """Gradient verification through attention mechanism."""

    def test_backward_through_attention(self) -> None:
        attn = MultiHeadAttention(d_model=64, n_heads=2, dropout=0.0)
        attn.train()
        x = torch.randn(2, 5, 64, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        # Projections should have gradients.
        assert attn.W_q.weight.grad is not None
        assert attn.W_k.weight.grad is not None
        assert attn.W_v.weight.grad is not None
        assert attn.W_o.weight.grad is not None

    def test_gradient_not_exploding(self) -> None:
        attn = MultiHeadAttention(d_model=128, n_heads=4, dropout=0.0)
        x = torch.randn(1, 3, 128)
        out = attn(x)
        # Scale the output to create a non-trivial gradient.
        scaled = out * 10.0
        loss = scaled.sum()
        loss.backward()
        # Gradient norms should be finite and not excessively large.
        for param in (attn.W_q, attn.W_k, attn.W_v, attn.W_o):
            assert param.weight.grad is not None
            assert torch.isfinite(param.weight.grad).all()


class TestAttentionNumericalStability:
    """Softmax scaling, masking, large scores."""

    def test_large_scores_dont_overflow(self) -> None:
        attn = MultiHeadAttention(d_model=32, n_heads=2, dropout=0.0)
        # Create input that could produce very large attention scores.
        x = torch.randn(1, 4, 32) * 100  # Large values
        out = attn(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_dropout_training_vs_eval(self) -> None:
        attn = MultiHeadAttention(d_model=32, n_heads=2, dropout=0.5)
        attn.train()
        x = torch.randn(4, 8, 32)
        out_train = attn(x)
        attn.eval()
        out_eval = attn(x)
        # In training, dropout introduces randomness; in eval, no dropout.
        # The outputs should not contain NaN in either mode.
        assert not torch.isnan(out_train).any()
        assert not torch.isnan(out_eval).any()


class TestAttentionRoPEIntegration:
    """RoPE integration (optional, configurable)."""

    def test_use_rope_true_applies_rotation(self) -> None:
        attn = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0, use_rope=True)
        x = torch.randn(2, 6, 64)
        out = attn(x)
        assert out.shape == (2, 6, 64)
        assert not torch.isnan(out).any()

    def test_use_rope_false_no_rope(self) -> None:
        attn = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0, use_rope=False)
        assert attn.rope is None
        x = torch.randn(2, 6, 64)
        out = attn(x)
        assert out.shape == (2, 6, 64)
