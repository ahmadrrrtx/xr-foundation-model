"""
Tests for multi-head attention (v0.6.0) — now returns (output, present_kv).
"""

import pytest
import torch

from model.attention.multi_head import MultiHeadAttention


class TestAttentionShape:
    def test_standard_forward_shape(self):
        attn = MultiHeadAttention(d_model=256, n_heads=8, dropout=0.1)
        x = torch.randn(2, 10, 256)
        out, pkv = attn(x)
        assert out.shape == (2, 10, 256)
        assert pkv is None  # use_cache=False by default

    def test_with_cache_returns_present_kv(self):
        attn = MultiHeadAttention(d_model=128, n_heads=4, dropout=0.0)
        x = torch.randn(1, 5, 128)
        out, pkv = attn(x, use_cache=True)
        assert out.shape == (1, 5, 128)
        assert pkv is not None
        assert isinstance(pkv, tuple)
        assert len(pkv) == 2
        k, v = pkv
        assert k.shape == (1, 4, 5, 32)  # (batch, heads, seq, d_head)
        assert v.shape == (1, 4, 5, 32)

    def test_past_kv_concatenation(self):
        attn = MultiHeadAttention(d_model=64, n_heads=2, dropout=0.0)
        # First pass: cache full sequence
        x_full = torch.randn(1, 4, 64)
        _, pkv = attn(x_full, use_cache=True)
        k_cache, v_cache = pkv
        # Second pass: single token with cached KV
        x_new = torch.randn(1, 1, 64)
        out, pkv2 = attn(x_new, past_kv=pkv, use_cache=True)
        assert out.shape == (1, 1, 64)
        k_full, v_full = pkv2
        assert k_full.shape == (1, 2, 5, 32)  # 4 cached + 1 new
        assert v_full.shape == (1, 2, 5, 32)


class TestAttentionInputValidation:
    def test_invalid_dimensions(self):
        attn = MultiHeadAttention(d_model=256, n_heads=8)
        with pytest.raises(ValueError, match="3D input"):
            attn(torch.randn(2, 10))

    def test_d_model_mismatch(self):
        attn = MultiHeadAttention(d_model=256, n_heads=8)
        with pytest.raises(ValueError):
            attn(torch.randn(2, 5, 128))

    def test_d_model_not_divisible(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadAttention(d_model=100, n_heads=8)

    def test_invalid_dropout(self):
        with pytest.raises(ValueError, match="dropout"):
            MultiHeadAttention(d_model=128, n_heads=4, dropout=1.5)


class TestAttentionMasking:
    def test_mask_passed(self):
        attn = MultiHeadAttention(d_model=32, n_heads=2, dropout=0.0)
        x = torch.randn(2, 4, 32)
        mask = torch.ones(2, 1, 4, 4)
        out, _ = attn(x, mask=mask)
        assert not torch.isnan(out).any()

    def test_invalid_mask_dims(self):
        attn = MultiHeadAttention(d_model=32, n_heads=2)
        x = torch.randn(2, 4, 32)
        mask = torch.ones(2, 4, 4, 4, 4)
        with pytest.raises(ValueError, match="Mask must have"):
            attn(x, mask=mask)


class TestAttentionNumericalStability:
    def test_no_nan_inf(self):
        attn = MultiHeadAttention(d_model=32, n_heads=2, dropout=0.0)
        x = torch.randn(1, 4, 32) * 100
        out, _ = attn(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_dropout_training_vs_eval(self):
        attn = MultiHeadAttention(d_model=32, n_heads=2, dropout=0.5)
        attn.train()
        x = torch.randn(4, 8, 32)
        o1, _ = attn(x)
        attn.eval()
        o2, _ = attn(x)
        assert not torch.isnan(o1).any()
        assert not torch.isnan(o2).any()

    def test_gradient_flow(self):
        attn = MultiHeadAttention(d_model=64, n_heads=2, dropout=0.0)
        attn.train()
        x = torch.randn(2, 5, 64, requires_grad=True)
        out, _ = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert attn.W_q.weight.grad is not None


class TestAttentionRoPE:
    def test_rope_enabled(self):
        attn = MultiHeadAttention(d_model=64, n_heads=4, use_rope=True)
        x = torch.randn(2, 6, 64)
        out, _ = attn(x)
        assert out.shape == (2, 6, 64)

    def test_rope_disabled(self):
        attn = MultiHeadAttention(d_model=64, n_heads=4, use_rope=False)
        assert attn.rope is None
        x = torch.randn(2, 6, 64)
        out, _ = attn(x)
        assert out.shape == (2, 6, 64)

    def test_rope_with_cache_offset(self):
        """RoPE should use correct positions when cache is provided."""
        attn = MultiHeadAttention(d_model=64, n_heads=2, dropout=0.0, use_rope=True)
        # First pass
        x1 = torch.randn(1, 4, 64)
        _, pkv = attn(x1, use_cache=True)
        # Second pass: new token should get position 4 (offset)
        x2 = torch.randn(1, 1, 64)
        out, _ = attn(x2, past_kv=pkv, use_cache=True)
        assert not torch.isnan(out).any()
