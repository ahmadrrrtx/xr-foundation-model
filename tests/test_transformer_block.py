"""
Tests for TransformerBlock (v0.6.0) — now returns (output, present_kv).
"""

import pytest
import torch

from model.layers.transformer_block import TransformerBlock


class TestTransformerBlock:
    def test_standard_forward(self):
        block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
        x = torch.randn(2, 10, 256)
        out, pkv = block(x)
        assert out.shape == (2, 10, 256)
        assert pkv is None

    def test_with_cache(self):
        block = TransformerBlock(d_model=128, n_heads=4, d_ff=256, dropout=0.0)
        x = torch.randn(1, 5, 128)
        out, pkv = block(x, use_cache=True)
        assert out.shape == (1, 5, 128)
        assert pkv is not None
        assert isinstance(pkv, tuple)
        assert len(pkv) == 2

    def test_cache_concatenation(self):
        block = TransformerBlock(d_model=64, n_heads=2, d_ff=128, dropout=0.0)
        # First pass
        x1 = torch.randn(1, 4, 64)
        _, pkv = block(x1, use_cache=True)
        # Second pass with cache
        x2 = torch.randn(1, 1, 64)
        out, pkv2 = block(x2, past_kv=pkv, use_cache=True)
        assert out.shape == (1, 1, 64)
        assert pkv2 is not None

    def test_residual_identity(self):
        block = TransformerBlock(d_model=64, n_heads=2, d_ff=128, dropout=0.0)
        x = torch.randn(2, 5, 64, requires_grad=True)
        out, _ = block(x)
        loss = out.sum()
        loss.backward()
        assert torch.isfinite(x.grad).all()

    def test_invalid_input_dim(self):
        block = TransformerBlock(d_model=32, n_heads=2, d_ff=64)
        with pytest.raises(ValueError, match="3D input"):
            block(torch.randn(2, 10))


class TestTransformerBlockConfig:
    def test_rmsnorm_false(self):
        block = TransformerBlock(d_model=32, n_heads=2, d_ff=64, use_rmsnorm=False)
        assert isinstance(block.norm1, torch.nn.LayerNorm)

    def test_rope_false(self):
        block = TransformerBlock(d_model=32, n_heads=2, d_ff=64, use_rope=False)
        assert block.attn.rope is None

    def test_invalid_params(self):
        with pytest.raises(ValueError, match="divisible"):
            TransformerBlock(d_model=30, n_heads=4, d_ff=64)
