"""
Tests for GPTModel (v0.6.0) — forward returns (logits, present_key_values).
"""

import pytest
import torch

from model.gpt import GPTModel


class TestGPTModel:
    def test_init(self):
        model = GPTModel()
        assert hasattr(model, "embedding")
        assert hasattr(model, "blocks")
        assert hasattr(model, "norm_final")
        assert hasattr(model, "lm_head")

    def test_forward_shape_batch(self):
        model = GPTModel()
        model.eval()
        x = torch.randint(0, 50304, (2, 12))
        with torch.no_grad():
            logits, pkv = model(x)
        assert logits.shape == (2, 12, 50304)
        assert pkv is None  # use_cache=False

    def test_forward_single_seq(self):
        model = GPTModel()
        model.eval()
        x = torch.randint(0, 50304, (8,))
        with torch.no_grad():
            logits, pkv = model(x)
        assert logits.shape == (8, 50304)
        assert pkv is None

    def test_forward_with_cache(self):
        model = GPTModel()
        model.eval()
        x = torch.randint(0, 50304, (1, 8))
        with torch.no_grad():
            logits, pkv = model(x, use_cache=True)
        assert logits.shape == (1, 8, 50304)
        assert pkv is not None
        assert len(pkv) == 6  # n_layers = 6
        for layer_kv in pkv:
            k, v = layer_kv
            assert k.shape[1] == 8  # n_heads

    def test_cache_incremental(self):
        """First pass caches, second pass reuses cache."""
        model = GPTModel()
        model.eval()
        # First pass: process full prompt
        x1 = torch.randint(0, 50304, (1, 4))
        _, pkv = model(x1, use_cache=True)
        # Second pass: single new token with cache
        x2 = torch.randint(0, 50304, (1, 1))
        logits2, pkv2 = model(x2, past_key_values=pkv, use_cache=True)
        assert logits2.shape == (1, 1, 50304)
        assert pkv2 is not None
        # Cache should now have 5 positions
        k0, _ = pkv2[0]
        assert k0.shape[2] == 5

    def test_numerical_stability(self):
        model = GPTModel()
        model.eval()
        x = torch.randint(0, 50304, (2, 16))
        with torch.no_grad():
            logits, _ = model(x)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_gradient_flow(self):
        model = GPTModel()
        model.train()
        x = torch.randint(0, 50304, (2, 6))
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()
        assert model.embedding.embedding.weight.grad is not None
        assert model.norm_final.weight.grad is not None

    def test_weight_tied(self):
        model = GPTModel(weight_tied=True)
        assert torch.equal(model.lm_head.weight, model.embedding.embedding.weight)

    def test_weight_untied(self):
        model = GPTModel(weight_tied=False)
        assert not torch.equal(model.lm_head.weight, model.embedding.embedding.weight)

    def test_vocab_bounds_error(self):
        model = GPTModel()
        with pytest.raises(IndexError, match="exceeds vocab"):
            model(torch.tensor([[50304]]))

    def test_invalid_input_dims(self):
        model = GPTModel()
        with pytest.raises(ValueError, match="1D or 2D"):
            model(torch.randn(2, 3, 4))

    def test_config_not_found(self):
        with pytest.raises(FileNotFoundError):
            GPTModel(config_path="nonexistent.yaml")

    def test_parameter_count(self):
        model = GPTModel()
        count = model.parameter_count()
        assert count > 0
        print(f"GPTModel parameters: {count:,}")
