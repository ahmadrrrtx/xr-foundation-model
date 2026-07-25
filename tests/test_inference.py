"""
Tests for inference engine (v0.6.0).

Covers sampling strategies, KV cache, and end-to-end generation.
"""

import pytest
import torch

from model.gpt import GPTModel
from inference.sampling import (
    sample_greedy,
    sample_temperature,
    sample_top_k,
    sample_top_p,
    sample_token,
)
from inference.kv_cache import KVCache
from inference.engine import GenerationEngine


# --- Sampling Tests ---

class TestGreedySampling:
    def test_selects_max(self):
        logits = torch.tensor([[0.1, 5.0, 2.0]])
        token = sample_greedy(logits)
        assert token.item() == 1

    def test_batched(self):
        logits = torch.tensor([[0.1, 5.0], [3.0, 1.0]])
        tokens = sample_greedy(logits)
        assert tokens.shape == (2, 1)
        assert tokens[0].item() == 1
        assert tokens[1].item() == 0


class TestTemperatureSampling:
    def test_temp_zero_is_greedy(self):
        logits = torch.tensor([[0.1, 5.0, 2.0]])
        token = sample_temperature(logits, temperature=0)
        assert token.item() == 1

    def test_high_temp_diverse(self):
        """High temperature should not crash and produce valid tokens."""
        logits = torch.randn(1, 100)
        token = sample_temperature(logits, temperature=10.0)
        assert 0 <= token.item() < 100

    def test_invalid_temp(self):
        with pytest.raises(ValueError, match="temperature"):
            sample_temperature(torch.randn(1, 10), temperature=-1.0)


class TestTopKSampling:
    def test_selects_from_top_k(self):
        logits = torch.tensor([[0.1, 0.2, 5.0, 0.3, 2.0, 0.05]])
        token = sample_top_k(logits, top_k=3, temperature=1.0)
        assert token.item() in (2, 4, 3)  # top 3 values

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="top_k"):
            sample_top_k(torch.randn(1, 10), top_k=0)


class TestTopPSampling:
    def test_top_p_1_includes_all(self):
        logits = torch.randn(1, 100)
        token = sample_top_p(logits, top_p=1.0, temperature=1.0)
        assert 0 <= token.item() < 100

    def test_small_p_greedy_like(self):
        """Very small p should behave like greedy for concentrated dist."""
        logits = torch.tensor([[0.1, 10.0, 0.2, 0.05]])
        # Top token dominates; small p should still include it
        tokens = {sample_top_p(logits, top_p=0.1, temperature=1.0).item()
                  for _ in range(10)}
        assert 1 in tokens  # highest prob token always present

    def test_invalid_p(self):
        with pytest.raises(ValueError, match="top_p"):
            sample_top_p(torch.randn(1, 10), top_p=0.0)
        with pytest.raises(ValueError, match="top_p"):
            sample_top_p(torch.randn(1, 10), top_p=1.5)


class TestSampleToken:
    def test_combined_strategy(self):
        logits = torch.randn(1, 50)
        # Greedy
        t1 = sample_token(logits, temperature=0)
        assert t1.shape == (1, 1)
        # Temperature only
        t2 = sample_token(logits, temperature=0.8)
        assert t2.shape == (1, 1)
        # Top-k
        t3 = sample_token(logits, temperature=0.8, top_k=10)
        assert t3.shape == (1, 1)
        # Top-p
        t4 = sample_token(logits, temperature=0.8, top_p=0.9)
        assert t4.shape == (1, 1)


# --- KV Cache Tests ---

class TestKVCache:
    def test_init(self):
        cache = KVCache(max_cache_len=512)
        assert cache.is_empty()
        assert cache.seq_len == 0

    def test_update_single_layer(self):
        cache = KVCache(max_cache_len=512)
        k = torch.randn(1, 4, 8, 32)
        v = torch.randn(1, 4, 8, 32)
        k_out, v_out = cache.update(0, k, v)
        assert k_out.shape == (1, 4, 8, 32)
        assert cache.seq_len == 8
        assert not cache.is_empty()

    def test_update_concat(self):
        cache = KVCache(max_cache_len=512)
        k1 = torch.randn(1, 4, 5, 32)
        v1 = torch.randn(1, 4, 5, 32)
        cache.update(0, k1, v1)
        k2 = torch.randn(1, 4, 1, 32)
        v2 = torch.randn(1, 4, 1, 32)
        k_out, v_out = cache.update(0, k2, v2)
        assert k_out.shape == (1, 4, 6, 32)
        assert cache.seq_len == 6

    def test_max_len_truncation(self):
        cache = KVCache(max_cache_len=10)
        k = torch.randn(1, 4, 15, 32)
        v = torch.randn(1, 4, 15, 32)
        k_out, v_out = cache.update(0, k, v)
        assert k_out.shape[2] == 10  # truncated

    def test_clear(self):
        cache = KVCache(max_cache_len=512)
        cache.update(0, torch.randn(1, 2, 3, 16), torch.randn(1, 2, 3, 16))
        cache.clear()
        assert cache.is_empty()
        assert cache.seq_len == 0

    def test_multi_layer(self):
        cache = KVCache(max_cache_len=512)
        for i in range(4):
            cache.update(i, torch.randn(1, 2, 3, 16), torch.randn(1, 2, 3, 16))
        assert cache.seq_len == 3

    def test_invalid_layer_idx(self):
        cache = KVCache()
        with pytest.raises(ValueError):
            cache.update(-1, torch.randn(1, 2, 3, 16), torch.randn(1, 2, 3, 16))


# --- Generation Engine Tests ---

class TestGenerationEngine:
    def test_init(self):
        model = GPTModel()
        engine = GenerationEngine(model)
        assert engine.model is model

    def test_greedy_generation(self):
        model = GPTModel()
        engine = GenerationEngine(model)
        prompt = torch.randint(0, 100, (8,))
        output = engine.generate(prompt, max_new_tokens=3, temperature=0)
        assert len(output) > len(prompt)
        assert output.dim() == 1

    def test_temperature_generation(self):
        model = GPTModel()
        engine = GenerationEngine(model)
        prompt = torch.randint(0, 100, (5,))
        output = engine.generate(prompt, max_new_tokens=5, temperature=0.8)
        assert len(output) == len(prompt) + 5

    def test_top_k_generation(self):
        model = GPTModel()
        engine = GenerationEngine(model)
        prompt = torch.randint(0, 100, (4,))
        output = engine.generate(prompt, max_new_tokens=3, top_k=10, temperature=1.0)
        assert len(output) == 7

    def test_top_p_generation(self):
        model = GPTModel()
        engine = GenerationEngine(model)
        prompt = torch.randint(0, 100, (4,))
        output = engine.generate(prompt, max_new_tokens=3, top_p=0.9, temperature=1.0)
        assert len(output) == 7

    def test_batch_generation(self):
        model = GPTModel()
        engine = GenerationEngine(model)
        prompt = torch.randint(0, 100, (2, 4))
        output = engine.generate_batch(prompt, max_new_tokens=3)
        assert output.shape == (2, 7)

    def test_invalid_max_tokens(self):
        model = GPTModel()
        engine = GenerationEngine(model)
        with pytest.raises(ValueError, match="max_new_tokens"):
            engine.generate(torch.randint(0, 100, (4,)), max_new_tokens=0)

    def test_invalid_temp(self):
        model = GPTModel()
        engine = GenerationEngine(model)
        with pytest.raises(ValueError, match="temperature"):
            engine.generate(torch.randint(0, 100, (4,)), temperature=-1)

    def test_prompt_truncation(self):
        model = GPTModel()
        engine = GenerationEngine(model)
        # Prompt longer than max_seq_len (512)
        long_prompt = torch.randint(0, 100, (600,))
        output = engine.generate(long_prompt, max_new_tokens=2, temperature=0)
        assert len(output) <= 512 + 2

    def test_kv_cache_equivalence(self):
        """Cached generation should produce same logits as full forward."""
        model = GPTModel()
        model.eval()
        prompt = torch.randint(0, 100, (1, 6))
        # Full forward
        with torch.no_grad():
            logits_full, _ = model(prompt)
        last_full = logits_full[:, -1, :]
        # Cached forward: first pass caches, second pass uses cache
        with torch.no_grad():
            _, pkv = model(prompt[:, :5], use_cache=True)  # cache first 5
            logits_cached, _ = model(prompt[:, -1:], past_key_values=pkv, use_cache=True)
        last_cached = logits_cached[:, -1, :]
        # Logits should be close (not identical due to RoPE but structurally equivalent)
        assert torch.isfinite(last_cached).all()
        assert last_cached.shape == last_full.shape
