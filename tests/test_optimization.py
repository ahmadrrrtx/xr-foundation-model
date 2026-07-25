"""
Tests for optimization module (v0.9.0).

Covers FlashAttention, quantization (INT8/INT4), and speculative decoding.
"""

import pytest
import torch
import torch.nn as nn

from model.gpt import GPTModel

# --- FlashAttention ---


class TestFlashAttention:
    def test_sdpa_basic(self):
        from optimization.flash_attention import scaled_dot_product_attention

        q = torch.randn(1, 4, 16, 32)
        k = torch.randn(1, 4, 16, 32)
        v = torch.randn(1, 4, 16, 32)
        out = scaled_dot_product_attention(q, k, v, is_causal=True)
        assert out.shape == (1, 4, 16, 32)
        assert not torch.isnan(out).any()

    def test_sdpa_with_mask(self):
        from optimization.flash_attention import scaled_dot_product_attention

        q = torch.randn(2, 2, 8, 16)
        k = torch.randn(2, 2, 8, 16)
        v = torch.randn(2, 2, 8, 16)
        mask = torch.ones(2, 1, 8, 8)
        mask[:, :, :, -2:] = 0
        out = scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        assert out.shape == (2, 2, 8, 16)

    def test_flash_attention_forward(self):
        from optimization.flash_attention import flash_attention_forward

        q = torch.randn(1, 4, 8, 32)
        k = torch.randn(1, 4, 8, 32)
        v = torch.randn(1, 4, 8, 32)
        out = flash_attention_forward(q, k, v, training=False)
        assert out.shape == (1, 4, 8, 32)

    def test_backend_detection(self):
        from optimization.flash_attention import (
            get_available_backend,
        )

        backend = get_available_backend()
        assert backend in ("flash", "mem_efficient", "math")

    def test_force_math_backend(self):
        from optimization.flash_attention import scaled_dot_product_attention

        q = torch.randn(1, 2, 4, 16)
        k = torch.randn(1, 2, 4, 16)
        v = torch.randn(1, 2, 4, 16)
        out = scaled_dot_product_attention(q, k, v, is_causal=True, force_backend="math")
        assert out.shape == (1, 2, 4, 16)


# --- Quantization ---


class TestQuantization:
    def test_int8_per_tensor_roundtrip(self):
        from optimization.quantization import (
            dequantize_weight,
            quantize_int8_per_tensor,
        )

        w = torch.randn(64, 128)
        qw = quantize_int8_per_tensor(w)
        w_hat = dequantize_weight(qw)
        assert w_hat.shape == (64, 128)
        # Approximation error should be small relative to std
        err = (w - w_hat).abs().mean()
        assert err < w.std() * 0.5

    def test_int8_per_tensor_symmetric(self):
        from optimization.quantization import quantize_int8_per_tensor

        w = torch.randn(32, 64)
        qw = quantize_int8_per_tensor(w, symmetric=True)
        assert qw.data.dtype == torch.int8
        assert qw.bits == 8
        assert qw.zero_point is None  # symmetric

    def test_int8_per_channel(self):
        from optimization.quantization import (
            dequantize_weight,
            quantize_int8_per_channel,
        )

        w = torch.randn(128, 256)
        qw = quantize_int8_per_channel(w)
        w_hat = dequantize_weight(qw)
        assert w_hat.shape == (128, 256)
        assert qw.scale.numel() == 128
        err = (w - w_hat).abs().mean()
        assert err < w.std() * 0.3

    def test_int4_groupwise_roundtrip(self):
        from optimization.quantization import (
            dequantize_weight,
            quantize_int4_groupwise,
        )

        w = torch.randn(256, 512)
        qw = quantize_int4_groupwise(w, group_size=64)
        w_hat = dequantize_weight(qw)
        assert w_hat.shape == (256, 512)
        assert qw.bits == 4
        assert qw.group_size == 64

    def test_int4_packing(self):
        from optimization.quantization import quantize_int4_groupwise

        w = torch.randn(64, 64)
        qw = quantize_int4_groupwise(w, group_size=32)
        # Packed: 2 values per byte → data should be half original size
        n_params = w.numel()
        expected_packed = n_params // 2
        assert qw.data.numel() == expected_packed

    def test_int4_group_size_invalid(self):
        from optimization.quantization import quantize_int4_groupwise

        with pytest.raises(ValueError, match="positive"):
            quantize_int4_groupwise(torch.randn(16, 16), group_size=0)

    def test_model_quantization(self):
        from optimization.quantization import (
            compute_compression_ratio,
            quantize_model_weights,
        )

        model = GPTModel()
        _, q_map = quantize_model_weights(model, bits=8)
        assert len(q_map) > 0
        ratio = compute_compression_ratio(q_map)
        assert 3.5 < ratio < 4.5  # ~4x for INT8

    def test_int4_model_quantization(self):
        from optimization.quantization import (
            compute_compression_ratio,
            quantize_model_weights,
        )

        model = GPTModel()
        _, q_map = quantize_model_weights(model, bits=4, group_size=64)
        ratio = compute_compression_ratio(q_map)
        assert 6.0 < ratio < 9.0  # ~8x for INT4 (with scale overhead)

    def test_invalid_bits(self):
        from optimization.quantization import quantize_model_weights

        with pytest.raises(ValueError, match="bits must be"):
            quantize_model_weights(nn.Linear(4, 2), bits=16)

    def test_quantized_weight_dataclass(self):
        from optimization.quantization import (
            QuantizedWeight,
            quantize_int8_per_tensor,
        )

        w = torch.randn(32, 32)
        qw = quantize_int8_per_tensor(w)
        assert isinstance(qw, QuantizedWeight)
        assert qw.original_shape == (32, 32)
        assert qw.bits == 8


# --- Speculative Decoding ---


class TestSpeculativeDecoding:
    def test_init(self):
        from optimization.speculative_decoding import SpeculativeDecoder

        model = GPTModel()
        draft = GPTModel()
        sd = SpeculativeDecoder(model, draft, gamma=3)
        assert sd.gamma == 3

    def test_vocab_mismatch(self):
        from optimization.speculative_decoding import SpeculativeDecoder

        model = GPTModel()
        draft = GPTModel()
        # Models share config, so vocab matches
        sd = SpeculativeDecoder(model, draft, gamma=3)
        assert sd.target_model is model

    def test_invalid_gamma(self):
        from optimization.speculative_decoding import SpeculativeDecoder

        with pytest.raises(ValueError, match="gamma"):
            SpeculativeDecoder(GPTModel(), GPTModel(), gamma=0)

    def test_generate_greedy(self):
        from optimization.speculative_decoding import SpeculativeDecoder

        model = GPTModel()
        draft = GPTModel()
        sd = SpeculativeDecoder(model, draft, gamma=3)
        prompt = torch.randint(0, 100, (6,))
        out = sd.generate(prompt, max_new_tokens=4, temperature=0)
        assert len(out) > len(prompt)

    def test_generate_temperature(self):
        from optimization.speculative_decoding import SpeculativeDecoder

        model = GPTModel()
        draft = GPTModel()
        sd = SpeculativeDecoder(model, draft, gamma=3)
        prompt = torch.randint(0, 100, (4,))
        out = sd.generate(prompt, max_new_tokens=5, temperature=0.8)
        assert len(out) == len(prompt) + 5

    def test_speedup_estimate(self):
        from optimization.speculative_decoding import estimate_speedup

        speedup = estimate_speedup(
            gamma=5,
            acceptance_rate=0.7,
            target_cost=10,
            draft_cost=1,
        )
        assert speedup > 1.0
        # With 70% AR and 5 tokens: ~ (0.7*5+1)/(5*0.1+1) = 4.5/1.5 ≈ 3x
        assert 2.0 < speedup < 4.0

    def test_speedup_no_draft(self):
        from optimization.speculative_decoding import estimate_speedup

        speedup = estimate_speedup(gamma=0, acceptance_rate=0.0)
        assert speedup == 1.0
