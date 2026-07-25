"""
XRFM Optimization Module (v0.9.0).

Performance and efficiency optimizations:
- FlashAttention — 2-4x faster attention via scaled_dot_product_attention
- Quantization — INT8/INT4 weight compression (4-8x memory reduction)
- Speculative Decoding — draft-model accelerated generation (2-3x speedup)
"""

from optimization.flash_attention import (
    scaled_dot_product_attention,
    flash_attention_forward,
    is_flash_attention_available,
    get_available_backend,
)
from optimization.quantization import (
    QuantizedWeight,
    quantize_int8_per_tensor,
    quantize_int8_per_channel,
    quantize_int4_groupwise,
    dequantize_weight,
    quantize_model_weights,
    compute_compression_ratio,
)
from optimization.speculative_decoding import (
    SpeculativeDecoder,
    estimate_speedup,
)

__all__ = [
    "scaled_dot_product_attention",
    "flash_attention_forward",
    "is_flash_attention_available",
    "get_available_backend",
    "QuantizedWeight",
    "quantize_int8_per_tensor",
    "quantize_int8_per_channel",
    "quantize_int4_groupwise",
    "dequantize_weight",
    "quantize_model_weights",
    "compute_compression_ratio",
    "SpeculativeDecoder",
    "estimate_speedup",
]
