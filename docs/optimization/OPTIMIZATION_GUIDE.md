# XRFM Optimization Guide — v0.9.0

## Overview

Three performance optimization techniques for XRFM:

| Technique | Speedup | Memory Reduction | Quality Impact |
|---|---|---|---|
| FlashAttention | 2-4× | O(n) vs O(n²) attention | None (exact) |
| INT8 Quantization | ~2× (CPU) | 4× compression | < 0.5% accuracy loss |
| INT4 Quantization | – | 8× compression | 1–3% accuracy loss |
| Speculative Decoding | 2–3× | – | None (exact) |

## FlashAttention

Drop-in replacement for manual attention in `MultiHeadAttention`:

```python
from optimization.flash_attention import flash_attention_forward

# In MultiHeadAttention.forward():
# Replace: scores = Q @ K.T / sqrt(d_head); attn = softmax(scores); out = attn @ V
# With:
out = flash_attention_forward(Q, K, V, mask=mask, dropout_p=self.dropout_p, training=self.training)
```

PyTorch auto-selects the fastest backend (FlashAttention-2 on CUDA, MemoryEfficientAttention, or fallback math).

## Quantization

Compress model weights from FP32 to INT8/INT4:

```python
from optimization.quantization import quantize_model_weights, dequantize_weight, compute_compression_ratio

# INT8 (4x compression)
model, q_map = quantize_model_weights(model, bits=8)
ratio = compute_compression_ratio(q_map)  # ~3.9x

# INT4 with group-wise (8x compression)
model, q_map = quantize_model_weights(model, bits=4, group_size=128)

# Dequantize for inference
for name, qw in q_map.items():
    weight_fp32 = qw.dequantize()
```

### Quantization Schemes

| Scheme | INT8 | INT4 |
|---|---|---|
| Per-tensor | One scale/zp for entire weight | Not recommended |
| Per-channel | One scale/zp per output channel | Not recommended |
| Group-wise | – | Groups of 64–128 elements |

## Speculative Decoding

Use a small draft model to accelerate the large target model:

```python
from optimization.speculative_decoding import SpeculativeDecoder

target = GPTModel("config/config.yaml")  # large model
draft = GPTModel("config/config.yaml")  # small model (same vocab)

decoder = SpeculativeDecoder(target, draft, gamma=5)
output = decoder.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
)
```

### How It Works

```
Cycle: Draft generates 5 tokens → Target verifies all 5 in parallel
       → Accept ~3.5 tokens on average → Repeat

Speedup ≈ (γ × acceptance_rate + 1) / (γ × draft_cost/target_cost + 1)
```

The output distribution is mathematically identical to standard autoregressive decoding (rejection sampling preserves exact distribution).

## Integration

All optimizations are **opt-in**: existing code continues to work without any changes.

### Add FlashAttention to model

```python
# In model/attention/multi_head.py, replace the manual attention with:
from optimization.flash_attention import flash_attention_forward
# ... use flash_attention_forward(Q, K, V, ...) instead of manual loop
```

### Quantize before deployment

```python
from optimization.quantization import quantize_model_weights, compute_compression_ratio

model, q_map = quantize_model_weights(model, bits=8)
print(f"Compression: {compute_compression_ratio(q_map):.1f}x")
```

## References

- Dao et al. (2022) — FlashAttention
- Dao (2023) — FlashAttention-2
- Jacob et al. (2018) — Quantization and Training of Neural Networks
- Dettmers et al. (2022) — LLM.int8()
- Frantar et al. (2023) — GPTQ
- Leviathan et al. (2023) — Speculative Decoding
