# XRFM Inference Engine — v0.6.0

## Overview

The inference engine provides autoregressive text generation with **KV cache acceleration**, reducing per-step attention from O(n²) to O(n). It supports greedy, temperature, top-k, and nucleus (top-p) sampling strategies.

## Architecture

```
Prompt → GPTModel (full forward, cache K/V)
   ↓
┌──────────────────────────────────┐
│  Generation Loop                 │
│  for each new token:             │
│    1. Forward single token       │
│       with past_key_values       │
│    2. Sample from logits         │
│    3. Append to sequence         │
│    4. Update KV cache            │
└──────────────────────────────────┘
   ↓
Generated Sequence
```

## Components

| Module | Path | Purpose |
|---|---|---|
| GenerationEngine | `inference/engine.py` | Main generation loop |
| KVCache | `inference/kv_cache.py` | Key-Value tensor storage per layer |
| Sampling | `inference/sampling.py` | Greedy, temperature, top-k, top-p |

## Quick Start

```python
from model.gpt import GPTModel
from inference.engine import GenerationEngine

model = GPTModel()
engine = GenerationEngine(model)

# Encode prompt
prompt_ids = torch.tensor([1, 2, 3, 4, 5])

# Generate with temperature + top-p
output = engine.generate(
    prompt_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9,
)
```

## Sampling Strategies

### Greedy (`temperature=0`)
Selects the most probable token at each step. Deterministic.

### Temperature (`temperature > 0`)
Scales logits before softmax. Higher values → more diversity. Default: 1.0.

### Top-K (`top_k=N`)
Restricts sampling to the N highest-probability tokens.

### Top-P / Nucleus (`top_p=P`)
Keeps the smallest set of tokens whose cumulative probability exceeds P. Dynamically adjusts the candidate pool size.

### Combined
Priority: `temperature=0` (greedy) > `top_k` > `top_p` > temperature-only.

## KV Cache

The KV cache stores previously computed Key and Value tensors for each transformer layer. On each generation step, only the new token's projections are computed. The cached K/V from all previous tokens are reused.

**Without KV cache:** O(n²) per step (recomputing attention for all tokens).
**With KV cache:** O(n) per step (only new token's Q attends to cached K/V).

Cache shape per layer: `(batch=1, n_heads, cached_seq_len, d_head)`.

## Model Changes for v0.6.0

- `GPTModel.forward()` returns `(logits, present_key_values)` when `use_cache=True`
- `MultiHeadAttention.forward()` accepts `past_kv` and returns `(output, present_kv)`
- `TransformerBlock.forward()` passes through KV cache parameters
- `RoPE` accepts `offset` parameter for correct position encoding with cached sequences

## Performance

| Mode | Per-Step Complexity | Memory |
|---|---|---|
| Full forward (no cache) | O(L²) | O(L) |
| KV cache (single token) | O(L) | O(L) |
| KV cache (max_seq_len=512) | O(512) | ~4MB per layer |

## Future Extensions (deferred)

- Streaming token output (yield per token)
- Beam search decoding
- Repetition penalty
- vLLM / PagedAttention integration (Phase 9)
- Speculative decoding (Phase 9)

## References

- Su et al. (2023) — RoPE position encoding
- Holtzman et al. (2020) — Nucleus sampling
- Meta AI (2024) — Llama 3 inference design
- HuggingFace transformers — KV cache pattern
