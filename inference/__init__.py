"""
XRFM Inference Engine (v0.6.0).

Provides autoregressive text generation with KV cache acceleration,
temperature, top-k, and nucleus (top-p) sampling.

Usage:
    from inference import GenerationEngine
    from model.gpt import GPTModel

    model = GPTModel()
    engine = GenerationEngine(model)
    output_ids = engine.generate(
        input_ids, max_new_tokens=50, temperature=0.8, top_p=0.9
    )
"""

from inference.engine import GenerationEngine
from inference.kv_cache import KVCache
from inference.sampling import (
    sample_greedy,
    sample_temperature,
    sample_token,
    sample_top_k,
    sample_top_p,
)

__all__ = [
    "GenerationEngine",
    "KVCache",
    "sample_greedy",
    "sample_temperature",
    "sample_top_k",
    "sample_top_p",
    "sample_token",
]
