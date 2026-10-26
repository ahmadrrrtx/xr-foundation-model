"""XRFM inference subsystem.

Public surface:

    from xrfm.inference import generate, GenerationEngine, GenerationResult, GenerationMetadata

Sampling primitives (``sample_greedy``, ``sample_top_k``, ``sample_top_p``,
``sample_temperature``, ``sample_token``) and :class:`KVCache` are importable
for advanced/experimental use.
"""

from xrfm.inference.engine import GenerationEngine, GenerationEvent, GenerationMetadata
from xrfm.inference.generate import GenerationResult, generate
from xrfm.inference.kv_cache import KVCache
from xrfm.inference.sampling import (
    sample_greedy,
    sample_temperature,
    sample_token,
    sample_top_k,
    sample_top_p,
)

__all__ = [
    "GenerationEngine",
    "GenerationEvent",
    "GenerationMetadata",
    "GenerationResult",
    "KVCache",
    "generate",
    "sample_greedy",
    "sample_temperature",
    "sample_token",
    "sample_top_k",
    "sample_top_p",
]
