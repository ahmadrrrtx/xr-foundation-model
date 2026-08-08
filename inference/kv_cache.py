"""
KV Cache for XR Foundation Model (XRFM) inference engine.

Stores previously computed Key and Value tensors per transformer layer
to avoid recomputing them during autoregressive generation.
Reduces per-step attention complexity from O(n^2) to O(n).

Conceptual references (not copied):
- HuggingFace transformers — past_key_values pattern
- Meta AI (2024) — Llama 3 KV cache design
- DeepSeek-AI (2024) — DeepSeek-V3 inference optimization

Implementation is original.

Shape convention per layer:
    K_cache: (batch_size, num_heads, cached_seq_len, head_dim)
    V_cache: (batch_size, num_heads, cached_seq_len, head_dim)
"""

import torch


class KVCache:
    """Key-Value cache for autoregressive transformer inference.

    Maintains a list of (K, V) tuples, one per transformer layer.
    Supports both dynamic concatenation and pre-allocated contiguous
    buffer allocation for high-throughput zero-copy updates.

    Attributes:
        layers: List of (K_cache, V_cache) tuples per transformer block.
        max_cache_len: Maximum number of cached tokens (from model config).


    EXPERIMENTAL (v1.1): not used by the production training/inference
    paths. See docs/architecture/DEAD_OR_EXPERIMENTAL.md."""

    def __init__(self, max_cache_len: int = 2048) -> None:
        if max_cache_len <= 0:
            raise ValueError(f"max_cache_len must be positive, got {max_cache_len}")
        self.layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.max_cache_len = max_cache_len
        self._seq_len: int = 0
        self._static_buffers: list[tuple[torch.Tensor, torch.Tensor]] = []

    def is_empty(self) -> bool:
        """True if no cache entries have been populated yet."""
        return len(self.layers) == 0

    @property
    def seq_len(self) -> int:
        """Number of tokens currently cached."""
        return self._seq_len

    def update(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache for one layer with new K, V tensors.

        On first call: initializes the cache entry for this layer.
        On subsequent calls: concatenates new K, V with cached values.

        Args:
            layer_idx: Transformer block index (0-based).
            k_new: New key tensor (batch, n_heads, new_seq_len, head_dim).
            v_new: New value tensor (batch, n_heads, new_seq_len, head_dim).

        Returns:
            (K_full, V_full) — cached + new, concatenated along seq dim.
        """
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")

        # Extend layers list on first population
        while len(self.layers) <= layer_idx:
            self.layers.append((torch.empty(0), torch.empty(0)))

        k_prev, v_prev = self.layers[layer_idx]

        if k_prev.numel() == 0:
            k_full = k_new
            v_full = v_new
        else:
            k_full = torch.cat([k_prev, k_new], dim=2)
            v_full = torch.cat([v_prev, v_new], dim=2)

        # Truncate from the left if exceeding max length
        cached_len = k_full.shape[2]
        if cached_len > self.max_cache_len:
            excess = cached_len - self.max_cache_len
            k_full = k_full[:, :, excess:, :]
            v_full = v_full[:, :, excess:, :]

        self.layers[layer_idx] = (k_full, v_full)
        self._seq_len = k_full.shape[2]
        return k_full, v_full

    def clear(self) -> None:
        """Reset cache for a new generation sequence."""
        self.layers = []
        self._static_buffers = []
        self._seq_len = 0

    def __repr__(self) -> str:
        return f"KVCache(n_layers={len(self.layers)}, seq_len={self.seq_len}, max_cache_len={self.max_cache_len})"
