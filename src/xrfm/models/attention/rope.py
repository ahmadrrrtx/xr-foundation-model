"""
Rotary Positional Embedding (RoPE) for XRFM (v0.6.0).

Encodes relative token positions into query/key vectors via rotation.
Supports position offset for KV cache during autoregressive inference,
non-persistent FP32 ``cos``/``sin`` table caching with dynamic sequence
extension, and strict input dtype preservation for FP16/BF16 mixed precision.

Conceptual references (not copied):
- Su et al. (2023) — RoFormer (arXiv:2104.09864)
- Meta AI (2024) — Llama 3 RoPE design
- DeepSeek-AI (2024) — DeepSeek-V3 RoPE usage

Implementation is original.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Rotary Positional Embedding with KV cache position offset and FP32 table cache.

    For feature dimension d, pairs (i, i + d/2) are rotated by
    theta_i = base^(-2*i/d) for relative position encoding.

    Trigonometric tables are cached in ``float32`` (``persistent=False`` so
    existing checkpoint state_dicts remain 100% compatible) and dynamically
    extended if ``offset + seq_len`` exceeds ``max_seq_len``. Rotation math is
    performed in ``float32`` and cast back to ``x.dtype`` so ``fp16``/``bf16``
    Query and Key tensors never upcast and mismatch Value in SDPA.
    """

    inv_freq: torch.Tensor
    _cos_cached: torch.Tensor
    _sin_cached: torch.Tensor

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if base <= 0:
            raise ValueError(f"base must be positive, got {base}")

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = float(base)
        self.scale_factor = float(scale_factor)

        half_dim = d_model // 2
        freq_indices = torch.arange(0, half_dim, dtype=torch.float32) * self.scale_factor
        inv_freq = (
            1.0 / (self.base ** (freq_indices / half_dim)) if half_dim > 0 else torch.ones(0, dtype=torch.float32)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("_cos_cached", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("_sin_cached", torch.empty(0, dtype=torch.float32), persistent=False)
        self._build_cache(max_seq_len, device=inv_freq.device)

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        """Populate FP32 ``_cos_cached`` and ``_sin_cached`` up to ``seq_len``."""
        self.max_seq_len = max(self.max_seq_len, seq_len)
        if self.inv_freq.device != device or self.inv_freq.dtype != torch.float32:
            half_dim = self.d_model // 2
            freq_indices = torch.arange(0, half_dim, dtype=torch.float32, device=device) * self.scale_factor
            inv_freq = (
                1.0 / (self.base ** (freq_indices / half_dim))
                if half_dim > 0
                else torch.ones(0, dtype=torch.float32, device=device)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(0, self.max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        cos_full = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
        sin_full = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
        self.register_buffer("_cos_cached", cos_full, persistent=False)
        self.register_buffer("_sin_cached", sin_full, persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Split last dim in half and rotate: (-x2, x1)."""
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_emb(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> torch.Tensor:
        """Apply rotary embedding with optional position offset, preserving ``x.dtype``.

        Args:
            x: Input (..., seq_len, d_head).
            seq_len: Number of new token positions.
            offset: Position offset for KV cache (cache length).

        Returns:
            Rotated tensor of same shape and dtype as ``x``.
        """
        orig_dtype = x.dtype
        d = x.shape[-1]
        needed_len = offset + seq_len

        if d == self.d_model:
            if (
                self._cos_cached.numel() == 0
                or self._cos_cached.shape[0] < needed_len
                or self._cos_cached.device != x.device
            ):
                self._build_cache(needed_len, device=x.device)
            cos_slice = self._cos_cached[offset:needed_len]
            sin_slice = self._sin_cached[offset:needed_len]
        else:
            # Fallback if called directly with a tensor whose last dim != self.d_model
            half_dim = d // 2
            freq_indices = torch.arange(0, half_dim, dtype=torch.float32, device=x.device) * self.scale_factor
            inv_freq = (
                1.0 / (self.base ** (freq_indices / half_dim))
                if half_dim > 0
                else torch.ones(0, device=x.device, dtype=torch.float32)
            )
            t = torch.arange(offset, needed_len, device=x.device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            cos_slice = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
            sin_slice = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)

        # Reshape for broadcasting: (1, ..., 1, seq_len, d_head)
        view_shape = (1,) * (x.dim() - 2) + (seq_len, d)
        cos = cos_slice.view(view_shape)
        sin = sin_slice.view(view_shape)

        x_fp32 = x.float()
        rotated = x_fp32 * cos + self.rotate_half(x_fp32) * sin
        return rotated.to(orig_dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embedding to Q and K with optional cache offset.

        Args:
            q: Query tensor (batch, n_heads, seq_len, d_head).
            k: Key tensor (batch, n_kv_heads, seq_len, d_head).
            seq_len: Number of new token positions.
            offset: Position offset from KV cache (0 for full-sequence).

        Returns:
            (Q_rotated, K_rotated) — same shapes and dtypes as inputs.
        """
        return (
            self.apply_rotary_emb(q, seq_len, offset=offset),
            self.apply_rotary_emb(k, seq_len, offset=offset),
        )
