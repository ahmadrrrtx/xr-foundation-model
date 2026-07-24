"""
Rotary Positional Embedding (RoPE) for XR Foundation Model (XRFM).

Purpose: Encode relative token positions into query and key vectors using
rotation matrices. RoPE ensures that attention scores depend only on the
relative distance between tokens (m - n), not absolute positions.

Conceptual references (NOT copied):
- Su, J., Ahmed, M., Lu, Y., et al. (2023). RoFormer: Enhanced Transformer
  with Rotary Position Embedding. arXiv:2104.09864.
- Llama 3 Technical Report (Meta AI, 2024) — RoPE design, frequency base.
- DeepSeek-AI (2024). DeepSeek-V3 Technical Report — RoPE usage.

Implementation is original. No source code copied from reference repositories.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Original Rotary Positional Embedding.

    Applies rotation matrices to query/key tensors. For feature dimension d,
    pairs (i, i + d/2) are rotated by angle theta_i = base^(-2*i/d) for
    relative position encoding.

    Design notes:
    - Configurable (base, max_seq_len, scale_factor).
    - Numerical stability: frequency computation uses float32; rotation
      uses sin/cos which are bounded in [-1, 1].
    - Stable interface: forward(q, k, seq_len) returns rotated Q, K.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        scale_factor: float = 1.0,
    ) -> None:
        super(RoPE, self).__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        self.scale_factor = scale_factor
        half_dim = d_model // 2
        freq_indices = torch.arange(0, half_dim, dtype=torch.float32)
        freq_indices = freq_indices * scale_factor
        inv_freq = 1.0 / (base ** (freq_indices / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Split in half and rotate: (-x2, x1)."""
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_emb(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotation to x of shape (..., d_model)."""
        d = x.shape[-1]
        half_dim = d // 2
        # Compute frequency based on the actual feature dimension of x.
        # This ensures RoPE works correctly regardless of whether x has
        # full d_model or d_head dimension.
        freq_indices = torch.arange(0, half_dim, dtype=torch.float32, device=x.device)
        freq_indices = freq_indices * self.scale_factor
        inv_freq = 1.0 / (self.base ** (freq_indices / half_dim)) if half_dim > 0 else torch.ones(0, device=x.device, dtype=torch.float32)
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos_pair = torch.cos(freqs)
        sin_pair = torch.sin(freqs)

        # Pair first half and second half with same frequency.
        # cos_full and sin_full each have shape (seq_len, d)
        cos_full = torch.cat([cos_pair, cos_pair], dim=-1)
        sin_full = torch.cat([sin_pair, sin_pair], dim=-1)

        cos = cos_full.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d)
        sin = sin_full.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d)

        # x shape: (batch, heads, seq_len, d) or similar
        # We need to handle arbitrary batch/head dimensions.
        # Add dimensions to match: unsqueeze at appropriate positions.
        # The last dimension is d. We need cos/sin to broadcast over batch/head.
        # cos shape: (1, 1, seq_len, d) broadcasts to (batch, heads, seq_len, d)
        rotated = x * cos + self.rotate_half(x) * sin
        return rotated

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rotary_emb(q, seq_len), self.apply_rotary_emb(k, seq_len)
