"""
Rotary Positional Embedding (RoPE) for XRFM (v0.6.0).

Encodes relative token positions into query/key vectors via rotation.
Supports position offset for KV cache during autoregressive inference.

Conceptual references (not copied):
- Su et al. (2023) — RoFormer (arXiv:2104.09864)
- Meta AI (2024) — Llama 3 RoPE design
- DeepSeek-AI (2024) — DeepSeek-V3 RoPE usage

Implementation is original.
"""

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Rotary Positional Embedding with KV cache position offset.

    For feature dimension d, pairs (i, i + d/2) are rotated by
    theta_i = base^(-2*i/d) for relative position encoding.

    When offset > 0 (KV cache mode), positions are shifted by offset
    so that new tokens receive correct absolute position embeddings.
    """

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
        """Split last dim in half and rotate: (-x2, x1)."""
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_emb(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> torch.Tensor:
        """Apply rotary embedding with optional position offset.

        Args:
            x: Input (..., seq_len, d_head).
            seq_len: Number of new token positions.
            offset: Position offset for KV cache (cache length).

        Returns:
            Rotated tensor of same shape as x.
        """
        d = x.shape[-1]
        half_dim = d // 2

        # Compute inverse frequencies for this device/dtype
        freq_indices = torch.arange(0, half_dim, dtype=torch.float32, device=x.device)
        freq_indices = freq_indices * self.scale_factor

        if half_dim > 0:
            inv_freq = 1.0 / (self.base ** (freq_indices / half_dim))
        else:
            inv_freq = torch.ones(0, device=x.device, dtype=torch.float32)

        # Position indices: offset, offset+1, ..., offset+seq_len-1
        t = torch.arange(offset, offset + seq_len, device=x.device, dtype=torch.float32)

        # freqs: (seq_len, half_dim)
        freqs = torch.outer(t, inv_freq)

        # Duplicate for paired dimensions: (seq_len, d_head)
        cos_full = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
        sin_full = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)

        # Reshape for broadcasting: (1, 1, seq_len, d_head)
        cos = cos_full.unsqueeze(0).unsqueeze(0)
        sin = sin_full.unsqueeze(0).unsqueeze(0)

        return x * cos + self.rotate_half(x) * sin

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
            k: Key tensor (batch, n_heads, seq_len, d_head).
            seq_len: Number of new token positions.
            offset: Position offset from KV cache (0 for full-sequence).

        Returns:
            (Q_rotated, K_rotated) — same shapes as inputs.
        """
        return (
            self.apply_rotary_emb(q, seq_len, offset=offset),
            self.apply_rotary_emb(k, seq_len, offset=offset),
        )
