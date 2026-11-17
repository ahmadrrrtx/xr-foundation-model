"""
Transformer block for XRFM (v0.6.0+).

Pre-norm block with Multi-Head / Grouped-Query Attention, SwiGLU (or standard
GELU) FFN, residual connections, and KV cache pass-through for autoregressive
inference.

Conceptual references (not copied):
- Vaswani et al. (2017) — Transformer
- Touvron et al. (2023) — Llama 2 pre-norm design
- DeepSeek-AI (2024) — DeepSeek-V3 block architecture

Implementation is original.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xrfm.models.attention.multi_head import MultiHeadAttention
from xrfm.models.layers.rmsnorm import RMSNorm
from xrfm.models.layers.swiglu import SwiGLU


class StandardFFN(nn.Module):
    """Standard 2-layer GELU feed-forward network used when ``use_swiglu=False``."""

    def __init__(self, d_model: int, d_ff: int, bias: bool = True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.fc2(self.act(self.fc1(x)))
        return out


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with KV cache and optional GQA / SwiGLU support.

    Architecture: norm1 -> attention -> residual -> norm2 -> FFN -> residual.

    Supports incremental inference via use_cache/past_kv parameters
    passed through to the attention sub-layer.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_rmsnorm: bool = True,
        use_rope: bool = True,
        use_swiglu: bool = True,
        bias: bool = True,
        n_kv_heads: int | None = None,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if d_model <= 0 or n_heads <= 0 or d_ff <= 0:
            raise ValueError("d_model, n_heads, d_ff must be positive")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        # Pre-normalization layers
        self.norm1: nn.Module = RMSNorm(d_model, eps=norm_eps) if use_rmsnorm else nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2: nn.Module = RMSNorm(d_model, eps=norm_eps) if use_rmsnorm else nn.LayerNorm(d_model, eps=norm_eps)

        # Sub-layers
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_rope=use_rope,
            bias=bias,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )
        self.ffn: nn.Module = (
            SwiGLU(d_model=d_model, d_ff=d_ff, bias=bias)
            if use_swiglu
            else StandardFFN(d_model=d_model, d_ff=d_ff, bias=bias)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        """Apply transformer block with optional KV cache."""
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq, d_model), got {x.dim()}D")

        # Attention sub-layer with pre-norm
        h, present_kv = self.attn(self.norm1(x), mask=mask, past_kv=past_kv, use_cache=use_cache)
        x = x + self.dropout(h)

        # FFN sub-layer with pre-norm
        h = self.ffn(self.norm2(x))
        x = x + self.dropout(h)

        return x, present_kv
