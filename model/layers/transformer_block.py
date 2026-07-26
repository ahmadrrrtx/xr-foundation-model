"""
Transformer block for XRFM (v0.6.0).

Pre-norm block with attention, SwiGLU FFN, residual connections,
and KV cache pass-through for autoregressive inference.

Conceptual references (not copied):
- Vaswani et al. (2017) — Transformer
- Touvron et al. (2023) — Llama 2 pre-norm design
- DeepSeek-AI (2024) — DeepSeek-V3 block architecture

Implementation is original.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from model.attention.multi_head import MultiHeadAttention
from model.layers.rmsnorm import RMSNorm
from model.layers.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with KV cache support.

    Architecture: norm1 -> attention -> residual -> norm2 -> SwiGLU -> residual.

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
    ) -> None:
        super().__init__()

        if d_model <= 0 or n_heads <= 0 or d_ff <= 0:
            raise ValueError("d_model, n_heads, d_ff must be positive")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        # Pre-normalization layers
        self.norm1 = RMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)
        self.norm2 = RMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)

        # Sub-layers
        self.attn = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout, use_rope=use_rope
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Apply transformer block with optional KV cache.

        Args:
            x: Input (batch, seq, d_model).
            mask: Optional attention mask.
            past_kv: Cached (K, V) from previous steps, or None.
            use_cache: Whether to return updated KV for caching.

        Returns:
            (output, present_kv) where output is (batch, seq, d_model)
            and present_kv is the updated KV cache for this layer.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq, d_model), got {x.dim()}D"
            )

        # Attention sub-layer with pre-norm
        h, present_kv = self.attn(
            self.norm1(x), mask=mask, past_kv=past_kv, use_cache=use_cache
        )
        x = x + self.dropout(h)

        # SwiGLU sub-layer with pre-norm
        h = self.ffn(self.norm2(x))
        x = x + self.dropout(h)

        return x, present_kv
