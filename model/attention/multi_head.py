"""
Multi-Head Attention for XRFM (v0.6.0).

Manual multi-head self-attention with KV cache support for efficient
autoregressive inference. Supports RoPE, masking, and dropout.

Conceptual references (not copied):
- Vaswani et al. (2017) — Attention Is All You Need
- Meta AI (2024) — Llama 3 attention + KV cache design
- DeepSeek-AI (2024) — DeepSeek-V3 attention patterns

Implementation is original.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.rope import RoPE


class MultiHeadAttention(nn.Module):
    """Manual multi-head self-attention with KV cache support.

    Splits input into heads, computes scaled dot-product attention,
    and supports incremental generation via past_kv parameter.

    Shape conventions:
        Input:  (batch, seq, d_model)
        Output: (batch, seq, d_model)
        K/V:    (batch, n_heads, seq, d_head)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_rope: bool = True,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout
        self.use_rope = use_rope

        # Q/K/V/O projections
        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

        # RoPE applied to Q/K after head splitting
        self.rope = RoPE(d_model=d_model // n_heads) if use_rope else None

    def _init_weights(self) -> None:
        for layer in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute multi-head self-attention with optional KV cache.

        When use_cache=True and past_kv is provided:
            - Only the new token's K/V are computed
            - They are concatenated with cached K/V from previous steps
            - Q attends to the full (cached + new) K/V sequence

        Args:
            x: Input (batch, seq, d_model).
            mask: Optional attention mask.
            past_kv: Tuple of (K_cached, V_cached) from previous step, or None.
            use_cache: Whether to return present KV for caching.

        Returns:
            (output, present_kv) where output is (batch, seq, d_model) and
            present_kv is (K_full, V_full) or None if use_cache=False.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq, d_model), got {x.dim()}D"
            )
        batch_size, seq_len, d_model_in = x.shape
        if d_model_in != self.d_model:
            raise ValueError(
                f"Input dim ({d_model_in}) != model dim ({self.d_model})"
            )

        # Project to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads: (batch, seq, d_model) -> (batch, n_heads, seq, d_head)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE before concatenating with cache.
        # When using cache, positions are offset by cache length.
        cache_len = past_kv[0].shape[2] if (past_kv is not None and past_kv[0].numel() > 0) else 0
        if self.use_rope and self.rope is not None:
            Q, K = self.rope(Q, K, seq_len, offset=cache_len)

        # Concatenate with cached K/V if provided
        if past_kv is not None and past_kv[0].numel() > 0:
            K_cache, V_cache = past_kv
            K_full = torch.cat([K_cache, K], dim=2)
            V_full = torch.cat([V_cache, V], dim=2)
        else:
            K_full = K
            V_full = V

        # Build present KV for caching
        present_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if use_cache:
            present_kv = (K_full, V_full)

        # Scaled dot-product attention (FlashAttention when available)
        # Try FlashAttention for the full (no explicit mask) or initial case
        use_flash = False
        if mask is None and cache_len == 0:
            try:
                from optimization.flash_attention import flash_attention_forward
                out = flash_attention_forward(
                    Q, K_full, V_full,
                    dropout_p=self.dropout_p,
                    training=self.training,
                )
                use_flash = True
            except ImportError:
                pass

        if not use_flash:
            scores = torch.matmul(Q, K_full.transpose(-2, -1)) / math.sqrt(self.d_head)

            if mask is not None:
                if mask.dim() not in (2, 3, 4):
                    raise ValueError(
                        f"Mask must have 2-4 dims, got {mask.dim()}"
                    )
                scores = scores.masked_fill(mask == 0, float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, V_full)

        # Concatenate heads: (batch, n_heads, seq, d_head) -> (batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_o(out)

        return output, present_kv

# To use: replace the softmax+matmul block in forward() with:
#