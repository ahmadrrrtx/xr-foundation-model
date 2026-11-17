"""
Multi-Head and Grouped-Query Attention (MHA / GQA) for XRFM (v0.6.0+).

Manual multi-head / grouped-query self-attention with KV cache support for
efficient autoregressive inference. Supports RoPE, native ``is_causal=True``
FlashAttention/MemEfficient SDPA when unmasked, explicit causal+padding mask
composition when masked, and dropout.

Conceptual references (not copied):
- Vaswani et al. (2017) — Attention Is All You Need
- Ainslie et al. (2023) — GQA: Training Generalized Multi-Query Transformer Models
- Meta AI (2024) — Llama 3 attention + KV cache design
- DeepSeek-AI (2024) — DeepSeek-V3 attention patterns

Implementation is original.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from xrfm.models.attention.rope import RoPE


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match Query heads for Grouped-Query Attention (GQA).

    Args:
        hidden_states: ``(batch, n_kv_heads, seq, d_head)``.
        n_rep: ``n_heads // n_kv_heads``.

    Returns:
        ``(batch, n_heads, seq, d_head)``.
    """
    if n_rep == 1:
        return hidden_states
    batch, n_kv_heads, slen, d_head = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, n_kv_heads, n_rep, slen, d_head)
        .reshape(batch, n_kv_heads * n_rep, slen, d_head)
    )


class MultiHeadAttention(nn.Module):
    """Multi-Head / Grouped-Query Self-Attention with KV cache support.

    Shape conventions:
        Input:  (batch, seq, d_model)
        Output: (batch, seq, d_model)
        Cached K/V: (batch, n_kv_heads, seq, d_head)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        bias: bool = True,
        n_kv_heads: int | None = None,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        resolved_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if resolved_kv_heads <= 0 or resolved_kv_heads > n_heads or (n_heads % resolved_kv_heads != 0):
            raise ValueError(f"n_kv_heads ({resolved_kv_heads}) must be positive and divide n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = resolved_kv_heads
        self.n_rep = n_heads // resolved_kv_heads
        self.d_head = d_model // n_heads
        self.kv_dim = self.n_kv_heads * self.d_head
        self.dropout_p = dropout
        self.use_rope = use_rope

        # Q/K/V/O projections (K/V project to n_kv_heads * d_head for GQA; equals d_model when MHA)
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, self.kv_dim, bias=bias)
        self.W_v = nn.Linear(d_model, self.kv_dim, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.bias = bias
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

        # RoPE applied to Q/K after head splitting
        self.rope = RoPE(d_model=self.d_head, max_seq_len=max_seq_len, base=rope_theta) if use_rope else None

    def _init_weights(self) -> None:
        for layer in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Compute self-attention with optional KV cache and GQA."""
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq, d_model), got {x.dim()}D")
        batch_size, seq_len, d_model_in = x.shape
        if d_model_in != self.d_model:
            raise ValueError(f"Input dim ({d_model_in}) != model dim ({self.d_model})")

        # Project to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads: Q -> (B, n_heads, seq, d_head), K/V -> (B, n_kv_heads, seq, d_head)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)

        # Apply RoPE before concatenating with cache.
        cache_len = past_kv[0].shape[2] if (past_kv is not None and past_kv[0].numel() > 0) else 0
        if self.use_rope and self.rope is not None:
            Q, K = self.rope(Q, K, seq_len, offset=cache_len)

        # Concatenate with cached K/V (stored compactly at n_kv_heads!)
        if past_kv is not None and past_kv[0].numel() > 0:
            K_cache, V_cache = past_kv
            K_full = torch.cat([K_cache, K], dim=2)
            V_full = torch.cat([V_cache, V], dim=2)
        else:
            K_full = K
            V_full = V

        present_kv: tuple[torch.Tensor, torch.Tensor] | None = None
        if use_cache:
            present_kv = (K_full, V_full)

        # Expand KV heads to match Q heads when n_kv_heads < n_heads (no-op for MHA)
        K_attn = _repeat_kv(K_full, self.n_rep)
        V_attn = _repeat_kv(V_full, self.n_rep)

        q_len = Q.shape[2]
        kv_len = K_attn.shape[2]

        # ------------------------------------------------------------------
        # Fast-path vs explicit-mask path:
        # When `mask is None`:
        #   - If `cache_len == 0` (standard full-sequence training/prefill,
        #     `q_len == kv_len`), pass `sdpa_mask = None` so PyTorch SDPA
        #     dispatches native `is_causal=True` FlashAttention-2 / MemEfficient
        #     kernels with O(L) memory instead of materializing an O(L^2) tensor!
        #   - If `cache_len > 0` and `q_len == 1` (single-token decode step),
        #     the single query at `kv_len - 1` attends to all `0..kv_len - 1`
        #     keys (`diagonal = cache_len + 1 = kv_len` masks nothing), so an
        #     all-zero (1, kv_len) mask or `is_causal=False` is exact.
        #   - Otherwise (chunked prefill with `cache_len > 0, q_len > 1` or
        #     any explicit `mask`), build the exact causal + padding additive mask.
        # ------------------------------------------------------------------
        additive_mask: torch.Tensor | None
        if mask is None and cache_len == 0:
            additive_mask = None
        else:
            causal_mask = torch.triu(
                torch.full((q_len, kv_len), float("-inf"), dtype=Q.dtype, device=Q.device),
                diagonal=cache_len + 1,
            )
            additive_mask = causal_mask
            if mask is not None:
                if mask.dim() not in (2, 3, 4):
                    raise ValueError(f"Mask must have 2-4 dims, got {mask.dim()}")
                if not bool(torch.all((mask == 0) | (mask == 1))):
                    raise ValueError("attention mask values must be 0 or 1 (1=attend)")
                if mask.dim() == 2:
                    if mask.shape[0] == batch_size and mask.shape[1] == kv_len:
                        mask = mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,kv) padding
                    elif tuple(mask.shape) == (q_len, kv_len):
                        mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,q,kv) custom
                    else:
                        raise ValueError(
                            f"2-D mask must be (batch={batch_size}, kv={kv_len}) or (q={q_len}, kv={kv_len}); "
                            f"got {tuple(mask.shape)}"
                        )
                elif mask.dim() == 3:
                    if (
                        mask.shape[0] not in (1, batch_size)
                        or mask.shape[1] not in (1, q_len)
                        or mask.shape[2] != kv_len
                    ):
                        raise ValueError(f"3-D mask must broadcast to (B,q,kv)=({batch_size},{q_len},{kv_len})")
                    mask = mask.unsqueeze(1)  # (B,1,q,kv)
                else:
                    if mask.shape[-1] != kv_len or mask.shape[-2] not in (1, q_len):
                        raise ValueError(f"4-D mask must end in (q,kv)=({q_len},{kv_len})")
                mask = mask.to(device=Q.device, dtype=Q.dtype)
                explicit_mask = torch.zeros_like(mask, dtype=Q.dtype).masked_fill(mask == 0, float("-inf"))
                additive_mask = causal_mask + explicit_mask

        use_flash = False
        try:
            from xrfm.optimization.flash_attention import flash_attention_forward

            use_dropout = self.dropout_p if self.training else 0.0
            out = flash_attention_forward(
                Q,
                K_attn,
                V_attn,
                mask=additive_mask,
                dropout_p=use_dropout,
                training=self.training,
            )
            use_flash = True
        except ImportError:
            use_flash = False

        if not use_flash:
            scores = torch.matmul(Q, K_attn.transpose(-2, -1)) / math.sqrt(self.d_head)
            if additive_mask is None:
                additive_mask = torch.triu(
                    torch.full((q_len, kv_len), float("-inf"), dtype=Q.dtype, device=Q.device),
                    diagonal=cache_len + 1,
                )
            if additive_mask.dim() == 2:
                additive_mask = additive_mask.unsqueeze(0).unsqueeze(0)
            elif additive_mask.dim() == 3:
                additive_mask = additive_mask.unsqueeze(1)
            scores = scores + additive_mask

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, V_attn)

        # Concatenate heads: (batch, n_heads, seq, d_head) -> (batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_o(out)

        return output, present_kv
