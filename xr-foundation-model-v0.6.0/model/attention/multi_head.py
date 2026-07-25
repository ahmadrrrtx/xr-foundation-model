"""
Multi-Head Attention mechanism for XR Foundation Model (XRFM).

Purpose: Implement manual multi-head self-attention for the decoder-only
transformer architecture. This is the core mechanism that allows the model
to relate tokens across the sequence.

Conceptual references (NOT copied):
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need.
- Shazeer, N. (2019). Fast Transformer Decoding: Multi-Query Attention (GQA reference — not implemented in this phase).
- Llama 3 Technical Report (Meta, 2024) — Multi-head attention design,
  Grouped Query Attention concepts (deferred to optional Phase 8+).
- DeepSeek-V3 Technical Report (DeepSeek-AI, 2024) — Attention patterns,
  sparse attention concepts (deferred to RESEARCH-ONLY Phase 9+).

Implementation is original. The attention mechanism includes:
- Query/Key/Value linear projections.
- Attention score computation with `sqrt(d_k)` scaling.
- Softmax normalization.
- Multi-head concatenation and output projection.
- Masking support (padding mask) for batched sequences.

Design principles (Phase 4 architecture freeze):
- Configurable (`d_model`, `n_heads`, `dropout` from `ConfigLoader`).
- Stable interface (`MultiHeadAttention.forward(x, mask)` returns output tensor).
- Original manual implementation (no hidden behavior from `nn.MultiheadAttention`).
- Numerical stability (softmax scaling, masking, gradient-safe design).
- Future extensibility (GQA can replace Q/K projections; FlashAttention can replace score computation; sliding window can modify masking — all without changing `TransformerBlock` interface).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.rope import RoPE


class MultiHeadAttention(nn.Module):
    """Original manual multi-head self-attention for XRFM.

    This module splits the input into multiple attention heads, computes
    scaled dot-product attention for each head, concatenates the results,
    and applies a final output projection. Masking is supported for batched
    sequences with padding tokens.

    Design note: Using manual implementation (rather than `nn.MultiheadAttention`)
    ensures full control over Q/K/V projections, attention scoring, and masking.
    This is essential for future extensions (GQA, sliding window attention,
    FlashAttention optimization, custom masking patterns) that require
    direct access to attention score matrices.

    Attributes:
        n_heads (int): Number of attention heads (`ConfigLoader.get_model_config()["n_heads"]`).
        d_model (int): Model hidden dimension (`ConfigLoader.get_model_config()["d_model"]`).
        d_head (int): Dimension per head (`d_model // n_heads`).
        W_q (nn.Linear): Query projection matrix.
        W_k (nn.Linear): Key projection matrix.
        W_v (nn.Linear): Value projection matrix.
        W_o (nn.Linear): Output projection matrix.
        dropout (nn.Dropout): Dropout layer applied to attention weights.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_rope: bool = True,
    ) -> None:
        """Initialize multi-head attention module.

        Args:
            d_model: Hidden dimension size (`d_model`).
            n_heads: Number of attention heads. Must divide `d_model` evenly.
            dropout: Dropout probability applied to attention scores.

        Raises:
            ValueError: If `d_model` is not divisible by `n_heads`.
            ValueError: If `n_heads` or `d_model` is non-positive.
            ValueError: If `dropout` is not in [0, 1).
        """
        super(MultiHeadAttention, self).__init__()

        # Input validation: ensure valid hyperparameter values.
        if d_model <= 0:
            raise ValueError(
                f"d_model must be positive, got {d_model}. "
                f"Check ConfigLoader settings (model.d_model)."
            )
        if n_heads <= 0:
            raise ValueError(
                f"n_heads must be positive, got {n_heads}. "
                f"Check ConfigLoader settings (model.n_heads)."
            )
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}). "
                f"This is a fundamental requirement for multi-head attention. "
                f"Check ConfigLoader model configuration."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1), got {dropout}. "
                f"Check ConfigLoader settings (model.dropout)."
            )
        if not isinstance(use_rope, bool):
            raise ValueError(
                f"use_rope must be bool, got {type(use_rope).__name__}. "
                f"Check ConfigLoader settings (model.use_rope)."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # Dimension per attention head.
        self.dropout_p = dropout
        self.use_rope = use_rope

        # Linear projection matrices for Query, Key, Value, and Output.
        # These are standard linear transformations: `output = input @ W.T + b`.
        # No bias is used in many modern LLM designs (Llama, Mistral, DeepSeek);
        # however, a small bias can improve numerical stability. For simplicity
        # and to match modern practices closely, we include bias.
        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)

        # Dropout applied to attention scores (before softmax, in some variants;
        # here applied after attention weights for standard behavior).
        self.dropout = nn.Dropout(p=dropout)

        # Initialize projection weights with Xavier initialization for stability.
        # This ensures initial attention scores have appropriate variance,
        # preventing early divergence or saturation.
        self._init_weights()

        # Optional RoPE (Rotary Positional Embedding) for position encoding.
        # Applied to Q/K after splitting into heads. Configurable (`use_rope`).
        self.rope = RoPE(d_model=d_model // n_heads) if use_rope else None

    def _init_weights(self) -> None:
        """Initialize projection weights with Xavier (Glorot) uniform initialization.

        This private method applies `nn.init.xavier_uniform_` to all linear
        projection matrices. It is called during `__init__` to ensure numerical
        stability at the start of training.
        """
        for linear_layer in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(linear_layer.weight)
            # Initialize bias to zero for numerical stability.
            if linear_layer.bias is not None:
                nn.init.zeros_(linear_layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head self-attention output.

        The computation follows the standard scaled dot-product attention
        procedure defined in Vaswani et al. (2017):
        1. Project input `x` to Q, K, V.
        2. Split Q, K, V into `n_heads` heads.
        3. Compute attention scores: `Q @ K.T / sqrt(d_head)`.
        4. Apply mask (optional) to prevent attention to padding tokens.
        5. Apply softmax to obtain attention weights.
        6. Apply dropout to attention weights.
        7. Compute weighted sum: `attention_weights @ V`.
        8. Concatenate heads and apply output projection.

        Args:
            x: Input tensor of shape `(batch_size, sequence_length, d_model)`.
            mask: Optional attention mask of shape `(batch_size, 1, sequence_length, sequence_length)`
                or broadcastable shape. Values should be non-zero for positions
                that should be attended to, and zero for masked positions.

        Returns:
            Attention output tensor of shape `(batch_size, sequence_length, d_model)`.

        Raises:
            ValueError: If input dimensions do not match expected shape.
        """
        # Validate input dimensions to provide early, clear error messages.
        if x.dim() != 3:
            raise ValueError(
                f"Expected input tensor of dimension 3 (batch, seq, d_model), "
                f"got dimension {x.dim()} with shape {x.shape}. "
                f"Check dataset loader output format and model input expectations."
            )
        batch_size, seq_len, d_model_input = x.shape
        if d_model_input != self.d_model:
            raise ValueError(
                f"Input feature dimension ({d_model_input}) does not match "
                f"model dimension ({self.d_model}). Check ConfigLoader settings."
            )

        # Step 1: Project input to Query, Key, Value.
        # Linear projection transforms `(batch, seq, d_model)` -> `(batch, seq, d_model)`.
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: Split Q, K, V into `n_heads` heads.
        # Reshape from `(batch, seq, d_model)` to `(batch, n_heads, seq, d_head)`.
        # Note: `d_head = d_model // n_heads` by design.
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        # After transpose: shape `(batch_size, n_heads, seq_len, d_head)`.

        # Apply RoPE (Rotary Positional Embedding) to Q and K when configured.
        # This encodes relative position information into attention scores.
        if self.use_rope and self.rope is not None:
            Q, K = self.rope(Q, K, seq_len)

        # Step 3: Compute scaled dot-product attention scores.
        # `torch.matmul` broadcasts the batch and head dimensions automatically.
        # The scaling factor `sqrt(d_head)` prevents softmax saturation (numerical stability).
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Step 4: Apply mask (if provided) to prevent attention to padding tokens.
        # Mask values: non-zero for valid positions, zero for masked positions.
        # We use `masked_fill` to set masked positions to a very large negative number,
        # ensuring their softmax probability is approximately zero.
        if mask is not None:
            # Validate mask shape for helpful error messages.
            if mask.dim() not in (2, 3, 4):
                raise ValueError(
                    f"Attention mask must have 2, 3, or 4 dimensions, got {mask.dim()}. "
                    f"Check masking logic in dataset loader or model input pipeline."
                )
            # Ensure mask is broadcastable to `(batch, n_heads, seq, seq)`.
            # Standard mask shapes: `(batch, 1, 1, seq)` or `(batch, 1, seq, seq)`.
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Step 5: Apply softmax to obtain attention weights.
        # Softmax converts scores to probabilities that sum to 1 across the key dimension.
        attention_weights = F.softmax(scores, dim=-1)

        # Step 6: Apply dropout to attention weights.
        # Dropout prevents overfitting by randomly zeroing attention weights during training.
        # Note: Dropout is not applied during inference (`self.training == False`).
        attention_weights = self.dropout(attention_weights)

        # Step 7: Compute weighted sum of values.
        # `torch.matmul(attention_weights, V)` applies the attention weights to values,
        # producing a weighted representation for each query position.
        out = torch.matmul(attention_weights, V)
        # `out` shape: `(batch_size, n_heads, seq_len, d_head)`.

        # Step 8: Concatenate heads and apply output projection.
        # Transpose back to `(batch_size, seq_len, n_heads, d_head)`, then reshape
        # to `(batch_size, seq_len, n_heads * d_head) = (batch_size, seq_len, d_model)`.
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # Apply output projection to combine head outputs into final representation.
        output = self.W_o(out)
        # `output` shape: `(batch_size, seq_len, d_model)`.

        return output
