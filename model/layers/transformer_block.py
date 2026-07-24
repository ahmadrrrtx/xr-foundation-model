"""
Transformer block for XR Foundation Model (XRFM).

Purpose: Combine multi-head attention, feed-forward network (SwiGLU),
residual connections, and pre-normalization into a single modular layer.
This block is stacked multiple times (`n_layers`) to build the full model.

Conceptual references (NOT copied):
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need.
  (Transformer block architecture — standard design; implementation original.)
- Touvron, H., et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models.
  (Pre-normalization design — reference only; no copied code.)
- DeepSeek-AI (2024). DeepSeek-V3 Technical Report. (Block architecture — conceptual reference only.)

Implementation is original. The block follows modern best practices:
- Pre-normalization (`norm` applied before attention and before FFN).
- Residual connections (`x + sublayer(x)`) to prevent vanishing gradients.
- Dropout applied after attention weights and after FFN output.
- Configurable dropout rate (`model.dropout` from ConfigLoader).

Design principles (Phase 4 architecture freeze):
- Stable interface (`TransformerBlock.forward(x, mask)`).
- Configurable (`d_model`, `n_heads`, `d_ff`, `dropout`, `use_rope`, `use_rmsnorm`, `use_swiglu`).
- No hard-coded hyperparameters (all from ConfigLoader).
- Clean separation: attention module, normalization module, FFN module are independent.
- Original code (no copied implementations from Llama, Mistral, or other repositories).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from model.attention.multi_head import MultiHeadAttention
from model.layers.rmsnorm import RMSNorm
from model.layers.swiglu import SwiGLU
from model.attention.rope import RoPE


class TransformerBlock(nn.Module):
    """Original pre-normalization transformer block for XRFM.

    The block follows the modern architecture used in Llama 3, Mistral,
    Qwen, and DeepSeek: normalization before each sub-layer, residual
    connections around sub-layers, dropout applied after sub-layer outputs.

    Mathematical structure (for a single block):
    `h' = Norm(x)`
    `h = Attention(h')`
    `x = x + Dropout(h)`  (residual + dropout)
    `h' = Norm(x)`
    `h = SwiGLU(h')`
    `x = x + Dropout(h)`  (residual + dropout)
    `output = Norm(x)` (optional final normalization; handled by model-level final norm in GPTModel)

    Note: RoPE (Rotary Positional Embedding) is applied to query and key
    tensors inside the attention mechanism (see `MultiHeadAttention.forward()`
    which uses `RoPE.forward()`). The `TransformerBlock` does not apply
    RoPE directly; it relies on the attention module for position encoding.

    Attributes:
        norm1 (RMSNorm): Pre-norm before attention.
        norm2 (RMSNorm): Pre-norm before feed-forward.
        attn (MultiHeadAttention): Attention mechanism.
        ffn (SwiGLU): Feed-forward network.
        dropout (nn.Dropout): Dropout layer applied after attention and FFN outputs.
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
        """Initialize transformer block.

        Args:
            d_model: Hidden dimension (`ConfigLoader.get_model_config()["d_model"]`).
            n_heads: Number of attention heads (`ConfigLoader.get_model_config()["n_heads"]`).
            d_ff: Feed-forward hidden dimension (`ConfigLoader.get_model_config()["d_ff"]`).
            dropout: Dropout probability (`ConfigLoader.get_model_config()["dropout"]`).
            use_rmsnorm: Whether to use RMSNorm (True, standard) or LayerNorm (False, optional for comparison/research).

        Raises:
            ValueError: If any dimension parameter is non-positive or if `dropout` is outside [0, 1).
        """
        super(TransformerBlock, self).__init__()
        if d_model <= 0:
            raise ValueError(
                f"d_model must be positive, got {d_model}. "
                f"Check ConfigLoader settings."
            )
        if n_heads <= 0:
            raise ValueError(
                f"n_heads must be positive, got {n_heads}. Check ConfigLoader settings."
            )
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}). "
                f"This ensures equal head dimensions. Check ConfigLoader."
            )
        if d_ff <= 0:
            raise ValueError(
                f"d_ff must be positive, got {d_ff}. Check ConfigLoader settings."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1), got {dropout}. Check ConfigLoader settings."
            )

        if not isinstance(use_rmsnorm, bool):
            raise ValueError(f"use_rmsnorm must be bool, got {type(use_rmsnorm).__name__}.")
        if not isinstance(use_rope, bool):
            raise ValueError(f"use_rope must be bool, got {type(use_rope).__name__}.")

        # Pre-normalization layers: applied before sub-layer (attention and FFN).
        # RMSNorm is preferred (standard in modern LLMs: Llama, Mistral, DeepSeek).
        # Standard LayerNorm can be used as an optional fallback.
        self.norm1 = RMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)
        self.norm2 = RMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)

        # Attention mechanism (manual multi-head for full control and extensibility).
        # RoPE is configurable (`use_rope`) for future comparison with ALiBi/learned embeddings.
        self.attn = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout, use_rope=use_rope
        )

        # Feed-forward network (SwiGLU for modern performance; configurable).
        # The FFN dimension `d_ff` is typically `4 * d_model` in standard architectures.
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

        # Dropout applied after attention and after FFN outputs.
        # Dropout prevents overfitting by randomly zeroing activations during training.
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply transformer block with pre-normalization and residual connections.

        The computation follows the modern pre-norm architecture:
        1. `h = Attention(Norm(x))`
        2. `x = x + Dropout(h)` (residual)
        3. `h = SwiGLU(Norm(x))`
        4. `x = x + Dropout(h)` (residual)
        5. Return `x` (final normalization is handled at model level by `GPTModel`, not in each block).

        Note: RoPE (Rotary Positional Embedding) is applied inside the attention
        mechanism (`MultiHeadAttention.forward()` uses `RoPE.forward()`), not
        at the block level. This keeps the block design clean and focuses on
        the sub-layer structure.

        Args:
            x: Input tensor of shape `(batch_size, sequence_length, d_model)`.
            mask: Optional attention mask. See `MultiHeadAttention.forward()` for details.

        Returns:
            Output tensor of shape `(batch_size, sequence_length, d_model)`.

        Raises:
            ValueError: If input feature dimension does not match `d_model`.
        """
        # Validate input dimension for early error detection.
        if x.dim() != 3:
            raise ValueError(
                f"TransformerBlock expects 3D input (batch, seq, d_model), "
                f"got {x.dim()}D with shape {x.shape}. Check dataset loader output."
            )
        batch_size, seq_len, d_model_input = x.shape
        if d_model_input != self.attn.W_q.in_features:
            # W_q.in_features equals d_model (passed to MultiHeadAttention constructor).
            raise ValueError(
                f"Input feature dimension ({d_model_input}) does not match "
                f"model dimension ({self.attn.W_q.in_features}). Check config consistency."
            )

        # Step 1: Pre-normalization + Attention + Residual + Dropout.
        # `norm1(x)` normalizes the input before passing it to attention.
        # `self.attn(...)` computes multi-head attention (with RoPE applied to Q/K inside).
        # `self.dropout(...)` applies dropout to the attention output.
        # `x + ...` is the residual connection (preserves gradient flow).
        # Note: The addition `norm1(x) + x` is NOT performed; the residual is
        # `x + Dropout(Attention(Norm(x)))`, which is the standard pre-norm design.
        h = self.attn(self.norm1(x), mask=mask)
        x = x + self.dropout(h)

        # Step 2: Pre-normalization + SwiGLU FFN + Residual + Dropout.
        # `norm2(x)` normalizes the updated representation.
        # `self.ffn(...)` applies SwiGLU feed-forward transformation.
        # `self.dropout(...)` applies dropout to FFN output.
        # `x + ...` is the second residual connection.
        h = self.ffn(self.norm2(x))
        x = x + self.dropout(h)

        # Return the updated representation (no final normalization here;
        # `GPTModel` handles final normalization if needed, but modern architectures
        # typically apply normalization only before sub-layers, not after the last block,
        # unless a final output normalization is explicitly configured).
        return x
