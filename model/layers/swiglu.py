"""
SwiGLU (Swish + Gated Linear Unit) feed-forward layer for XR Foundation Model (XRFM).

Purpose: Replace the standard GELU/ReLU feed-forward network with SwiGLU,
which introduces a gating mechanism: `SwiGLU(x) = (W_1(x) ⊗ σ(W_2(x))) · W_3`.
This provides more expressive power and has been shown to improve language
model performance compared to standard GELU/ReLU FFNs.

Conceptual references (NOT copied):
- Shazeer, N. (2020). GLU Variants Improve Transformer. (SwiGLU concept — reference only.)
- Llama 3 Technical Report (Meta AI, 2024) — SwiGLU usage in modern architectures.
- DeepSeek-V3 Technical Report (DeepSeek-AI, 2024) — SwiGLU in Mixture-of-Experts architectures.
- Mistral 7B Technical Report — SwiGLU as standard FFN in modern LLMs.

Implementation is original. The SwiGLU module uses three linear projections:
`W_1` (gate input), `W_2` (value input), and `W_3` (output projection), with
`SiLU` (Swish) activation applied to the gate before gating the value.

Design principles (Phase 4 architecture freeze):
- Configurable via `ConfigLoader.get_model_config()` (`d_model`, `d_ff`).
- Stable interface (`nn.Module` subclass with `forward(x)`).
- Original code (no copied implementations from Llama, Mistral, DeepSeek repositories).
- Numerical stability (standard initialization with Xavier/Kaiming; no special numerical risks).
- Performance (efficient linear projections; no unnecessary operations).
"""

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """Original SwiGLU feed-forward layer for XRFM.

    SwiGLU combines a Swish (SiLU) activation with a gated linear mechanism:
    `SwiGLU(x) = (W_1(x) ⊗ σ(W_2(x))) · W_3`
    where `σ` is the sigmoid function (used in `SiLU` activation) and `⊗` is element-wise multiplication.

    This architecture introduces a gating mechanism similar to LSTMs but simpler
    and fully parallelizable. It allows the network to learn selective information
    flow through the feed-forward layer.

    The module uses Xavier initialization for all linear projections, ensuring
    stable gradient flow during training.

    Attributes:
        W_1 (nn.Linear): First linear projection (gating input, dimension `d_model` -> `d_ff`).
        W_2 (nn.Linear): Second linear projection (value input, dimension `d_model` -> `d_ff`).
        W_3 (nn.Linear): Output linear projection (dimension `d_ff` -> `d_model`).
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        """Initialize SwiGLU layer.

        Args:
            d_model: Model hidden dimension (`ConfigLoader.get_model_config()["d_model"]`).
            d_ff: Feed-forward hidden dimension (`ConfigLoader.get_model_config()["d_ff"]`).
                Standard practice: `d_ff = 4 * d_model` (configurable).

        Raises:
            ValueError: If `d_model` or `d_ff` is non-positive.
        """
        super().__init__()
        if d_model <= 0:
            raise ValueError(
                f"d_model must be positive, got {d_model}. Check ConfigLoader settings."
            )
        if d_ff <= 0:
            raise ValueError(
                f"d_ff must be positive, got {d_ff}. Check ConfigLoader settings (d_ff)."
            )

        # Three linear projections: W_1 (gate), W_2 (value), W_3 (output).
        # W_3 projects from expanded dimension `d_ff` back to `d_model`.
        self.W_1 = nn.Linear(d_model, d_ff, bias=True)
        self.W_2 = nn.Linear(d_model, d_ff, bias=True)
        self.W_3 = nn.Linear(d_ff, d_model, bias=True)

        # Initialize weights with Xavier uniform initialization for stability.
        # This prevents gradient explosion or vanishing at the start of training,
        # which is critical for deep networks (32+ layers in large models like XRFM-7B).
        for linear_layer in (self.W_1, self.W_2, self.W_3):
            nn.init.xavier_uniform_(linear_layer.weight)
            if linear_layer.bias is not None:
                nn.init.zeros_(linear_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU feed-forward transformation.

        The computation follows:
        `gate = SiLU(W_1(x))`
        `value = W_2(x)`
        `output = W_3(gate ⊗ value)`
        where `⊗` is element-wise multiplication and `SiLU(x) = x · σ(x)`.

        Args:
            x: Input tensor of shape `(batch_size, sequence_length, d_model)`.

        Returns:
            Output tensor of shape `(batch_size, sequence_length, d_model)`.

        Raises:
            ValueError: If input feature dimension does not match `d_model`.
        """
        # Validate input dimension for clear error diagnostics.
        if x.shape[-1] != self.W_1.in_features:
            # Note: W_1.in_features equals d_model (passed to constructor).
            # We check against the actual linear layer's input feature count.
            expected = self.W_1.in_features
            actual = x.shape[-1]
            raise ValueError(
                f"SwiGLU input feature dimension ({actual}) does not match "
                f"expected dimension ({expected}). Check ConfigLoader (d_model) "
                f"and model architecture consistency."
            )

        # Apply linear projections.
        # `W_1(x)` and `W_2(x)` expand from `d_model` to `d_ff`.
        gate_input = self.W_1(x)  # (batch, seq, d_ff)
        value_input = self.W_2(x)  # (batch, seq, d_ff)

        # Apply SiLU (Swish) activation to gate input.
        # `SiLU(x) = x · sigmoid(x)` — smooth activation that outperforms ReLU/GELU.
        gate = torch.nn.functional.silu(gate_input)

        # Element-wise gating: multiply activated gate with value.
        gated = gate * value_input

        # Project back to model dimension `d_model`.
        output = self.W_3(gated)

        return output
