"""
Root Mean Square Normalization (RMSNorm) for XR Foundation Model (XRFM).

Purpose: Normalize layer inputs to stabilize training in deep networks.
RMSNorm removes the mean-centering step of standard LayerNorm, reducing
computational overhead while maintaining similar performance.

Conceptual references (NOT copied):
- Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization.
  *NeurIPS* (reference only; no code copied).
- Llama 3 Technical Report (Meta AI, 2024) — RMSNorm usage in modern LLM architecture.
- DeepSeek-V3 Technical Report (DeepSeek-AI, 2024) — RMSNorm in very large-scale models.

Implementation is original. The module includes configurable epsilon (`eps`)
and learnable scale parameter (`weight`).

Design principles (from Phase 4 architecture freeze):
- Configurable (`use_rmsnorm` in `config/config.yaml`).
- Stable interface (`nn.Module` subclass with `forward(x)`).
- Original code (no copied implementation from Llama or other repositories).
- Numerical stability (epsilon prevents division by zero for very small inputs).
- Pre-norm architecture support (`TransformerBlock` applies normalization before sub-layer).
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Original Root Mean Square Normalization implementation.

    RMSNorm normalizes by dividing by the root mean square of the input,
    without subtracting the mean. This reduces computation compared to
    standard LayerNorm (`(x - mean) / sqrt(var + eps)`) while achieving
    similar stabilization effects in deep networks.

    The module uses a learnable scale parameter `weight` (gamma) but no
    learnable bias (beta), matching modern practices (Llama 3, Qwen, DeepSeek).

    Attributes:
        weight (nn.Parameter): Learnable scale parameter of shape `(dim,)`.
        eps (float): Small constant for numerical stability (default: 1e-6).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize RMSNorm layer.

        Args:
            dim: Normalization dimension (`d_model` from config).
            eps: Small constant added to variance for numerical stability.
                Default: 1e-6 (standard in modern LLM architectures).

        Raises:
            ValueError: If `dim` or `eps` is invalid (non-positive `dim`, non-positive `eps`).
        """
        super().__init__()
        if dim <= 0:
            raise ValueError(
                f"Normalization dimension must be positive, got {dim}. Check ConfigLoader model settings (d_model)."
            )
        if eps <= 0:
            raise ValueError(f"Epsilon must be positive, got {eps}. Check RMSNorm initialization.")
        self.dim = dim
        self.eps = eps
        # Learnable scale parameter (gamma). No bias parameter (beta) —
        # this matches modern LLM architectures (Llama 3, Mistral, DeepSeek).
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization to input tensor.

        Mathematical formulation:
            `rms = sqrt(mean(x^2) + eps)`
            `output = weight * x / rms`

        The mean is not subtracted (`x - mean` is omitted compared to LayerNorm),
        reducing computation by avoiding mean computation. The `eps` ensures
        numerical stability when all elements are zero or very small.

        Args:
            x: Input tensor of arbitrary shape; normalization applied over
                the last dimension (`dim`).

        Returns:
            Normalized tensor with same shape as input.
        """
        # Compute root mean square over the last dimension.
        # `mean(x^2, dim=-1, keepdim=True)` computes the mean of squares.
        # Adding `eps` prevents division by zero.
        # `torch.sqrt(...)` computes the root mean square.
        # The division `x / rms` normalizes the input.
        # The learnable `weight` scales the output (learned during training).
        return self.weight * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
