"""
Optimizer module for XR Foundation Model (XRFM) training engine.

Purpose: Configure and initialize `AdamW` optimizer for model parameters,
with decoupled weight decay and numerical stability checks.

Conceptual references (NOT copied):
- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization.
- Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization (`AdamW`).

Implementation is original. `torch.optim.AdamW` is used as the underlying
optimizer (standard library dependency; no source code copied). This module
provides a stable, config-driven interface for optimizer initialization and
parameter grouping.

Design principles (Phase 5 architecture freeze):
- Config-driven (`ConfigLoader.get_training_config()` provides `learning_rate`,
  `weight_decay`, `gradient_clip`).
- Stable interface (`OptimizerConfig` / `OptimizerLoader`).
- Numerical stability (`AdamW` uses `eps = 1e-8`; `gradient_clip` prevents explosion).
- Security: No hidden dependencies; only `torch` + `typing`.
- Original attribution in module docstring.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim


def build_decay_param_groups(
    model_or_params: nn.Module | Any,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Partition parameters into decayed (>=2D matrices) and non-decayed (<2D norms/biases).

    Follows standard transformer optimization practice (Karpathy nanoGPT, Llama 3,
    Qwen): 1-D scale parameters (``RMSNorm``, ``LayerNorm``) and linear biases
    receive ``weight_decay = 0.0`` while 2-D+ projection/embedding matrices
    receive ``weight_decay``. Deduplicates shared/weight-tied parameters cleanly.
    """
    if isinstance(model_or_params, nn.Module):
        iterable = (p for p in model_or_params.parameters() if p.requires_grad)
    else:
        params_list = list(model_or_params)
        if params_list and isinstance(params_list[0], dict):
            return params_list
        iterable = (p for p in params_list if getattr(p, "requires_grad", True))

    seen: set[int] = set()
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for p in iterable:
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    groups: list[dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups if groups else [{"params": [], "weight_decay": weight_decay}]


class OptimizerLoader:
    """Production-quality optimizer loader for XRFM training pipeline.

    Design note: This loader creates an `AdamW` optimizer with configurable
    hyperparameters (`learning_rate`, `weight_decay`, `betas`, `eps`). It supports
    parameter grouping (e.g., different `weight_decay` for embeddings vs `FFN`)
    for future extensions (`RESEARCH-ONLY` comparison studies).

    Attributes:
        config: Training configuration (`ConfigLoader.get_training_config()`).
    """

    def __init__(
        self,
        model_params,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        """Initialize the optimizer loader.

        Args:
            model_params: Model parameters (`model.parameters()` or grouped parameters).
            learning_rate: Base learning rate (`ConfigLoader.get_training_config()["learning_rate"]`).
            weight_decay: Weight decay coefficient (`ConfigLoader.get_training_config()["weight_decay"]`).
            betas: `AdamW` momentum coefficients (`(beta1, beta2)`).
            eps: `AdamW` epsilon (`1e-8` standard) to prevent division by zero.

        Raises:
            ValueError: If any hyperparameter is invalid (`learning_rate` non-positive,
                `weight_decay` negative, `betas` out of range, `eps` non-positive).
        """
        if learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {learning_rate}. Check ConfigLoader training settings."
            )
        if weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {weight_decay}. Check ConfigLoader training settings."
            )
        if len(betas) != 2 or not (0.0 <= betas[0] < 1.0) or not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"betas must be a tuple of two values in [0, 1), got {betas}.")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}. Check AdamW initialization.")

        # Initialize AdamW optimizer (`torch.optim.AdamW`) with decoupled
        # weight decay excluded from 1-D norms and biases.
        grouped_params = build_decay_param_groups(model_params, weight_decay=weight_decay)
        self.optimizer = optim.AdamW(
            grouped_params,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps

    def step(self) -> None:
        """Perform optimizer step (update parameters).

        Note: `gradient_clip` should be applied before calling `step()`.
        See `GradientClipper.clip_gradients(model)` (reserved for Phase 5 design).
        """
        self.optimizer.step()

    def zero_grad(self) -> None:
        """Clear parameter gradients (`optimizer.zero_grad()`)."""
        self.optimizer.zero_grad()

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer state (`optimizer.state_dict()`) for checkpointing."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state (`optimizer.load_state_dict()`) from checkpoint."""
        self.optimizer.load_state_dict(state_dict)
