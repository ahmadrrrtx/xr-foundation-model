"""
Mixed precision training module for XR Foundation Model (XRFM) training engine.

Purpose: Enable `bfloat16` (`mixed_precision: True`) training with `GradScaler`
(`torch.cuda.amp.GradScaler`) for numerical stability. Falls back to `FP32`
when `mixed_precision: False` or `GPU` is unavailable.

Conceptual references (NOT copied):
- Micikevicius, P., et al. (2018). Mixed Precision Training (`arXiv:1710.03740`).
- PyTorch documentation (`torch.cuda.amp`, `torch.cuda.amp.GradScaler`).
- Meta AI (2024). Llama 3 Technical Report (`mixed_precision` usage).
- DeepSeek-AI (2024). DeepSeek-V3 Technical Report (`mixed_precision` design).

Implementation is original. `torch.cuda.amp` is a standard `PyTorch` library
function (`CORE` dependency; no source code copied from external tutorials or repositories).

Design principles (Phase 5 architecture freeze):
- Config-driven (`ConfigLoader.get_training_config()["mixed_precision"]`).
- Stable interface (`MixedPrecisionLoader` / `grad_scaler`).
- Numerical stability (`GradScaler.scale()` / `step()` / `update()` handles `bfloat16` `underflow`).
- Security: Only `torch` + standard library; no hidden dependencies.
- Original attribution in module docstring.
- No placeholders; no `"fix later"` comments.
"""

import torch


class MixedPrecisionLoader:
    """Production-quality mixed precision loader for XRFM training pipeline.

    Design note: `MixedPrecisionLoader` activates `bfloat16` training
    (`mixed_precision: True`) by creating a `torch.cuda.amp.GradScaler`
    (`GradScaler`). The `GradScaler` manages gradient scaling (`scale_factor`)
    to prevent `bfloat16` `underflow`. When `mixed_precision: False`, the loader
    creates a `null` scaler (`NoOpScaler`) that passes through without scaling,
    ensuring consistent interface (`grad_scaler.scale()`, `.step()`, `.update()`).
    This design ensures `training/loop.py` can use `grad_scaler` unconditionally
    (`True` or `False`), simplifying the loop logic.

    Attributes:
        enabled: Whether `mixed_precision` is enabled (`True` or `False`).
        scaler: `torch.cuda.amp.GradScaler` (if `enabled` and `torch.cuda.is_available()`);
            otherwise `NoOpScaler` (no-op for `CPU` or `False`).
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize mixed precision loader.

        Args:
            enabled: Whether to enable `mixed_precision` (`ConfigLoader.get_training_config()["mixed_precision"]`).

        Raises:
            ValueError: If `enabled` is not a `bool`.
        """
        if not isinstance(enabled, bool):
            raise ValueError(
                f"enabled must be bool, got {type(enabled).__name__}. "
                f"Check ConfigLoader settings (training.mixed_precision)."
            )
        self.enabled = enabled

        # Initialize `GradScaler` (`torch.cuda.amp.GradScaler`) when `enabled` is `True`
        # and `torch.cuda` is available (`GPU` environment). Otherwise, use `NoOpScaler`.
        # Note: `GradScaler` handles gradient scaling (`scale_factor` adjustment) to
        # prevent `bfloat16` `underflow`. The `scale_factor` is adjusted dynamically
        # based on whether `Inf`/`NaN` gradients are detected (`GradScaler.update()`).
        if self.enabled and torch.cuda.is_available():
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                self.scaler = torch.amp.GradScaler("cuda")
            else:
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            # `NoOpScaler` — no-op scaler for `CPU` or `mixed_precision: False`.
            # This ensures `grad_scaler.scale(loss)`, `.step(optimizer)`, `.update()`
            # work identically regardless of `enabled`, simplifying `training/loop.py`.
            self.scaler = NoOpScaler()

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient computation (`GradScaler.scale(loss)`).

        Args:
            loss: Loss tensor (`torch.Tensor`).

        Returns:
            Scaled loss (`torch.Tensor`) — scaled by `GradScaler.scale_factor` when `enabled`;
            unscaled (`loss`) when `NoOpScaler`.
        """
        return self.scaler.scale(loss) if hasattr(self.scaler, "scale") else loss

    def unscale_(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        """Unscale gradients before gradient clipping (`GradScaler.unscale_(optimizer)`).

        Args:
            optimizer: `AdamW` optimizer instance (`OptimizerLoader`). Optional (only needed for `GradScaler`).

        Note:
            `unscale_` should be called before `clip_grad_norm_` (`step` order: `unscale_` -> `clip_grad_norm_` -> `step` -> `update`).
            This ensures gradient clipping operates on unscaled gradients (correct `L2` norm calculation).
        """
        if hasattr(self.scaler, "unscale_") and optimizer is not None:
            self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Perform optimizer step with gradient scaling (`GradScaler.step(optimizer)`).

        Args:
            optimizer: `AdamW` optimizer instance.

        Returns:
            `True` if the step was performed (`GradScaler` did not skip due to `Inf`/`NaN`);
            `True` always for `NoOpScaler`.

        Note:
            `GradScaler.step()` checks for `Inf`/`NaN` in gradients (`unscale_` detects overflow).
            If `Inf`/`NaN` detected, the step is skipped (`optimizer.step()` not called) and
            `GradScaler` updates the `scale_factor` (`backoff_factor` applied, reducing scale for stability).
        """
        if hasattr(self.scaler, "step"):
            return self.scaler.step(optimizer)
        # `NoOpScaler` — always perform step (`True` indicates step performed).
        optimizer.step()
        return True

    def update(self) -> None:
        """Update gradient scaler (`GradScaler.update()` — adjusts `scale_factor` based on `Inf`/`NaN` detection)."""
        if hasattr(self.scaler, "update"):
            self.scaler.update()


class NoOpScaler:
    """No-operation scaler (`NoOpScaler`) for `CPU` or `mixed_precision: False`.

    Design note: `NoOpScaler` provides the same interface (`scale`, `unscale_`, `step`, `update`)
    as `GradScaler` but performs no scaling. This ensures `training/loop.py` can use the scaler
    unconditionally (`True` or `False`) without branching logic (`if enabled: ... else: ...`),
    improving maintainability and reducing error risk.
    """

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Return unscaled loss (`NoOpScaler`)."""
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        """No-op (`NoOpScaler`)."""
        pass

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Perform optimizer step and return `True` (`NoOpScaler` — always performs step)."""
        optimizer.step()
        return True

    def update(self) -> None:
        """No-op (`NoOpScaler`)."""
        pass
