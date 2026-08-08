"""
Learning rate scheduler module for XR Foundation Model (XRFM) training engine.

Purpose: Configure and apply `cosine` decay schedule with `linear` warmup.
Future extensions (`constant`, `exponential`, `cyclic`) supported via interface.

Conceptual references (NOT copied):
- Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models.
- Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models (`Chinchilla`).
- PyTorch documentation (`torch.optim.lr_scheduler.CosineAnnealingLR`).

Implementation is original. `torch.optim.lr_scheduler` is used as the underlying
library dependency (standard; no source code copied).

Design principles (Phase 5 architecture freeze):
- Config-driven (`ConfigLoader.get_training_config()` provides `learning_rate`,
  `warmup_steps`, `max_steps`).
- Stable interface (`SchedulerLoader` / `get_scheduler`).
- Numerical stability (`cosine` decay prevents sudden `lr` drops; `warmup` prevents divergence).
- Security: No hidden dependencies; `torch` standard library only.
"""

import math
from typing import Any

import torch.optim as optim


class SchedulerLoader:
    """Production-quality learning rate scheduler loader for XRFM.

    Design note: This loader creates a `cosine` annealing schedule with `linear`
    warmup (`training/loop.py` uses `scheduler.step()` after each optimizer step).
    The schedule ensures smooth `lr` decay (`cosine`) and safe startup (`warmup`),
    matching modern LLM training practices (`Llama 3`, `DeepSeek-V3`, `Chinchilla`).

    Attributes:
        optimizer: `torch.optim.Optimizer` instance (`AdamW` from `OptimizerLoader`).
        warmup_steps: Linear warmup duration (steps).
        max_steps: Total training steps (`ConfigLoader.get_training_config()["max_steps"]`).
        base_lr: Initial learning rate.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        base_lr: float = 0.001,
        warmup_steps: int = 1000,
        max_steps: int = 50000,
    ) -> None:
        """Initialize the scheduler loader.

        Args:
            optimizer: `AdamW` optimizer instance (`OptimizerLoader`).
            base_lr: Base learning rate (`ConfigLoader` setting).
            warmup_steps: Linear warmup steps (`ConfigLoader` setting; default `1000`).
            max_steps: Maximum training steps (`ConfigLoader` setting; default `50000`).

        Raises:
            ValueError: If `warmup_steps` or `max_steps` is non-positive, or if
                `warmup_steps >= max_steps` (invalid schedule configuration).
        """
        if warmup_steps <= 0:
            raise ValueError(
                f"warmup_steps must be positive, got {warmup_steps}. Check ConfigLoader training settings."
            )
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}. Check ConfigLoader training settings.")
        if warmup_steps >= max_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be less than max_steps ({max_steps}). "
                f"Check ConfigLoader settings (training.max_steps > training.warmup_steps)."
            )
        if base_lr <= 0:
            raise ValueError(f"base_lr must be positive, got {base_lr}. Check ConfigLoader settings.")

        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.current_step = 0

    def get_lr(self, step: int | None = None) -> float:
        """Compute the current learning rate (`cosine` + `warmup`).

        Mathematical formulation:
        - Warmup phase (`step <= warmup_steps`):
          `lr = base_lr * step / warmup_steps`
        - Cosine decay phase (`step > warmup_steps`):
          `lr = base_lr * 0.5 * (1 + cos(pi * (step - warmup_steps) / (max_steps - warmup_steps)))`

        Args:
            step: Current training step (`ConfigLoader` provides `max_steps`);
                if `None`, uses `self.current_step`.

        Returns:
            Learning rate (`float`).

        Design notes:
        - `cosine` decay ensures smooth `lr` reduction (`Kaplan 2020`; `Chinchilla`).
        - `warmup` prevents early divergence by gradually increasing `lr` (`Meta AI 2024`).
        - The formula is standard (`Llama 3`, `DeepSeek-V3`, `Qwen2.5` use similar schedules).
        """
        step = step if step is not None else self.current_step
        if step < self.warmup_steps:
            # Linear warmup: `lr` increases linearly from `0` to `base_lr`.
            return self.base_lr * step / self.warmup_steps
        # Cosine annealing: `lr` decays smoothly from `base_lr` to `0`.
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self) -> None:
        """Update optimizer learning rate (`scheduler.step()` called after `optimizer.step()`).

        Note: This updates the `learning_rate` for all parameter groups in the optimizer.
        If `grouped` parameters (e.g., different `lr` for embeddings) are used (`OPTIONAL`),
        this method updates each group's `lr` based on `base_lr` and the schedule.
        """
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    # ------------------------------------------------------------------
    # State serialization (forensic-audit fix, F-24): without these,
    # CheckpointLoader silently skips scheduler state and resume restarts
    # the LR schedule from step 0.
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state for checkpointing."""
        return {
            "current_step": self.current_step,
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scheduler state from a checkpoint."""
        self.current_step = int(state_dict.get("current_step", 0))
        if "base_lr" in state_dict:
            self.base_lr = float(state_dict["base_lr"])
        if "warmup_steps" in state_dict:
            self.warmup_steps = int(state_dict["warmup_steps"])
        if "max_steps" in state_dict:
            self.max_steps = int(state_dict["max_steps"])
        # Re-apply the restored schedule to the optimizer's param groups.
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
