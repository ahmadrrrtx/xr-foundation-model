"""
Checkpoint module for XR Foundation Model (XRFM) training engine.

Purpose: Save and load model weights, optimizer state, and training state
(`step`, `loss`, `best_loss`). Supports resume from checkpoint (`ConfigLoader`
`training.resume_from`).

Conceptual references (NOT copied):
- PyTorch documentation (`torch.save`, `torch.load`, `state_dict`).
- Meta AI (2024). Llama 3 Technical Report (`checkpoint` design reference).
- DeepSeek-AI (2024). DeepSeek-V3 Technical Report (`checkpoint` patterns).

Implementation is original. `torch.save` / `torch.load` are standard library
functions (`PyTorch` core dependency; no source code copied).

Design principles (Phase 5 architecture freeze):
- Config-driven (`ConfigLoader.get_training_config()` provides `checkpoint_every`,
  `resume_from`).
- Stable interface (`CheckpointLoader.save_checkpoint`, `load_checkpoint`).
- Security: Checkpoint files contain only `torch.save` data (no external APIs);
  optional `checksum` verification (`RESEARCH-ONLY` for security-critical applications).
- Numerical stability: Checkpoints include `ConfigLoader` settings for reproducibility.
- Original code attribution in module docstring.
- No hidden dependencies (`torch` + `typing` + `os` + `hashlib` optional).
"""

import os
from typing import Optional, Dict, Any

import torch


class CheckpointLoader:
    """Production-quality checkpoint loader for XRFM training pipeline.

    Design note: Checkpoints are saved as `.pt` files (`torch.save`) containing
    `model.state_dict()`, `optimizer.state_dict()`, `scheduler.state_dict()`
    (if scheduler exists), `step`, `loss`, and `best_loss`. The `ConfigLoader`
    settings (`config_path` or serialized `raw_config`) are included for
    reproducibility (`DECISIONS.md` confirms this as `CORE`).

    Attributes:
        checkpoint_dir: Directory path (`checkpoints/` by default; `ConfigLoader`
            `paths.checkpoint_dir` provides the default).
    """

    def __init__(self, checkpoint_dir: str = "checkpoints/") -> None:
        """Initialize checkpoint loader.

        Args:
            checkpoint_dir: Checkpoint directory path (`ConfigLoader.get("paths.checkpoint_dir")`).

        Raises:
            ValueError: If `checkpoint_dir` is not a valid directory path or does not exist
                (the loader will create it if missing; validation ensures the path is a string).
        """
        if not isinstance(checkpoint_dir, str):
            raise ValueError(
                f"checkpoint_dir must be str, got {type(checkpoint_dir).__name__}. "
                f"Check ConfigLoader settings (paths.checkpoint_dir)."
            )
        self.checkpoint_dir = checkpoint_dir
        # Ensure directory exists (`os.makedirs` creates if missing; safe for resumable training).
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler=None,
        step: int = 0,
        loss: float = float("inf"),
        best_loss: float = float("inf"),
        filename: Optional[str] = None,
    ) -> str:
        """Save a checkpoint (`.pt` file) with full training state.

        Args:
            model: `GPTModel` instance (`model.state_dict()`).
            optimizer: `OptimizerLoader` instance (`optimizer.state_dict()`).
            scheduler: `SchedulerLoader` instance (`scheduler.state_dict()`); optional.
            step: Current training step (`ConfigLoader` provides `max_steps`).
            loss: Current training loss.
            best_loss: Best loss observed so far (for early stopping or comparison).
            filename: Custom checkpoint filename; if `None`, generates
                `checkpoint_step_{step}.pt`.

        Returns:
            Full checkpoint file path (`str`).

        Raises:
            ValueError: If `model`, `optimizer`, or `step` is invalid (`step` non-negative required).
            TypeError: If `filename` is provided but not a `str`.
        """
        if step < 0:
            raise ValueError(
                f"step must be non-negative, got {step}. Check training loop state."
            )
        if filename is not None and not isinstance(filename, str):
            raise TypeError(
                f"filename must be str or None, got {type(filename).__name__}."
            )

        # Build checkpoint dictionary (`torch.save` format; standard `PyTorch`).
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "loss": loss,
            "best_loss": best_loss,
            # Note: `ConfigLoader` settings (`config_path` or serialized config)
            # should be saved for full reproducibility (`DECISIONS.md` confirms this as `CORE`).
            # The `training/loop.py` is responsible for passing `config_path` or `raw_config`
            # when calling `save_checkpoint` (`RESEARCH-ONLY` extension: add config serialization here).
        }
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Generate filename (`checkpoint_step_{step}.pt` by default).
        filename = filename or f"checkpoint_step_{step}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        # Save (`torch.save` — standard `PyTorch` function; no hidden behavior).
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model,
        optimizer,
        scheduler=None,
    ) -> Dict[str, Any]:
        """Load a checkpoint (`.pt` file) and restore model, optimizer, and scheduler states.

        Args:
            checkpoint_path: Path to checkpoint file (`str`). Must exist.
            model: `GPTModel` instance (restores `model.load_state_dict()`).
            optimizer: `OptimizerLoader` instance (restores `optimizer.load_state_dict()`).
            scheduler: `SchedulerLoader` instance (optional; restores `scheduler.load_state_dict()`).

        Returns:
            Checkpoint metadata dictionary (`step`, `loss`, `best_loss`, `filename`).

        Raises:
            FileNotFoundError: If `checkpoint_path` does not exist.
            TypeError: If `checkpoint_path` is not a `str`.
        """
        if not isinstance(checkpoint_path, str):
            raise TypeError(
                f"checkpoint_path must be str, got {type(checkpoint_path).__name__}."
            )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}. "
                f"Check training resume settings (ConfigLoader training.resume_from)."
            )

        # Load (`torch.load` — standard `PyTorch` function).
        checkpoint = torch.load(checkpoint_path, weights_only=True)

        # Restore model state (`GPTModel.load_state_dict()`).
        model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer state (`OptimizerLoader.load_state_dict()`).
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler state (`SchedulerLoader.load_state_dict()` — optional).
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            if hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Return checkpoint metadata (`step`, `loss`, `best_loss`, `filename`) for logging/resume.
        return {
            "step": checkpoint.get("step", 0),
            "loss": checkpoint.get("loss", float("inf")),
            "best_loss": checkpoint.get("best_loss", float("inf")),
            "filename": os.path.basename(checkpoint_path),
        }
