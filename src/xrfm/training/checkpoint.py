"""Atomic, weights-only-safe checkpoints with complete resume state."""

from __future__ import annotations

import os
import random
import tempfile
from typing import Any

import numpy as np
import torch

_FORMAT_VERSION = 2
_NDARRAY_MARKER = "__xrfm_numpy_ndarray__"
_TUPLE_MARKER = "__xrfm_tuple__"


def _safe_nested(value: Any) -> Any:
    """Convert RNG containers to primitives accepted by weights_only load."""
    if isinstance(value, np.ndarray):
        return {
            _NDARRAY_MARKER: True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, tuple):
        return {_TUPLE_MARKER: True, "items": [_safe_nested(item) for item in value]}
    if isinstance(value, list):
        return [_safe_nested(item) for item in value]
    if isinstance(value, dict):
        return {key: _safe_nested(item) for key, item in value.items()}
    return value


def _restore_nested(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_nested(item) for item in value]
    if isinstance(value, dict):
        if value.get(_NDARRAY_MARKER):
            array = np.asarray(value["data"], dtype=np.dtype(value["dtype"]))
            return array.reshape(tuple(value["shape"]))
        if value.get(_TUPLE_MARKER):
            return tuple(_restore_nested(item) for item in value["items"])
        return {key: _restore_nested(item) for key, item in value.items()}
    return value


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": _safe_nested(random.getstate()),
        "numpy": _safe_nested(np.random.get_state()),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(_restore_nested(state["python"]))
    if "numpy" in state:
        np.random.set_state(_restore_nested(state["numpy"]))
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    cuda_states = state.get("cuda", [])
    if cuda_states and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                f"checkpoint has {len(cuda_states)} CUDA RNG states but this host has "
                f"{torch.cuda.device_count()} devices; exact resume requires the same device count"
            )
        torch.cuda.set_rng_state_all([state_tensor.cpu() for state_tensor in cuda_states])


class CheckpointLoader:
    """Save and restore model, optimization, scaler, RNG, and run metadata."""

    def __init__(self, checkpoint_dir: str = "checkpoints/") -> None:
        if not isinstance(checkpoint_dir, str):
            raise ValueError(f"checkpoint_dir must be str, got {type(checkpoint_dir).__name__}")
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _checkpoint_path(self, filename: str) -> str:
        if not isinstance(filename, str) or not filename:
            raise ValueError("filename must be a non-empty string")
        base = os.path.abspath(self.checkpoint_dir)
        path = os.path.abspath(os.path.join(base, filename))
        if os.path.commonpath((base, path)) != base:
            raise ValueError("checkpoint filename must remain inside checkpoint_dir")
        return path

    def save_checkpoint(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any = None,
        step: int = 0,
        loss: float = float("inf"),
        best_loss: float = float("inf"),
        filename: str | None = None,
        extra: dict[str, Any] | None = None,
        scaler: Any = None,
    ) -> str:
        """Atomically save a complete format-v2 checkpoint.

        The temporary file is written and fsynced in the destination directory,
        then atomically renamed.  Readers therefore see either the previous
        complete checkpoint or the new complete checkpoint, never a partial
        torch archive.
        """
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError(f"step must be a non-negative int, got {step!r}")
        filename = filename or f"checkpoint_step_{step}.pt"
        checkpoint_path = self._checkpoint_path(filename)
        destination_dir = os.path.dirname(checkpoint_path)
        os.makedirs(destination_dir, exist_ok=True)

        checkpoint: dict[str, Any] = {
            "format_version": _FORMAT_VERSION,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "loss": loss,
            "best_loss": best_loss,
            "rng_state": _capture_rng_state(),
            "extra": extra or {},
        }
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None and hasattr(scaler, "state_dict"):
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(checkpoint_path)}.", suffix=".tmp", dir=destination_dir
        )
        os.close(fd)
        try:
            torch.save(checkpoint, temporary_path)
            with open(temporary_path, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary_path, checkpoint_path)
            try:
                directory_fd = os.open(destination_dir, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                # Some filesystems do not support directory fsync; the atomic
                # replace still prevents partial-file visibility.
                pass
        except Exception:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise
        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: Any,
        optimizer: Any,
        scheduler: Any = None,
        scaler: Any = None,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Restore a checkpoint without executing pickled Python objects."""
        if not isinstance(checkpoint_path, str):
            raise TypeError(f"checkpoint_path must be str, got {type(checkpoint_path).__name__}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        format_version = int(checkpoint.get("format_version", 1))
        rng_restored = False
        if restore_rng and "rng_state" in checkpoint:
            _restore_rng_state(checkpoint["rng_state"])
            rng_restored = True

        return {
            "step": checkpoint.get("step", 0),
            "loss": checkpoint.get("loss", float("inf")),
            "best_loss": checkpoint.get("best_loss", float("inf")),
            "filename": os.path.basename(checkpoint_path),
            "format_version": format_version,
            "extra": checkpoint.get("extra", {}),
            "scaler_state_dict": checkpoint.get("scaler_state_dict"),
            "rng_restored": rng_restored,
        }
