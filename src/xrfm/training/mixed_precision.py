"""Capability-aware mixed-precision policy and gradient-scaler adapter.

``mixed_precision=True`` is a request, not a promise that every device supports
BF16.  ``precision='auto'`` selects BF16 only when CUDA reports support and
otherwise selects FP16; CPU and disabled runs remain FP32.  Explicit reduced
precision requests fail clearly when the required device is unavailable.

The scaler adapter deliberately determines step success from the scale before
and after ``GradScaler.update()``.  ``GradScaler.step()`` returns ``None`` for
optimizers such as AdamW even when their step succeeded, so its return value is
not a success signal.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

_PRECISIONS = {"auto", "fp32", "fp16", "bf16"}


class MixedPrecisionLoader:
    """Own autocast, scaling, and optimizer-step outcome for one training loop."""

    def __init__(self, enabled: bool = True, precision: str = "auto") -> None:
        if not isinstance(enabled, bool):
            raise ValueError(
                f"enabled must be bool, got {type(enabled).__name__}. "
                "Check ConfigLoader settings (training.mixed_precision)."
            )
        if precision not in _PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(_PRECISIONS)}, got {precision!r}")
        if not enabled and precision not in {"auto", "fp32"}:
            raise ValueError("an explicit reduced precision requires mixed_precision=True")

        self.enabled = enabled
        self.requested_precision = precision
        self.scaler: Any = NoOpScaler()
        self._update_already_done = False

        cuda_available = torch.cuda.is_available()
        if not enabled or precision == "fp32" or not cuda_available:
            if enabled and precision in {"fp16", "bf16"} and not cuda_available:
                raise RuntimeError(f"training.precision={precision!r} requires CUDA; no CUDA device is available")
            self.precision = "fp32"
            self.active = False
            self._autocast_dtype = torch.float32
            self._autocast_device = "cpu"
            return

        if precision == "auto":
            supports_bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            self.precision = "bf16" if supports_bf16 else "fp16"
        else:
            self.precision = precision

        if self.precision == "bf16" and not bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
            raise RuntimeError("training.precision='bf16' is not supported by this CUDA device")

        self.active = True
        self._autocast_device = "cuda"
        self._autocast_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        # BF16 has a wide exponent range and does not need gradient scaling.
        # FP16 uses GradScaler to detect overflow and skip unsafe updates.
        if self.precision == "fp16":
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                self.scaler = torch.amp.GradScaler("cuda")
            else:  # pragma: no cover - compatibility with older supported torch
                self.scaler = torch.cuda.amp.GradScaler()

    def autocast_ctx(self) -> Any:
        """Return the active autocast context, or a no-op FP32 context."""
        if self.active:
            return torch.autocast(device_type=self._autocast_device, dtype=self._autocast_dtype)
        return nullcontext()

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale a loss when FP16 gradient scaling is active."""
        return self.scaler.scale(loss) if hasattr(self.scaler, "scale") else loss

    def unscale_(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        """Unscale before gradient clipping, as required by GradScaler."""
        if hasattr(self.scaler, "unscale_") and optimizer is not None:
            self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Perform an optimizer step and return whether it really happened.

        For a scale-aware scaler, ``update()`` is performed here so the scale
        comparison includes overflow detection.  A subsequent legacy call to
        :meth:`update` is consumed as a no-op; this preserves the old public
        call sequence without updating the scaler twice.
        """
        if isinstance(self.scaler, NoOpScaler):
            return bool(self.scaler.step(optimizer))
        if not hasattr(self.scaler, "get_scale"):
            raise RuntimeError("scale-aware optimizer stepping requires a scaler with get_scale()")

        scale_before = float(self.scaler.get_scale())
        self.scaler.step(optimizer)
        self.scaler.update()
        self._update_already_done = True
        scale_after = float(self.scaler.get_scale())
        # GradScaler decreases the scale iff inf/nan gradients caused a skip.
        # Growth and an unchanged scale both denote a successful update.
        return scale_after >= scale_before

    def update(self) -> None:
        """Compatibility wrapper; avoid a second update after :meth:`step`."""
        if self._update_already_done:
            self._update_already_done = False
            return
        if hasattr(self.scaler, "update"):
            self.scaler.update()

    def state_dict(self) -> dict[str, Any]:
        """Return precision and scaler state for an exact-resume checkpoint."""
        state = self.scaler.state_dict() if hasattr(self.scaler, "state_dict") else {}
        return {"precision": self.precision, "scaler": state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scaler state without silently changing the device policy."""
        saved_precision = state.get("precision")
        if saved_precision is not None and saved_precision != self.precision:
            raise RuntimeError(f"checkpoint precision {saved_precision!r} != current precision {self.precision!r}")
        scaler_state = state.get("scaler", {})
        if scaler_state and hasattr(self.scaler, "load_state_dict"):
            self.scaler.load_state_dict(scaler_state)


class NoOpScaler:
    """FP32/BF16 scaler-compatible adapter that never scales gradients."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        return None

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        optimizer.step()
        return True

    def update(self) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise RuntimeError("cannot load non-empty gradient-scaler state into NoOpScaler")
