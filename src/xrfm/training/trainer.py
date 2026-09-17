"""
Public training API for XRFM.

``Trainer`` is the stable, typed boundary around the internal
:class:`~xrfm.training.loop.TrainingLoop`:

    from xrfm import Trainer, XRFMModel, load_config

    cfg = load_config("config/tiny.yaml")
    model = XRFMModel(cfg)
    trainer = Trainer(model, config=cfg)
    result = trainer.train(dataset)          # dataset: xrfm.data.TextDataset

It builds the optimizer/scheduler from the config when not supplied, owns
device placement (config ``training.device`` or auto), and returns a result
dict with the final loss/step and checkpoint path. Advanced users can reach
the underlying loop via :attr:`Trainer.loop` — the facade is intentionally
thin, not hiding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from xrfm.config.schema import TrainingConfig, XRFMConfig
from xrfm.training.distributed import xrfm_collate_fn
from xrfm.training.loop import TrainingLoop
from xrfm.training.optimizer import OptimizerLoader
from xrfm.training.scheduler import SchedulerLoader

__all__ = ["Trainer", "TrainResult"]


class TrainResult(dict):
    """Training outcome dict with attribute access (``result.final_loss``)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(name) from exc


class Trainer:
    """Train an XRFM model from a typed configuration.

    Args:
        model: An ``XRFMModel`` (or any ``nn.Module`` exposing ``config``
            and accepting ``(input_ids, mask=...) -> (logits, ...)``).
        config: :class:`XRFMConfig` (preferred), a bare
            :class:`TrainingConfig`, a YAML path (legacy), or ``None``
            (packaged default).
        optimizer: Optional prebuilt :class:`OptimizerLoader`.
        scheduler: Optional prebuilt :class:`SchedulerLoader`.
        checkpoint_dir: Where checkpoints are written.
        device: Explicit device override (string or ``torch.device``).
            ``None`` → config ``training.device`` → auto (cuda|cpu).

    Example:
        >>> trainer = Trainer(model, config=cfg)          # doctest: +SKIP
        >>> out = trainer.train(train_ds, max_steps=100)  # doctest: +SKIP
        >>> out["final_loss"]                             # doctest: +SKIP
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: XRFMConfig | TrainingConfig | str | Path | None = None,
        optimizer: OptimizerLoader | None = None,
        scheduler: SchedulerLoader | None = None,
        checkpoint_dir: str = "checkpoints/",
        device: str | torch.device | None = None,
    ) -> None:
        if model is None:
            raise ValueError("model is required")

        self.model = model
        self._config = config
        self._device_override = str(device) if device is not None else None

        self.loop = TrainingLoop(
            config=config,  # type: ignore[arg-type]  # str | Path both accepted
            model=model,
            dataset=None,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dir=checkpoint_dir,
        )
        if self._device_override is not None:
            self.loop.device = torch.device(self._device_override)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> TrainingConfig:
        """The validated :class:`TrainingConfig` in effect."""
        return self.loop.training_config

    @property
    def device(self) -> torch.device:
        """Device batches will be moved to (the model's device)."""
        return self.loop.device

    def train(
        self,
        dataset: Any,
        max_steps: int | None = None,
        checkpoint_every: int | None = None,
        log_interval: int = 10,
        validation_fn: Any = None,
        metrics_writer: Any = None,
    ) -> TrainResult:
        """Run training over ``dataset`` (a torch ``Dataset``).

        Args:
            dataset: Map-style dataset yielding ``(input_ids, targets)`` —
                typically :class:`xrfm.data.TextDataset`.
            max_steps: Override ``training.max_steps``.
            checkpoint_every: Override ``training.checkpoint_every``.
            log_interval: Steps between log lines.
            validation_fn: Optional callable(loop) -> {"loss", "perplexity"}.
            metrics_writer: Optional sink with ``write_metrics(step, metrics)``.

        Returns:
            :class:`TrainResult` with ``final_loss``, ``best_loss``,
            ``final_step``, ``checkpoint_path``.
        """
        if dataset is None:
            raise ValueError("dataset is required")
        self.loop.dataset = dataset
        self.loop.validation_fn = validation_fn
        self.loop.metrics_writer = metrics_writer
        result = self.loop.training_loop(
            max_steps=max_steps,
            checkpoint_every=checkpoint_every,
            log_interval=log_interval,
        )
        return TrainResult(result)

    def train_dataloader(self, dataloader: Any, max_steps: int | None = None, log_interval: int = 10) -> TrainResult:
        """Run training over a prepared ``DataLoader`` (advanced use).

        The loader must yield ``(input_ids, targets)`` 2D batches (e.g. via
        :func:`xrfm.training.distributed.xrfm_collate_fn`).
        """
        if dataloader is None:
            raise ValueError("dataloader is required")
        result = self.loop.training_loop(max_steps=max_steps, log_interval=log_interval, dataloader=dataloader)
        return TrainResult(result)

    def train_step(self, input_ids: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        """Single optimization step (exposes the loop's step function)."""
        return self.loop.train_step(input_ids, targets, mask)

    def save_checkpoint(self, step: int | None = None, loss: float = float("nan")) -> str:
        """Checkpoint now (raw model + optimizer + scheduler + metadata)."""
        from xrfm.training.distributed import get_raw_model, is_main_process

        if not is_main_process():
            raise RuntimeError("save_checkpoint is only available on the main process")
        return self.loop.checkpoint_loader.save_checkpoint(
            get_raw_model(self.loop.model),
            self.loop.optimizer,
            self.loop.scheduler,
            step=step if step is not None else self.loop.current_step,
            loss=loss,
            best_loss=self.loop.best_loss,
            extra=self.loop._checkpoint_extra(),
        )


class _LoaderAsDataset:
    """Adapts an iterable-of-batches DataLoader to the loop's dataset contract.

    The loop iterates ``for bi, bt in dataloader`` internally per epoch; a
    DataLoader *is* iterable, but the loop's dataset contract wants a
    re-iterable object with ``__len__``. Most torch DataLoaders satisfy
    both; this wrapper guarantees it.
    """

    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def __iter__(self):  # noqa: ANN201
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    def __getitem__(self, idx: int):  # pragma: no cover - contract shim
        raise IndexError("batch-level pseudo-dataset does not support scalar indexing")


# Re-exported for callers constructing loaders manually.
__all__ += ["xrfm_collate_fn"]
