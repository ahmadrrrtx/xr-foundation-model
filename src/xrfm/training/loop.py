"""
Training loop for XR Foundation Model (v0.8.0).

Supports DDP/FSDP, gradient accumulation, mixed precision,
gradient clipping, checkpointing, and resume.
"""

import hashlib
import json
import logging
import random
from contextlib import nullcontext
from typing import Any, NoReturn

import numpy as np
import torch
import torch.nn as nn

from xrfm.training.checkpoint import CheckpointLoader, _capture_rng_state, _restore_rng_state


def _set_seed(seed: int) -> None:
    """Seed all global RNGs for reproducible runs (forensic-audit fix, F-25)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


from xrfm.config.loader import load_config
from xrfm.config.schema import TrainingConfig, XRFMConfig
from xrfm.training.distributed import (
    barrier,
    create_distributed_dataloader,
    get_raw_model,
    is_distributed,
    is_main_process,
    set_dataloader_epoch,
    wrap_model_ddp,
    wrap_model_fsdp,
)
from xrfm.training.mixed_precision import MixedPrecisionLoader
from xrfm.training.optimizer import OptimizerLoader
from xrfm.training.scheduler import SchedulerLoader

logger = logging.getLogger("xrfm.training")


class TrainingOOMError(RuntimeError):
    """A CUDA allocation failure that invalidates the current training attempt."""


class _ConfigFacade:
    """Compat facade exposing dot-notation ``get()`` over the typed config.

    Pre-Phase-0 code accessed ``loop.config_loader.get("training.x")``;
    typed config is now the source of truth and this adapter serves legacy
    readers only.
    """

    def __init__(self, cfg: XRFMConfig | None) -> None:
        self._raw: dict = cfg.to_dict() if cfg is not None else {}

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self._raw
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value


class TrainingLoop:
    """Training loop for XRFM (v0.8.0).

    Features: causal next-token prediction, gradient accumulation,
    DDP/FSDP wrapping, distributed-aware logging, checkpoint barrier.
    """

    def __init__(
        self,
        config: XRFMConfig | TrainingConfig | str | None = None,
        model: nn.Module | None = None,
        dataset: Any | None = None,
        optimizer: OptimizerLoader | None = None,
        scheduler: SchedulerLoader | None = None,
        checkpoint_dir: str = "checkpoints/",
        config_path: str | None = None,
        data_manifest_hash: str | None = None,
        tokenizer_hash: str | None = None,
    ) -> None:
        """Create a training loop from a typed config (or legacy YAML path).

        Args:
            config: :class:`xrfm.config.schema.XRFMConfig` (uses its
                ``training`` section), a bare :class:`TrainingConfig`, a YAML
                path (legacy spelling), or ``None`` for the packaged default.
            config_path: legacy keyword alias for a YAML path.
        """
        if model is None:
            raise ValueError("model is required")

        # Resolve config → typed TrainingConfig (+ full XRFMConfig when
        # available). No CWD dependence: None means the packaged default.
        if config is not None and config_path is not None:
            raise ValueError("pass either config or config_path, not both")
        if config_path is not None and not isinstance(config, (XRFMConfig, TrainingConfig)):
            # Legacy spelling: TrainingLoop(config_path="config/tiny.yaml").
            config = config_path

        full_cfg: XRFMConfig | None
        if isinstance(config, TrainingConfig):
            train_cfg, full_cfg = config, None
        elif isinstance(config, XRFMConfig):
            train_cfg, full_cfg = config.training, config
        elif isinstance(config, str):
            full_cfg = load_config(config)
            train_cfg = full_cfg.training
        elif config is None:
            full_cfg = load_config(None)
            train_cfg = full_cfg.training
        else:
            raise TypeError(
                f"config must be XRFMConfig, TrainingConfig, str (YAML path), or None; got {type(config).__name__}"
            )

        self.training_config: TrainingConfig = train_cfg
        self.config: XRFMConfig | None = full_cfg
        self.config_path = config if isinstance(config, str) else config_path

        # Wrap model for distributed if applicable
        self.model = model
        if is_distributed():
            if train_cfg.use_fsdp:
                logger.info("Wrapping model with FSDP")
                self.model = wrap_model_fsdp(model)
            elif train_cfg.use_ddp:
                logger.info("Wrapping model with DDP")
                self.model = wrap_model_ddp(model)

        self.dataset: Any = dataset
        self.checkpoint_dir = checkpoint_dir
        self.data_manifest_hash = (
            data_manifest_hash
            or getattr(dataset, "data_manifest_hash", None)
            or getattr(dataset, "manifest_hash", None)
        )
        self.tokenizer_hash = tokenizer_hash or getattr(dataset, "tokenizer_hash", None)
        self.epoch: int = 0
        self._batch_in_epoch: int = 0
        self._resume_rng_state: dict[str, Any] | None = None
        self._resume_loader_kind: str | None = None
        self._active_loader_kind = "internal"
        self._resolved_batch_size: int | None = None
        self._resolved_drop_last: bool | None = None

        # Optimizer on raw model params
        raw_model = get_raw_model(self.model)
        if optimizer is not None:
            if not isinstance(optimizer, OptimizerLoader):
                raise TypeError(f"optimizer must be OptimizerLoader, got {type(optimizer).__name__}")
            self.optimizer = optimizer
        else:
            self.optimizer = OptimizerLoader(
                raw_model.parameters(),
                learning_rate=train_cfg.learning_rate,
                weight_decay=train_cfg.weight_decay,
            )

        # Scheduler
        if scheduler is not None:
            if not isinstance(scheduler, SchedulerLoader):
                raise TypeError(f"scheduler must be SchedulerLoader, got {type(scheduler).__name__}")
            self.scheduler = scheduler
        else:
            self.scheduler = SchedulerLoader(
                self.optimizer.optimizer,
                base_lr=train_cfg.learning_rate,
                warmup_steps=train_cfg.warmup_steps,
                max_steps=train_cfg.max_steps,
            )

        self.checkpoint_loader = CheckpointLoader(checkpoint_dir)
        self.mixed_precision_loader = MixedPrecisionLoader(
            enabled=train_cfg.mixed_precision,
            precision=train_cfg.precision,
        )

        # All hyperparameters come from the validated typed config — no
        # code-level duplicated defaults (single source of truth).
        self.max_steps: int = train_cfg.max_steps
        self.warmup_steps: int = train_cfg.warmup_steps
        self.gradient_clip: float = train_cfg.gradient_clip
        self.checkpoint_every: int = train_cfg.checkpoint_every
        self.batch_size: int = train_cfg.batch_size
        self.grad_accum_steps: int = train_cfg.grad_accum_steps
        self.resume_from: str | None = train_cfg.resume_from
        self.seed: int = train_cfg.seed
        self.error_if_nonfinite: bool = train_cfg.error_if_nonfinite
        self.max_optimizer_skips: int = train_cfg.max_optimizer_skips
        # Loss masking: targets equal to this value are ignored by
        # cross_entropy (padding positions). -100 is PyTorch's convention.
        self.ignore_index: int = train_cfg.ignore_index
        self.eval_every: int = train_cfg.eval_every

        # ---- GPU-scale dataloader / device options (Phase 27 hardening) ----
        self.num_workers: int = train_cfg.num_workers
        self.pin_memory: bool = train_cfg.pin_memory
        self.prefetch_factor: int | None = train_cfg.prefetch_factor
        self.persistent_workers: bool = train_cfg.persistent_workers
        # None -> auto (CUDA if available else CPU).
        if train_cfg.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(str(train_cfg.device))
        # Deterministic execution (only meaningful where supported; warn-only
        # so unsupported ops degrade gracefully instead of crashing).
        self.deterministic: bool = train_cfg.deterministic

        # Dataset contract validation + grad accumulation sanity.
        # (dataset may be None at construction; training_loop() requires one.)
        if self.dataset is not None and (
            not hasattr(self.dataset, "__getitem__") or not hasattr(self.dataset, "__len__")
        ):
            raise ValueError(f"dataset must implement __getitem__/__len__. Got: {type(self.dataset).__name__}")
        if self.grad_accum_steps <= 0:
            raise ValueError(f"grad_accum_steps must be positive: {self.grad_accum_steps}")

        # Resume-from-checkpoint (restored to __init__ in v1.1: a previous
        # refactor had orphaned this block after a return inside
        # _checkpoint_extra, silently disabling config-driven resume).
        # NOTE: this block intentionally runs AFTER the state initializers
        # (current_step=0 etc.) so the checkpoint values take precedence.

        # Determinism (forensic-audit fix, F-25): seed the global RNGs BEFORE
        # the dataloader is constructed so shuffle order is reproducible.
        _set_seed(self.seed)
        if self.deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:  # older torch
                torch.use_deterministic_algorithms(True)

        self.current_step: int = 0
        self.best_loss: float = float("inf")
        self._micro_counter: int = 0
        self._accum_loss: float = 0.0
        self._consecutive_optimizer_skips: int = 0
        # Optional callable(callable[TrainingLoop] -> dict with loss/perplexity).
        self.validation_fn: Any | None = None
        # Optional metrics sink with write_metrics(step, metrics).
        self.metrics_writer: Any | None = None

        self.resume_exact: bool | None = None
        if self.resume_from is not None:
            meta = self.checkpoint_loader.load_checkpoint(
                self.resume_from,
                raw_model,
                self.optimizer,
                self.scheduler,
                scaler=self.mixed_precision_loader,
            )
            self.current_step = int(meta.get("step", 0))
            self.best_loss = float(meta.get("best_loss", float("inf")))
            if int(meta.get("format_version", 1)) >= 2:
                extra = meta.get("extra") or {}
                if not meta.get("rng_restored"):
                    raise RuntimeError("format-v2 checkpoint did not restore RNG state; exact resume is unavailable")
                if extra.get("config_hash") != self._config_identity():
                    raise RuntimeError("checkpoint config identity does not match the current run")
                for field in ("data_manifest_hash", "tokenizer_hash"):
                    saved_identity = extra.get(field)
                    current_identity = getattr(self, field)
                    if not saved_identity or not current_identity or saved_identity != current_identity:
                        raise RuntimeError(
                            f"exact resume requires matching {field}; "
                            "pass the original data/tokenizer identity to TrainingLoop"
                        )
                data_state = extra.get("data_state")
                required = {"epoch", "batch_in_epoch", "micro_counter", "accum_loss", "loader_kind"}
                if not isinstance(data_state, dict) or not required.issubset(data_state):
                    raise RuntimeError(
                        "format-v2 checkpoint lacks complete dataloader position; exact resume is unavailable"
                    )
                self.epoch = int(data_state["epoch"])
                self._batch_in_epoch = int(data_state["batch_in_epoch"])
                self._micro_counter = int(data_state["micro_counter"])
                self._accum_loss = float(data_state["accum_loss"])
                self._resume_loader_kind = str(data_state["loader_kind"])
                self._resume_rng_state = _capture_rng_state()
                self.resume_exact = True
            else:
                self.resume_exact = False
                logger.warning(
                    "legacy checkpoint restored without exact data/RNG identity; this is not an exact resume"
                )
            logger.info(
                "Resumed from step %d, best_loss=%.4f (config resume_from=%s, exact=%s)",
                self.current_step,
                self.best_loss,
                self.resume_from,
                self.resume_exact,
            )

    @property
    def config_loader(self) -> "_ConfigFacade":
        """Legacy accessor: dot-notation reads over the typed config."""
        return _ConfigFacade(self.config)

    def _config_identity(self) -> str:
        """Hash the run config while excluding the resume pointer itself."""
        raw = self.config.to_dict() if self.config is not None else {"training": self.training_config.to_dict()}
        if isinstance(raw.get("training"), dict):
            raw["training"]["resume_from"] = None
        return hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]

    def _checkpoint_extra(self) -> dict[str, Any]:
        """Reproducibility, identity, and iterator state for exact resume."""
        raw_config = self.config.to_dict() if self.config is not None else {"training": self.training_config.to_dict()}
        return {
            "config_path": self.config_path,
            "config": raw_config,
            "config_hash": self._config_identity(),
            "data_manifest_hash": self.data_manifest_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "identity_complete": bool(self.data_manifest_hash and self.tokenizer_hash),
            "seed": self.seed,
            "ignore_index": self.ignore_index,
            "data_state": {
                "epoch": self.epoch,
                "batch_in_epoch": self._batch_in_epoch,
                "micro_counter": self._micro_counter,
                "accum_loss": self._accum_loss,
                "loader_kind": self._active_loader_kind,
            },
            # str() is important: TorchVersion objects are not weights_only-safe.
            "pytorch_version": str(torch.__version__),
        }

    # -- grad accum helpers --

    def _is_ddp_model(self) -> bool:
        return isinstance(self.model, nn.parallel.DistributedDataParallel)

    def _no_sync_ctx(self):
        if not self._is_ddp_model():
            return nullcontext()
        is_last = ((self._micro_counter + 1) % self.grad_accum_steps) == 0
        if is_last:
            return nullcontext()
        return self.model.no_sync()

    def _ready_to_step(self) -> bool:
        return self._micro_counter % self.grad_accum_steps == 0

    # -- train step --

    def _abort_oom(self, exc: torch.cuda.OutOfMemoryError) -> NoReturn:
        """Invalidate partial gradients and stop the coordinated attempt.

        There is no safe local "skip this batch" path for DDP: ranks can
        diverge or block in a collective.  Callers must terminate the attempt
        and restart every rank from the last complete checkpoint.
        """
        self.optimizer.optimizer.zero_grad(set_to_none=True)
        self._accum_loss = 0.0
        self._micro_counter = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.error(
            "CUDA OOM at step %d; aborting training attempt (distributed=%s). "
            "Resume all ranks from the last complete checkpoint.",
            self.current_step,
            is_distributed(),
        )
        raise TrainingOOMError("CUDA OOM aborted the training attempt; partial gradients were discarded") from exc

    def train_step(
        self,
        batch_input_ids: torch.Tensor,
        batch_target_ids: torch.Tensor,
        batch_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if self.optimizer is None or self.model is None:
            raise RuntimeError("model/optimizer not initialized")
        if batch_input_ids.dim() != 2 or batch_target_ids.dim() != 2:
            raise ValueError("batch tensors must be 2D")
        if batch_input_ids.shape != batch_target_ids.shape:
            raise ValueError("language-model input and target batch shapes must match")

        raw = get_raw_model(self.model)
        device = next(raw.parameters()).device
        non_blocking = self.pin_memory and device.type == "cuda"
        if batch_input_ids.device != device:
            batch_input_ids = batch_input_ids.to(device, non_blocking=non_blocking)
        if batch_target_ids.device != device:
            batch_target_ids = batch_target_ids.to(device, non_blocking=non_blocking)
        if batch_mask is not None and batch_mask.device != device:
            batch_mask = batch_mask.to(device, non_blocking=non_blocking)
        vocab_size = int(getattr(raw.embedding, "vocab_size"))
        if (batch_input_ids < 0).any() or (batch_input_ids >= vocab_size).any():
            raise IndexError(f"Token ID outside [0, {vocab_size})")

        if self._micro_counter % self.grad_accum_steps == 0:
            self.optimizer.optimizer.zero_grad(set_to_none=True)

        self.model.train()
        try:
            with self._no_sync_ctx():
                with self.mixed_precision_loader.autocast_ctx():
                    logits, _ = self.model(batch_input_ids, mask=batch_mask)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    batch_target_ids.view(-1),
                    ignore_index=self.ignore_index,
                )
                self._accum_loss += loss.item()
                scaled = loss / self.grad_accum_steps
                self.mixed_precision_loader.scale(scaled).backward()
        except torch.cuda.OutOfMemoryError as exc:
            self._abort_oom(exc)
        self._micro_counter += 1

        if not self._ready_to_step():
            lr = self.scheduler.get_lr() if self.scheduler else self.optimizer.learning_rate
            return {
                "loss": self._accum_loss / self._micro_counter,
                "learning_rate": lr,
                "step": self.current_step,
                "step_performed": False,
            }

        try:
            self.mixed_precision_loader.unscale_(self.optimizer.optimizer)
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    raw.parameters(),
                    max_norm=self.gradient_clip,
                    norm_type=2.0,
                    error_if_nonfinite=self.error_if_nonfinite,
                )
            ok = self.mixed_precision_loader.step(self.optimizer.optimizer)
            self.mixed_precision_loader.update()
        except torch.cuda.OutOfMemoryError as exc:
            self._abort_oom(exc)

        mean_loss = self._accum_loss / self.grad_accum_steps
        self._accum_loss = 0.0
        self._micro_counter = 0
        if not ok:
            # An overflow skip consumes no optimizer/scheduler/global step.
            # Do not carry partial gradients into the next accumulation cycle.
            self.optimizer.optimizer.zero_grad(set_to_none=True)
            self._consecutive_optimizer_skips += 1
            if self._consecutive_optimizer_skips > self.max_optimizer_skips:
                raise RuntimeError(
                    f"more than {self.max_optimizer_skips} consecutive optimizer steps were skipped; "
                    "inspect loss/gradient scaling before continuing"
                )
            return {
                "loss": mean_loss,
                "learning_rate": self.scheduler.get_lr() if self.scheduler else self.optimizer.learning_rate,
                "step": self.current_step,
                "step_performed": False,
            }

        self._consecutive_optimizer_skips = 0
        if self.scheduler:
            self.scheduler.step()
        self.current_step += 1
        if mean_loss < self.best_loss:
            self.best_loss = mean_loss
        lr = self.scheduler.get_lr() if self.scheduler else self.optimizer.learning_rate
        return {
            "loss": mean_loss,
            "learning_rate": lr,
            "step": self.current_step,
            "step_performed": True,
        }

    def _build_internal_dataloader(self, epoch: int) -> Any:
        """Build one deterministic epoch iterator with the model pad contract."""
        if self.dataset is None:
            raise RuntimeError("internal dataloader requires a dataset")
        raw = get_raw_model(self.model)
        model_config = getattr(raw, "config", None)
        model_pad = getattr(model_config, "pad_token_id", None)
        dataset_pad = getattr(self.dataset, "pad_id", None)
        if model_pad is not None and dataset_pad is not None and model_pad != dataset_pad:
            raise ValueError(f"model pad_token_id={model_pad} != dataset pad_id={dataset_pad}")
        pad_id = model_pad if model_pad is not None else (dataset_pad if dataset_pad is not None else 0)

        generator = torch.Generator()
        generator.manual_seed((self.seed + epoch) % (2**63 - 1))

        def _worker_init(worker_id: int) -> None:
            worker_seed = (self.seed + epoch * 1_000_003 + worker_id * 1000) % (2**32)
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        kwargs: dict[str, Any] = {
            "generator": generator,
            "worker_init_fn": _worker_init,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "pad_id": pad_id,
            "ignore_index": self.ignore_index,
            "return_attention_mask": True,
            "sampler_seed": self.seed,
        }
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers

        if self._resolved_batch_size is None:
            batch_size = self.batch_size
            drop_last = True
            loader = create_distributed_dataloader(
                self.dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last,
                **kwargs,
            )
            if len(loader) == 0:
                logger.warning(
                    "Dataset length (%d) < batch_size (%d) with drop_last=True; retrying with drop_last=False.",
                    len(self.dataset),
                    batch_size,
                )
                batch_size = min(batch_size, len(self.dataset))
                if batch_size <= 0:
                    raise ValueError("dataset is empty")
                drop_last = False
                loader = create_distributed_dataloader(
                    self.dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=drop_last,
                    **kwargs,
                )
            self._resolved_batch_size = batch_size
            self._resolved_drop_last = drop_last
            return loader

        return create_distributed_dataloader(
            self.dataset,
            batch_size=self._resolved_batch_size,
            shuffle=True,
            drop_last=bool(self._resolved_drop_last),
            **kwargs,
        )

    # -- full loop --

    def training_loop(
        self,
        max_steps: int | None = None,
        checkpoint_every: int | None = None,
        log_interval: int = 10,
        dataloader: Any = None,
    ) -> dict[str, Any]:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("model and optimizer required")

        max_steps = self.max_steps if max_steps is None else max_steps
        checkpoint_every = self.checkpoint_every if checkpoint_every is None else checkpoint_every
        if max_steps <= 0 or checkpoint_every <= 0:
            raise ValueError("max_steps and checkpoint_every must be positive")
        if self.dataset is None and dataloader is None:
            raise ValueError("dataset (or dataloader) is required")

        loader_kind = "external" if dataloader is not None else "internal"
        if self._resume_loader_kind is not None and self._resume_loader_kind != loader_kind:
            raise RuntimeError(
                f"checkpoint was saved with {self._resume_loader_kind} dataloader, "
                f"but resume requested {loader_kind}; exact resume is unavailable"
            )
        self._active_loader_kind = loader_kind
        if dataloader is not None and len(dataloader) == 0:
            raise ValueError("Provided dataloader is empty.")

        # If this is a continuation, positioning the iterator must not consume
        # the RNG state that belongs to the next training computation.
        position_rng = self._resume_rng_state
        if position_rng is None and self._batch_in_epoch > 0:
            position_rng = _capture_rng_state()

        eff_bs = self.batch_size * self.grad_accum_steps
        if is_distributed():
            eff_bs *= torch.distributed.get_world_size()
        if is_main_process():
            logger.info(
                "Starting: max_steps=%d per_dev_bs=%d accum=%d eff_bs=%d",
                max_steps,
                self.batch_size,
                self.grad_accum_steps,
                eff_bs,
            )

        final_ckpt = None
        metrics: dict[str, Any] = {
            "loss": 0.0,
            "learning_rate": 0.0,
            "step": self.current_step,
            "step_performed": False,
        }
        self._metrics_records: list[dict[str, float]] = []

        while self.current_step < max_steps:
            epoch_loader = dataloader if dataloader is not None else self._build_internal_dataloader(self.epoch)
            if len(epoch_loader) == 0:
                raise ValueError("dataloader is empty")
            set_dataloader_epoch(epoch_loader, self.epoch)
            iterator = iter(epoch_loader)

            # Restore a persisted batch position without training on skipped
            # examples. If dataset access uses global RNG, undo that access
            # afterward; it is bookkeeping, not a second consumption event.
            for _ in range(self._batch_in_epoch):
                try:
                    next(iterator)
                except StopIteration as exc:
                    raise RuntimeError("checkpoint dataloader position exceeds the current epoch") from exc
            if position_rng is not None:
                _restore_rng_state(position_rng)
                position_rng = None
                self._resume_rng_state = None

            epoch_exhausted = False
            while self.current_step < max_steps:
                try:
                    batch = next(iterator)
                except StopIteration:
                    self.epoch += 1
                    self._batch_in_epoch = 0
                    epoch_exhausted = True
                    break

                if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
                    raise ValueError("dataloader must yield (inputs, targets) or (inputs, targets, mask)")
                batch_input, batch_target = batch[0], batch[1]
                batch_mask = batch[2] if len(batch) == 3 else None
                metrics = self.train_step(batch_input, batch_target, batch_mask)
                self._batch_in_epoch += 1
                if not metrics["step_performed"]:
                    continue

                if self.eval_every > 0 and self.validation_fn is not None:
                    if self.current_step % self.eval_every == 0:
                        try:
                            val_metrics = self.validation_fn(self)
                            metrics["val_loss"] = val_metrics.get("loss", float("nan"))
                            metrics["val_ppl"] = val_metrics.get("perplexity", float("nan"))
                        except Exception:  # noqa: BLE001 - validation must not kill training
                            logger.exception("Validation failed at step %d", self.current_step)

                self._metrics_records.append(dict(metrics))
                if self.metrics_writer is not None:
                    self.metrics_writer.write_metrics(self.current_step, metrics)
                if is_main_process() and self.current_step % log_interval == 0:
                    extra = ""
                    if "val_loss" in metrics:
                        extra = (
                            f" | ValLoss: {metrics.get('val_loss', float('nan')):.4f}"
                            f" | ValPPL: {metrics.get('val_ppl', float('nan')):.2f}"
                        )
                    logger.info(
                        "Step %d/%d | Loss: %.4f | LR: %.6f | EffBS: %d%s",
                        self.current_step,
                        max_steps,
                        metrics["loss"],
                        metrics["learning_rate"],
                        eff_bs,
                        extra,
                    )

                if (self.current_step % checkpoint_every == 0) or (self.current_step >= max_steps):
                    if is_main_process():
                        final_ckpt = self.checkpoint_loader.save_checkpoint(
                            get_raw_model(self.model),
                            self.optimizer,
                            self.scheduler,
                            step=self.current_step,
                            loss=metrics["loss"],
                            best_loss=self.best_loss,
                            extra=self._checkpoint_extra(),
                            scaler=self.mixed_precision_loader,
                        )
                        logger.info("Checkpoint: %s", final_ckpt)
                    barrier()

            if not epoch_exhausted and self.current_step >= max_steps:
                break

        if is_main_process():
            final_ckpt = self.checkpoint_loader.save_checkpoint(
                get_raw_model(self.model),
                self.optimizer,
                self.scheduler,
                step=self.current_step,
                loss=metrics["loss"],
                best_loss=self.best_loss,
                extra=self._checkpoint_extra(),
                scaler=self.mixed_precision_loader,
            )
            logger.info(
                "Training complete: loss=%.4f best=%.4f | Saved final checkpoint: %s",
                metrics["loss"],
                self.best_loss,
                final_ckpt,
            )
        barrier()
        return {
            "final_loss": metrics["loss"],
            "best_loss": self.best_loss,
            "final_step": self.current_step,
            "checkpoint_path": final_ckpt,
            "epoch": self.epoch,
            "batch_in_epoch": self._batch_in_epoch,
            "resume_exact": self.resume_exact,
        }
