"""
Training loop for XR Foundation Model (v0.8.0).

Supports DDP/FSDP, gradient accumulation, mixed precision,
gradient clipping, checkpointing, and resume.
"""

import logging
import random
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn

from training.checkpoint import CheckpointLoader


def _set_seed(seed: int) -> None:
    """Seed all global RNGs for reproducible runs (forensic-audit fix, F-25)."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


from training.distributed import (
    barrier,
    create_distributed_dataloader,
    get_raw_model,
    is_distributed,
    is_main_process,
    wrap_model_ddp,
    wrap_model_fsdp,
)
from training.mixed_precision import MixedPrecisionLoader
from training.optimizer import OptimizerLoader
from training.scheduler import SchedulerLoader
from xrfm.config.loader import ConfigLoader

logger = logging.getLogger("xrfm.training")


class TrainingLoop:
    """Training loop for XRFM (v0.8.0).

    Features: causal next-token prediction, gradient accumulation,
    DDP/FSDP wrapping, distributed-aware logging, checkpoint barrier.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        model: nn.Module | None = None,
        dataset: Any | None = None,
        optimizer: OptimizerLoader | None = None,
        scheduler: SchedulerLoader | None = None,
        checkpoint_dir: str = "checkpoints/",
    ) -> None:
        if model is None:
            raise ValueError("model is required")
        if dataset is None:
            raise ValueError("dataset is required")

        try:
            loader = ConfigLoader(config_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Config not found: '{config_path}'") from exc

        self.config_path = config_path
        self.config_loader = loader

        # Wrap model for distributed if applicable
        use_ddp = loader.get("training.use_ddp", False)
        use_fsdp = loader.get("training.use_fsdp", False)
        self.model = model

        if is_distributed():
            if use_fsdp:
                logger.info("Wrapping model with FSDP")
                self.model = wrap_model_fsdp(model)
            elif use_ddp:
                logger.info("Wrapping model with DDP")
                self.model = wrap_model_ddp(model)

        self.dataset = dataset
        self.checkpoint_dir = checkpoint_dir

        # Optimizer on raw model params
        raw_model = get_raw_model(self.model)
        if optimizer is not None:
            if not isinstance(optimizer, OptimizerLoader):
                raise TypeError(f"optimizer must be OptimizerLoader, got {type(optimizer).__name__}")
            self.optimizer = optimizer
        else:
            self.optimizer = OptimizerLoader(
                raw_model.parameters(),
                learning_rate=loader.get("training.learning_rate", 0.001),
                weight_decay=loader.get("training.weight_decay", 0.01),
            )

        # Scheduler
        if scheduler is not None:
            if not isinstance(scheduler, SchedulerLoader):
                raise TypeError(f"scheduler must be SchedulerLoader, got {type(scheduler).__name__}")
            self.scheduler = scheduler
        else:
            self.scheduler = SchedulerLoader(
                self.optimizer.optimizer,
                base_lr=loader.get("training.learning_rate", 0.001),
                warmup_steps=loader.get("training.warmup_steps", 1000),
                max_steps=loader.get("training.max_steps", 50000),
            )

        self.checkpoint_loader = CheckpointLoader(checkpoint_dir)
        amp = loader.get("training.mixed_precision", True)
        self.mixed_precision_loader = MixedPrecisionLoader(enabled=amp)

        self.max_steps: int = loader.get("training.max_steps", 50000)
        self.warmup_steps: int = loader.get("training.warmup_steps", 1000)
        self.gradient_clip: float = loader.get("training.gradient_clip", 1.0)
        self.checkpoint_every: int = loader.get("training.checkpoint_every", 1000)
        self.batch_size: int = loader.get("training.batch_size", 32)
        self.grad_accum_steps: int = loader.get("training.grad_accum_steps", 1)
        self.resume_from: str | None = loader.get("training.resume_from", None)
        self.seed: int = loader.get("training.seed", 42)
        self.error_if_nonfinite: bool = loader.get("training.error_if_nonfinite", False)
        # Loss masking: targets equal to this value are ignored by
        # cross_entropy (padding positions). -100 is PyTorch's convention.
        self.ignore_index: int = loader.get("training.ignore_index", -100)
        self.eval_every: int = loader.get("training.eval_every", 0)

        # ---- GPU-scale dataloader / device options (Phase 27 hardening) ----
        self.num_workers: int = loader.get("training.num_workers", 0)
        self.pin_memory: bool = loader.get("training.pin_memory", False)
        self.prefetch_factor: int | None = loader.get("training.prefetch_factor", None)
        self.persistent_workers: bool = loader.get("training.persistent_workers", False)
        # None -> auto (CUDA if available else CPU).
        self.device = loader.get("training.device", None)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(str(self.device))
        # Deterministic execution (only meaningful where supported; warn-only
        # so unsupported ops degrade gracefully instead of crashing).
        self.deterministic: bool = loader.get("training.deterministic", False)

        # Dataset contract validation + grad accumulation sanity.
        if not hasattr(self.dataset, "__getitem__") or not hasattr(self.dataset, "__len__"):
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
        # Optional callable(callable[TrainingLoop] -> dict with loss/perplexity).
        self.validation_fn: Any | None = None
        # Optional metrics sink with write_metrics(step, metrics).
        self.metrics_writer: Any | None = None

        if self.resume_from is not None:
            meta = self.checkpoint_loader.load_checkpoint(self.resume_from, raw_model, self.optimizer, self.scheduler)
            self.current_step = meta.get("step", 0)
            self.best_loss = meta.get("best_loss", float("inf"))
            logger.info(
                "Resumed from step %d, best_loss=%.4f (config resume_from=%s)",
                self.current_step,
                self.best_loss,
                self.resume_from,
            )

    def _checkpoint_extra(self) -> dict[str, Any]:
        """Reproducibility metadata stored in every checkpoint (F-33)."""
        try:
            raw_config = self.config_loader._config
        except Exception:  # noqa: BLE001
            raw_config = None
        return {
            "config_path": self.config_path,
            "config": raw_config,
            "seed": self.seed,
            "ignore_index": self.ignore_index,
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

    def train_step(
        self,
        batch_input_ids: torch.Tensor,
        batch_target_ids: torch.Tensor,
        batch_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        if self.optimizer is None or self.model is None:
            raise RuntimeError("model/optimizer not initialized")
        if batch_input_ids.dim() != 2 or batch_target_ids.dim() != 2:
            raise ValueError("batch tensors must be 2D")

        raw = get_raw_model(self.model)
        device = next(raw.parameters()).device
        # non_blocking=True overlaps H2D copies with compute when the loader
        # uses pin_memory (Phase 27 hardening; harmless on CPU).
        non_blocking = self.pin_memory and device.type == "cuda"
        if batch_input_ids.device != device:
            batch_input_ids = batch_input_ids.to(device, non_blocking=non_blocking)
        if batch_target_ids.device != device:
            batch_target_ids = batch_target_ids.to(device, non_blocking=non_blocking)
        if batch_mask is not None and batch_mask.device != device:
            batch_mask = batch_mask.to(device, non_blocking=non_blocking)
        vocab_size = int(getattr(raw.embedding, "vocab_size"))
        if (batch_input_ids >= vocab_size).any():
            raise IndexError(f"Token ID exceeds vocab size ({vocab_size})")

        # zero_grad at start of accumulation cycle
        if self._micro_counter % self.grad_accum_steps == 0:
            self.optimizer.zero_grad()

        # forward + loss (DDP no_sync on intermediate micro-batches)
        self.model.train()
        try:
            with self._no_sync_ctx():
                with self.mixed_precision_loader.autocast_ctx():
                    logits, _ = self.model(batch_input_ids, mask=batch_mask)
        except torch.cuda.OutOfMemoryError:
            # CUDA-only: skip the micro-batch and continue instead of dying.
            # (Phase 27 hardening — prevents a single oversized batch from
            # killing a long GPU run.)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.warning(
                "CUDA OOM at step %d micro %d; skipping batch",
                self.current_step,
                self._micro_counter,
            )
            self._micro_counter += 1
            return {
                "loss": float("nan"),
                "learning_rate": self.scheduler.get_lr() if self.scheduler else float("nan"),
                "step": self.current_step,
                "step_performed": False,
            }

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            batch_target_ids.view(-1),
            ignore_index=self.ignore_index,
        )
        self._accum_loss += loss.item()
        scaled = loss / self.grad_accum_steps
        self.mixed_precision_loader.scale(scaled).backward()
        self._micro_counter += 1

        if not self._ready_to_step():
            lr = self.scheduler.get_lr() if self.scheduler else self.optimizer.learning_rate
            return {
                "loss": self._accum_loss / self._micro_counter,
                "learning_rate": lr,
                "step": self.current_step,
                "step_performed": False,
            }

        # optimizer step
        self.mixed_precision_loader.unscale_(self.optimizer.optimizer)
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                raw.parameters(),
                max_norm=self.gradient_clip,
                norm_type=2.0,
                error_if_nonfinite=self.error_if_nonfinite,
            )
        ok = self.mixed_precision_loader.step(self.optimizer.optimizer)
        if self.scheduler:
            self.scheduler.step()
        self.mixed_precision_loader.update()

        # Report the mean loss over the accumulation cycle. The value is the
        # per-rank average (distributed gradients already synchronize across
        # ranks; per-rank losses are expected to differ slightly).
        mean_loss = self._accum_loss / self.grad_accum_steps
        self._accum_loss = 0.0
        self.current_step += 1
        if mean_loss < self.best_loss:
            self.best_loss = mean_loss
        lr = self.scheduler.get_lr() if self.scheduler else self.optimizer.learning_rate
        return {
            "loss": mean_loss,
            "learning_rate": lr,
            "step": self.current_step,
            "step_performed": ok,
        }

    # -- full loop --

    def training_loop(
        self,
        max_steps: int | None = None,
        checkpoint_every: int | None = None,
        log_interval: int = 10,
    ) -> dict[str, Any]:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("model and optimizer required")

        max_steps = max_steps or self.max_steps
        checkpoint_every = checkpoint_every or self.checkpoint_every
        if max_steps <= 0 or checkpoint_every <= 0:
            raise ValueError("max_steps and checkpoint_every must be positive")
        if self.dataset is None:
            raise ValueError("dataset is required")

        # Deterministic shuffling (F-25): a generator seeded from `self.seed`
        # + an epoch-derived offset keeps batch order reproducible across runs.
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        def _worker_init(worker_id: int) -> None:
            seed = (self.seed + worker_id * 1000) % (2**32)
            random.seed(seed)
            torch.manual_seed(seed)

        dl_kwargs: dict[str, Any] = {
            "generator": generator,
            "worker_init_fn": _worker_init,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = self.prefetch_factor
        if self.num_workers > 0:
            dl_kwargs["persistent_workers"] = self.persistent_workers

        dataloader = create_distributed_dataloader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            **dl_kwargs,
        )
        if len(dataloader) == 0:
            logger.warning(
                "Dataset length (%d) < batch_size (%d) with drop_last=True. Retrying with drop_last=False.",
                len(self.dataset),
                self.batch_size,
            )
            dataloader = create_distributed_dataloader(
                self.dataset,
                batch_size=min(self.batch_size, len(self.dataset)),
                shuffle=True,
                drop_last=False,
                **dl_kwargs,
            )
        if len(dataloader) == 0:
            raise ValueError(f"Dataset length ({len(self.dataset)}) is too small for training.")
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
        metrics = {"loss": 0.0, "learning_rate": 0.0, "step": 0, "step_performed": False}
        self._metrics_records: list[dict[str, float]] = []

        while self.current_step < max_steps:
            for bi, bt in dataloader:
                if self.current_step >= max_steps:
                    break
                metrics = self.train_step(bi, bt)
                if not metrics["step_performed"]:
                    continue

                # Validation hook (F-27): called periodically if configured.
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
                        extra = f" | ValLoss: {metrics.get('val_loss', float('nan')):.4f} | ValPPL: {metrics.get('val_ppl', float('nan')):.2f}"
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
                        )
                        logger.info("Checkpoint: %s", final_ckpt)
                    barrier()

        if is_main_process():
            final_ckpt = self.checkpoint_loader.save_checkpoint(
                get_raw_model(self.model),
                self.optimizer,
                self.scheduler,
                step=self.current_step,
                loss=metrics["loss"],
                best_loss=self.best_loss,
                extra=self._checkpoint_extra(),
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
        }
