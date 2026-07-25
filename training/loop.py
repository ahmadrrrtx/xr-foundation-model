"""
Training loop for XR Foundation Model (v0.8.0).

Supports DDP/FSDP, gradient accumulation, mixed precision,
gradient clipping, checkpointing, and resume.
"""

from typing import Optional, Dict, Any
from contextlib import nullcontext
import logging

import torch
import torch.nn as nn

from xrfm.config.loader import ConfigLoader
from training.optimizer import OptimizerLoader
from training.scheduler import SchedulerLoader
from training.mixed_precision import MixedPrecisionLoader
from training.checkpoint import CheckpointLoader
from training.distributed import (
    is_main_process,
    is_distributed,
    reduce_loss,
    wrap_model_ddp,
    wrap_model_fsdp,
    get_raw_model,
    create_distributed_dataloader,
    barrier,
)

logger = logging.getLogger("xrfm.training")


class TrainingLoop:
    """Training loop for XRFM (v0.8.0).

    Features: causal next-token prediction, gradient accumulation,
    DDP/FSDP wrapping, distributed-aware logging, checkpoint barrier.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        model: Optional[nn.Module] = None,
        dataset: Optional[Any] = None,
        optimizer: Optional[OptimizerLoader] = None,
        scheduler: Optional[SchedulerLoader] = None,
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
        self.resume_from: Optional[str] = loader.get("training.resume_from", None)

        self.current_step: int = 0
        self.best_loss: float = float("inf")
        self._micro_counter: int = 0

        if not hasattr(dataset, "__getitem__") or not hasattr(dataset, "__len__"):
            raise ValueError(f"dataset must implement __getitem__/__len__. Got: {type(dataset).__name__}")
        if self.grad_accum_steps <= 0:
            raise ValueError(f"grad_accum_steps must be positive: {self.grad_accum_steps}")

        if self.resume_from is not None:
            meta = self.checkpoint_loader.load_checkpoint(
                self.resume_from, raw_model, self.optimizer, self.scheduler
            )
            self.current_step = meta.get("step", 0)
            self.best_loss = meta.get("best_loss", float("inf"))
            logger.info("Resumed from step %d, best_loss=%.4f", self.current_step, self.best_loss)

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
        return (self._micro_counter + 1) % self.grad_accum_steps == 0

    # -- train step --

    def train_step(
        self,
        batch_input_ids: torch.Tensor,
        batch_target_ids: torch.Tensor,
        batch_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        if self.optimizer is None or self.model is None:
            raise RuntimeError("model/optimizer not initialized")
        if batch_input_ids.dim() != 2 or batch_target_ids.dim() != 2:
            raise ValueError("batch tensors must be 2D")

        raw = get_raw_model(self.model)
        if (batch_input_ids >= raw.embedding.vocab_size).any():
            raise IndexError(f"Token ID exceeds vocab size ({raw.embedding.vocab_size})")

        # zero_grad at start of accumulation cycle
        if self._micro_counter % self.grad_accum_steps == 0:
            self.optimizer.zero_grad()

        # forward + loss (DDP no_sync on intermediate micro-batches)
        self.model.train()
        with self._no_sync_ctx():
            logits, _ = self.model(batch_input_ids, mask=batch_mask)

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            batch_target_ids.view(-1),
        )
        scaled = loss / self.grad_accum_steps
        self.mixed_precision_loader.scale(scaled).backward()
        self._micro_counter += 1

        if not self._ready_to_step():
            lr = self.scheduler.get_lr() if self.scheduler else self.optimizer.learning_rate
            return {"loss": loss.item(), "learning_rate": lr, "step": self.current_step, "step_performed": False}

        # optimizer step
        self.mixed_precision_loader.unscale_(self.optimizer.optimizer)
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw.parameters(), max_norm=self.gradient_clip, norm_type=2.0, error_if_nonfinite=True)
        ok = self.mixed_precision_loader.step(self.optimizer.optimizer)
        if self.scheduler:
            self.scheduler.step()
        self.mixed_precision_loader.update()

        reduced = reduce_loss(loss.detach()) if is_distributed() else loss.item()
        self.current_step += 1
        if reduced < self.best_loss:
            self.best_loss = reduced
        lr = self.scheduler.get_lr() if self.scheduler else self.optimizer.learning_rate
        return {"loss": reduced, "learning_rate": lr, "step": self.current_step, "step_performed": ok}

    # -- full loop --

    def training_loop(
        self,
        max_steps: Optional[int] = None,
        checkpoint_every: Optional[int] = None,
        log_interval: int = 10,
    ) -> Dict[str, Any]:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("model and optimizer required")

        max_steps = max_steps or self.max_steps
        checkpoint_every = checkpoint_every or self.checkpoint_every
        if max_steps <= 0 or checkpoint_every <= 0:
            raise ValueError("max_steps and checkpoint_every must be positive")
        if self.dataset is None:
            raise ValueError("dataset is required")

        dataloader = create_distributed_dataloader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        eff_bs = self.batch_size * self.grad_accum_steps
        if is_distributed():
            eff_bs *= torch.distributed.get_world_size()

        if is_main_process():
            logger.info("Starting: max_steps=%d per_dev_bs=%d accum=%d eff_bs=%d", max_steps, self.batch_size, self.grad_accum_steps, eff_bs)

        final_ckpt = None
        metrics = {"loss": 0.0, "learning_rate": 0.0, "step": 0, "step_performed": False}

        while self.current_step < max_steps:
            for bi, bt in dataloader:
                if self.current_step >= max_steps:
                    break
                metrics = self.train_step(bi, bt)
                if not metrics["step_performed"]:
                    continue

                if is_main_process() and self.current_step % log_interval == 0:
                    logger.info("Step %d/%d | Loss: %.4f | LR: %.6f | EffBS: %d",
                                self.current_step, max_steps, metrics["loss"], metrics["learning_rate"], eff_bs)

                if (self.current_step % checkpoint_every == 0) or (self.current_step >= max_steps):
                    if is_main_process():
                        final_ckpt = self.checkpoint_loader.save_checkpoint(
                            get_raw_model(self.model), self.optimizer, self.scheduler,
                            step=self.current_step, loss=metrics["loss"], best_loss=self.best_loss)
                        logger.info("Checkpoint: %s", final_ckpt)
                    barrier()

        if is_main_process():
            logger.info("Training complete: loss=%.4f best=%.4f", metrics["loss"], self.best_loss)
        barrier()
        return {"final_loss": metrics["loss"], "best_loss": self.best_loss, "final_step": self.current_step, "checkpoint_path": final_ckpt}
