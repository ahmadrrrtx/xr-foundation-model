"""
Distributed training utilities for XRFM (v0.8.0).

Provides DDP/FSDP wrappers, gradient accumulation, distributed
sampler setup, and multi-process launching.

All modules are designed to work in single-GPU mode by default
(backward compatible). Distributed features activate only when
torch.distributed is initialized and world_size > 1.

Conceptual references (not copied):
- PyTorch DistributedDataParallel official docs
- PyTorch FSDP (FullyShardedDataParallel) API
- Meta AI (2024) — Llama 3 distributed training patterns
- HuggingFace Accelerate — gradient accumulation design
- DeepSpeed — ZeRO optimization concepts (influence only)

Implementation is original.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger("xrfm.distributed")


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------


def is_distributed() -> bool:
    """Return True if torch.distributed is initialized and available."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    """Return the global rank of the current process, or 0 if not distributed."""
    if is_distributed():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    """Return total number of processes, or 1 if not distributed."""
    if is_distributed():
        return torch.distributed.get_world_size()
    return 1


def is_main_process() -> bool:
    """Return True if this is rank 0 (the main/coordinator process)."""
    return get_rank() == 0


def init_distributed(
    backend: str | None = None,
    timeout_seconds: int = 600,
) -> None:
    """Initialize the distributed process group.

    Reads MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK
    from environment (set by torchrun). Safe to call even if not
    distributed — does nothing if WORLD_SIZE == 1.

    Args:
        backend: NCCL (GPU), GLOO (CPU), or None for auto-detect.
        timeout_seconds: Process group timeout in seconds.
    """
    if not torch.distributed.is_available():
        logger.info("torch.distributed not available; running in single-process mode")
        return

    if torch.distributed.is_initialized():
        logger.info("torch.distributed already initialized (rank=%d)", get_rank())
        return

    # Check if we're in a torchrun-launched environment
    if "LOCAL_RANK" not in os.environ:
        logger.info("LOCAL_RANK not set; running in single-process mode")
        return

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    torch.distributed.init_process_group(
        backend=backend,
        timeout=torch.distributed.Timedelta(seconds=timeout_seconds),
    )

    # Set the default CUDA device for this process
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    logger.info(
        "Distributed initialized: rank=%d/%d, local_rank=%d, backend=%s, device=%s",
        get_rank(),
        get_world_size(),
        local_rank,
        backend,
        device,
    )


def cleanup_distributed() -> None:
    """Destroy the process group. Call at end of training."""
    if is_distributed():
        torch.distributed.destroy_process_group()
        logger.info("Distributed process group destroyed")


def barrier() -> None:
    """Synchronize all processes. No-op if not distributed."""
    if is_distributed():
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# Model wrapping
# ---------------------------------------------------------------------------


def wrap_model_ddp(
    model: nn.Module,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Wrap a model with DistributedDataParallel.

    In single-GPU mode, returns the model unchanged.
    Backward compatible — existing training code works with
    the returned model identically.

    Args:
        model: The PyTorch model (already on the correct device).
        find_unused_parameters: If True, DDP traverses the autograd
            graph to find params not used in loss. Slower but safer
            for models with conditional computation.

    Returns:
        DDP-wrapped model or original model if not distributed.
    """
    if not is_distributed():
        return model

    device = next(model.parameters()).device
    return nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
    )


def wrap_model_fsdp(
    model: nn.Module,
    sharding_strategy: str = "FULL_SHARD",
    cpu_offload: bool = False,
) -> nn.Module:
    """Wrap a model with FullyShardedDataParallel.

    FSDP shards model parameters, gradients, and optimizer states
    across GPUs, enabling training of larger models than DDP alone.

    Args:
        model: The PyTorch model.
        sharding_strategy: "FULL_SHARD" (ZeRO-3), "SHARD_GRAD_OP" (ZeRO-2),
            or "NO_SHARD" (DDP equivalent).
        cpu_offload: If True, offload parameters to CPU when not in use.

    Returns:
        FSDP-wrapped model or original model if not distributed.
    """
    if not is_distributed():
        return model

    try:
        from torch.distributed.fsdp import (
            CPUOffload,
            ShardingStrategy,
        )
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
        )

        strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }

        strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

        fsdp_kwargs = {"sharding_strategy": strategy}

        if cpu_offload:
            fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

        return FSDP(model, **fsdp_kwargs)

    except ImportError:
        logger.warning("FSDP not available in this PyTorch version. " "Falling back to DDP.")
        return wrap_model_ddp(model)


def get_raw_model(model: nn.Module) -> nn.Module:
    """Unwrap DDP/FSDP to get the original model.

    Args:
        model: Possibly DDP or FSDP wrapped model.

    Returns:
        The underlying GPTModel.
    """
    # Check for FSDP
    if hasattr(model, "_fsdp_wrapped_module"):
        return model._fsdp_wrapped_module

    # Check for DDP
    if hasattr(model, "module"):
        return model.module

    return model


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------


class GradientAccumulator:
    """Manages gradient accumulation for simulating larger batch sizes.

    Divides loss by accumulation_steps so that gradients from
    micro-batches are summed, effectively training with:
        effective_batch_size = per_device_batch_size * accumulation_steps

    When combined with DDP, uses no_sync() on non-sync steps to
    avoid unnecessary all-reduce communication.

    Usage:
        accum = GradientAccumulator(model, steps=4)
        for batch in dataloader:
            loss = model(batch) / accum.steps
            accum.backward(loss)
            if accum.should_step(step_idx):
                accum.step(optimizer)
    """

    def __init__(
        self,
        model: nn.Module,
        steps: int = 1,
    ) -> None:
        if steps <= 0:
            raise ValueError(f"accumulation_steps must be positive, got {steps}")
        self.model = model
        self.steps = steps
        self._counter: int = 0

    def should_sync(self) -> bool:
        """True on the last micro-batch of an accumulation cycle."""
        return (self._counter + 1) % self.steps == 0

    def no_sync_context(self):
        """Context manager that disables gradient sync for DDP.

        On micro-batches that aren't the last in an accumulation
        cycle, wraps forward in model.no_sync() to avoid all-reduce.
        On the final micro-batch (or if not DDP), grad sync is enabled.
        """
        if self.should_sync() or not self._is_ddp():
            return nullcontext()
        return self.model.no_sync()

    def _is_ddp(self) -> bool:
        return isinstance(self.model, nn.parallel.DistributedDataParallel)

    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass. Call after loss is divided by steps."""
        loss.backward()
        self._counter += 1

    def should_step(self) -> bool:
        """True when gradients are ready for optimizer step."""
        return self._counter > 0 and self._counter % self.steps == 0

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        scaler: Any | None = None,
    ) -> bool:
        """Perform optimizer step (and scheduler step).

        Resets counter after stepping.

        Args:
            optimizer: The PyTorch optimizer.
            scaler: Optional GradScaler for mixed precision.

        Returns:
            True if the step was performed.
        """
        if scaler is not None and hasattr(scaler, "step"):
            scaler.step(optimizer)
        else:
            optimizer.step()
        return True

    def reset(self) -> None:
        """Reset counter. Call after optimizer.zero_grad()."""
        self._counter = 0

    @property
    def current_step(self) -> int:
        return self._counter


# ---------------------------------------------------------------------------
# Distributed DataLoader
# ---------------------------------------------------------------------------


def create_distributed_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
    **dataloader_kwargs,
) -> DataLoader:
    """Create a DataLoader with optional DistributedSampler.

    When distributed, each process gets a unique subset of data.
    set_epoch() must be called on the sampler each epoch.

    Args:
        dataset: The PyTorch Dataset.
        batch_size: Per-device batch size.
        shuffle: Shuffle data (handled by DistributedSampler).
        num_workers: Data loading worker processes.
        pin_memory: Pin memory for faster GPU transfer.
        drop_last: Drop incomplete last batch.

    Returns:
        DataLoader with DistributedSampler if distributed, else
        standard DataLoader.
    """
    if is_distributed():
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **dataloader_kwargs,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **dataloader_kwargs,
        )


# ---------------------------------------------------------------------------
# Distributed logging & metrics
# ---------------------------------------------------------------------------


def reduce_loss(loss: torch.Tensor, device: torch.device | None = None) -> float:
    """Average a loss value across all distributed processes.

    Args:
        loss: Scalar loss tensor (on any device).
        device: Device for the reduction tensor.

    Returns:
        Average loss across all ranks as a Python float.
    """
    if not is_distributed():
        return loss.item()

    loss_tensor = torch.tensor([loss.item()], device=device or loss.device)
    torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
    return loss_tensor.item() / get_world_size()


# ---------------------------------------------------------------------------
# FSDP checkpoint helpers
# ---------------------------------------------------------------------------


def save_fsdp_checkpoint(
    model: nn.Module,
    path: str,
    rank0_only: bool = True,
) -> None:
    """Save a full (consolidated) checkpoint from an FSDP model.

    Args:
        model: FSDP-wrapped model.
        path: Save path.
        rank0_only: If True, only rank 0 saves to disk.
    """
    try:
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            StateDictType,
        )
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
        )

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = model.state_dict()
            if not rank0_only or get_rank() == 0:
                torch.save(state_dict, path)
                logger.info("FSDP full checkpoint saved: %s", path)
    except ImportError:
        logger.warning("FSDP not available; falling back to standard save")
        torch.save(get_raw_model(model).state_dict(), path)


def load_fsdp_checkpoint(
    model: nn.Module,
    path: str,
) -> None:
    """Load a full checkpoint into an FSDP model.

    Args:
        model: FSDP-wrapped model.
        path: Checkpoint path.
    """
    try:
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            StateDictType,
        )
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
        )

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            logger.info("FSDP checkpoint loaded: %s", path)
    except ImportError:
        logger.warning("FSDP not available; falling back to standard load")
        get_raw_model(model).load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
