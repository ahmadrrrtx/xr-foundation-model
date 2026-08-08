# XRFM Distributed Training Guide — v0.8.0

> **AUDIT REMEDIATION NOTE (2026-08-08):** this document describes the original
> design. A forensic audit found and fixed several issues (implicit causal
> masking, character-level tokenizer, padding-loss, resume/scheduler state,
> API import, version chaos). See `docs/audit/FORENSIC_AUDIT.md`,
> `docs/audit/GAP_ANALYSIS.md`, and `docs/implementation/REMEDIATION_PLAN.md`
> for the authoritative current state. Historical claims below are preserved
> as evidence, not as current truth.


## Overview

XRFM supports distributed training across multiple GPUs using PyTorch's
DistributedDataParallel (DDP) and FullyShardedDataParallel (FSDP).

All distributed features are **opt-in**: single-GPU/single-process mode
is the default and works identically to previous versions.

## Architecture

```
torchrun
  ├── Rank 0 (GPU 0) ── DataLoader (shard 0) ── Model Replica 0
  ├── Rank 1 (GPU 1) ── DataLoader (shard 1) ── Model Replica 1
  ├── ...
  └── Rank N (GPU N) ── DataLoader (shard N) ── Model Replica N
                                  │
                          All-Reduce Gradients
                                  │
                          Synchronized Weights
```

## Components

| Module | Purpose |
|---|---|
| `training/distributed.py` | DDP/FSDP wrappers, gradient accumulation, distributed sampler, utilities |
| `training/loop.py` | Updated training loop with no_sync(), reduced logging, barrier checkpointing |
| `scripts/torchrun_launch.sh` | Launch script for torchrun |

## Quick Start

### Single-GPU (default — no changes needed)

```python
from training.loop import TrainingLoop

loop = TrainingLoop(model=model, dataset=dataset)
loop.training_loop()
```

### Multi-GPU with DDP

```bash
# config/config.yaml
training:
  use_ddp: true
  batch_size: 16        # per-GPU batch size
  grad_accum_steps: 4   # effective batch = 16 * 4 * N_GPUS

# Launch
torchrun --nproc_per_node=4 -m training.loop
```

### Multi-GPU with FSDP (large models)

```bash
# config/config.yaml
training:
  use_fsdp: true
  use_ddp: false
  batch_size: 4         # smaller per-GPU batch for large models
  grad_accum_steps: 8

# Launch
torchrun --nproc_per_node=8 -m training.loop
```

## Gradient Accumulation

Simulate larger batch sizes without increasing GPU memory:

```
Effective Batch Size = per_device_batch × grad_accum_steps × world_size
```

Example: `batch_size=8, grad_accum_steps=4, world_size=4` → effective batch = 128.

The training loop:
1. Divides loss by `grad_accum_steps` on each micro-batch
2. Uses `model.no_sync()` on intermediate DDP micro-batches (no all-reduce)
3. Only syncs gradients + steps optimizer on the last micro-batch

## Checkpointing in Distributed Mode

- **Save:** Only rank 0 writes to disk. `barrier()` ensures all ranks sync.
- **Load:** All ranks load the checkpoint. `barrier()` prevents races.
- **Raw model access:** Use `get_raw_model(model)` to unwrap DDP/FSDP.

## Performance Considerations

| Config | When to Use |
|---|---|
| DDP | Up to ~8 GPUs, model fits on single GPU |
| FSDP | Larger models that don't fit on single GPU |
| Gradient Accumulation | Limited GPU memory, need larger effective batches |
| DDP + Grad Accum | Combine both for maximum effective batch size |

## Future Extensions (deferred)

- DeepSpeed ZeRO integration (Phase 9+)
- Gradient checkpointing (Phase 9)
- Pipeline parallelism (Phase 10+)
- Multi-node training (Phase 10+)

## References

- PyTorch DDP documentation
- PyTorch FSDP documentation
- Meta AI — Llama 3 training infrastructure
- Li et al. (2020) — PyTorch Distributed
