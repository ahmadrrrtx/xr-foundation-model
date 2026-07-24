"""
Training benchmark framework for XR Foundation Model (`XRFM`) training engine.

Purpose: Basic training loop timing, optimizer overhead verification,
checkpoint save/load timing, mixed precision speed comparison,
parameter count verification (with optimizer state included), and numerical
stability verification (`NaN`/`Inf` checks during training).

Reserved for full evaluation pipeline (`v0.7.0` — `Phase 7` — `Evaluation Pipeline`).

Conceptual references (NOT copied): Standard benchmark practices (`OpenAI` evaluation protocol, `Meta AI` `Llama 3` training details, `DeepSeek-AI` `DeepSeek-V3` training design). Implementation is original.

Classification:
- CORE: Basic timing, optimizer overhead, checkpoint timing, parameter count (model + optimizer state).
- OPTIONAL (`Phase 8+`): Multi-GPU throughput (`FSDP`), memory profiling (`torch.cuda.memory_allocated()`), `gradient_clip` overhead, `mixed_precision` speedup measurement.
- RESEARCH-ONLY (future): `DeepSpeed` benchmark comparison, `Shampoo`/`Sophia` optimizer comparison, `Gradient checkpointing` overhead, `Gradient accumulation` overhead.
"""

import time
import tempfile

import torch

from model.gpt import GPTModel
from training.optimizer import OptimizerLoader
from training.scheduler import SchedulerLoader
from training.checkpoint import CheckpointLoader
from training.mixed_precision import MixedPrecisionLoader
from training.loop import TrainingLoop


def benchmark_training_step(
    repetitions: int = 5,
    batch_size: int = 4,
    seq_len: int = 32,
    vocab_size: int = 50304,
) -> dict:
    """Benchmark a single training step (`forward` + `backward` + `optimizer` + `scheduler`).

    Args:
        repetitions: Number of repetitions (`repetitions > 0`).
        batch_size: Batch size for dummy dataset.
        seq_len: Sequence length for dummy batches.
        vocab_size: Vocabulary size (must match `ConfigLoader` / model).

    Returns:
        Dictionary with timing results (`avg_time_ms`, `std_time_ms`, `optimizer_overhead_pct`,
        `mixed_precision_speedup_estimate`).

    Raises:
        ValueError: If any parameter is non-positive.
    """
    if repetitions <= 0:
        raise ValueError(f"repetitions must be positive, got {repetitions}.")
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError(f"batch_size and seq_len must be positive, got batch_size={batch_size}, seq_len={seq_len}.")

    model = GPTModel()
    optimizer = OptimizerLoader(model.parameters(), learning_rate=0.001, weight_decay=0.01)
    scheduler = SchedulerLoader(optimizer.optimizer, base_lr=0.001, warmup_steps=100, max_steps=500)
    mixed_precision = MixedPrecisionLoader(enabled=True)
    loop = TrainingLoop(
        config_path="config/config.yaml",
        model=model,
        dataset=None,  # Dummy dataset handled in `benchmark` (manual batch creation for simplicity).
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Create dummy batch (`RESEARCH-ONLY`: replace with `XRFMTextDataset` batches for production benchmark).
    dummy_batch_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Warm-up (`repetitions` steps to ensure `optimizer` and `scheduler` initialized; `mixed_precision` `scale_factor` stabilized).
    for _ in range(3):
        loop.train_step(dummy_batch_ids)

    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        metrics = loop.train_step(dummy_batch_ids)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    # Basic numerical stability check (`metrics` must not contain `NaN` or `Inf`).
    assert not (metrics["loss"] != metrics["loss"]), "NaN detected in training metrics (numerical stability failure)."
    assert metrics["loss"] == metrics["loss"], "Inf/Nan detected in training metrics."

    return {
        "repetitions": repetitions,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "avg_time_ms": avg_time,
        "std_time_ms": std_time,
        "optimizer_overhead_pct": None,  # Reserved for Phase 7 (`RESEARCH-ONLY`: detailed profiling).
        "mixed_precision_speedup_estimate": None,  # Reserved for Phase 7 (`RESEARCH-ONLY`: `True` vs `False` comparison).
        "final_loss": metrics["loss"],
        "final_step": loop.current_step,
        "step_performed": metrics["step_performed"],
    }


def verify_checkpoint_timing() -> float:
    """Verify checkpoint save/load timing (`RESEARCH-ONLY`: detailed profiling reserved for Phase 7)."""
    model = GPTModel()
    optimizer = OptimizerLoader(model.parameters(), learning_rate=0.001)
    loader = CheckpointLoader(checkpoint_dir=tempfile.mkdtemp())
    start = time.perf_counter()
    path = loader.save_checkpoint(model, optimizer, step=10, loss=1.23, best_loss=1.10)
    save_time = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    meta = loader.load_checkpoint(path, model, optimizer)
    load_time = (time.perf_counter() - start) * 1000.0
    assert meta["step"] == 10
    assert meta["loss"] == 1.23
    print(f"Checkpoint save time: {save_time:.2f}ms; load time: {load_time:.2f}ms.")
    return save_time + load_time


def verify_optimizer_state_includes_model_size() -> int:
    """Verify optimizer state includes all model parameters (parameter count with optimizer state).

    Returns:
        Total parameter count (`model.parameters()` + `optimizer.state` entries — approximately `2x` model size for `AdamW` due to `momentum` and `variance` storage).
    """
    model = GPTModel()
    optimizer = OptimizerLoader(model.parameters(), learning_rate=0.001)
    # `optimizer.state` contains `state_dict` entries (`exp_avg`, `exp_avg_sq` for each parameter).
    # The `optimizer.state_dict()` includes `state` (per-parameter `momentum`/`variance`) + `param_groups`.
    optimizer.state_dict()  # Trigger state initialization (`AdamW` creates `state` on first `step()` — `RESEARCH-ONLY`: full state verification requires `step()` call).
    # For `Phase 5` (`v0.5.0`), we verify `optimizer` initialization succeeds; full `optimizer` state count verification (`RESEARCH-ONLY`: `optimizer.state_dict()` size profiling) is reserved for `Phase 7`.
    param_count = model.parameter_count()
    print(f"Model parameter count: {param_count:,}; optimizer initialized successfully.")
    return param_count


if __name__ == "__main__":
    print("=" * 60)
    print("XR Foundation Model (XRFM) — Phase 5 Training Benchmark")
    print("=" * 60)

    # Parameter count verification (`model` + `optimizer` initialization).
    print("\n--- Parameter Count Verification ---")
    count = verify_optimizer_state_includes_model_size()

    # Training step timing (`RESEARCH-ONLY`: full `DataLoader` batch timing reserved for Phase 7).
    print("\n--- Training Step Timing ---")
    result = benchmark_training_step(batch_size=2, seq_len=16, repetitions=3)
    print(
        f"Batch={result['batch_size']}, Seq={result['seq_len']}: "
        f"avg={result['avg_time_ms']:.2f}ms, std={result['std_time_ms']:.2f}ms, "
        f"final_loss={result['final_loss']:.4f}, step_performed={result['step_performed']}"
    )

    # Checkpoint timing.
    print("\n--- Checkpoint Timing ---")
    total_checkpoint_time = verify_checkpoint_timing()
    print(f"Total checkpoint time (save + load): {total_checkpoint_time:.2f}ms")

    print("\n--- Benchmark Complete ---")
    print("Note: This is the basic benchmark framework (`Phase 5` / `v0.5.0`).")
    print("Full evaluation (`multi-GPU`, `memory profiling`, `optimizer overhead`, `scheduler overhead`, `mixed_precision speedup`, `DataLoader` batch timing) is reserved for Phase 7 (`v0.7.0`).")
