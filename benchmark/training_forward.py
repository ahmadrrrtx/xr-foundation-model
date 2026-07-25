"""
Training benchmark for XRFM (v0.7.0).

Measures training step timing, checkpoint I/O, and numerical stability.
Uses a minimal DummyDataset for benchmark isolation.
"""

import time
import tempfile

import torch

from model.gpt import GPTModel
from training.optimizer import OptimizerLoader
from training.scheduler import SchedulerLoader
from training.checkpoint import CheckpointLoader
from training.loop import TrainingLoop


class DummyDataset:
    """Minimal dataset for benchmark — provides (input_ids, target_ids) tuples."""

    def __init__(self, vocab_size: int = 50304, seq_len: int = 32, size: int = 100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return ids[:-1], ids[1:]  # input, shifted target


def benchmark_training_step(
    repetitions: int = 5,
    batch_size: int = 4,
    seq_len: int = 32,
    vocab_size: int = 50304,
) -> dict:
    if repetitions <= 0 or batch_size <= 0 or seq_len <= 0:
        raise ValueError("all parameters must be positive")

    model = GPTModel()
    optimizer = OptimizerLoader(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = SchedulerLoader(optimizer.optimizer, base_lr=0.001, warmup_steps=100, max_steps=500)
    dataset = DummyDataset(vocab_size=vocab_size, seq_len=seq_len)

    loop = TrainingLoop(
        config_path="config/config.yaml",
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Create a batch for isolated step timing
    batch_input = torch.randint(0, vocab_size, (batch_size, seq_len - 1))
    batch_target = torch.randint(0, vocab_size, (batch_size, seq_len - 1))

    # Warm-up
    for _ in range(3):
        loop.train_step(batch_input, batch_target)

    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        metrics = loop.train_step(batch_input, batch_target)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    assert metrics["loss"] == metrics["loss"], "NaN/Inf in training metrics"

    return {
        "repetitions": repetitions,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "avg_time_ms": avg_time,
        "std_time_ms": std_time,
        "final_loss": metrics["loss"],
        "final_step": loop.current_step,
        "step_performed": metrics["step_performed"],
    }


def verify_checkpoint_timing() -> float:
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
    print(f"Checkpoint save: {save_time:.2f}ms, load: {load_time:.2f}ms")
    return save_time + load_time


def verify_parameter_count() -> int:
    model = GPTModel()
    count = model.parameter_count()
    print(f"Model parameters: {count:,}")
    return count


if __name__ == "__main__":
    print("=" * 60)
    print("XRFM v0.7.0 — Training Benchmark")
    print("=" * 60)

    print("\n--- Parameter Count ---")
    verify_parameter_count()

    print("\n--- Training Step Timing ---")
    result = benchmark_training_step(batch_size=2, seq_len=16, repetitions=3)
    print(
        f"Batch={result['batch_size']}, Seq={result['seq_len']}: "
        f"avg={result['avg_time_ms']:.2f}ms, std={result['std_time_ms']:.2f}ms, "
        f"loss={result['final_loss']:.4f}"
    )

    print("\n--- Checkpoint Timing ---")
    total = verify_checkpoint_timing()
    print(f"Total checkpoint time: {total:.2f}ms")

    print("\n--- Benchmark Complete ---")
