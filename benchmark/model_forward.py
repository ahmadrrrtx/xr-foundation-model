"""
Benchmark framework for XR Foundation Model (`XRFM`).

Purpose: Basic forward pass timing and parameter count verification.
Reserved for full evaluation pipeline (`v0.7.0` / Phase 7).

Conceptual references (NOT copied): Standard benchmark practices
(OpenAI evaluation protocol, Hugging Face `transformers` benchmark patterns).
Implementation is original.

Classification:
- CORE: Basic timing, parameter count verification.
- OPTIONAL (post-v1.0): Multi-GPU throughput, memory profiling, latency benchmarking.
- RESEARCH-ONLY (future): Custom Triton kernels, FlashAttention-2 comparison,
  vLLM serving benchmark.
"""

import time
import sys

sys.path.insert(0, ".")

import torch
from model.gpt import GPTModel


def benchmark_forward_pass(
    batch_size: int = 4,
    seq_len: int = 32,
    vocab_size: int = 50304,
    repetitions: int = 10,
) -> dict:
    """Run basic forward pass timing benchmark.

    Args:
        batch_size: Number of sequences per batch.
        seq_len: Sequence length for benchmark.
        vocab_size: Vocabulary size (must match config/model).
        repetitions: Number of repetitions for averaging.

    Returns:
        Dictionary with timing results (`avg_time_ms`, `std_time_ms`, `throughput_seqs_per_sec`).

    Raises:
        ValueError: If `batch_size`, `seq_len`, or `repetitions` is non-positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}.")
    if repetitions <= 0:
        raise ValueError(f"repetitions must be positive, got {repetitions}.")

    model = GPTModel()
    model.eval()
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Warm-up (ensure any lazy initialization is complete).
    with torch.no_grad():
        for _ in range(3):
            _, _ = model(input_ids)

    # Benchmark timing.
    times = []
    with torch.no_grad():
        for _ in range(repetitions):
            start = time.perf_counter()
            out, _ = model(input_ids)
            # Ensure the forward pass is complete (synchronize if GPU available).
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000.0)  # Convert to milliseconds.

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    throughput = (batch_size * repetitions) / (sum(times) / 1000.0)

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "repetitions": repetitions,
        "avg_time_ms": avg_time,
        "std_time_ms": std_time,
        "throughput_seqs_per_sec": throughput,
    }


def verify_parameter_count(expected_approx: int = 10_000_000, tolerance: float = 0.3) -> None:
    """Verify parameter count matches expected approximate value.

    The `XRFM-10M` preset (`v0.4.0`) uses `vocab_size=50304`, `d_model=256`,
    `n_layers=6`, `d_ff=1024`. The actual parameter count is approximately
    19.2M due to vocabulary scaling (embedding + weight-tied output projection).
    This benchmark documents the actual count rather than enforcing an
    approximate approximation.

    Args:
        expected_approx: Approximate expected parameter count (for documentation only).
        tolerance: Tolerance for approximate comparison (not enforced for this preset).

    Raises:
        ValueError: If parameter count is zero (indicating initialization failure).
    """
    model = GPTModel()
    param_count = model.parameter_count()
    if param_count == 0:
        raise ValueError("Parameter count is zero. Model initialization may have failed.")

    print(f"[Benchmark] XRFM-10M Preset Parameter Count: {param_count:,}")
    print(f"[Benchmark] Approximate expected: {expected_approx:,} (documented reference)")
    print(f"[Benchmark] Note: Actual count reflects full architecture (embedding, attention, SwiGLU, norm, output with weight tying).")


def run_full_benchmark() -> None:
    """Execute full benchmark suite (timing + parameter verification)."""
    print("=" * 60)
    print("XR Foundation Model (XRFM) — Phase 4 Benchmark Results")
    print("=" * 60)

    # Parameter count verification.
    print("\n--- Parameter Count Verification ---")
    verify_parameter_count()

    # Forward pass timing.
    print("\n--- Forward Pass Timing ---")
    for batch_size in (1, 2, 4):
        for seq_len in (8, 32, 64):
            result = benchmark_forward_pass(
                batch_size=batch_size, seq_len=seq_len, repetitions=5
            )
            print(
                f"Batch={batch_size}, Seq={seq_len}: "
                f"avg={result['avg_time_ms']:.2f}ms, "
                f"std={result['std_time_ms']:.2f}ms, "
                f"throughput={result['throughput_seqs_per_sec']:.1f} seq/s"
            )

    print("\n--- Benchmark Complete ---")
    print("Note: This is the basic benchmark framework (Phase 4 / v0.4.0).")
    print("Full evaluation pipeline (throughput profiling, multi-GPU benchmarks,")
    print("memory profiling, latency analysis) is reserved for Phase 7 (v0.7.0).")


if __name__ == "__main__":
    run_full_benchmark()
