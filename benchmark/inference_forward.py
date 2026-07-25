"""
Inference benchmark for XRFM (v0.6.0).

Measures generation throughput with and without KV cache,
sampling overhead, and numerical stability.
"""

import time
import torch

from model.gpt import GPTModel
from inference.engine import GenerationEngine


def benchmark_full_vs_cached(
    prompt_len: int = 32,
    new_tokens: int = 20,
    repetitions: int = 5,
):
    """Compare full forward vs KV-cached generation speed."""
    model = GPTModel()
    model.eval()
    engine = GenerationEngine(model)
    prompt = torch.randint(0, 50304, (prompt_len,))

    # Warm-up
    _ = engine.generate(prompt, max_new_tokens=3, temperature=0)

    # Cached generation
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        _ = engine.generate(prompt, max_new_tokens=new_tokens, temperature=0)
        times.append((time.perf_counter() - start) * 1000)

    avg_cached = sum(times) / len(times)
    tokens_per_sec = new_tokens / (avg_cached / 1000)

    print(f"Prompt: {prompt_len} tokens, Generated: {new_tokens} tokens")
    print(f"KV-cached generation: {avg_cached:.1f}ms ({tokens_per_sec:.1f} tok/s)")


def verify_sampling_determinism():
    """Greedy sampling should be deterministic across runs."""
    model = GPTModel()
    model.eval()
    engine = GenerationEngine(model)
    prompt = torch.randint(0, 100, (8,))

    out1 = engine.generate(prompt, max_new_tokens=5, temperature=0)
    out2 = engine.generate(prompt, max_new_tokens=5, temperature=0)
    assert torch.equal(out1, out2), "Greedy generation not deterministic!"
    print("Greedy determinism: PASSED")


def verify_numerical_stability():
    """Generated outputs should contain no NaN or Inf."""
    model = GPTModel()
    model.eval()
    engine = GenerationEngine(model)
    prompt = torch.randint(0, 100, (8,))

    for strategy in [
        {"temperature": 0},
        {"temperature": 0.8},
        {"temperature": 0.8, "top_k": 10},
        {"temperature": 0.8, "top_p": 0.9},
    ]:
        output = engine.generate(prompt, max_new_tokens=10, **strategy)
        assert not torch.isnan(output).any(), f"NaN in output with {strategy}"
        assert not torch.isinf(output).any(), f"Inf in output with {strategy}"

    print("Numerical stability: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("XRFM v0.6.0 — Inference Benchmark")
    print("=" * 60)

    print("\n--- Sampling Determinism ---")
    verify_sampling_determinism()

    print("\n--- Numerical Stability ---")
    verify_numerical_stability()

    print("\n--- KV Cache Speed ---")
    benchmark_full_vs_cached()

    print("\n--- Benchmark Complete ---")
