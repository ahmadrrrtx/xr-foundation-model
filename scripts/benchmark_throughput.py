"""
XRFM throughput benchmark (Phase 37).

Measures forward+backward training-step throughput for a given config across
sequence lengths and batch sizes, reporting:
  - tokens/sec
  - step time (ms)
  - peak VRAM (CUDA only)

Reproducible: fixed config, fixed seed, fixed warmup/trials, greedy-free
(random-logits training step with real loss/backward).

Usage:
    python scripts/benchmark_throughput.py --config config/v1.1-medium.yaml \
        --seq_lens 512 1024 --batch_sizes 8 16 [--device cuda]
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F

from model.gpt import GPTModel


def bench_one(config_path: str, vocab: int, seq_len: int, batch: int, device, steps: int) -> dict:
    torch.manual_seed(0)
    model = GPTModel(config_path, vocab_size=vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, vocab, (batch, seq_len), device=device)
    y = torch.randint(0, vocab, (batch, seq_len), device=device)

    # warmup
    for _ in range(2):
        opt.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        loss.backward()
        opt.step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tokens = batch * seq_len * steps
    return {
        "seq_len": seq_len,
        "batch": batch,
        "steps": steps,
        "step_ms": round(dt / steps * 1000, 1),
        "tokens_per_sec": round(tokens / dt, 1),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1) if device.type == "cuda" else None,
    }


def main(
    config_path: str,
    seq_lens: list[int],
    batch_sizes: list[int],
    device_name: str | None,
    steps: int,
    vocab: int,
):
    dev = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {dev}  config: {config_path}  steps/run: {steps}  vocab: {vocab}")

    model = GPTModel(config_path, vocab_size=vocab)
    print(f"params: {model.parameter_count():,} ({model.parameter_count() / 1e6:.2f}M)")

    results = []
    for seq in seq_lens:
        for bs in batch_sizes:
            try:
                r = bench_one(config_path, vocab, seq, bs, dev, steps)
                results.append(r)
                vram = f"vram={r['peak_vram_mb']}MB" if r["peak_vram_mb"] else "vram=n/a"
                print(
                    f"  seq={seq:5d} batch={bs:3d}: {r['step_ms']:7.1f} ms/step | "
                    f"{r['tokens_per_sec']:>9,.0f} tok/s | {vram}"
                )
            except (MemoryError, RuntimeError) as e:
                # Report memory-blocked configurations instead of dying
                # (e.g. a 1.9GB sandbox cannot hold seq>=1024 activations).
                print(f"  seq={seq:5d} batch={bs:3d}: MEMORY BLOCKED ({type(e).__name__})")
                results.append(
                    {
                        "seq_len": seq,
                        "batch": bs,
                        "steps": 0,
                        "step_ms": None,
                        "tokens_per_sec": None,
                        "peak_vram_mb": None,
                        "blocked": str(e)[:80],
                    }
                )

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/v1.1-medium.yaml")
    ap.add_argument("--seq_lens", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--vocab", type=int, default=8192, help="vocab for model build")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = main(args.config, args.seq_lens, args.batch_sizes, args.device, args.steps, args.vocab)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print("saved", args.out)
