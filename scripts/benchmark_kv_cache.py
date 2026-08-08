"""
XRFM KV-cache benchmark (Phase 32).

Compares autoregressive generation WITH KV cache vs WITHOUT (recompute full
prefix every step), using the SAME model and prompts.

Measures:
  - prompt length
  - generated tokens
  - latency per token (ms)
  - tokens/sec
  - peak VRAM (CUDA only)

Reproducible: fixed model config, fixed tokenizer, fixed prompts, fixed seeds,
fixed number of repetitions.

Usage:
    python scripts/benchmark_kv_cache.py [--checkpoint path] [--prompt_len 32]
                                         [--new_tokens 32] [--reps 3]
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402


def main(checkpoint: str | None, prompt_len: int, new_tokens: int, reps: int, device: str) -> dict:
    from model.gpt import GPTModel
    from tokenizer.bpe import BytePairEncoder

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Tokenizer + model (checkpoint optional: benchmark is shape/throughput,
    # so a randomly-initialized model is acceptable and deterministic).
    tok = BytePairEncoder()
    if os.path.exists("tokenizer/vocab.json"):
        tok.load("tokenizer/vocab.json")
    model = GPTModel("config/config.yaml", vocab_size=tok.vocab_size()).to(dev)
    if checkpoint and os.path.exists(checkpoint):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        print(f"[bench] loaded checkpoint {checkpoint} (step={ck.get('step')})")
    model.eval()

    torch.manual_seed(0)
    prompt = torch.randint(0, tok.vocab_size() - 4, (prompt_len,), device=dev)
    prompt_ids = prompt.unsqueeze(0)

    results: dict = {"device": str(dev), "prompt_len": prompt_len, "new_tokens": new_tokens}

    # ---------------------------------------------------------------
    # 1. WITH KV cache (GenerationEngine)
    # ---------------------------------------------------------------
    from inference.engine import GenerationEngine

    engine = GenerationEngine(model)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        with torch.no_grad():
            engine.generate(prompt_ids, max_new_tokens=new_tokens, temperature=0.0)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    with_cache = sum(times) / len(times)
    results["with_cache"] = {
        "avg_s": round(with_cache, 4),
        "ms_per_token": round(with_cache / new_tokens * 1000, 2),
        "tokens_per_sec": round(new_tokens / with_cache, 1),
    }

    # ---------------------------------------------------------------
    # 2. WITHOUT KV cache (recompute full prefix each step)
    # ---------------------------------------------------------------
    times_no = []
    for _ in range(reps):
        gen = prompt_ids.clone()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(new_tokens):
                logits, _ = model(gen, use_cache=False)
                nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen = torch.cat([gen, nxt], dim=1)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times_no.append(time.perf_counter() - t0)
    without_cache = sum(times_no) / len(times_no)
    results["without_cache"] = {
        "avg_s": round(without_cache, 4),
        "ms_per_token": round(without_cache / new_tokens * 1000, 2),
        "tokens_per_sec": round(new_tokens / without_cache, 1),
    }
    results["speedup"] = round(without_cache / with_cache, 2)

    # ---------------------------------------------------------------
    # 3. Peak VRAM (CUDA only)
    # ---------------------------------------------------------------
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        with torch.no_grad():
            engine.generate(prompt_ids, max_new_tokens=new_tokens, temperature=0.0)
        results["peak_vram_mb"] = round(torch.cuda.max_memory_allocated(dev) / 1e6, 1)
    else:
        results["peak_vram_mb"] = None  # not measurable on CPU

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--prompt_len", type=int, default=32)
    ap.add_argument("--new_tokens", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default=None, help="JSON output path")
    args = ap.parse_args()
    res = main(args.checkpoint, args.prompt_len, args.new_tokens, args.reps, args.device)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"saved {args.out}")
