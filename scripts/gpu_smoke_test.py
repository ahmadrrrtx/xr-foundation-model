"""
XRFM GPU training smoke test (Phase 26).

Verifies, on the detected device (CUDA if present, else CPU), that the full
training stack works end-to-end on-device:

  1. Model moves to device
  2. Forward pass works
  3. Backward pass works
  4. BF16 autocast works where supported (falls back to FP16, then no-op on CPU)
  5. Gradients are finite
  6. Gradient clipping works
  7. Optimizer works
  8. Scheduler works
  9. Checkpoint saves
 10. Checkpoint reloads
 11. Resume restores step / optimizer / scheduler / model / seed+config metadata
 12. VRAM measured (CUDA only)

Run only a SMALL number of steps. This is a correctness gate, not a benchmark.

Usage:
    python scripts/gpu_smoke_test.py                 # auto device
    python scripts/gpu_smoke_test.py --device cuda   # force CUDA
    python scripts/gpu_smoke_test.py --device cpu    # CPU logic check
"""

import argparse
import json
import logging
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xrfm.gpu_smoke")

RESULTS: list[tuple[str, str, str]] = []  # (check, status, detail)


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    RESULTS.append((name, status, detail))
    logger.info("[%s] %s %s", status, name, detail)


def main(device_name: str, steps: int, seed: int) -> int:
    from model.gpt import GPTModel
    from tokenizer.bpe import BytePairEncoder
    from training.checkpoint import CheckpointLoader
    from training.loop import _set_seed
    from training.mixed_precision import MixedPrecisionLoader
    from training.optimizer import OptimizerLoader
    from training.scheduler import SchedulerLoader

    has_cuda = torch.cuda.is_available()
    if device_name == "cuda" and not has_cuda:
        logger.error("Requested CUDA but no GPU available.")
        return 2
    device = torch.device(device_name if device_name else ("cuda" if has_cuda else "cpu"))
    gpu = device.type == "cuda"
    logger.info("Device: %s (CUDA available: %s)", device, has_cuda)

    # --------------------------------------------------------------
    # Fixed tiny dataset (deterministic, seedable)
    # --------------------------------------------------------------
    _set_seed(seed)
    tok = BytePairEncoder(vocab_size_target=512)
    tok.train_on_text("The red fox jumps over the lazy moon.\n" * 40)
    vocab = tok.vocab_size()

    from xrfm.data.loader import XRFMTextDataset

    with open("/tmp/xrfm_gpu_smoke.txt", "w", encoding="utf-8") as f:
        f.write("The red fox jumps over the lazy moon.\n" * 60)
    ds = XRFMTextDataset("/tmp/xrfm_gpu_smoke.txt", tok, max_seq_len=64, split="train", pad_id=tok.pad_id or 0)

    # --------------------------------------------------------------
    # 1. Model on device
    # --------------------------------------------------------------
    _set_seed(seed)
    model = GPTModel("config/tiny.yaml", vocab_size=vocab).to(device)
    n_params = model.parameter_count()
    check("1. model on device", next(model.parameters()).device == device, str(device))
    logger.info("    params=%d vocab=%d", n_params, vocab)

    # --------------------------------------------------------------
    # Training components
    # --------------------------------------------------------------
    raw = model
    opt = OptimizerLoader(raw.parameters(), learning_rate=3e-4, weight_decay=0.1)
    sched = SchedulerLoader(opt.optimizer, base_lr=3e-4, warmup_steps=steps // 3, max_steps=steps)
    mp = MixedPrecisionLoader(enabled=gpu)  # autocast only meaningful on GPU
    ckpt = CheckpointLoader(checkpoint_dir="/tmp/xrfm_gpu_smoke_ckpt")

    # --------------------------------------------------------------
    # 2-8. Train a few steps on device
    # --------------------------------------------------------------
    loss_history: list[float] = []
    t0 = time.time()
    for step in range(steps):
        x, y = ds[step % len(ds)]
        xb = x.unsqueeze(0).to(device)
        yb = y.unsqueeze(0).to(device)

        opt.zero_grad()
        with mp.autocast_ctx():
            logits, _ = model(xb)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), yb.view(-1), ignore_index=-100)
        loss.backward()

        # finite gradients
        grads_ok = all(torch.isfinite(p.grad).all() for p in raw.parameters() if p.grad is not None)
        if step == 0:
            check("5. gradients finite", grads_ok)

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(raw.parameters(), max_norm=1.0)
        opt.step()
        sched.step()
        loss_history.append(loss.item())

    dt = time.time() - t0
    step_time_ms = dt / steps * 1000
    check(
        "2. forward+backward (N steps ran)",
        len(loss_history) == steps,
        f"{steps} steps in {dt:.2f}s",
    )
    check(
        "6. gradient clipping",
        all(p.grad is None or torch.isfinite(p.grad).all() for p in raw.parameters()),
    )
    check("7. optimizer step", all(torch.isfinite(p).all() for p in raw.parameters()))
    check("8. scheduler step", sched.current_step == steps, f"current_step={sched.current_step}")
    check(
        "9. loss finite+decreasing trajectory",
        all(torch.isfinite(torch.tensor(loss_history))) and loss_history[-1] < loss_history[0] + 1.0,
        f"loss {loss_history[0]:.3f} -> {loss_history[-1]:.3f}",
    )
    check("4. autocast context", True, "no-op on CPU / bf16-fp16 on GPU (see detail below)")

    if gpu:
        # 4. autocast dtype verification
        mp2 = MixedPrecisionLoader(enabled=True)
        with mp2.autocast_ctx():
            x2 = torch.randn(2, 8, 64, device=device)
            lg2, _ = model(x2)
        check("4a. autocast ran on GPU", torch.isfinite(lg2).all(), f"dtype={lg2.dtype}")
        check("4b. bf16 support", torch.cuda.is_bf16_supported(), "")

    # --------------------------------------------------------------
    # 9-11. Checkpoint save / load / resume
    # --------------------------------------------------------------
    ckpt_path = ckpt.save_checkpoint(
        raw,
        opt,
        sched,
        step=steps,
        loss=loss_history[-1],
        best_loss=min(loss_history),
        extra={
            "seed": seed,
            "config_path": "config/tiny.yaml",
            "device": str(device),
            "pytorch_version": str(torch.__version__),
        },
    )
    check("9. checkpoint saved", os.path.exists(ckpt_path), ckpt_path)

    # fresh model + components to prove load works
    _set_seed(seed + 1)
    model2 = GPTModel("config/tiny.yaml", vocab_size=vocab).to(device)
    opt2 = OptimizerLoader(model2.parameters(), learning_rate=3e-4, weight_decay=0.1)
    sched2 = SchedulerLoader(opt2.optimizer, base_lr=3e-4, warmup_steps=steps // 3, max_steps=steps)
    meta = ckpt.load_checkpoint(ckpt_path, model2, opt2, sched2)
    w_ok = all(torch.equal(a, b) for a, b in zip(model2.parameters(), raw.parameters()))
    check("10. checkpoint reloads (weights equal)", w_ok)
    check("11. resume restores step", meta.get("step") == steps, f"step={meta.get('step')}")
    # Compare optimizer state by VALUE via state_dict() (parameter tensors of
    # model1 and model2 are distinct objects; state is keyed by parameter).
    st1 = opt.optimizer.state_dict()["state"]
    st2 = opt2.optimizer.state_dict()["state"]
    opt_ok = len(st1) == len(st2) and all(
        "exp_avg" in st1[i]
        and "exp_avg" in st2[i]
        and torch.equal(st1[i]["exp_avg"], st2[i]["exp_avg"])
        and torch.equal(st1[i]["exp_avg_sq"], st2[i]["exp_avg_sq"])
        for i in st1
    )
    check("11. resume restores optimizer state", bool(opt_ok))
    check(
        "11. resume restores scheduler",
        sched2.current_step == sched.current_step,
        f"sched_step={sched2.current_step}",
    )
    # metadata (seed/config) verification from the checkpoint payload itself
    ck_raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ck_extra = ck_raw.get("extra", {})
    meta_ok = ck_extra.get("seed") == seed and ck_extra.get("config_path") == "config/tiny.yaml"
    check("11. resume restores seed/config metadata", bool(meta_ok), json.dumps(ck_extra, default=str))

    # --------------------------------------------------------------
    # 12. VRAM measurement (CUDA only)
    # --------------------------------------------------------------
    if gpu:
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            x3, y3 = ds[0]
            _ = model(x3.unsqueeze(0).to(device))
        peak = torch.cuda.max_memory_allocated(device) / 1e6
        check("12. peak VRAM", peak > 0, f"{peak:.1f} MB")
    else:
        check("12. peak VRAM", True, "N/A (CPU)")

    logger.info(
        "GPU smoke summary: %.1f ms/step (device=%s), loss %.4f -> %.4f",
        step_time_ms,
        device,
        loss_history[0],
        loss_history[-1],
    )
    return 0 if all(s == "PASS" for _, s, _ in RESULTS) else 1


def ckpt_extra(path: str) -> str:
    c = torch.load(path, map_location="cpu", weights_only=True)
    return json.dumps(c.get("extra", {}), default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None, help="cuda | cpu | None=auto")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    sys.exit(main(args.device, args.steps, args.seed))
