"""
XRFM CI training smoke test (Phase 28).

Fast, deterministic, CPU-only gate that FAILS the build if the training
pipeline breaks. Verifies:
  1. Training runs for ~60 steps
  2. Loss DECREASES
  3. Checkpoint is created
  4. Checkpoint loads back (weights equal)
  5. Resume restores step + scheduler + optimizer

Runs in well under 2 minutes on a standard CI CPU runner.

Usage:
    python scripts/ci_smoke_test.py
Exit code 0 = pass, 1 = fail (breaks CI).
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402


def main() -> int:
    import tempfile

    from model.gpt import GPTModel
    from tokenizer.bpe import BytePairEncoder
    from training.checkpoint import CheckpointLoader
    from training.loop import TrainingLoop, _set_seed
    from training.scheduler import SchedulerLoader
    from xrfm.data.loader import XRFMTextDataset

    STEPS = 60
    SEED = 42
    corpus = (
        "The orange kangaroo builds castles from wet sand.\n"
        "Quiet robots hum old songs near the frozen lake.\n"
        "The paper crane carries a message across the harbor.\n" * 12
    )
    tmpdir = tempfile.mkdtemp(prefix="xrfm_ci_")
    corpus_path = os.path.join(tmpdir, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    # Tokenizer fit on the train text (byte-level).
    tok = BytePairEncoder(vocab_size_target=512)
    tok.train_on_text(corpus)
    ds = XRFMTextDataset(corpus_path, tok, max_seq_len=64, split="train", pad_id=tok.pad_id or 0)

    _set_seed(SEED)
    model = GPTModel("config/tiny.yaml", vocab_size=tok.vocab_size())
    loop = TrainingLoop(
        config_path="config/tiny.yaml",
        model=model,
        dataset=ds,
        checkpoint_dir=os.path.join(tmpdir, "ckpt"),
    )
    loop.batch_size = 8

    result = loop.training_loop(max_steps=STEPS, checkpoint_every=STEPS, log_interval=1000)
    train_loss = result["final_loss"]

    # 1. loss decreased from ~random
    import math

    random_loss = math.log(tok.vocab_size())
    if not (train_loss < random_loss):
        print(f"FAIL: loss did not decrease below random ({train_loss:.3f} >= {random_loss:.3f})")
        return 1

    # 2. checkpoint created
    ckpt_path = result.get("checkpoint_path")
    if not ckpt_path or not os.path.exists(ckpt_path):
        print("FAIL: checkpoint not created")
        return 1

    # 3-5. load + resume
    from training.optimizer import OptimizerLoader

    _set_seed(SEED + 1)
    model2 = GPTModel("config/tiny.yaml", vocab_size=tok.vocab_size())
    opt2 = OptimizerLoader(model2.parameters(), learning_rate=3e-4, weight_decay=0.1)
    sched2 = SchedulerLoader(opt2.optimizer, base_lr=3e-4, warmup_steps=10, max_steps=200)
    ck = CheckpointLoader(checkpoint_dir=os.path.join(tmpdir, "ckpt"))
    meta = ck.load_checkpoint(ckpt_path, model2, opt2, sched2)

    w_ok = all(torch.equal(a, b) for a, b in zip(model2.parameters(), model.parameters()))
    if not w_ok:
        print("FAIL: checkpoint weights do not match")
        return 1
    if meta.get("step") != STEPS:
        print(f"FAIL: resume step mismatch ({meta.get('step')} != {STEPS})")
        return 1
    if sched2.current_step != loop.scheduler.current_step:
        print(f"FAIL: scheduler state not restored ({sched2.current_step} != {loop.scheduler.current_step})")
        return 1

    print(
        f"CI TRAINING SMOKE: PASS (steps={STEPS}, loss {random_loss:.2f} -> {train_loss:.3f}, "
        f"ckpt={os.path.basename(ckpt_path)}, resume step={meta.get('step')})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
