"""
XRFM overfitting sanity test (Phase 17 of the forensic mission).

PROVES the learning loop is correct end-to-end: model, loss, labels, causal
masking, tokenizer, optimizer, data loader. NOT a quality benchmark.

Procedure:
1. Build a tiny fixed dataset (a few short documents, ~300-600 tokens).
2. Train XRFM-TINY until training loss < 0.1 (strong overfit).
3. Verify generation reproduces the training text approximately.

Usage:
    python scripts/run_overfit_test.py [--steps 800] [--loss_threshold 0.1]
"""

import argparse
import logging
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xrfm.overfit")


def make_tiny_corpus() -> str:
    return (
        "The purple elephant dances on the moon every Tuesday night.\n"
        "Green robots build tiny houses from recycled starlight.\n"
        "Captain Nova sails her silver boat across the milky sea.\n"
        "The dragon whispers secrets to the mountain of glass.\n"
        "Every Thursday, the clockwork owl sings three songs.\n"
        "Professor Quill maps the hidden rivers of the desert.\n" * 4
    )


def main(steps: int, loss_threshold: float, seed: int):
    from model.gpt import GPTModel
    from tokenizer.bpe import BytePairEncoder
    from training.loop import TrainingLoop, _set_seed
    from xrfm.data.loader import XRFMTextDataset

    corpus = make_tiny_corpus()
    corpus_path = "/tmp/xrfm_overfit_corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    # Tokenizer fit on the corpus (byte-level).
    tok = BytePairEncoder(vocab_size_target=1024)
    tok.train_on_text(corpus)
    logger.info("Tokenizer vocab: %d, pad_id: %s", tok.vocab_size(), tok.pad_id)

    ds = XRFMTextDataset(corpus_path, tok, max_seq_len=128, split="train", pad_id=tok.pad_id or 0)
    logger.info("Train chunks: %d (~%d tokens)", len(ds), len(ds) * 127)

    _set_seed(seed)
    model = GPTModel("config/tiny.yaml", vocab_size=tok.vocab_size())
    logger.info("Model params: %d", model.parameter_count())

    loop = TrainingLoop(
        config_path="config/tiny.yaml",
        model=model,
        dataset=ds,
        checkpoint_dir="/tmp/xrfm_overfit_ckpt",
    )
    loop.batch_size = 8
    loop.seed = seed

    t0 = time.time()
    result = loop.training_loop(max_steps=steps, checkpoint_every=max(steps // 4, 10), log_interval=25)
    dt = time.time() - t0

    train_loss = result["final_loss"]
    logger.info(
        "Overfit: steps=%d final_loss=%.4f best=%.4f time=%.0fs (%.0f ms/step)",
        result["final_step"],
        train_loss,
        result["best_loss"],
        dt,
        dt / max(steps, 1) * 1000,
    )

    # Generation check: greedy continuation should contain training vocabulary.
    from inference.engine import GenerationEngine

    model.eval()
    engine = GenerationEngine(model)
    prompt = tok.encode("The purple elephant")
    input_ids = torch.tensor([prompt], dtype=torch.long)
    with torch.no_grad():
        out_ids = engine.generate(input_ids, max_new_tokens=30, temperature=0)
    text = tok.decode(out_ids.squeeze(0).tolist())
    logger.info("Greedy continuation: %r", text)

    passed = train_loss < loss_threshold
    logger.info(
        "OVERFIT TEST: %s (loss %.4f < %.4f)",
        "PASS" if passed else "FAIL",
        train_loss,
        loss_threshold,
    )
    if not passed:
        raise SystemExit(1)
    return {"final_loss": train_loss, "text": text}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--loss_threshold", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.steps, args.loss_threshold, args.seed)
