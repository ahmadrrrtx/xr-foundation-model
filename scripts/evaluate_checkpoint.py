"""
Evaluate a trained XRFM checkpoint: val PPL, loss, and generation samples.

Usage:
    python scripts/evaluate_checkpoint.py --checkpoint checkpoints/checkpoint_step_5000.pt
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
logger = logging.getLogger("xrfm.eval")


def main(checkpoint_path: str, max_new_tokens: int, temperature: float, num_samples: int):
    from torch.utils.data import DataLoader

    from evaluation.perplexity import compute_perplexity
    from inference.engine import GenerationEngine
    from model.gpt import GPTModel
    from tokenizer.bpe import BytePairEncoder
    from training.distributed import xrfm_collate_fn
    from xrfm.data.loader import XRFMTextDataset

    # Tokenizer (current repo tokenizer is the one used by the run).
    tok = BytePairEncoder()
    vocab_path = "tokenizer/vocab.json"
    if os.path.exists(vocab_path):
        tok.load(vocab_path)
        logger.info("Tokenizer loaded: vocab=%d pad=%s", tok.vocab_size(), tok.pad_id)
    else:
        raise SystemExit("tokenizer/vocab.json not found")

    # Model built to match the tokenizer, then load checkpoint weights.
    model = GPTModel("config/config.yaml", vocab_size=tok.vocab_size())
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    logger.info(
        "Loaded checkpoint step=%s loss=%s params=%d",
        ckpt.get("step"),
        ckpt.get("loss"),
        model.parameter_count(),
    )
    extra = ckpt.get("extra", {})
    if extra:
        logger.info(
            "Checkpoint metadata: seed=%s config_path=%s",
            extra.get("seed"),
            extra.get("config_path"),
        )

    # Val perplexity
    val_ds = XRFMTextDataset("data/datasets/corpus.txt", tok, max_seq_len=256, split="val", pad_id=tok.pad_id or 0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=xrfm_collate_fn)
    t0 = time.time()
    eval_result = compute_perplexity(model, val_loader)
    logger.info(
        "Val: loss=%.4f PPL=%.2f tokens=%d (%.1fs)",
        eval_result["loss"],
        eval_result["perplexity"],
        eval_result["total_tokens"],
        time.time() - t0,
    )

    # Generation samples
    engine = GenerationEngine(model)
    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "It was the best of times,",
        "The scientists discovered",
        "def compute_average(values):",
    ]
    samples = []
    for p in prompts:
        ids = tok.encode(p)
        input_ids = torch.tensor([ids], dtype=torch.long)
        for temp in [0.0] if temperature is None else [temperature]:
            with torch.no_grad():
                out = engine.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temp)
            text = tok.decode(out.squeeze(0).tolist())
            samples.append({"prompt": p, "temperature": temp, "text": text})
            logger.info("--- prompt: %r (T=%s)", p, temp)
            logger.info("--- output: %s", text)

    result = {
        "checkpoint": checkpoint_path,
        "step": ckpt.get("step"),
        "val_loss": eval_result["loss"],
        "val_ppl": eval_result["perplexity"],
        "total_tokens_evaluated": eval_result["total_tokens"],
        "samples": samples,
    }
    with open("logs/eval_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved logs/eval_result.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--num_samples", type=int, default=5)
    args = ap.parse_args()
    main(args.checkpoint, args.max_new_tokens, args.temperature, args.num_samples)
