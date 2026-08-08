"""
XRFM scaling mini-experiment (Phase 20).

Trains XRFM-TINY and XRFM-SMALL on the SAME corpus with the SAME tokenizer,
SAME steps, SAME batch/seq, and records train/val loss, val PPL, and
throughput. Isolates model size as the only variable.

Usage:
    python scripts/scaling_experiment.py --steps 1500
"""

import argparse
import logging
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("xrfm.scaling")


def run_size(name: str, config_path: str, tokenizer, corpus_path: str, steps: int, seed: int):
    from torch.utils.data import DataLoader

    from evaluation.perplexity import compute_perplexity
    from model.gpt import GPTModel
    from training.distributed import xrfm_collate_fn
    from training.loop import TrainingLoop, _set_seed
    from xrfm.data.loader import XRFMTextDataset

    ds = XRFMTextDataset(corpus_path, tokenizer, max_seq_len=256, split="train", pad_id=tokenizer.pad_id or 0)
    val_ds = XRFMTextDataset(corpus_path, tokenizer, max_seq_len=256, split="val", pad_id=tokenizer.pad_id or 0)

    _set_seed(seed)
    model = GPTModel(config_path, vocab_size=tokenizer.vocab_size())
    n_params = model.parameter_count()

    loop = TrainingLoop(config_path=config_path, model=model, dataset=ds)
    loop.batch_size = 8
    loop.seed = seed

    def _validate(loop_obj):
        loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=xrfm_collate_fn)
        return compute_perplexity(loop_obj.model, loader)

    loop.validation_fn = _validate
    loop.eval_every = 250

    t0 = time.time()
    result = loop.training_loop(max_steps=steps, checkpoint_every=steps // 2, log_interval=100)
    dt = time.time() - t0
    tokens = steps * 8 * 256
    throughput = tokens / dt

    # Final val PPL
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=xrfm_collate_fn)
    final_eval = compute_perplexity(model, val_loader)

    return {
        "name": name,
        "params": n_params,
        "steps": steps,
        "tokens": tokens,
        "train_loss": result["final_loss"],
        "val_loss": final_eval["loss"],
        "val_ppl": final_eval["perplexity"],
        "throughput_tok_s": round(throughput, 1),
        "wall_s": round(dt, 1),
    }


def main(steps: int, seed: int):
    from tokenizer.bpe import BytePairEncoder

    corpus = "data/datasets/corpus.txt"
    # ONE shared tokenizer for both sizes (isolates model size).
    with open(corpus, encoding="utf-8") as f:
        text = f.read()
    n = 8
    chunk = 40000
    sample = "\n\n".join(text[i * (len(text) // n) : i * (len(text) // n) + chunk] for i in range(n))
    tok = BytePairEncoder(vocab_size_target=2048)
    tok.train_on_text(sample)
    logger.info("shared tokenizer vocab=%d", tok.vocab_size())

    results = []
    for name, cfg in (("XRFM-TINY", "config/tiny.yaml"), ("XRFM-SMALL", "config/config.yaml")):
        logger.info("=== %s ===", name)
        r = run_size(name, cfg, tok, corpus, steps, seed)
        results.append(r)
        print(
            f"{name}: params={r['params']:,} train_loss={r['train_loss']:.4f} "
            f"val_loss={r['val_loss']:.4f} val_ppl={r['val_ppl']:.1f} "
            f"throughput={r['throughput_tok_s']} tok/s wall={r['wall_s']}s"
        )

    print("\n=== SCALING COMPARISON ===")
    print(f"{'model':12s} {'params':>10s} {'tokens':>10s} {'train':>8s} {'val':>8s} {'PPL':>8s} {'tok/s':>8s}")
    for r in results:
        print(
            f"{r['name']:12s} {r['params']:>10,} {r['tokens']:>10,} {r['train_loss']:>8.3f} "
            f"{r['val_loss']:>8.3f} {r['val_ppl']:>8.1f} {r['throughput_tok_s']:>8,.0f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.steps, args.seed)
