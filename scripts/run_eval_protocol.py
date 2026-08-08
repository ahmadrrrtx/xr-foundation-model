"""
XRFM v1.1 evaluation protocol runner (Phase 39).

Implements the documented protocol: intrinsic metrics on a held-out slice
(masked val loss, perplexity, top-1 accuracy) + fixed-prompt generation with
repetition/diversity analysis + EOS/stop behavior accounting.

Usage:
    python scripts/run_eval_protocol.py --checkpoint <path> --dataset <path>
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

FIXED_PROMPTS = [
    "The history of the telescope begins",
    "def quicksort(arr):",
    "Why do leaves change color in autumn?",
    "Once upon a time in a small village",
    "The three laws of thermodynamics state that",
    "To bake sourdough bread, first",
    "The capital of Australia is",
    "Machine learning models learn by",
    "She opened the letter and read:",
    "if __name__ == '__main__':",
]


def main(checkpoint: str, dataset: str, config: str, max_new_tokens: int, seed: int) -> dict:
    from torch.utils.data import DataLoader

    from evaluation.benchmarks import TextCompletionAccuracy
    from evaluation.perplexity import compute_perplexity
    from inference.engine import GenerationEngine
    from model.gpt import GPTModel
    from tokenizer.bpe import BytePairEncoder
    from training.distributed import xrfm_collate_fn
    from xrfm.data.loader import XRFMTextDataset

    torch.manual_seed(seed)
    tok = BytePairEncoder()
    tok.load("tokenizer/vocab.json")
    vocab = tok.vocab_size()

    model = GPTModel(config, vocab_size=vocab)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    print(f"[eval] checkpoint step={ck.get('step')} loss={ck.get('loss')} params={model.parameter_count():,}")

    # ---- intrinsic metrics on held-out val slice (padding excluded) ----
    # Protocol rule: evaluate at the TRAINING context length (evaluating at a
    # longer context than trained degrades numbers for non-long-context models).
    ctx = model.max_seq_len
    print(f"[eval] context (train) = {ctx}")
    val_ds = XRFMTextDataset(dataset, tok, max_seq_len=ctx, split="val", pad_id=tok.pad_id or 0)
    loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=xrfm_collate_fn)
    ppl = compute_perplexity(model, loader)
    acc = TextCompletionAccuracy().compute(model, loader)
    print(
        f"[eval] val loss={ppl['loss']:.4f} PPL={ppl['perplexity']:.2f} top1={acc['accuracy']:.4f} tokens={ppl['total_tokens']}"
    )

    # ---- generation protocol ----
    engine = GenerationEngine(model)
    gen_records = []
    for prompt in FIXED_PROMPTS:
        ids = tok.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long)
        for temp, top_p, rp in [(0.0, None, 1.0), (0.8, 0.9, 1.0), (0.8, 0.9, 1.2)]:
            with torch.no_grad():
                out = engine.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    top_p=top_p,
                    repetition_penalty=rp,
                    stop_token_id=tok.eos_id,
                    decode_fn=tok.decode,
                )
            new_ids = out.squeeze(0)[len(ids) :].tolist()
            text = tok.decode(new_ids)
            gen_records.append(
                {
                    "prompt": prompt,
                    "temperature": temp,
                    "top_p": top_p,
                    "repetition_penalty": rp,
                    "generated_tokens": len(new_ids),
                    "stopped_by": "eos" if new_ids and new_ids[-1] == (tok.eos_id or -1) else "length",
                    "text": text,
                }
            )

    # ---- repetition analysis (4-gram diversity on generated spans) ----
    for r in gen_records:
        tokens = r["text"].split()
        n = len(tokens)
        if n >= 4:
            four = [" ".join(tokens[i : i + 4]) for i in range(n - 3)]
            r["rep4"] = round(1 - len(set(four)) / max(len(four), 1), 4)
        else:
            r["rep4"] = 0.0

    summary = {
        "val_loss": ppl["loss"],
        "val_ppl": ppl["perplexity"],
        "top1_accuracy": acc["accuracy"],
        "val_tokens": ppl["total_tokens"],
        "generations": len(gen_records),
        "mean_rep4": round(sum(r["rep4"] for r in gen_records) / len(gen_records), 4),
        "eos_rate": round(sum(1 for r in gen_records if r["stopped_by"] == "eos") / len(gen_records), 3),
        "mean_generated_tokens": round(sum(r["generated_tokens"] for r in gen_records) / len(gen_records), 1),
    }
    print(
        f"[eval] mean rep4={summary['mean_rep4']} eos_rate={summary['eos_rate']} "
        f"mean_gen_tokens={summary['mean_generated_tokens']}"
    )
    return {"summary": summary, "generations": gen_records}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="data/datasets/corpus.txt")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="logs/eval_result.json")
    args = ap.parse_args()
    res = main(args.checkpoint, args.dataset, args.config, args.max_new_tokens, args.seed)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print("saved", args.out)
