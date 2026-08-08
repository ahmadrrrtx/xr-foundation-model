"""
Script to train your own custom XRFM model from scratch on your dataset.

Forensic-audit fixes applied (see docs/audit):
- F-13: the model is built with the tokenizer's ACTUAL vocabulary size
  (previously: tokenizer trained to 1024 but model built with vocab 50304).
- F-18/contamination: the tokenizer is trained on the TRAIN split only.
- F-31/F-33: the tokenizer file and a JSONL metrics log are saved alongside
  the checkpoint so training is reproducible.

Usage:
    python scripts/train_custom_model.py --dataset_path data/datasets/<file>.txt --max_steps 500
"""

import argparse
import logging
import os
import sys
import time

# Ensure repository root is on sys.path for Windows & cross-platform imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from model.gpt import GPTModel  # noqa: E402
from tokenizer.bpe import BytePairEncoder  # noqa: E402
from training.loop import TrainingLoop  # noqa: E402
from training.metrics import MetricsWriter  # noqa: E402
from xrfm.data.loader import XRFMTextDataset, split_dataset_lines  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xrfm.train")


def train_custom_model(
    dataset_path: str,
    max_steps: int = 1000,
    batch_size: int = 8,
    max_seq_len: int = 256,
    vocab_size_target: int = 2048,
    seed: int = 42,
    eval_every: int = 100,
    checkpoint_every: int = 200,
    resume_from: str | None = None,
    config_path: str = "config/config.yaml",
    train_ratio: float = 0.9,
    tokenizer_out: str | None = None,
):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' not found. Provide a text file to train on.")

    # --- Tokenizer: train on the TRAIN split only (no contamination) ---
    logger.info("Reading dataset for tokenizer training (train split only)...")
    with open(dataset_path, encoding="utf-8") as f:
        all_lines = f.read().splitlines()
    train_lines, _, _ = split_dataset_lines(all_lines, train_ratio=0.9, seed=seed)
    if len(train_lines) < 10:
        raise ValueError("Train split too small for tokenizer training.")

    # Sample the train text for BPE fitting (standard tokenizer practice:
    # merge statistics converge on a representative sample; training on the
    # full corpus is O(merges x corpus) in pure Python and needlessly slow).
    train_text = "\n".join(train_lines)
    max_chars = 400_000
    if len(train_text) > max_chars:
        n = 8
        step = len(train_text) // n
        slices = [train_text[i * step : i * step + max_chars // n] for i in range(n)]
        train_text = "\n\n".join(slices)
        logger.info("Tokenizer training text sampled to %d chars", len(train_text))

    tokenizer = BytePairEncoder(vocab_size_target=vocab_size_target)
    tokenizer.train_on_text(train_text)
    tokenizer_path = tokenizer_out or "tokenizer/vocab.json"
    os.makedirs(os.path.dirname(tokenizer_path) or ".", exist_ok=True)
    tokenizer.save(tokenizer_path)
    logger.info("Tokenizer trained: vocab=%d, saved to %s", tokenizer.vocab_size(), tokenizer_path)

    # --- Dataset (all splits from the same file, line-boundary split) ---
    dataset = XRFMTextDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        split="train",
        split_ratio=train_ratio,
        pad_id=tokenizer.pad_id or 0,
    )
    val_dataset = XRFMTextDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        split="val",
        split_ratio=train_ratio,
        pad_id=tokenizer.pad_id or 0,
    )
    logger.info("Train chunks: %d | Val chunks: %d", len(dataset), len(val_dataset))

    # --- Model: vocabulary MUST match the tokenizer (F-13) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using compute device: %s", device)
    model = GPTModel(config_path, vocab_size=tokenizer.vocab_size()).to(device)
    logger.info("Model params: %d (vocab=%d)", model.parameter_count(), tokenizer.vocab_size())

    loop = TrainingLoop(
        config_path=config_path,
        model=model,
        dataset=dataset,
        checkpoint_dir="checkpoints/",
    )
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)
        meta = loop.checkpoint_loader.load_checkpoint(resume_from, model, loop.optimizer, loop.scheduler)
        loop.current_step = meta.get("step", 0)
        loop.best_loss = meta.get("best_loss", float("inf"))
        loop.resume_from = resume_from
    loop.batch_size = batch_size
    loop.seed = seed
    loop.ignore_index = -100

    # Validation hook: val loss + perplexity every 100 steps.
    from torch.utils.data import DataLoader

    from evaluation.perplexity import compute_perplexity
    from training.distributed import xrfm_collate_fn

    def _validate(loop_obj):
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=xrfm_collate_fn)
        return compute_perplexity(
            loop_obj.model if not hasattr(loop_obj.model, "module") else loop_obj.model.module,
            val_loader,
        )

    loop.validation_fn = _validate
    loop.eval_every = eval_every
    loop.metrics_writer = MetricsWriter(
        "logs/training_metrics.jsonl",
        header={"dataset": dataset_path, "tokenizer": tokenizer_path, "seed": seed},
    )

    # Dataset provenance manifest (Phase 33): written next to the metrics log
    # so every run is identifiable from its artifacts alone.
    from xrfm.data.manifest import build_dataset_manifest

    manifest = build_dataset_manifest(
        dataset_path,
        tokenizer,
        name=f"xrfm_{os.path.basename(dataset_path)}",
        version="1.0.0",
        source=os.getenv("XRFM_DATASET_SOURCE", "see data/datasets/README.md"),
        license=os.getenv("XRFM_DATASET_LICENSE", "see data/datasets/README.md"),
        train_ratio=0.9,
        val_ratio=0.05,
    )
    manifest_path = "logs/dataset_manifest.json"
    manifest.save(manifest_path)
    logger.info(
        "Dataset manifest: %s (docs=%d, tokens=%d, sha256=%s…)",
        manifest_path,
        manifest.num_documents,
        manifest.total_tokens,
        manifest.sha256[:12],
    )

    # Experiment record (Phase 40): complete machine-readable run manifest.
    from xrfm.config.loader import ConfigLoader
    from xrfm.experiment.tracking import ExperimentRecord, git_commit

    _cfg = ConfigLoader(config_path)
    _model_cfg = _cfg.model_config()
    _train_cfg = _cfg.training_config()
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-seed{seed}"
    record = ExperimentRecord(
        run_id=run_id,
        git_commit=git_commit(),
        model_config={
            "vocab_size": tokenizer.vocab_size(),
            "d_model": _model_cfg.d_model,
            "n_layers": _model_cfg.n_layers,
            "n_heads": _model_cfg.n_heads,
            "d_ff": _model_cfg.d_ff,
            "max_seq_len": _model_cfg.max_seq_len,
            "dropout": _model_cfg.dropout,
            "use_bias": _model_cfg.use_bias,
            "params": model.parameter_count(),
        },
        tokenizer_version=tokenizer._version,
        tokenizer_vocab_size=tokenizer.vocab_size(),
        dataset_name=manifest.name,
        dataset_version=manifest.version,
        dataset_manifest=manifest_path,
        seed=seed,
        optimizer="adamw",
        learning_rate=_train_cfg.learning_rate,
        scheduler="cosine+warmup",
        batch_size=batch_size,
        grad_accum_steps=_train_cfg.grad_accum_steps,
        sequence_length=max_seq_len,
        precision="bf16" if (torch.cuda.is_available() and _train_cfg.mixed_precision) else "fp32",
        device=str(device),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    )
    _t_start = time.time()
    try:
        result = loop.training_loop(max_steps=max_steps, checkpoint_every=checkpoint_every, log_interval=10)
    finally:
        loop.metrics_writer.close()

    # Finalize the record with measured results.
    record.finished_at = time.time()
    record.steps = result["final_step"]
    record.tokens = result["final_step"] * batch_size * max_seq_len
    record.training_loss = result["final_loss"]
    record.tokens_per_sec = round(record.tokens / max(time.time() - _t_start, 1e-9), 1)
    record.checkpoint_path = result.get("checkpoint_path", "")
    record_path = "logs/run_manifest.json"
    record.save(record_path)
    logger.info("Experiment record saved: %s (commit %s, run %s)", record_path, record.git_commit, run_id)

    logger.info("Custom model training finished! Final loss: %.4f", result["final_loss"])
    logger.info("Checkpoint saved at: %s", result["checkpoint_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom XRFM Model")
    parser.add_argument("--dataset_path", type=str, default="data/datasets/sample.txt")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--vocab_size_target", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--checkpoint_every", type=int, default=200)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--tokenizer_out", type=str, default=None)
    args = parser.parse_args()

    train_custom_model(
        args.dataset_path,
        args.max_steps,
        args.batch_size,
        args.max_seq_len,
        args.vocab_size_target,
        args.seed,
        args.eval_every,
        args.checkpoint_every,
        args.resume_from,
        args.config,
        args.train_ratio,
        args.tokenizer_out,
    )
