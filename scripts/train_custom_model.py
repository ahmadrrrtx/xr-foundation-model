"""
Script to train your own custom XRFM foundation model from scratch on your dataset.

Usage:
    python scripts/train_custom_model.py --dataset_path data/datasets/sample.txt --max_steps 500
"""

import argparse
import logging
import os
import sys

# Ensure repository root is on sys.path for Windows & cross-platform imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from model.gpt import GPTModel  # noqa: E402
from tokenizer.bpe import BytePairEncoder  # noqa: E402
from training.loop import TrainingLoop  # noqa: E402
from xrfm.data.loader import XRFMTextDataset  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xrfm.train")


def train_custom_model(dataset_path: str, max_steps: int = 1000, batch_size: int = 8):
    if not os.path.exists(dataset_path):
        logger.info(f"Dataset path '{dataset_path}' not found. Creating sample dataset...")
        os.makedirs(os.path.dirname(dataset_path) or ".", exist_ok=True)
        with open(dataset_path, "w", encoding="utf-8") as f:
            f.write(
                "XR Foundation Model (XRFM) is an original custom language model built in PyTorch. "
                "It uses Rotary Position Embeddings (RoPE), SwiGLU activations, RMSNorm pre-normalization, "
                "and weight-tied embeddings. You can train this model on any text dataset to create your "
                "own custom ChatGPT-like AI assistant!\n" * 100
            )

    logger.info("Initializing custom BPE tokenizer...")
    tokenizer = BytePairEncoder(vocab_size_target=1024)
    tokenizer.train(dataset_path)
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer.save("tokenizer/vocab.json")

    logger.info("Preparing dataset...")
    dataset = XRFMTextDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_seq_len=256,
        split="train",
    )

    logger.info("Instantiating XRFM Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")
    model = GPTModel("config/config.yaml").to(device)

    logger.info(f"Starting custom training loop for {max_steps} steps...")
    loop = TrainingLoop(
        config_path="config/config.yaml",
        model=model,
        dataset=dataset,
        checkpoint_dir="checkpoints/",
    )
    loop.batch_size = batch_size
    result = loop.training_loop(max_steps=max_steps, checkpoint_every=200)

    logger.info(f"Custom model training finished! Final loss: {result['final_loss']:.4f}")
    logger.info(f"Checkpoint saved at: {result['checkpoint_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom XRFM Model")
    parser.add_argument("--dataset_path", type=str, default="data/datasets/sample.txt")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    train_custom_model(args.dataset_path, args.max_steps, args.batch_size)
