"""
Dataset loader for XR Foundation Model (XRFM).

Provides XRFMTextDataset for loading and processing text datasets,
with support for train/val/test splits, chunking, and manifest generation.
"""

import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from tokenizer.interface import TokenizerInterface


@dataclass
class DatasetConfig:
    """Typed dataset configuration."""

    dataset_name: str
    dataset_path: str
    max_seq_len: int = 512
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    shuffle: bool = True
    seed: int = 42


def load_config_for_dataset(config_loader) -> DatasetConfig:
    """Load dataset configuration from ConfigLoader."""
    datasets_cfg = config_loader.get("datasets", {})
    model_cfg = config_loader.get("model", {})

    return DatasetConfig(
        dataset_name=datasets_cfg.get("default", "tiny_shakespeare"),
        dataset_path=datasets_cfg.get("path", "data/datasets/"),
        max_seq_len=model_cfg.get("max_seq_len", 512),
        train_ratio=datasets_cfg.get("train_ratio", 0.9),
        val_ratio=datasets_cfg.get("val_ratio", 0.05),
        test_ratio=datasets_cfg.get("test_ratio", 0.05),
        shuffle=datasets_cfg.get("shuffle", True),
        seed=datasets_cfg.get("seed", 42),
    )


def verify_text_file(file_path: str) -> None:
    """Verify that a text file exists and is valid."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Dataset file not found: '{file_path}'")

    file_size = os.path.getsize(file_path)
    if file_size < 10:  # Arbitrary minimum size
        raise ValueError(f"Dataset file too short: '{file_path}' ({file_size} bytes)")


def normalize_text(text: str) -> str:
    """Normalize whitespace in text."""
    import re

    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_dataset(
    text: str,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[str, str, str]:
    """Split text into train/val/test sets."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Split ratios must sum to ~1.0, got {total}")

    import random

    random.seed(seed)

    text_len = len(text)
    train_end = int(text_len * train_ratio)
    val_end = train_end + int(text_len * val_ratio)

    train_text = text[:train_end]
    val_text = text[train_end:val_end]
    test_text = text[val_end:]

    return train_text, val_text, test_text


def chunk_text(
    text: str,
    max_seq_len: int,
    tokenizer: TokenizerInterface,
    overlap: int = 0,
) -> list[list[int]]:
    """Chunk text into fixed-length sequences."""
    # Tokenize full text
    token_ids = tokenizer.encode(text)

    chunks = []
    start = 0
    while start < len(token_ids):
        end = start + max_seq_len
        chunk = token_ids[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = end

    return chunks


class XRFMTextDataset(Dataset):
    """Dataset for XRFM training."""

    def __init__(
        self,
        dataset_path: str,
        tokenizer: TokenizerInterface,
        max_seq_len: int = 512,
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 42,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: '{split}'. Must be train, val, or test")

        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split

        # Load and process dataset
        verify_text_file(dataset_path)
        with open(dataset_path, encoding="utf-8") as f:
            text = f.read()

        text = normalize_text(text)
        train_text, val_text, test_text = split_dataset(
            text,
            train_ratio=split_ratio,
            val_ratio=(1 - split_ratio) / 2,
            test_ratio=(1 - split_ratio) / 2,
            seed=seed,
        )

        if split == "train":
            self.chunks = chunk_text(train_text, max_seq_len, tokenizer)
        elif split == "val":
            self.chunks = chunk_text(val_text, max_seq_len, tokenizer)
        else:
            self.chunks = chunk_text(test_text, max_seq_len, tokenizer)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self.chunks):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.chunks)}")

        chunk = self.chunks[idx]

        # Ensure chunk has at least 2 tokens (input and target)
        if len(chunk) < 2:
            chunk = chunk + [0] * (2 - len(chunk))

        # Input: all tokens except last, Target: all tokens except first
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)

        return input_ids, target_ids


def build_manifest(
    dataset_name: str,
    dataset_path: str,
    tokenizer: TokenizerInterface,
    split_info: dict[str, int],
) -> dict:
    """Build dataset manifest for reproducibility."""
    manifest = {
        "version": "xrfm-manifest-v1",
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "tokenizer": {
            "class_name": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size() if hasattr(tokenizer, "vocab_size") else 256,
        },
        "split_info": split_info,
    }
    return manifest


def save_manifest(manifest: dict, path: str) -> None:
    """Save manifest to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        import json

        json.dump(manifest, f, indent=2)
