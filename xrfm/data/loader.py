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
    """Normalize whitespace in text (kept for backward compatibility).

    NOTE (forensic-audit fix, F-19): the dataset no longer calls this — it
    destroys newline/paragraph structure which a language model needs to
    learn. This function is retained for callers that explicitly want
    collapsed text.
    """
    import re

    text = text.replace("\r\n", "\n").replace("\t", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_dataset_lines(
    lines: list[str],
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    dedup: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """Split text by LINE BOUNDARIES into train/val/test.

    Forensic-audit fix (F-17/F-18): the previous implementation split a single
    string by character index — arbitrary mid-sentence cuts that also leaked
    identical repeated text into every split. Splitting by line keeps
    documents/sentences intact and, with optional exact-line dedup, prevents
    identical lines from appearing in more than one split.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Split ratios must sum to ~1.0, got {total}")

    if dedup:
        seen: set[str] = set()
        unique: list[str] = []
        for ln in lines:
            if ln not in seen:
                seen.add(ln)
                unique.append(ln)
        lines = unique

    n = len(lines)
    if n == 0:
        return [], [], []

    # Guarantee at least one training line for tiny corpora (a single-line
    # file would otherwise produce train_end == 0).
    train_end = min(n, max(1, int(n * train_ratio)))
    val_end = min(n, train_end + int(n * val_ratio))
    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]
    test_lines = lines[val_end:]
    return train_lines, val_lines, test_lines


def split_dataset(
    text: str,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[str, str, str]:
    """Split text into train/val/test sets by line boundaries (see above)."""
    lines = text.splitlines()
    tr, va, te = split_dataset_lines(
        lines, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    return "\n".join(tr), "\n".join(va), "\n".join(te)


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
    """Dataset for XRFM training.

    Forensic-audit fixes:
    - F-18/F-19: splits by line boundaries and preserves newlines/paragraphs.
    - F-15/F-20: pads with a dedicated ``pad_id`` and emits ``-100`` targets
      for padded positions so cross_entropy(ignore_index=-100) never trains on
      padding; also pads short chunks up to ``max_seq_len`` (uniform batches).
    """

    def __init__(
        self,
        dataset_path: str,
        tokenizer: TokenizerInterface,
        max_seq_len: int = 512,
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 42,
        pad_id: int = 0,
        dedup: bool = True,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: '{split}'. Must be train, val, or test")

        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.pad_id = pad_id
        self._ignore: int = -100

        # Load and process dataset
        verify_text_file(dataset_path)
        with open(dataset_path, encoding="utf-8") as f:
            text = f.read()

        # Keep the raw text structure: split by lines, dedup exact duplicates,
        # then rejoin so newlines survive into the token stream.
        lines = text.splitlines()
        train_lines, val_lines, test_lines = split_dataset_lines(
            lines,
            train_ratio=split_ratio,
            val_ratio=(1 - split_ratio) / 2,
            test_ratio=(1 - split_ratio) / 2,
            seed=seed,
            dedup=dedup,
        )
        if split == "train":
            split_text = "\n".join(train_lines)
        elif split == "val":
            split_text = "\n".join(val_lines)
        else:
            split_text = "\n".join(test_lines)

        self.chunks = chunk_text(split_text, max_seq_len, tokenizer)
        # NOTE: an empty split is allowed (tiny corpora may have no val/test
        # lines). Callers must guard against empty dataloaders.

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self.chunks):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.chunks)}")

        seq = self.chunks[idx][: self.max_seq_len]

        # Pad the input with pad_id up to max_seq_len.
        padded = seq + [self.pad_id] * (self.max_seq_len - len(seq))

        # Targets: next-token. Padded input positions get -100 (ignored by CE).
        # The last real token predicts pad_id (sequence end).
        targets: list[int] = []
        for i in range(self.max_seq_len):
            if padded[i] == self.pad_id:
                targets.append(self._ignore)
            elif i + 1 < len(seq):
                targets.append(seq[i + 1])
            else:
                targets.append(self.pad_id)

        return torch.tensor(padded, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


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
