"""
Dataset representation for XRFM.

``TextDataset`` owns *dataset representation only*: it loads a text corpus,
splits it (``xrfm.data.splits``), packs it (``xrfm.data.packing``), and
serves padded ``(input_ids, targets)`` pairs for next-token training.

Contract
--------
* Inputs are long tensors of shape ``(max_seq_len,)``; targets align with
  inputs such that ``targets[i]`` is the token the model should predict at
  position ``i`` (i.e. ``inputs[i + 1]``).
* Padding is applied with the tokenizer's **contract** pad id
  (``tokenizer.pad_token_id``), never a hard-coded constant.
* Target masking is **position-based**: positions ``>= len(chunk)`` (and the
  final real position, whose continuation is undefined) are masked to
  ``ignore_index`` (default ``-100``). Tokens that merely *equal* the pad id
  are content and keep their targets — this fixes the pre-Phase-0 bug where
  any token with the pad id lost its target.
* The final real token of each chunk carries ``ignore_index`` as its target:
  "what follows the end of a chunk" is undefined, and teaching the model to
  emit the padding token was a pre-Phase-0 artifact (documented behavior
  change).

Every knob of :class:`xrfm.config.schema.DatasetConfig` has a real effect;
the dataset is constructed *from* that config object.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from xrfm.config.schema import DatasetConfig
from xrfm.data.packing import chunk_token_ids
from xrfm.data.splits import split_dataset_lines
from xrfm.tokenization.interface import Tokenizer

__all__ = ["TextDataset", "verify_text_file", "IGNORE_INDEX"]


#: Default target mask value (PyTorch ``cross_entropy(ignore_index=...)``).
IGNORE_INDEX = -100


def verify_text_file(file_path: str) -> None:
    """Verify that a text file exists and is non-trivially sized."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Dataset file not found: '{file_path}'")
    file_size = os.path.getsize(file_path)
    if file_size < 10:  # Arbitrary minimum size
        raise ValueError(f"Dataset file too short: '{file_path}' ({file_size} bytes)")


class TextDataset(Dataset):
    """Packed next-token dataset over a plain-text corpus.

    Usage:
        cfg = load_config("config/tiny.yaml")
        tokenizer = BytePairEncoder(); tokenizer.load("vocab.json")
        train_ds = TextDataset(cfg.data, tokenizer, split="train")

    Args:
        config: :class:`xrfm.config.schema.DatasetConfig` (or ``None`` to
            pass every field explicitly — keyword form kept for tests).
        tokenizer: Any :class:`xrfm.tokenization.interface.Tokenizer`.
        split: ``"train"`` | ``"val"`` | ``"test"``.
        dataset_path: Explicit corpus path (overrides ``config.path``).
        max_seq_len: Explicit chunk length (overrides ``config.max_seq_len``).
        split_ratio: Train fraction; val/test split the remainder evenly
            (kept for backward compatibility with the pre-Phase-0 API).
        pad_id: Explicit pad token id. Defaults to the tokenizer contract
            value (``tokenizer.pad_token_id``); an error is raised when
            neither defines one.
        seed: Split shuffle seed (defaults to ``config.seed``). Consumed
            whenever ``shuffle=True``.
        shuffle: Whether to shuffle lines before splitting (defaults to
            ``config.shuffle``).
        dedup: Exact-line dedup before splitting (defaults to ``config.dedup``).
        ignore_index: Target mask value (default ``-100``).
    """

    def __init__(
        self,
        config: DatasetConfig | None = None,
        tokenizer: Tokenizer | None = None,
        split: str = "train",
        dataset_path: str | None = None,
        max_seq_len: int | None = None,
        split_ratio: float | None = None,
        pad_id: int | None = None,
        seed: int | None = None,
        shuffle: bool | None = None,
        dedup: bool | None = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        if tokenizer is None:
            raise TypeError("TextDataset requires a tokenizer (xrfm.tokenization.interface.Tokenizer)")
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: '{split}'. Must be train, val, or test")

        cfg = config or DatasetConfig()
        if isinstance(cfg, (str, Path)):
            # Legacy/positional spelling: TextDataset("corpus.txt", tokenizer).
            if dataset_path is None:
                dataset_path = str(cfg)
            cfg = DatasetConfig()
        if not isinstance(cfg, DatasetConfig):
            raise TypeError(f"config must be DatasetConfig, got {type(cfg).__name__}")

        self.config = cfg
        self.tokenizer = tokenizer
        self.split = split
        self.max_seq_len = max_seq_len if max_seq_len is not None else cfg.max_seq_len
        self.split_ratio = split_ratio if split_ratio is not None else cfg.train_ratio
        self.seed = seed if seed is not None else cfg.seed
        self.shuffle = shuffle if shuffle is not None else cfg.shuffle
        self.dedup = dedup if dedup is not None else cfg.dedup
        self.ignore_index = int(ignore_index)

        if not isinstance(self.max_seq_len, int) or isinstance(self.max_seq_len, bool) or self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be a positive int, got {self.max_seq_len!r}")
        if not (0.0 < self.split_ratio <= 1.0):
            raise ValueError(f"split_ratio must be in (0, 1], got {self.split_ratio!r}")

        # Pad id: explicit argument > tokenizer contract. Never hard-coded.
        if pad_id is not None:
            self.pad_id = int(pad_id)
        else:
            contract_pad = getattr(tokenizer, "pad_token_id", None)
            if contract_pad is None:
                raise ValueError(
                    "No pad token available: the tokenizer does not define one "
                    "(tokenizer.pad_token_id is None). Pass pad_id=... explicitly, or use a "
                    "tokenizer trained with special tokens (BytePairEncoder reserves "
                    "<|pad|> by default)."
                )
            self.pad_id = int(contract_pad)
        vocab = tokenizer.vocab_size()
        if not (0 <= self.pad_id < vocab):
            raise ValueError(f"pad_id ({self.pad_id}) must be in [0, tokenizer.vocab_size()={vocab})")

        # Load and process dataset.
        path = dataset_path if dataset_path is not None else cfg.path
        verify_text_file(path)
        with open(path, encoding="utf-8") as f:
            text = f.read()

        # Split by line boundaries; val/test evenly split the remainder.
        remainder = (1.0 - self.split_ratio) / 2.0
        lines = text.splitlines()
        train_lines, val_lines, test_lines = split_dataset_lines(
            lines,
            train_ratio=self.split_ratio,
            val_ratio=remainder,
            test_ratio=remainder,
            seed=self.seed,
            shuffle=self.shuffle,
            dedup=self.dedup,
        )
        split_text = {"train": train_lines, "val": val_lines, "test": test_lines}[split]
        joined = "\n".join(split_text)

        self.chunks = chunk_token_ids(tokenizer.encode(joined), max_seq_len=self.max_seq_len, overlap=0)
        # NOTE: an empty split is allowed (tiny corpora may have no val/test
        # lines). Callers must guard against empty dataloaders.

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self.chunks):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.chunks)}")

        seq = self.chunks[idx]
        n_real = len(seq)

        # Inputs: chunk padded to max_seq_len with the contract pad id.
        padded = seq + [self.pad_id] * (self.max_seq_len - n_real)

        # Targets: next real token where defined; masked elsewhere.
        # Position-based masking (NOT token-identity based): positions whose
        # continuation is undefined (last real token + all padding) are
        # ignored by the loss even if their *input* token equals pad_id.
        targets = [seq[i + 1] if i + 1 < n_real else self.ignore_index for i in range(self.max_seq_len)]

        return torch.tensor(padded, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


# Historical name (pre-Phase-0 module: xrfm.data.loader).
XRFMTextDataset = TextDataset
