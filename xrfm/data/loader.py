"""
Dataset loader for XR Foundation Model (XRFM).

Purpose: Load text datasets (Tiny Shakespeare, WikiText, OpenWebText, future
custom/multilingual/code datasets) into a format compatible with the XRFM
training pipeline. The loader uses the `TokenizerInterface` (Phase 2) and
produces fixed-length chunks (`max_seq_len` from config) for model training.

Conceptual references (NOT copied):
- Hugging Face `datasets` library — dataset loading patterns, dataset cards,
  streaming concepts. No code copied; interface designed independently.
- MosaicML `Streaming` — streaming dataset design concepts (optional future).
- Raschka (2024) — dataset preparation and chunking concepts.
- WebDataset / Apache Arrow — future data format concepts (not adopted in Phase 3).

Implementation is original. The pipeline follows clean architecture:
Source -> Verify -> Normalize -> Tokenize -> Chunk -> PyTorch Dataset.

Design principles (from TDR-002, Phase 3 research):
- Config-driven (dataset name, split ratios, sequence length configurable).
- No hard-coded dataset names or file paths.
- Stable interface: future dataset formats (multilingual, instruction,
  multimodal metadata) can be added without rewriting loader logic.
- Manifest generation: every dataset load produces reproducible metadata.
"""

import os
import json
import random
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from tokenizer.interface import TokenizerInterface
from tokenizer.bpe import BytePairEncoder
from xrfm.config.loader import ConfigLoader


@dataclass
class DatasetConfig:
    """Typed dataset configuration loaded from YAML config file."""
    dataset_name: str
    dataset_path: str
    max_seq_len: int
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    shuffle: bool = True
    streaming: bool = False  # Reserved for Phase 8+ (scaling)
    seed: int = 42


def load_config_for_dataset(config: ConfigLoader) -> DatasetConfig:
    """Extract dataset-related settings from global XRFM config.

    Args:
        config: Loaded `ConfigLoader` instance (`config/config.yaml`).

    Returns:
        `DatasetConfig` with dataset parameters.
    """
    data_cfg = config.get("datasets", {})
    model_cfg = config.get("model", {})
    return DatasetConfig(
        dataset_name=str(data_cfg.get("default", "tiny_shakespeare")),
        dataset_path=str(data_cfg.get("path", "data/datasets/")),
        max_seq_len=int(model_cfg.get("max_seq_len", 512)),
        train_ratio=float(data_cfg.get("train_ratio", 0.9)),
        val_ratio=float(data_cfg.get("val_ratio", 0.05)),
        test_ratio=float(data_cfg.get("test_ratio", 0.05)),
        shuffle=bool(data_cfg.get("shuffle", True)),
        streaming=bool(data_cfg.get("streaming", False)),
        seed=int(data_cfg.get("seed", 42)),
    )


def verify_text_file(file_path: str) -> None:
    """Validate that a dataset file exists, is readable, and non-empty.

    Args:
        file_path: Path to text file.

    Raises:
        FileNotFoundError: If file is missing.
        ValueError: If file is empty or contains only whitespace.
        UnicodeDecodeError: If file has encoding issues (caught and converted
            to clear error message).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}. "
            f"Check dataset download or path in config."
        )
    try:
        with open(file_path, "r", encoding="utf-8", errors="strict") as f:
            content = f.read()
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Dataset file {file_path} has encoding issues (not UTF-8): {exc}"
        ) from exc
    if not content or len(content.strip()) < 10:
        raise ValueError(
            f"Dataset file {file_path} is empty or too short "
            f"({len(content)} chars). Check file content or download."
        )


def normalize_text(raw_text: str) -> str:
    """Normalize whitespace in raw dataset text.

    Normalizes multiple spaces/newlines into single spaces and strips
    leading/trailing whitespace. This ensures consistent tokenization.

    Args:
        raw_text: Raw text string from dataset file.

    Returns:
        Normalized text string.
    """
    # Normalize whitespace: replace any sequence of whitespace chars
    # with a single space.
    normalized = " ".join(raw_text.split())
    return normalized


def split_dataset(
    text: str,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[str, str, str]:
    """Split a single text dataset into train/validation/test parts.

    Splits by lines (paragraphs) to preserve document structure. For very
    small datasets, the split may produce very short validation/test sets.

    Args:
        text: Full normalized dataset text.
        train_ratio: Proportion of dataset for training.
        val_ratio: Proportion for validation.
        test_ratio: Proportion for testing.
        seed: Random seed for deterministic splitting.

    Returns:
        Tuple of (train_text, val_text, test_text).

    Raises:
        ValueError: If ratios do not sum to approximately 1.0 or if ratios are negative.
    """
    total = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total <= 1.01):
        raise ValueError(
            f"Dataset split ratios must sum to ~1.0, got {total} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )
    if any(r < 0 for r in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("Dataset split ratios must be non-negative.")

    # Split by lines (paragraphs) rather than by character count to preserve
    # document boundaries.
    lines = text.split("\n")
    lines = [line for line in lines if line.strip()]

    random.seed(seed)
    random.shuffle(lines)

    total_lines = len(lines)
    if total_lines == 0:
        raise ValueError("Dataset text produced no lines for splitting.")

    train_count = max(1, int(total_lines * train_ratio))
    val_count = max(1, int(total_lines * val_ratio))
    # Ensure we don't exceed total due to rounding; adjust last split.
    test_count = total_lines - train_count - val_count
    if test_count < 0:
        # If ratios are very small, adjust to ensure non-negative.
        test_count = 0
        val_count = total_lines - train_count

    train_text = "\n".join(lines[:train_count])
    val_text = "\n".join(lines[train_count : train_count + val_count])
    test_text = "\n".join(lines[train_count + val_count :])

    return train_text, val_text, test_text


def chunk_text(
    text: str,
    max_seq_len: int,
    tokenizer: TokenizerInterface,
    overlap: int = 0,
) -> List[List[int]]:
    """Split normalized text into fixed-length token sequences (chunks).

    Each chunk has length `max_seq_len` (or slightly shorter if the text
    is shorter than the chunk size). Overlap allows adjacent chunks to
    share context (optional, default 0 for simplicity in Phase 3).

    Args:
        text: Normalized dataset text.
        max_seq_len: Maximum sequence length (from config).
        tokenizer: Tokenizer instance implementing `TokenizerInterface`.
        overlap: Number of tokens to overlap between consecutive chunks.

    Returns:
        List of token ID sequences, each of length <= max_seq_len.
    """
    token_ids = tokenizer.encode(text)
    chunks: List[List[int]] = []
    step = max_seq_len - overlap if overlap > 0 else max_seq_len
    if step <= 0:
        step = max_seq_len  # Prevent non-positive step.

    for i in range(0, len(token_ids), step):
        chunk = token_ids[i : i + max_seq_len]
        if chunk:
            chunks.append(chunk)

    return chunks


class XRFMTextDataset(Dataset):
    """PyTorch Dataset for XRFM training pipeline.

    This dataset loads a text file, verifies it, splits it (optional),
    tokenizes it using the configured tokenizer, and produces fixed-length
    chunks compatible with the training loop.

    Design principle: The dataset loader is independent of the tokenizer
    algorithm. It receives a `TokenizerInterface` instance and uses only
    the stable interface methods (`encode`, `vocab_size`).

    Conceptual reference: Hugging Face `datasets` dataset design concepts
    (streaming, splitting, chunking). Implementation is original.
    """

    def __init__(
        self,
        dataset_path: str,
        tokenizer: TokenizerInterface,
        max_seq_len: int = 512,
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 42,
        overlap: int = 0,
    ) -> None:
        """Initialize dataset loader.

        Args:
            dataset_path: Path to dataset text file or directory.
            tokenizer: Tokenizer instance implementing `TokenizerInterface`.
            max_seq_len: Maximum token sequence length.
            split: Which split to load (`train`, `val`, `test`).
            split_ratio: Ratio for train/val/test split (only applies to single-file datasets without pre-split files).
            seed: Random seed for deterministic splitting.
            overlap: Overlap between consecutive chunks (0 for Phase 3).

        Raises:
            FileNotFoundError: If dataset file is missing.
            ValueError: If split name is invalid or dataset loading fails.
        """
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Invalid split '{split}'. Must be one of: train, val, test."
            )
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split_name = split
        self.seed = seed
        self.overlap = overlap

        # Load and process dataset.
        self.chunks: List[List[int]] = []
        self._load_and_process()

    def _load_and_process(self) -> None:
        """Load dataset file, verify, split, tokenize, and chunk.

        This method performs all pipeline stages in sequence, ensuring
        that any failure produces a clear error message with context
        (dataset path, split, file name, tokenizer info).
        """
        # Determine file to load.
        # For Phase 3, we expect dataset_path to point to a single file
        # or a directory containing dataset files. If directory, we load
        # the first `.txt` file found.
        file_path = self.dataset_path
        if os.path.isdir(file_path):
            txt_files = [
                f for f in os.listdir(file_path)
                if f.endswith(".txt")
            ]
            if not txt_files:
                raise ValueError(
                    f"No `.txt` dataset files found in directory: {file_path}"
                )
            txt_files.sort()
            file_path = os.path.join(file_path, txt_files[0])

        # Verify dataset file.
        verify_text_file(file_path)

        # Read and normalize text.
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
        normalized_text = normalize_text(raw_text)

        # Split into train/val/test (for single file datasets).
        # For simplicity in Phase 3, we split by line and then reconstruct.
        # More sophisticated splitting (by document, by chunk) is reserved
        # for future versions.
        train_text, val_text, test_text = split_dataset(
            normalized_text,
            train_ratio=0.9,
            val_ratio=0.05,
            test_ratio=0.05,
            seed=self.seed,
        )

        # Select split text based on configuration.
        if self.split_name == "train":
            split_text = train_text
        elif self.split_name == "val":
            split_text = val_text
        else:
            split_text = test_text

        # Tokenize and chunk.
        self.chunks = chunk_text(
            split_text,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
            overlap=self.overlap,
        )

    def __len__(self) -> int:
        """Return number of chunks in the dataset."""
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a single chunk and its target (next-token prediction target).

        For language modeling, the input is all tokens except the last,
        and the target is all tokens except the first. This ensures the
        model learns to predict the next token at every position.

        Args:
            idx: Chunk index.

        Returns:
            Tuple of `(input_tensor, target_tensor)` both of shape `(len(chunk)-1,)`.
        """
        if idx < 0 or idx >= len(self.chunks):
            raise IndexError(
                f"Dataset index {idx} out of range (dataset size: {len(self.chunks)}). "
                f"Check dataset configuration and split settings."
            )
        chunk = self.chunks[idx]
        if len(chunk) < 2:
            # Very short chunks cannot produce meaningful input/target pairs.
            # We return a single-token pair (input and target are the same token).
            # This is a fallback; ideally, dataset should not produce chunks < 2 tokens.
            input_ids = chunk
            target_ids = chunk
        else:
            input_ids = chunk[:-1]
            target_ids = chunk[1:]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


def build_manifest(
    dataset_name: str,
    dataset_path: str,
    tokenizer: TokenizerInterface,
    split_info: Dict[str, int],
    file_checksums: Optional[Dict[str, str]] = None,
    git_commit: Optional[str] = None,
) -> Dict:
    """Generate a dataset manifest file for reproducibility.

    The manifest records dataset identity, source, tokenizer version,
    split statistics, and optional file checksums. It is saved to
    `data/manifests/` and should be committed to version control or
    referenced in experiment tracking.

    Args:
        dataset_name: Name of the dataset (e.g., `tiny_shakespeare`).
        dataset_path: Path to dataset file or directory.
        tokenizer: Tokenizer instance used for processing.
        split_info: Dictionary with split names and chunk counts
            (e.g., `{"train": 100, "val": 10, "test": 10}`).
        file_checksums: Optional dictionary mapping file paths to SHA-256 hashes.
        git_commit: Optional git commit hash (for full reproducibility).

    Returns:
        Dictionary representing the manifest content.
    """
    manifest = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "version": "xrfm-manifest-v1",
        "tokenizer": {
            "class_name": tokenizer.__class__.__name__,
            "vocab_size": tokenizer.vocab_size(),
        },
        "split_info": split_info,
        "file_checksums": file_checksums or {},
        "git_commit": git_commit,
    }
    return manifest


def save_manifest(manifest: Dict, manifest_path: str) -> None:
    """Save manifest dictionary to JSON file.

    Args:
        manifest: Manifest dictionary (from `build_manifest()`).
        manifest_path: Path for JSON output.

    Raises:
        FileNotFoundError: If parent directory missing.
        OSError: If file write fails.
    """
    parent_dir = os.path.dirname(manifest_path)
    if parent_dir and not os.path.exists(parent_dir):
        raise FileNotFoundError(
            f"Manifest directory does not exist: {parent_dir}"
        )
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        raise OSError(f"Failed to save manifest to {manifest_path}: {exc}") from exc
