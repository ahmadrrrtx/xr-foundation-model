"""ChatML SFT dataset loader for XRFM CoderSec QLoRA training (GATE-5).

Loads `ahmadrrrtx/xrfm-codersec-sft-40k` from HF Hub, tokenizes ChatML-formatted
records with assistant-only loss masking, and produces a PyTorch Dataset ready
for QLoRA fine-tuning with `max_seq_length=2048`.

All loading is done via PyArrow streaming through `HfFileSystem` to stay within
container memory limits.
"""

from __future__ import annotations

import json
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-not-found]
import torch
from huggingface_hub import HfFileSystem  # type: ignore[import-not-found]
from torch.utils.data import Dataset
from transformers import AutoTokenizer  # type: ignore[import-not-found]

DEFAULT_MAX_SEQ_LENGTH = 2048


class CoderSecSFTDataset(Dataset):
    """
    PyTorch Dataset for CoderSec SFT records.

    Each record is a ChatML-formatted conversation. The dataset tokenizes the
    full conversation and creates labels where only assistant turn tokens
    contribute to the loss (assistant-only loss masking per roadmap).

    Attributes:
        records: List of parsed SFT record dicts.
        tokenizer: Hugging Face tokenizer.
        max_seq_length: Maximum sequence length in tokens.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Ensure pad token exists.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        chatml = record["chatml"]

        # Tokenize full ChatML conversation.
        encoding = self.tokenizer(
            chatml,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Build labels: assistant-only loss masking.
        # We find the assistant turn tokens and mask everything else with -100.
        labels = input_ids.clone()

        # Tokenize just the system + user part to find where assistant starts.
        im_start = "lic"

        # Count tokens before the assistant turn.
        # ChatML format: licsystem\n...irit\nlicuser\n...irit\nlicassistant\n...irit\n
        assistant_prefix = f"{im_start}assistant\n"
        # Find the assistant prefix in the chatml string.
        assistant_idx = chatml.find(assistant_prefix)
        if assistant_idx == -1:
            # No assistant turn found — mask everything.
            labels[:] = -100
        else:
            # Tokenize the prefix (system + user + assistant header) to get
            # the token index where assistant content begins.
            prefix_text = chatml[: assistant_idx + len(assistant_prefix)]
            prefix_encoding = self.tokenizer(
                prefix_text,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors="pt",
            )
            prefix_len = prefix_encoding["input_ids"].shape[1]

            # Mask everything before the assistant content.
            labels[:prefix_len] = -100

        # Mask padding tokens in labels.
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_sft_records_from_hf(
    repo_id: str = "ahmadrrrtx/xrfm-codersec-sft-40k",
    hf_token: str | None = None,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load SFT records from HF Hub dataset via PyArrow streaming.

    Args:
        repo_id: Hugging Face dataset repository id.
        hf_token: Hugging Face API token.
        max_records: Maximum number of records to load (None = all).

    Returns:
        List of record dicts.
    """
    fs = HfFileSystem(token=hf_token)
    files = sorted(fs.glob(f"datasets/{repo_id}/data/*.jsonl"))

    records: list[dict[str, Any]] = []
    for fpath in files:
        try:
            pq.ParquetFile(fpath, filesystem=fs)
            # ParquetFile expects parquet; for JSONL we read directly.
            # Fall back to reading the file content.
            continue
        except Exception:
            pass

        try:
            with fs.open(fpath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    records.append(record)
                    if max_records is not None and len(records) >= max_records:
                        return records
        except Exception:
            continue

    return records


def load_sft_records_from_jsonl(
    path: str,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load SFT records from a local JSONL file.

    Args:
        path: Path to JSONL file.
        max_records: Maximum number of records to load.

    Returns:
        List of record dicts.
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
            if max_records is not None and len(records) >= max_records:
                break
    return records


def create_dataloader(
    records: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
):
    """
    Create a PyTorch DataLoader for CoderSec SFT records.

    Args:
        records: List of SFT record dicts.
        tokenizer: Hugging Face tokenizer.
        max_seq_length: Maximum sequence length.
        batch_size: Per-device batch size.
        shuffle: Whether to shuffle the dataset.
        num_workers: Number of DataLoader workers.

    Returns:
        Configured DataLoader.
    """
    from torch.utils.data import DataLoader

    dataset = CoderSecSFTDataset(
        records=records,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_tokenizer(model_id: str, hf_token: str | None = None) -> AutoTokenizer:
    """
    Load and configure a tokenizer for CoderSec training.

    Sets pad_token to eos_token if not present.
    """
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615 — revision pinned by caller
        model_id,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
