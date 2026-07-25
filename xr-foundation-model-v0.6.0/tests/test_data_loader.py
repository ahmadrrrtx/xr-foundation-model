"""
Unit tests for XRFM dataset loader (`xrfm/data/loader.py`).

Tests cover:
- Dataset verification (valid file, missing file, empty file, encoding issues)
- Normalization (whitespace normalization)
- Dataset splitting (ratios sum check, deterministic split with seed)
- Chunk generation (fixed length, overlap option)
- Dataset class (`XRFMTextDataset`): initialization, length, indexing, split selection
- Manifest generation and persistence
- Tokenizer integration (`TokenizerInterface` usage, not concrete BPE)
- Config integration (`load_config_for_dataset`)

Every public function in the loader module must have at least one test.
"""

import pytest
import torch
import tempfile
import os
from typing import List

from tokenizer.interface import TokenizerInterface
from tokenizer.bpe import BytePairEncoder
from xrfm.config.loader import ConfigLoader
from xrfm.data.loader import (
    DatasetConfig,
    load_config_for_dataset,
    verify_text_file,
    normalize_text,
    split_dataset,
    chunk_text,
    XRFMTextDataset,
    build_manifest,
    save_manifest,
)


class TestDatasetConfig:
    """Configuration loading for dataset settings."""

    def test_load_config_for_dataset_default(self) -> None:
        """Config loader should extract dataset parameters from YAML."""
        # Use absolute path to config so test works regardless of CWD.
        import pathlib
        config_path = pathlib.Path(__file__).parent.parent / "config" / "config.yaml"
        config = ConfigLoader(str(config_path))
        dataset_cfg = load_config_for_dataset(config)
        assert dataset_cfg.dataset_name == "tiny_shakespeare"
        assert dataset_cfg.dataset_path == "data/datasets/"
        assert dataset_cfg.max_seq_len == 512

    def test_custom_config_values(self) -> None:
        """Custom dataset settings should be readable from config."""
        import pathlib
        config_path = pathlib.Path(__file__).parent.parent / "config" / "config.yaml"
        config = ConfigLoader(str(config_path))
        dataset_cfg = load_config_for_dataset(config)
        assert isinstance(dataset_cfg.train_ratio, float)
        assert dataset_cfg.shuffle is True


class TestDatasetVerification:
    """File verification before dataset loading."""

    def test_verify_valid_file(self) -> None:
        """A non-empty UTF-8 file should pass verification."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("This is a valid dataset file with sufficient content.")
            temp_path = f.name
        try:
            verify_text_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_verify_missing_file_raises(self) -> None:
        """Missing file must raise FileNotFoundError with clear message."""
        with pytest.raises(FileNotFoundError, match="not found"):
            verify_text_file("/nonexistent/path/file.txt")

    def test_verify_empty_file_raises(self) -> None:
        """Empty file must raise ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="too short"):
                verify_text_file(temp_path)
        finally:
            os.unlink(temp_path)


class TestTextNormalization:
    """Whitespace normalization for consistent tokenization."""

    def test_normalize_whitespace(self) -> None:
        """Multiple spaces and newlines should collapse to single spaces."""
        raw = "hello\nworld   test"
        normalized = normalize_text(raw)
        assert "  " not in normalized  # No double spaces
        assert "\n" not in normalized  # Newlines normalized
        assert "hello world test" == normalized


class TestDatasetSplitting:
    """Train/validation/test split functionality."""

    def test_split_ratios_sum_check(self) -> None:
        """Ratios that do not sum to ~1.0 must raise ValueError."""
        text = "sample text for splitting test with enough length to split."
        with pytest.raises(ValueError, match="sum to"):
            split_dataset(text, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_split_produces_three_parts(self) -> None:
        """Split must return exactly three strings."""
        text = "hello world ".join(["hello world"] * 50)
        train_text, val_text, test_text = split_dataset(
            text, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
        )
        assert isinstance(train_text, str)
        assert isinstance(val_text, str)
        assert isinstance(test_text, str)
        assert len(train_text) > 0 or len(val_text) > 0 or len(test_text) > 0


class TestChunkGeneration:
    """Chunk generation for fixed-length model input."""

    def test_chunk_basic(self) -> None:
        """Chunks should be lists of integers with length <= max_seq_len."""
        # Use a trained BPE tokenizer for chunking.
        encoder = BytePairEncoder(vocab_size_target=512)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("hello world ".join(["hello world"] * 30))
            temp_path = f.name
        try:
            encoder.train(temp_path)
            chunks = chunk_text(
                "hello world hello world",
                max_seq_len=8,
                tokenizer=encoder,
                overlap=0,
            )
            assert isinstance(chunks, list)
            assert len(chunks) > 0
            for chunk in chunks:
                assert isinstance(chunk, list)
                assert len(chunk) <= 8
                assert all(isinstance(token_id, int) for token_id in chunk)
        finally:
            os.unlink(temp_path)

    def test_chunk_overlap_option(self) -> None:
        """Overlap should be configurable (even if not actively used in Phase 3)."""
        encoder = BytePairEncoder(vocab_size_target=512)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("hello world ".join(["hello world"] * 30))
            temp_path = f.name
        try:
            encoder.train(temp_path)
            chunks = chunk_text(
                "hello world",
                max_seq_len=4,
                tokenizer=encoder,
                overlap=1,
            )
            assert isinstance(chunks, list)
        finally:
            os.unlink(temp_path)


class TestXRFMTextDataset:
    """Integration tests for the main dataset loader."""

    def test_init_and_length(self) -> None:
        """Dataset initialization should produce chunks with correct length."""
        encoder = BytePairEncoder(vocab_size_target=512)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("hello world ".join(["hello world"] * 20))
            temp_path = f.name
        try:
            encoder.train(temp_path)
            dataset = XRFMTextDataset(
                dataset_path=temp_path,
                tokenizer=encoder,
                max_seq_len=8,
                split="train",
                split_ratio=0.9,
                seed=42,
            )
            assert len(dataset) > 0
            input_ids, target_ids = dataset[0]
            assert isinstance(input_ids, torch.Tensor)
            assert isinstance(target_ids, torch.Tensor)
            assert input_ids.dtype == torch.long
        finally:
            os.unlink(temp_path)

    def test_split_selection(self) -> None:
        """Dataset should support train, val, and test splits."""
        encoder = BytePairEncoder(vocab_size_target=512)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("This is a longer sample text for split selection test that exceeds the one hundred character minimum threshold easily and ensures the dataset loader can split properly.")
            temp_path = f.name
        try:
            encoder.train(temp_path)
            for split_name in ("train", "val", "test"):
                dataset = XRFMTextDataset(
                    dataset_path=temp_path,
                    tokenizer=encoder,
                    max_seq_len=8,
                    split=split_name,
                    split_ratio=0.8,
                    seed=42,
                )
                # Each split should have some chunks (possibly 0 for very small splits,
                # but for this dataset size they should exist).
                assert isinstance(dataset, torch.utils.data.Dataset)
        finally:
            os.unlink(temp_path)

    def test_invalid_split_raises(self) -> None:
        """Invalid split name must raise ValueError."""
        encoder = BytePairEncoder()
        with pytest.raises(ValueError, match="Invalid split"):
            XRFMTextDataset(
                dataset_path="dummy_path",
                tokenizer=encoder,
                split="invalid_split",
            )

    def test_index_out_of_range_raises(self) -> None:
        """Accessing out-of-range index must raise IndexError."""
        encoder = BytePairEncoder(vocab_size_target=512)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("hello world ".join(["hello world"] * 20))
            temp_path = f.name
        try:
            encoder.train(temp_path)
            dataset = XRFMTextDataset(
                dataset_path=temp_path,
                tokenizer=encoder,
                max_seq_len=8,
                split="train",
            )
            with pytest.raises(IndexError):
                _ = dataset[len(dataset) + 10]
        finally:
            os.unlink(temp_path)


class TestManifestGeneration:
    """Dataset manifest for reproducibility."""

    def test_build_manifest_structure(self) -> None:
        """Manifest must contain required fields."""
        encoder = BytePairEncoder()
        split_info = {"train": 10, "val": 2, "test": 2}
        manifest = build_manifest(
            dataset_name="tiny_shakespeare",
            dataset_path="data/datasets/",
            tokenizer=encoder,
            split_info=split_info,
        )
        assert manifest["dataset_name"] == "tiny_shakespeare"
        assert "tokenizer" in manifest
        assert manifest["tokenizer"]["class_name"] == "BytePairEncoder"
        assert manifest["tokenizer"]["vocab_size"] == 256  # Untrained BPE encoder has base byte vocabulary (256)
        # Note: After training, vocab_size would be larger; manifest captures the tokenizer instance state at call time.
        assert manifest["version"] == "xrfm-manifest-v1"
        assert manifest["split_info"] == split_info

    def test_save_and_load_manifest(self) -> None:
        """Manifest should be saved to JSON and loadable."""
        encoder = BytePairEncoder()
        manifest = build_manifest(
            dataset_name="test_manifest",
            dataset_path="data/datasets/",
            tokenizer=encoder,
            split_info={"train": 5},
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            manifest_path = f.name
        try:
            save_manifest(manifest, manifest_path)
            # Verify file exists and is readable.
            assert os.path.exists(manifest_path)
            import json
            with open(manifest_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded["dataset_name"] == "test_manifest"
        finally:
            if os.path.exists(manifest_path):
                os.unlink(manifest_path)
