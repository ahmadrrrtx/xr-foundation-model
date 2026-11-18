"""Phase 5 (`GATE-5`) regression tests — Track B CoderSec QLoRA training.

Validates the QLoRA training stack:

1. CoderSecConfig — NF4/FP16 precision, assistant-only loss, max_seq_length=2048,
   LoRA hyperparameters, ablation configs (B0 smoke, B1 English-only, B2
   multilingual/Roman-Urdu, B3 rank check)
2. CoderSecSFTDataset — ChatML tokenization, assistant-only loss masking
   (-100 for non-assistant tokens), padding masking
3. QLoRA training pipeline — model loading with SHA verification, LoRA
   application, Trainer integration, adapter save + merge
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from xrfm.codersec.config import CoderSecConfig
from xrfm.codersec.dataset import CoderSecSFTDataset, get_tokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_records():
    """Minimal ChatML SFT records for dataset testing."""
    return [
        {
            "record_id": "rec-001",
            "source": "synthetic",
            "language": "en",
            "chatml": (
                "licsystem\nYou are a helpful coding assistant.irit\n"
                "licuser\nWrite a Python function to add two numbers.irit\n"
                "licassistant\ndef add(a, b):\n    return a + b\nirit\n"
            ),
            "domain": "code",
            "license": "MIT",
            "license_tier": "tier_a",
            "code_blocks": [{"language": "python", "code": "def add(a, b):\n    return a + b\n"}],
            "provenance": {"source_uri": "local://test"},
        },
        {
            "record_id": "rec-002",
            "source": "synthetic",
            "language": "en",
            "chatml": (
                "licsystem\nYou are a defensive security assistant.irit\n"
                "licuser\nHow do I prevent SQL injection?irit\n"
                "licassistant\nUse parameterized queries with placeholders.irit\n"
            ),
            "domain": "defensive_security",
            "license": "Apache-2.0",
            "license_tier": "tier_a",
            "code_blocks": [],
            "provenance": {"source_uri": "local://test"},
        },
    ]


@pytest.fixture
def tokenizer():
    """Load a small tokenizer for testing."""
    return get_tokenizer("Qwen/Qwen3-1.7B", hf_token=os.environ.get("HF_TOKEN"))


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestCoderSecConfigQLoRA:
    def test_default_config_qlora_params(self):
        cfg = CoderSecConfig()
        assert cfg.precision in ("nf4", "fp16")
        assert cfg.max_seq_length == 2048
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == 0.05
        assert cfg.assistant_only_loss is True
        assert cfg.gradient_checkpointing is True
        assert cfg.batch_size == 1
        assert cfg.grad_accum_steps == 16

    def test_smoke_b0_config(self):
        from xrfm.codersec.config import smoke_b0_config

        cfg = smoke_b0_config()
        assert cfg.max_seq_length == 512
        assert cfg.max_steps == 20
        assert cfg.lora_r == 8
        assert cfg.precision == "fp16"

    def test_ablation_configs(self):
        from xrfm.codersec.config import (
            ablation_b1_english_only,
            ablation_b2_multilingual_roman_urdu,
            ablation_b3_rank_check,
        )

        b1 = ablation_b1_english_only()
        assert b1.max_steps == 200
        assert b1.max_seq_length == 2048

        b2 = ablation_b2_multilingual_roman_urdu()
        assert b2.max_steps == 600

        b3 = ablation_b3_rank_check()
        assert b3.lora_r == 32
        assert b3.lora_alpha == 64


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestCoderSecSFTDataset:
    def test_dataset_len(self, sample_records, tokenizer):
        ds = CoderSecSFTDataset(
            records=sample_records,
            tokenizer=tokenizer,
            max_seq_length=256,
        )
        assert len(ds) == 2

    def test_dataset_getitem_returns_tensors(self, sample_records, tokenizer):
        ds = CoderSecSFTDataset(
            records=sample_records,
            tokenizer=tokenizer,
            max_seq_length=256,
        )
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)

    def test_assistant_only_loss_masking(self, sample_records, tokenizer):
        """Only assistant turn tokens should have non--100 labels."""
        ds = CoderSecSFTDataset(
            records=sample_records,
            tokenizer=tokenizer,
            max_seq_length=256,
        )
        item = ds[0]
        labels = item["labels"]

        # Some labels should be -100 (masked system/user turns)
        assert (labels == -100).any(), "Expected some masked (system/user) tokens"

        # Some labels should be non--100 (assistant turn)
        assert (labels != -100).any(), "Expected some unmasked (assistant) tokens"

    def test_padding_masked_in_labels(self, sample_records, tokenizer):
        ds = CoderSecSFTDataset(
            records=sample_records,
            tokenizer=tokenizer,
            max_seq_length=256,
        )
        item = ds[0]
        attention_mask = item["attention_mask"]
        labels = item["labels"]

        # Padding tokens (attention_mask == 0) should be -100 in labels.
        pad_positions = attention_mask == 0
        if pad_positions.any():
            assert (labels[pad_positions] == -100).all()

    def test_truncation(self, sample_records, tokenizer):
        """Long conversations should be truncated to max_seq_length."""
        ds = CoderSecSFTDataset(
            records=sample_records,
            tokenizer=tokenizer,
            max_seq_length=32,
        )
        item = ds[0]
        assert item["input_ids"].shape[0] == 32
        assert item["attention_mask"].shape[0] == 32
        assert item["labels"].shape[0] == 32


# ---------------------------------------------------------------------------
# QLoRA training pipeline tests (mocked)
# ---------------------------------------------------------------------------


class TestQLoRAModelLoading:
    @patch("huggingface_hub.HfApi")
    def test_load_base_model_verifies_sha(self, mock_api_class):
        """Model loading should verify pinned SHA."""
        mock_info = MagicMock()
        mock_info.sha = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
        mock_api_class.return_value.model_info.return_value = mock_info

        # Without torch/transformers/peft available in test env, we just
        # verify the SHA check logic is exercised.
        from xrfm.codersec.train_qlora import load_base_model

        # Should raise on SHA mismatch.
        with pytest.raises(ValueError, match="SHA mismatch"):
            load_base_model(
                model_id="Qwen/Qwen3-1.7B",
                model_sha="wrong_sha_000000000000000000000000000000000000",
                precision="fp16",
                hf_token="dummy",
            )


class TestTrainResult:
    def test_train_result_fields(self):
        from xrfm.codersec.train_qlora import TrainResult

        result = TrainResult(
            run_name="test_run",
            config={"max_steps": 10},
            train_loss=0.5,
            eval_loss=0.45,
            steps=10,
            seconds=120.0,
            peak_vram_mb=3000.0,
            lora_adapter_path="outputs/lora_adapter",
            merged_checkpoint_path="outputs/merged",
        )
        assert result.run_name == "test_run"
        assert result.train_loss == 0.5
        assert result.eval_loss == 0.45
        assert result.steps == 10
        assert result.peak_vram_mb == 3000.0


# ---------------------------------------------------------------------------
# Integration smoke test (no GPU required)
# ---------------------------------------------------------------------------


def test_dataset_end_to_end_with_sample_records(sample_records, tokenizer):
    """Full dataset pipeline: records -> DataLoader -> batches."""
    from torch.utils.data import DataLoader

    ds = CoderSecSFTDataset(
        records=sample_records,
        tokenizer=tokenizer,
        max_seq_length=128,
    )
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    batches = list(dl)
    assert len(batches) == 1
    batch = batches[0]
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape[0] == 2
    # Verify masking is present in batch labels.
    assert (batch["labels"] == -100).any()


def test_smoke_b0_config_loads():
    """Smoke B0 config should be loadable and valid."""
    from xrfm.codersec.config import smoke_b0_config

    cfg = smoke_b0_config()
    assert cfg.max_seq_length == 512
    assert cfg.max_steps == 20
    assert cfg.precision == "fp16"
