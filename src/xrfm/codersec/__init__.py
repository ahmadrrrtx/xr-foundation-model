"""XRFM CoderSec (Track B) package — defensive security coding model toolkit.

Provides QLoRA training configuration, SFT data validation, dataset loading,
QLoRA training, and model merging for the XRFM CoderSec base-model bake-off
and fine-tuning pipeline (GATE-4 through GATE-5).
"""

from __future__ import annotations

from xrfm.codersec.config import CoderSecConfig
from xrfm.codersec.dataset import DEFAULT_MAX_SEQ_LENGTH, CoderSecSFTDataset, get_tokenizer
from xrfm.codersec.merge import merge_and_publish, merge_from_config
from xrfm.codersec.sft_validator import SFTRecord, SFTValidator, ValidationReport
from xrfm.codersec.train_qlora import TrainResult, run_smoke_b0, train_qlora

__all__ = [
    "CoderSecConfig",
    "CoderSecSFTDataset",
    "get_tokenizer",
    "DEFAULT_MAX_SEQ_LENGTH",
    "merge_and_publish",
    "merge_from_config",
    "SFTValidator",
    "SFTRecord",
    "ValidationReport",
    "train_qlora",
    "run_smoke_b0",
    "TrainResult",
]
