"""XRFM CoderSec (Track B) package — defensive security coding model toolkit.

Provides QLoRA training configuration, SFT data validation, and training
utilities for the XRFM CoderSec base-model bake-off and fine-tuning pipeline
(GATE-4 through GATE-5).
"""

from __future__ import annotations

from xrfm.codersec.config import CoderSecConfig
from xrfm.codersec.sft_validator import SFTRecord, SFTValidator, ValidationReport

__all__ = ["CoderSecConfig", "SFTValidator", "SFTRecord", "ValidationReport"]
