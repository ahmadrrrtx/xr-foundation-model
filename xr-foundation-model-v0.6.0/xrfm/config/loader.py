"""
Loader for XRFM configuration system.

Design: Config-driven architecture. Single source of truth.
Every model component reads hyperparameters from this loader,
ensuring that changing from XRFM-10M to XRFM-1B requires only
configuration updates — no code rewrites.

Reference concepts: Professional AI lab configuration patterns
(Hugging Face config.json, Meta internal config systems).
Implementation is original.
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Structured model architecture parameters."""
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int
    dropout: float
    use_rope: bool = True
    use_rmsnorm: bool = True
    use_swiglu: bool = True


@dataclass
class TrainingConfig:
    """Structured training hyperparameters."""
    batch_size: int
    max_steps: int
    warmup_steps: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    mixed_precision: bool = True
    checkpoint_every: int = 1000
    resume_from: Optional[str] = None


class ConfigLoader:
    """Production-quality configuration loader for XRFM."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"XRFM config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)
        self.raw = raw
        # Basic validation
        for section in ("project", "model", "training", "paths"):
            if section not in raw:
                raise ValueError(f"XRFM config missing required section: {section}")

    def get(self, key_path: str, default: Optional[Any] = None) -> Any:
        """Retrieve nested value via dot notation (e.g., 'model.d_model')."""
        keys = key_path.split(".")
        value: Any = self.raw
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def model_config(self) -> ModelConfig:
        """Return typed model architecture config."""
        data = self.get("model", {})
        return ModelConfig(
            vocab_size=int(data.get("vocab_size", 50304)),
            d_model=int(data.get("d_model", 256)),
            n_layers=int(data.get("n_layers", 6)),
            n_heads=int(data.get("n_heads", 8)),
            d_ff=int(data.get("d_ff", 1024)),
            max_seq_len=int(data.get("max_seq_len", 512)),
            dropout=float(data.get("dropout", 0.1)),
            use_rope=bool(data.get("use_rope", True)),
            use_rmsnorm=bool(data.get("use_rmsnorm", True)),
            use_swiglu=bool(data.get("use_swiglu", True)),
        )

    def training_config(self) -> TrainingConfig:
        """Return typed training hyperparameter config."""
        data = self.get("training", {})
        return TrainingConfig(
            batch_size=int(data.get("batch_size", 32)),
            max_steps=int(data.get("max_steps", 50000)),
            warmup_steps=int(data.get("warmup_steps", 1000)),
            learning_rate=float(data.get("learning_rate", 0.001)),
            weight_decay=float(data.get("weight_decay", 0.01)),
            gradient_clip=float(data.get("gradient_clip", 1.0)),
            mixed_precision=bool(data.get("mixed_precision", True)),
            checkpoint_every=int(data.get("checkpoint_every", 1000)),
            resume_from=data.get("resume_from"),
        )


class ConfigPresets:
    """Preset profiles for rapid model scaling."""

    @staticmethod
    def xrfm_10m() -> Dict[str, Any]:
        return {
            "model": {
                "vocab_size": 50304,
                "d_model": 256,
                "n_layers": 6,
                "n_heads": 8,
                "d_ff": 1024,
                "max_seq_len": 512,
                "dropout": 0.1,
            },
            "training": {
                "batch_size": 32,
                "max_steps": 50000,
                "learning_rate": 0.001,
            },
        }

    @staticmethod
    def xrfm_100m() -> Dict[str, Any]:
        return {
            "model": {
                "vocab_size": 50304,
                "d_model": 768,
                "n_layers": 12,
                "n_heads": 12,
                "d_ff": 3072,
                "max_seq_len": 1024,
                "dropout": 0.1,
            },
            "training": {
                "batch_size": 16,
                "max_steps": 100000,
                "learning_rate": 0.0005,
            },
        }

    @staticmethod
    def xrfm_1b() -> Dict[str, Any]:
        return {
            "model": {
                "vocab_size": 100000,
                "d_model": 2048,
                "n_layers": 24,
                "n_heads": 16,
                "d_ff": 8192,
                "max_seq_len": 2048,
                "dropout": 0.1,
            },
            "training": {
                "batch_size": 8,
                "max_steps": 300000,
                "learning_rate": 0.0001,
            },
        }
