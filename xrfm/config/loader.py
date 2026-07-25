"""
ConfigLoader for XR Foundation Model (XRFM).

Loads and validates YAML configuration, provides typed dataclass configs,
and exposes dot-notation access for all hyperparameters.
"""

import os
from dataclasses import dataclass
from typing import Any

import yaml


class ConfigLoader:
    """
    Loads config.yaml and provides dot-notation access.

    Usage:
        loader = ConfigLoader("config/config.yaml")
        model_cfg = loader.model_config()
        lr = loader.get("training.learning_rate")
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        if not isinstance(config_path, str):
            raise TypeError(f"config_path must be str, got {type(config_path).__name__}")

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config not found: '{config_path}'")

        with open(config_path) as f:
            self._config = yaml.safe_load(f)

        if self._config is None:
            raise ValueError(f"Config file '{config_path}' is empty or invalid")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value using dot notation.

        Args:
            key: Dot-separated key (e.g., "training.learning_rate").
            default: Default value if key not found.

        Returns:
            The config value, or default if not found.
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def model_config(self) -> "ModelConfig":
        """Return a ModelConfig dataclass from the config."""
        m = self._config.get("model", {})
        return ModelConfig(
            vocab_size=m.get("vocab_size", 50304),
            d_model=m.get("d_model", 256),
            n_layers=m.get("n_layers", 6),
            n_heads=m.get("n_heads", 8),
            d_ff=m.get("d_ff", 1024),
            max_seq_len=m.get("max_seq_len", 512),
            dropout=m.get("dropout", 0.1),
            use_rope=m.get("use_rope", True),
            use_rmsnorm=m.get("use_rmsnorm", True),
            use_swiglu=m.get("use_swiglu", True),
        )

    def training_config(self) -> "TrainingConfig":
        """Return a TrainingConfig dataclass from the config."""
        t = self._config.get("training", {})
        return TrainingConfig(
            batch_size=t.get("batch_size", 32),
            max_steps=t.get("max_steps", 50000),
            warmup_steps=t.get("warmup_steps", 1000),
            learning_rate=t.get("learning_rate", 0.001),
            weight_decay=t.get("weight_decay", 0.01),
            gradient_clip=t.get("gradient_clip", 1.0),
            mixed_precision=t.get("mixed_precision", True),
            checkpoint_every=t.get("checkpoint_every", 1000),
            resume_from=t.get("resume_from", None),
            grad_accum_steps=t.get("grad_accum_steps", 1),
            use_ddp=t.get("use_ddp", False),
            use_fsdp=t.get("use_fsdp", False),
        )

    def dataset_config(self) -> dict[str, Any]:
        """Return the dataset config dictionary."""
        return self._config.get("datasets", {})


@dataclass
class ModelConfig:
    """Typed model configuration."""

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int
    dropout: float
    use_rope: bool
    use_rmsnorm: bool
    use_swiglu: bool


@dataclass
class TrainingConfig:
    """Typed training configuration."""

    batch_size: int
    max_steps: int
    warmup_steps: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    mixed_precision: bool
    checkpoint_every: int
    resume_from: str | None
    grad_accum_steps: int
    use_ddp: bool
    use_fsdp: bool
