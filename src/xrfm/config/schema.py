"""
Typed, validated configuration schema for XRFM.

This module is the single definition of *what a valid XRFM configuration
is*. Every subsystem (model, data, training) is constructed from these
dataclasses — never from raw YAML dictionaries or file paths — so an
invalid configuration fails here, with a precise message, before any
torch object is allocated.

Design notes
------------
* Plain stdlib dataclasses (no pydantic): XRFM keeps a minimal dependency
  set and the config surface is small enough to validate explicitly.
  See ``docs/research/PHASE_0_BEST_PRACTICES.md``.
* ``validate()`` is idempotent and called automatically in
  ``__post_init__``; cross-field invariants (e.g. ``d_model % n_heads``)
  live next to the fields they constrain.
* Serialization round-trips exactly: ``XRFMConfig.to_dict() ->
  from_dict()`` is identity-preserving, which makes configs stable,
  hashable inputs to reproducible runs.

Example:
    >>> from xrfm.config.schema import ModelConfig
    >>> cfg = ModelConfig(vocab_size=2048, d_model=128, n_layers=4,
    ...                   n_heads=4, d_ff=512, max_seq_len=256,
    ...                   dropout=0.1, use_rope=True, use_rmsnorm=True,
    ...                   use_swiglu=True, use_bias=True)
    >>> cfg.d_model % cfg.n_heads == 0
    True
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any

__all__ = [
    "ConfigError",
    "ModelConfig",
    "TrainingConfig",
    "DatasetConfig",
    "XRFMConfig",
]


class ConfigError(ValueError):
    """Raised when a configuration value or combination of values is invalid.

    The message always names the offending field(s) and the received value,
    e.g. ``model.d_model (100) must be divisible by model.n_heads (6)``.
    """


def _err(section: str, name: str, message: str) -> ConfigError:
    return ConfigError(f"{section}.{name}: {message}")


@dataclass
class ModelConfig:
    """Architecture hyperparameters of the decoder-only transformer.

    Fields
    ------
    vocab_size:
        Number of embedding rows / logits columns. Must equal the trained
        tokenizer's real vocabulary size — ``xrfm.models.XRFMModel`` accepts
        an override, and training scripts derive it from the tokenizer.
    pad_token_id:
        Token id used for padding (from the tokenizer contract), or ``None``
        when the tokenizer defines no pad token. Passed to the embedding as
        ``padding_idx`` when set. Phase 0 note: this replaces the previous
        hard-coded ``padding_idx=0``.
    """

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
    use_bias: bool = True
    pad_token_id: int | None = None

    def validate(self) -> None:
        """Raise :class:`ConfigError` on any invalid field or combination."""
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "max_seq_len"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise _err("model", name, f"must be a positive int, got {value!r}")
        if not isinstance(self.dropout, (int, float)) or not (0.0 <= self.dropout < 1.0):
            raise _err("model", "dropout", f"must be in [0, 1), got {self.dropout!r}")
        if self.pad_token_id is not None:
            if not isinstance(self.pad_token_id, int) or isinstance(self.pad_token_id, bool):
                raise _err("model", "pad_token_id", f"must be an int or None, got {self.pad_token_id!r}")
            if not (0 <= self.pad_token_id < self.vocab_size):
                raise _err(
                    "model",
                    "pad_token_id",
                    f"must be in [0, vocab_size={self.vocab_size}), got {self.pad_token_id!r}",
                )
        if self.d_model % self.n_heads != 0:
            raise _err(
                "model",
                "d_model",
                f"({self.d_model}) must be divisible by model.n_heads ({self.n_heads})",
            )

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ConfigError(f"model: unknown field(s) {sorted(unknown)}")
        missing = required_fields(cls) - set(data)
        if missing:
            raise ConfigError(f"model: missing required field(s) {sorted(missing)}")
        return cls(**data)


@dataclass
class TrainingConfig:
    """Optimization / loop hyperparameters."""

    batch_size: int = 8
    max_steps: int = 5000
    warmup_steps: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    mixed_precision: bool = False
    checkpoint_every: int = 500
    resume_from: str | None = None
    grad_accum_steps: int = 1
    use_ddp: bool = False
    use_fsdp: bool = False
    seed: int = 42
    error_if_nonfinite: bool = False
    ignore_index: int = -100
    eval_every: int = 0
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    device: str | None = None
    deterministic: bool = False

    def validate(self) -> None:
        for name in ("batch_size", "max_steps", "warmup_steps", "checkpoint_every", "grad_accum_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise _err("training", name, f"must be a positive int, got {value!r}")
        if not (isinstance(self.learning_rate, (int, float)) and self.learning_rate > 0):
            raise _err("training", "learning_rate", f"must be > 0, got {self.learning_rate!r}")
        if not (isinstance(self.weight_decay, (int, float)) and self.weight_decay >= 0):
            raise _err("training", "weight_decay", f"must be >= 0, got {self.weight_decay!r}")
        if not (isinstance(self.gradient_clip, (int, float)) and self.gradient_clip >= 0):
            raise _err("training", "gradient_clip", f"must be >= 0, got {self.gradient_clip!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise _err("training", "seed", f"must be a non-negative int, got {self.seed!r}")
        if not isinstance(self.ignore_index, int) or isinstance(self.ignore_index, bool):
            raise _err("training", "ignore_index", f"must be an int, got {self.ignore_index!r}")
        if not isinstance(self.error_if_nonfinite, bool):
            raise _err("training", "error_if_nonfinite", f"must be a bool, got {self.error_if_nonfinite!r}")
        if self.eval_every < 0:
            raise _err("training", "eval_every", f"must be >= 0, got {self.eval_every!r}")
        if self.prefetch_factor is not None and (
            not isinstance(self.prefetch_factor, int) or self.prefetch_factor <= 0
        ):
            raise _err("training", "prefetch_factor", f"must be a positive int or None, got {self.prefetch_factor!r}")
        if not isinstance(self.num_workers, int) or isinstance(self.num_workers, bool) or self.num_workers < 0:
            raise _err("training", "num_workers", f"must be a non-negative int, got {self.num_workers!r}")
        if self.num_workers > 0 and self.prefetch_factor is None:
            # torch requires an explicit prefetch_factor when workers > 0 on
            # some versions; default to 2 like torch.utils.data does.
            self.prefetch_factor = 2
        if self.use_ddp and self.use_fsdp:
            raise _err("training", "use_ddp", "use_ddp and use_fsdp are mutually exclusive")
        if self.resume_from is not None and not isinstance(self.resume_from, str):
            raise _err("training", "resume_from", f"must be a str or None, got {self.resume_from!r}")
        if self.device is not None and not isinstance(self.device, str):
            raise _err("training", "device", f"must be a str or None, got {self.device!r}")
        # Cross-field: warmup cannot exceed the run length.
        if self.warmup_steps >= self.max_steps:
            raise _err(
                "training",
                "warmup_steps",
                f"({self.warmup_steps}) must be < training.max_steps ({self.max_steps})",
            )

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ConfigError(f"training: unknown field(s) {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class DatasetConfig:
    """Data-pipeline hyperparameters.

    Every field has a real effect:

    * ``name``/``path`` locate the raw text corpus.
    * ``max_seq_len`` fixes the packing/chunk length (and therefore the
      model's context during training).
    * ``train_ratio`` / ``val_ratio`` / ``test_ratio`` define the split; the
      remainder logic validates they are sane and sum to ~1.
    * ``shuffle`` controls whether documents (lines) are shuffled before the
      sequential split; when True the split is deterministic under ``seed``.
    * ``seed`` seeds the shuffle — it is *always* consumed (Phase 0 fix;
      previously accepted but ignored).
    * ``dedup`` enables exact-line deduplication before splitting.
    """

    name: str = "tiny_corpus"
    path: str = "data/datasets/corpus.txt"
    max_seq_len: int = 256
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    shuffle: bool = True
    seed: int = 42
    dedup: bool = True

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise _err("datasets", "name", "must be a non-empty str")
        if not isinstance(self.path, str) or not self.path:
            raise _err("datasets", "path", "must be a non-empty str")
        if not isinstance(self.max_seq_len, int) or isinstance(self.max_seq_len, bool) or self.max_seq_len <= 0:
            raise _err("datasets", "max_seq_len", f"must be a positive int, got {self.max_seq_len!r}")
        for name in ("train_ratio", "val_ratio", "test_ratio"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                raise _err("datasets", name, f"must be in [0, 1], got {value!r}")
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.01:
            raise _err("datasets", "train_ratio", f"ratios must sum to ~1.0, got {total}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise _err("datasets", "seed", f"must be a non-negative int, got {self.seed!r}")

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetConfig:
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ConfigError(f"datasets: unknown field(s) {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in allowed})


def required_fields(cls: type) -> set[str]:
    """Names of dataclass fields without defaults (required at construction)."""
    return {f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING}


@dataclass
class XRFMConfig:
    """Complete XRFM run configuration: model + training + dataset.

    This is the object that flows through the framework::

        cfg = load_config("config/tiny.yaml")
        model = XRFMModel(cfg)
        trainer = Trainer(model, config=cfg)
        dataset = TextDataset(cfg.data, tokenizer)

    ``version`` is intentionally *not* stored here — the single version
    source is ``xrfm.__version__``.
    """

    model: ModelConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DatasetConfig = field(default_factory=DatasetConfig)
    project: str = "XR Foundation Model"

    def validate(self) -> None:
        if not isinstance(self.model, ModelConfig):
            raise ConfigError(f"model: expected ModelConfig, got {type(self.model).__name__}")
        if not isinstance(self.training, TrainingConfig):
            raise ConfigError(f"training: expected TrainingConfig, got {type(self.training).__name__}")
        if not isinstance(self.data, DatasetConfig):
            raise ConfigError(f"data: expected DatasetConfig, got {type(self.data).__name__}")
        self.model.validate()
        self.training.validate()
        self.data.validate()
        # Cross-sub-config invariants.
        if self.data.max_seq_len != self.model.max_seq_len:
            raise ConfigError(
                f"datasets.max_seq_len ({self.data.max_seq_len}) must equal "
                f"model.max_seq_len ({self.model.max_seq_len}); the dataset packing "
                "length defines the model context during training"
            )

    def __post_init__(self) -> None:
        self.validate()

    # -- serialization ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Stable, JSON/YAML-serializable representation (nested dicts)."""
        return {
            "project": {"name": self.project},
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "datasets": self.data.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> XRFMConfig:
        if not isinstance(data, dict):
            raise ConfigError(f"config must be a mapping, got {type(data).__name__}")
        model_data = data.get("model")
        if not isinstance(model_data, dict):
            raise ConfigError("config: missing required 'model' section")
        project = (data.get("project") or {}).get("name", "XR Foundation Model")
        return cls(
            model=ModelConfig.from_dict(model_data),
            training=TrainingConfig.from_dict(data.get("training") or {}),
            data=DatasetConfig.from_dict(data.get("datasets") or {}),
            project=project,
        )

    def replace(self, **changes: Any) -> XRFMConfig:
        """Return a validated copy with nested overrides applied.

        Accepts dotted keys for nested sections::

            cfg.replace(**{"model.d_model": 256, "training.max_steps": 100})
        """
        model_data = self.model.to_dict()
        training_data = self.training.to_dict()
        data_data = self.data.to_dict()
        project = self.project
        targets = {"model": model_data, "training": training_data, "datasets": data_data}
        for key, value in changes.items():
            if "." in key:
                section, field_name = key.split(".", 1)
                if section not in targets or field_name not in targets[section]:
                    raise ConfigError(f"unknown config field: {key}")
                targets[section][field_name] = value
            elif key == "project":
                project = value
            elif key in ("data", "datasets") and isinstance(value, dict):
                targets["datasets"].update(value)
            elif key == "model" and isinstance(value, dict):
                targets["model"].update(value)
            elif key == "training" and isinstance(value, dict):
                targets["training"].update(value)
            else:
                raise ConfigError(f"unknown config field: {key}")
        return XRFMConfig.from_dict(
            {"project": {"name": project}, "model": model_data, "training": training_data, "datasets": data_data}
        )


def config_hash(cfg: XRFMConfig) -> str:
    """Deterministic hash of a configuration (for run manifests).

    Excludes nothing: two runs with identical configs share a hash; any
    field difference changes it.
    """
    import hashlib
    import json

    payload = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
