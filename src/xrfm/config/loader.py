"""
YAML → typed-config loading for XRFM.

Resolution order (Phase 0 contract — **no CWD dependence**):

1. ``load_config(path)`` / ``ConfigLoader(path)`` — explicit path, always
   wins. Relative paths are resolved against the process working directory
   *only when the caller passes one* (that is the caller's choice, not an
   implicit default).
2. ``load_config(None)`` / ``ConfigLoader()`` — the packaged default config
   resource (``xrfm/config/config.default.yaml``) is used. This works from
   any working directory and from an installed wheel.

The loader never returns raw dicts as configuration: it produces a
validated :class:`xrfm.config.schema.XRFMConfig`. Unknown YAML sections or
fields raise :class:`ConfigError` with the offending names, so typos fail
loudly instead of silently falling back to defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from xrfm.config.schema import (
    ConfigError,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    XRFMConfig,
)

__all__ = [
    "ConfigError",
    "DEFAULT_CONFIG_RESOURCE",
    "ConfigLoader",
    "default_config",
    "load_config",
]

DEFAULT_CONFIG_RESOURCE = "config.default.yaml"


def packaged_default_config_path() -> str:
    """Absolute path of the packaged default config resource."""
    from importlib import resources

    return str(resources.files("xrfm.config").joinpath(DEFAULT_CONFIG_RESOURCE))


def default_config() -> XRFMConfig:
    """Return the packaged default :class:`XRFMConfig` (XRFM-SMALL preset).

    Independent of the current working directory; works from a repo
    checkout or an installed wheel.
    """
    return load_config(packaged_default_config_path())


def load_config(config_path: str | Path | None = None) -> XRFMConfig:
    """Load and validate a YAML config file into an :class:`XRFMConfig`.

    Args:
        config_path: Path to a YAML file. ``None`` selects the packaged
            default resource (never the CWD).

    Raises:
        FileNotFoundError: path given but does not exist.
        ConfigError: YAML invalid, unknown fields/sections, or failed
            validation (message names the field).
    """
    if config_path is None:
        config_path = packaged_default_config_path()
    path = Path(os.fspath(config_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: '{path}'")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in '{path}': {exc}") from exc

    if raw is None:
        raise ConfigError(f"Config file '{path}' is empty")
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file '{path}' must contain a mapping at top level")

    known_sections = {"project", "paths", "model", "training", "datasets"}
    unknown = set(raw) - known_sections
    if unknown:
        raise ConfigError(f"Unknown config section(s) {sorted(unknown)} in '{path}'; expected {sorted(known_sections)}")

    datasets_raw = dict(raw.get("datasets") or {})
    model_raw = dict(raw.get("model") or {})
    # Legacy key mapping: `datasets.default` (pre-Phase-0 name) → `name`.
    if "default" in datasets_raw and "name" not in datasets_raw:
        datasets_raw["name"] = datasets_raw.pop("default")
    # `datasets.max_seq_len` used to live only on the model section; keep
    # accepting that spelling, but both must agree (validated below).
    if "max_seq_len" not in datasets_raw and "max_seq_len" in model_raw:
        datasets_raw["max_seq_len"] = model_raw["max_seq_len"]

    normalized = {
        "project": {"name": (raw.get("project") or {}).get("name", "XR Foundation Model")},
        "model": model_raw,
        "training": dict(raw.get("training") or {}),
        "datasets": datasets_raw,
    }
    cfg = XRFMConfig.from_dict(normalized)
    return cfg


class ConfigLoader:
    """Backward-compatible config facade returning typed sub-configs.

    Phase 0 behavior change: constructing ``ConfigLoader()`` with no argument
    loads the **packaged default resource**, not ``config/config.yaml``
    relative to the CWD. Pass an explicit path for anything else.

    Usage:
        loader = ConfigLoader("config/tiny.yaml")
        cfg: XRFMConfig = loader.config
        model_cfg: ModelConfig = loader.model_config()
        lr = loader.get("training.learning_rate")
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is not None and not isinstance(config_path, (str, Path)):
            raise TypeError(f"config_path must be str, Path, or None, got {type(config_path).__name__}")
        self.path: str | None = None if config_path is None else str(config_path)
        self.config: XRFMConfig = load_config(config_path)
        self._raw: dict[str, Any] = self.config.to_dict()

    # -- typed accessors ------------------------------------------------

    def model_config(self) -> ModelConfig:
        """Validated :class:`ModelConfig`."""
        return self.config.model

    def training_config(self) -> TrainingConfig:
        """Validated :class:`TrainingConfig`."""
        return self.config.training

    def dataset_config(self) -> DatasetConfig:
        """Validated :class:`DatasetConfig` (was a raw dict before Phase 0)."""
        return self.config.data

    # -- dynamic dot access (kept for compatibility) --------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-notation access over the *validated* config representation.

        Reads from the serialized config (``model.d_model``,
        ``training.learning_rate``, ``datasets.shuffle``). Sections map to
        the schema names: ``model``, ``training``, ``datasets``.
        """
        if not isinstance(key, str):
            raise TypeError(f"key must be str, got {type(key).__name__}")
        keys = key.split(".")
        value: Any = self._raw
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get_raw(self) -> dict[str, Any]:
        """Serializable dict form of the validated config (for manifests)."""
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self._raw.items()}

    def __repr__(self) -> str:  # pragma: no cover - trivial
        src = self.path if self.path is not None else "<packaged default>"
        return f"ConfigLoader({src})"
