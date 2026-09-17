"""XRFM configuration subsystem.

Public surface:

    from xrfm.config import XRFMConfig, ModelConfig, TrainingConfig, DatasetConfig
    from xrfm.config import load_config, default_config, ConfigLoader

Schema (``xrfm.config.schema``) defines the validated typed configuration;
``loader`` resolves YAML paths or the packaged default resource into those
objects. Never construct subsystem objects from raw YAML dicts.
"""

from xrfm.config.loader import (
    DEFAULT_CONFIG_RESOURCE,
    ConfigLoader,
    default_config,
    load_config,
    packaged_default_config_path,
)
from xrfm.config.schema import (
    ConfigError,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    XRFMConfig,
    config_hash,
)

__all__ = [
    "DEFAULT_CONFIG_RESOURCE",
    "ConfigError",
    "ConfigLoader",
    "DatasetConfig",
    "ModelConfig",
    "TrainingConfig",
    "XRFMConfig",
    "config_hash",
    "default_config",
    "load_config",
    "packaged_default_config_path",
]
