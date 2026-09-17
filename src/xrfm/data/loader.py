"""Backward-compatible path: ``xrfm.data.loader``.

Pre-Phase-0, all data functionality lived in this single module. It now
lives in :mod:`xrfm.data.dataset` / :mod:`.splits` / :mod:`.packing` /
:mod:`.manifest`; this module re-exports everything under the historical
names so existing code keeps working. Prefer importing from ``xrfm.data``.
"""

from xrfm.config.schema import DatasetConfig
from xrfm.data.dataset import IGNORE_INDEX, TextDataset, XRFMTextDataset, verify_text_file
from xrfm.data.manifest import build_manifest, save_manifest
from xrfm.data.packing import chunk_text, chunk_token_ids
from xrfm.data.splits import normalize_text, split_dataset, split_dataset_lines

__all__ = [
    "IGNORE_INDEX",
    "DatasetConfig",
    "TextDataset",
    "XRFMTextDataset",
    "build_manifest",
    "chunk_text",
    "chunk_token_ids",
    "normalize_text",
    "save_manifest",
    "split_dataset",
    "split_dataset_lines",
    "verify_text_file",
]


def load_config_for_dataset(config_loader) -> DatasetConfig:
    """Build a :class:`DatasetConfig` from a legacy ``ConfigLoader``-like object.

    Reads ``datasets.*`` and ``model.max_seq_len`` via dot-notation access,
    mapping pre-Phase-0 YAML keys (``datasets.default`` → ``name``).
    """
    datasets_cfg = config_loader.get("datasets", {}) or {}
    model_cfg = config_loader.get("model", {}) or {}
    max_seq_len = datasets_cfg.get("max_seq_len", model_cfg.get("max_seq_len", 256))
    return DatasetConfig(
        name=datasets_cfg.get("name", datasets_cfg.get("default", "corpus")),
        path=datasets_cfg.get("path", "data/datasets/corpus.txt"),
        max_seq_len=max_seq_len,
        train_ratio=datasets_cfg.get("train_ratio", 0.9),
        val_ratio=datasets_cfg.get("val_ratio", 0.05),
        test_ratio=datasets_cfg.get("test_ratio", 0.05),
        shuffle=datasets_cfg.get("shuffle", True),
        seed=datasets_cfg.get("seed", 42),
        dedup=datasets_cfg.get("dedup", True),
    )
