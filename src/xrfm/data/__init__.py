"""XRFM data subsystem: dataset representation, splitting, packing, manifests.

Public surface (use these, not the private module layout):

    from xrfm.data import TextDataset, DatasetConfig, split_dataset_lines, chunk_token_ids

Responsibilities are separated:

* :mod:`xrfm.data.splits`   — deterministic train/val/test splitting
* :mod:`xrfm.data.packing`  — validated fixed-length chunking
* :mod:`xrfm.data.dataset`  — torch Dataset over a packed corpus
* :mod:`xrfm.data.manifest` — dataset provenance manifests
"""

from xrfm.config.schema import DatasetConfig
from xrfm.data.dataset import IGNORE_INDEX, TextDataset, XRFMTextDataset, verify_text_file
from xrfm.data.manifest import (
    DatasetManifest,
    build_dataset_manifest,
    build_manifest,
    save_manifest,
)
from xrfm.data.packing import PackingError, chunk_text, chunk_token_ids
from xrfm.data.splits import SplitRatiosError, normalize_text, split_dataset, split_dataset_lines

__all__ = [
    "DatasetConfig",
    "DatasetManifest",
    "IGNORE_INDEX",
    "PackingError",
    "SplitRatiosError",
    "TextDataset",
    "XRFMTextDataset",
    "build_dataset_manifest",
    "build_manifest",
    "chunk_text",
    "chunk_token_ids",
    "normalize_text",
    "save_manifest",
    "split_dataset",
    "split_dataset_lines",
    "verify_text_file",
]
