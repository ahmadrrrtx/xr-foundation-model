"""
Checksums and immutability for XRFM datasets.

Every important artifact should be identifiable.
Use checksums for raw source snapshots, processed shards, manifests, tokenizer files, configuration.

Dataset identity derived from inputs:
dataset_id = hash(source_versions + processing_config + tokenizer_version + pipeline_version)
Same inputs + same pipeline version + same config = same dataset identity.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, List


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_config(config: Dict[str, Any]) -> str:
    """Stable hash of config dict (sorted keys)."""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return sha256_text(payload)


def compute_dataset_id(
    source_versions: Dict[str, str],
    processing_config: Dict[str, Any],
    tokenizer_version: str,
    pipeline_version: str,
    extra: Dict[str, Any] | None = None,
) -> str:
    """
    Compute stable dataset identity.
    Same inputs + same pipeline version + same config = same dataset ID.
    """
    components = {
        "sources": source_versions,
        "processing": processing_config,
        "tokenizer": tokenizer_version,
        "pipeline": pipeline_version,
    }
    if extra:
        components["extra"] = extra

    payload = json.dumps(components, sort_keys=True, ensure_ascii=False)
    full_hash = sha256_text(payload)
    # Short ID for readability
    return f"xrfm-ds-{full_hash[:16]}"


def verify_file_checksum(path: str, expected_sha256: str) -> bool:
    actual = sha256_file(path)
    return actual == expected_sha256


def generate_checksums_manifest(file_paths: List[str]) -> Dict[str, str]:
    """Generate dict of file -> sha256."""
    result = {}
    for p in file_paths:
        if os.path.isfile(p):
            result[p] = sha256_file(p)
    return result
