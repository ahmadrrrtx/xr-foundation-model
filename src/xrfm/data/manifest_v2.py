"""
Dataset manifest system for XRFM Phase 1.

This is one of the most important deliverables.
Every processed dataset must have machine-readable manifest.

Example:
{
  "dataset_name": "xrfm-pretrain",
  "dataset_version": "0.1.0",
  "created_at": "...",
  "git_commit": "...",
  "processing_pipeline_version": "...",
  "tokenizer": {...},
  "sources": [...],
  "documents": 123456,
  "tokens": 987654321,
  "shards": 128,
  "sequence_length": 4096,
  "split": {...},
  "checksums": [...]
}

Manifest must allow answering: Exactly what data produced this training corpus?
"""

from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

from xrfm.data.checksums import sha256_file, hash_config, compute_dataset_id


def _get_git_commit() -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


@dataclass
class SourceManifestEntry:
    name: str
    uri: str
    type: str
    license: str
    license_url: str
    version: str
    checksum: str
    language: str
    document_count: int
    token_count: int
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenizerManifestEntry:
    name: str
    version: str
    hash: str
    vocab_size: int
    pad_token_id: Optional[int]
    eos_token_id: Optional[int]
    bos_token_id: Optional[int]
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitManifestEntry:
    train_documents: int
    val_documents: int
    test_documents: int
    train_tokens: int
    val_tokens: int
    test_tokens: int
    method: str
    seed: int


@dataclass
class ShardManifestEntry:
    shard_id: int
    path: str
    num_sequences: int
    num_tokens: int
    sha256: str
    dtype: str


@dataclass
class DatasetManifestV2:
    dataset_name: str
    dataset_version: str
    dataset_id: str
    created_at: str
    git_commit: str
    processing_pipeline_version: str
    processing_config: Dict[str, Any]
    processing_config_hash: str
    tokenizer: TokenizerManifestEntry
    sources: List[SourceManifestEntry]
    documents_total: int
    documents_train: int
    documents_val: int
    documents_test: int
    tokens_total: int
    tokens_train: int
    tokens_val: int
    tokens_test: int
    shards: List[ShardManifestEntry]
    num_shards: int
    sequence_length: int
    split: SplitManifestEntry
    checksums: Dict[str, str]
    reports: Dict[str, str] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetManifestV2:
        # Reconstruct nested
        tokenizer_data = data.get("tokenizer", {})
        tokenizer = TokenizerManifestEntry(**tokenizer_data)

        sources = [SourceManifestEntry(**s) for s in data.get("sources", [])]
        shards = [ShardManifestEntry(**sh) for sh in data.get("shards", [])]
        split_data = data.get("split", {})
        split_entry = SplitManifestEntry(**split_data) if split_data else SplitManifestEntry(0, 0, 0, 0, 0, 0, "hash", 42)

        return cls(
            dataset_name=data["dataset_name"],
            dataset_version=data["dataset_version"],
            dataset_id=data["dataset_id"],
            created_at=data["created_at"],
            git_commit=data["git_commit"],
            processing_pipeline_version=data["processing_pipeline_version"],
            processing_config=data.get("processing_config", {}),
            processing_config_hash=data.get("processing_config_hash", ""),
            tokenizer=tokenizer,
            sources=sources,
            documents_total=data.get("documents_total", 0),
            documents_train=data.get("documents_train", 0),
            documents_val=data.get("documents_val", 0),
            documents_test=data.get("documents_test", 0),
            tokens_total=data.get("tokens_total", 0),
            tokens_train=data.get("tokens_train", 0),
            tokens_val=data.get("tokens_val", 0),
            tokens_test=data.get("tokens_test", 0),
            shards=shards,
            num_shards=data.get("num_shards", len(shards)),
            sequence_length=data.get("sequence_length", 0),
            split=split_entry,
            checksums=data.get("checksums", {}),
            reports=data.get("reports", {}),
            extra=data.get("extra", {}),
        )

    @classmethod
    def load(cls, path: str) -> DatasetManifestV2:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def build_manifest_v2(
    dataset_name: str,
    dataset_version: str,
    processing_config: Dict[str, Any],
    tokenizer_info,
    sources: List[Dict[str, Any]],
    splits: Dict[str, List],
    token_counts: Dict[str, int],
    shard_infos: List,
    sequence_length: int,
    split_config: Dict[str, Any],
    checksums: Dict[str, str] | None = None,
    pipeline_version: str = "xrfm-pipeline-v1",
) -> DatasetManifestV2:
    """
    Build manifest from pipeline artifacts.
    """

    # Compute dataset ID — exclude output_dir and other non-deterministic paths for reproducibility
    source_versions = {s.get("name", "unknown"): s.get("version", "unknown") for s in sources}
    tokenizer_version = getattr(tokenizer_info, "version", "unknown") if tokenizer_info else "unknown"
    # Sanitize processing_config for ID: remove output_dir, sharding.output_dir, etc.
    sanitized_config = dict(processing_config)
    sanitized_config.pop("output_dir", None)
    # Also remove sharding output_dir if present
    if "sharding" in sanitized_config and isinstance(sanitized_config["sharding"], dict):
        sh = dict(sanitized_config["sharding"])
        sh.pop("output_dir", None)
        sanitized_config["sharding"] = sh
    dataset_id = compute_dataset_id(
        source_versions=source_versions,
        processing_config=sanitized_config,
        tokenizer_version=tokenizer_version,
        pipeline_version=pipeline_version,
    )

    config_hash = hash_config(processing_config)

    # Tokenizer entry
    if tokenizer_info:
        tok_entry = TokenizerManifestEntry(
            name=getattr(tokenizer_info, "name", "unknown"),
            version=getattr(tokenizer_info, "version", "unknown"),
            hash=getattr(tokenizer_info, "hash", "unknown"),
            vocab_size=getattr(tokenizer_info, "vocab_size", 0),
            pad_token_id=getattr(tokenizer_info, "pad_token_id", None),
            eos_token_id=getattr(tokenizer_info, "eos_token_id", None),
            bos_token_id=getattr(tokenizer_info, "bos_token_id", None),
            config=getattr(tokenizer_info, "config", {}),
        )
    else:
        tok_entry = TokenizerManifestEntry(
            name="unknown", version="unknown", hash="unknown", vocab_size=0, pad_token_id=None, eos_token_id=None, bos_token_id=None
        )

    # Sources
    source_entries = []
    for s in sources:
        entry = SourceManifestEntry(
            name=s.get("name", "unknown"),
            uri=s.get("uri", ""),
            type=s.get("type", "text"),
            license=s.get("license", "unknown"),
            license_url=s.get("license_url", ""),
            version=s.get("version", "unknown"),
            checksum=s.get("checksum", ""),
            language=s.get("language", "en"),
            document_count=s.get("document_count", 0),
            token_count=s.get("token_count", 0),
            extra=s.get("extra", {}),
        )
        source_entries.append(entry)

    # Shards
    shard_entries = []
    for si in shard_infos:
        # si may be ShardInfo from sharding.py
        entry = ShardManifestEntry(
            shard_id=getattr(si, "shard_id", 0),
            path=getattr(si, "path", ""),
            num_sequences=getattr(si, "num_sequences", 0),
            num_tokens=getattr(si, "num_tokens", 0),
            sha256=getattr(si, "sha256", ""),
            dtype=getattr(si, "dtype", "uint32"),
        )
        shard_entries.append(entry)

    # Splits
    train_docs = len(splits.get("train", []))
    val_docs = len(splits.get("val", []))
    test_docs = len(splits.get("test", []))

    split_entry = SplitManifestEntry(
        train_documents=train_docs,
        val_documents=val_docs,
        test_documents=test_docs,
        train_tokens=token_counts.get("train", 0),
        val_tokens=token_counts.get("val", 0),
        test_tokens=token_counts.get("test", 0),
        method=split_config.get("method", "hash"),
        seed=split_config.get("seed", 42),
    )

    manifest = DatasetManifestV2(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_id=dataset_id,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_commit=_get_git_commit(),
        processing_pipeline_version=pipeline_version,
        processing_config=processing_config,
        processing_config_hash=config_hash,
        tokenizer=tok_entry,
        sources=source_entries,
        documents_total=train_docs + val_docs + test_docs,
        documents_train=train_docs,
        documents_val=val_docs,
        documents_test=test_docs,
        tokens_total=token_counts.get("total", 0),
        tokens_train=token_counts.get("train", 0),
        tokens_val=token_counts.get("val", 0),
        tokens_test=token_counts.get("test", 0),
        shards=shard_entries,
        num_shards=len(shard_entries),
        sequence_length=sequence_length,
        split=split_entry,
        checksums=checksums or {},
        reports={},
        extra={},
    )

    return manifest
