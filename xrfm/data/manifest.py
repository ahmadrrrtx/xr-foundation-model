"""
Dataset provenance manifests for XRFM (Phase 33).

Every training experiment must be able to identify:
    dataset name, version, source, license,
    preprocessing version, tokenizer version,
    number of documents, token count, train/val token counts,
    deduplication method, filtering method, checksum/hash.

`build_dataset_manifest()` computes all of these from the artifacts actually
used (file hash, line counts, token counts via the tokenizer), so the manifest
is verifiable rather than declarative.

Design: manifest v1 schema, JSON-serializable, stored next to run metadata.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

MANIFEST_VERSION = "xrfm-dataset-manifest-v1"


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file (streamed, memory-safe)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass
class DatasetManifest:
    """Provenance record for a dataset used in one training run."""

    name: str
    version: str
    path: str
    source: str
    license: str
    sha256: str
    preprocessing_version: str = "xrfm-preprocess-v1"
    tokenizer_version: str = ""
    tokenizer_vocab_size: int = 0
    num_documents: int = 0
    total_tokens: int = 0
    train_tokens: int = 0
    val_tokens: int = 0
    test_tokens: int = 0
    dedup_method: str = "exact-line-dedup"
    filtering_method: str = "gutenberg-header/footer-strip; none otherwise"
    language: str = "en"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": MANIFEST_VERSION,
            "name": self.name,
            "version": self.version,
            "path": self.path,
            "source": self.source,
            "license": self.license,
            "sha256": self.sha256,
            "preprocessing_version": self.preprocessing_version,
            "tokenizer_version": self.tokenizer_version,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "num_documents": self.num_documents,
            "total_tokens": self.total_tokens,
            "train_tokens": self.train_tokens,
            "val_tokens": self.val_tokens,
            "test_tokens": self.test_tokens,
            "dedup_method": self.dedup_method,
            "filtering_method": self.filtering_method,
            "language": self.language,
            "extra": self.extra,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


def build_dataset_manifest(
    dataset_path: str,
    tokenizer,
    name: str,
    version: str,
    source: str,
    license: str,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    language: str = "en",
    dedup_method: str = "exact-line-dedup",
    filtering_method: str = "gutenberg-header/footer-strip; none otherwise",
) -> DatasetManifest:
    """Build a DatasetManifest from the actual file + tokenizer.

    Token counts are computed per split using the SAME split logic as the
    dataset loader (line-boundary, exact-line dedup), so the manifest numbers
    match what training actually consumes.
    """
    from xrfm.data.loader import split_dataset_lines

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, encoding="utf-8") as f:
        text = f.read()
    lines = text.splitlines()

    train_lines, val_lines, test_lines = split_dataset_lines(
        lines,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=1 - train_ratio - val_ratio,
        seed=0,
        dedup=True,
    )

    def count_tokens(ls: list[str]) -> int:
        if not ls:
            return 0
        return len(tokenizer.encode("\n".join(ls)))

    total = count_tokens(lines)
    train_t = count_tokens(train_lines)
    val_t = count_tokens(val_lines)
    test_t = count_tokens(test_lines)

    return DatasetManifest(
        name=name,
        version=version,
        path=dataset_path,
        source=source,
        license=license,
        sha256=sha256_file(dataset_path),
        tokenizer_version=getattr(tokenizer, "_version", "xrfm-bpe-v2"),
        tokenizer_vocab_size=tokenizer.vocab_size(),
        num_documents=len(lines),
        total_tokens=total,
        train_tokens=train_t,
        val_tokens=val_t,
        test_tokens=test_t,
        dedup_method=dedup_method,
        filtering_method=filtering_method,
        language=language,
    )
