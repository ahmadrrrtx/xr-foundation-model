"""
Canonical document representation for XRFM data pipeline.

Design principles (from Dolma, FineWeb, OLMo research):
- Documents are atomic unit, not lines.
- Raw document separated from derived attributes.
- Stable IDs, hashes, provenance tracked.
- License and source are first-class.
- Metadata sidecar pattern.

This module defines:
- Document: canonical internal representation
- DocumentAttributes: derived scores/tags
- ProcessingDecision: final filtering decision + reason
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional


def _sha256_text(text: str) -> str:
    """SHA-256 hex of UTF-8 encoded text, used for content hashing."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _generate_doc_id(source: str, content_hash: str) -> str:
    """Stable document ID from source + content hash (deterministic)."""
    # Use first 16 chars of hash + source prefix for readability
    # Full uniqueness via hash.
    return f"{source}:{content_hash[:16]}"


@dataclass
class Document:
    """
    Canonical XRFM document.

    Fields:
        document_id: Stable unique ID (deterministic from source+content if not provided)
        source: Source name from registry (e.g. 'wikipedia', 'books', 'code')
        source_uri: Original URI / path / reference
        license: SPDX or human-readable license string
        license_url: URL to license if available
        collection: Logical collection grouping (optional)
        language: ISO 639-1 code or 'multilingual' etc.
        text: Raw document text (normalized but not aggressively cleaned)
        content_hash: SHA-256 of normalized text
        metadata: Arbitrary provenance dict
        created_at: ISO timestamp
        version: Document schema version
    """

    text: str
    source: str
    document_id: str = ""
    source_uri: str = ""
    license: str = "unknown"
    license_url: str = ""
    collection: str = ""
    language: str = "en"
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    version: str = "xrfm-doc-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(f"text must be str, got {type(self.text).__name__}")
        if not self.content_hash:
            self.content_hash = _sha256_text(self.text)
        if not self.document_id:
            self.document_id = _generate_doc_id(self.source, self.content_hash)
        if not self.source:
            raise ValueError("source must be non-empty")
        # Normalize language to lower
        if self.language:
            self.language = self.language.lower()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        # Handle legacy fields gracefully
        allowed = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in allowed}
        # Ensure required
        if "text" not in filtered or "source" not in filtered:
            raise ValueError(f"Document requires text and source, got {list(data.keys())}")
        return cls(**filtered)

    @classmethod
    def from_jsonl(cls, line: str) -> Document:
        return cls.from_dict(json.loads(line))

    def char_length(self) -> int:
        return len(self.text)

    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class LanguageScore:
    """Result of language identification."""

    language: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    detector: str = "heuristic"
    detector_version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityScore:
    """Quality scoring for a document."""

    # Individual heuristic scores
    char_length: int = 0
    word_count: int = 0
    symbol_ratio: float = 0.0
    number_ratio: float = 0.0
    repetition_score: float = 0.0
    boilerplate_score: float = 0.0
    malformed_score: float = 0.0
    overall_score: float = 0.0
    # Filter decisions
    is_too_short: bool = False
    is_too_long: bool = False
    has_excessive_repetition: bool = False
    has_excessive_symbols: bool = False
    is_boilerplate: bool = False
    is_malformed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PIIScore:
    """PII / safety detection result."""

    has_email: bool = False
    has_phone: bool = False
    has_secret: bool = False
    has_api_key: bool = False
    has_sensitive_id: bool = False
    flagged_patterns: list[str] = field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "low"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeduplicationInfo:
    """Deduplication metadata."""

    is_exact_duplicate: bool = False
    duplicate_of: str = ""  # document_id of canonical
    content_hash: str = ""
    is_near_duplicate: bool = False
    near_duplicate_of: str = ""
    near_duplicate_score: float = 0.0
    dedup_method: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentAttributes:
    """
    Derived attributes sidecar, stored separately from raw document.

    Mirrors Dolma's approach: documents + attributes separate.
    """

    document_id: str
    content_hash: str
    source: str
    language_score: Optional[LanguageScore] = None
    quality_score: Optional[QualityScore] = None
    pii_score: Optional[PIIScore] = None
    dedup_info: Optional[DeduplicationInfo] = None
    token_count: int = 0
    processing_version: str = "xrfm-pipeline-v1"
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Ensure nested dataclasses serialized
        return d

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentAttributes:
        # Reconstruct nested
        ls = data.get("language_score")
        if ls and isinstance(ls, dict):
            data["language_score"] = LanguageScore(**ls)
        qs = data.get("quality_score")
        if qs and isinstance(qs, dict):
            data["quality_score"] = QualityScore(**qs)
        ps = data.get("pii_score")
        if ps and isinstance(ps, dict):
            data["pii_score"] = PIIScore(**ps)
        di = data.get("dedup_info")
        if di and isinstance(di, dict):
            data["dedup_info"] = DeduplicationInfo(**di)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__.keys()})


@dataclass
class ProcessingDecision:
    """
    Final filtering decision for a document.
    Preserves reason for debugging corpus quality.
    """

    document_id: str
    keep: bool
    reason: str = ""  # e.g. "passed", "too_short", "language_mismatch", "pii_high_risk", "exact_duplicate"
    stage: str = ""  # which stage rejected it
    scores: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# Type alias for pipeline
DocumentBatch = list[Document]
