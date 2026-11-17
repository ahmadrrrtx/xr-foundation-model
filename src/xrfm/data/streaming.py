"""
Bounded-memory streaming ingestion and chunked pipeline helpers for XRFM.

Supports Hugging Face streaming / large JSONL corpora under strict Kaggle / CI
disk and RAM budgets (~10–20 GB temporary scratch space) by:
1. Yielding bounded chunks of ``Document`` records from any iterator or JSONL path.
2. Carrying stateful exact SHA-256 deduplication and Banded MinHash-LSH near-dedup
   across chunks so multi-million-document streams never materialize in RAM at once.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from xrfm.data.dedup import MinHashLSHIndex, NearDedupConfig, deduplicate_exact, deduplicate_near
from xrfm.data.language import HeuristicLanguageDetector, LanguageDetectorConfig, filter_by_language
from xrfm.data.normalization import normalize_documents
from xrfm.data.pii import PIIFilterConfig, filter_documents_pii
from xrfm.data.quality import QualityFilterConfig, filter_documents_quality
from xrfm.data.schema import Document


@dataclass
class StreamingIngestConfig:
    chunk_size: int = 2048
    max_documents: int | None = None
    preserve_code: bool = True


def iter_jsonl_documents(
    path: str,
    source_name: str = "jsonl_stream",
    text_field: str = "text",
    default_language: str = "unknown",
) -> Iterator[Document]:
    """Stream ``Document`` records lazily line-by-line from a JSONL file."""
    import json

    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            text = str(data.get(text_field, ""))
            if not text:
                continue
            yield Document(
                text=text,
                source=str(data.get("source", source_name)),
                document_id=str(data.get("document_id", data.get("id", f"{source_name}-{idx}"))),
                language=str(data.get("language", default_language)),
                license=str(data.get("license", "unknown")),
                metadata={k: v for k, v in data.items() if k not in {text_field, "text"}},
            )


def iter_bounded_document_chunks(
    documents: Iterable[Document],
    config: StreamingIngestConfig | None = None,
) -> Iterator[list[Document]]:
    """Yield bounded lists of at most ``config.chunk_size`` documents."""
    cfg = config or StreamingIngestConfig()
    if cfg.chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {cfg.chunk_size}")

    chunk: list[Document] = []
    emitted = 0

    for doc in documents:
        if cfg.max_documents is not None and emitted >= cfg.max_documents:
            break
        chunk.append(doc)
        emitted += 1
        if len(chunk) >= cfg.chunk_size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


class StreamingFilterState:
    """Cross-chunk state for streaming exact + LSH near-deduplication and counters."""

    def __init__(self, near_dedup_config: NearDedupConfig | None = None) -> None:
        self.seen_hashes: set[str] = set()
        nd = near_dedup_config or NearDedupConfig()
        self.near_dedup_config = nd
        self.lsh_index = MinHashLSHIndex(num_perm=nd.num_perm, num_bands=nd.num_bands)
        self.ingested_docs: int = 0
        self.retained_docs: int = 0
        self.exact_duplicates: int = 0
        self.near_duplicates: int = 0

    def process_chunk(
        self,
        raw_chunk: list[Document],
        *,
        language_config: LanguageDetectorConfig | None = None,
        quality_config: QualityFilterConfig | None = None,
        pii_config: PIIFilterConfig | None = None,
        exact_dedup: bool = True,
    ) -> tuple[list[Document], dict[str, Any]]:
        """Filter and deduplicate a single bounded chunk using persistent hash/LSH state."""
        self.ingested_docs += len(raw_chunk)
        norm_results = normalize_documents(raw_chunk)
        docs: list[Document] = [r.document for r in norm_results if not r.is_empty]
        if language_config is not None:
            docs, _, _ = filter_by_language(docs, HeuristicLanguageDetector(), language_config)
        if quality_config is not None:
            docs, _, _ = filter_documents_quality(docs, quality_config)
        if pii_config is not None:
            docs, _, _, _ = filter_documents_pii(docs, pii_config)

        if exact_dedup:
            ex_res = deduplicate_exact(docs, seen_hashes=self.seen_hashes)
            docs = ex_res.unique_docs
            self.exact_duplicates += ex_res.stats["duplicates"]

        if self.near_dedup_config.enabled:
            nd_res = deduplicate_near(docs, self.near_dedup_config, lsh_index=self.lsh_index)
            docs = nd_res.unique_docs
            self.near_duplicates += nd_res.stats["near_duplicates"]

        self.retained_docs += len(docs)
        return docs, {
            "ingested_total": self.ingested_docs,
            "retained_total": self.retained_docs,
            "exact_duplicates_total": self.exact_duplicates,
            "near_duplicates_total": self.near_duplicates,
        }
