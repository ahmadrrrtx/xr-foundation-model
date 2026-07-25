"""Document Indexer & Vector Store for XRFM Search Engine (RAG Module)."""

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """Represents a chunk of indexed text with metadata."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)


class SearchIndexer:
    """In-memory & disk-persisted hybrid indexer (BM25 + Dense Vectors)."""

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k1 = k1
        self.b = b
        self.chunks: dict[str, DocumentChunk] = {}
        self.doc_freqs: dict[str, int] = {}
        self.avg_dl: float = 0.0
        self.total_docs: int = 0

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        title: str = "",
        source_url: str | None = None,
    ) -> list[DocumentChunk]:
        words = text.split()
        if not words:
            return []

        chunks: list[DocumentChunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{doc_id}_chunk_{len(chunks)}"

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                title=title or doc_id,
                text=chunk_text,
                source_url=source_url,
                tokens=self.tokenize(chunk_text),
            )
            chunks.append(chunk)

        return chunks

    def add_document(
        self,
        doc_id: str,
        text: str,
        title: str = "",
        source_url: str | None = None,
    ) -> int:
        chunks = self.chunk_text(text, doc_id, title=title, source_url=source_url)
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            unique_tokens = set(chunk.tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.total_docs = len(self.chunks)
        if self.total_docs > 0:
            self.avg_dl = sum(len(c.tokens) for c in self.chunks.values()) / self.total_docs

        return len(chunks)

    def bm25_score(self, query_tokens: list[str], chunk: DocumentChunk) -> float:
        if self.total_docs == 0 or self.avg_dl == 0:
            return 0.0

        score = 0.0
        doc_len = len(chunk.tokens)
        token_counts = {}
        for token in chunk.tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        for q_token in query_tokens:
            if q_token not in token_counts:
                continue

            f = token_counts[q_token]
            n = self.doc_freqs.get(q_token, 0)
            idf = math.log((self.total_docs - n + 0.5) / (n + 0.5) + 1.0)

            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_dl))
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        query_tokens = self.tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        scored_chunks: list[tuple[DocumentChunk, float]] = []
        for chunk in self.chunks.values():
            score = self.bm25_score(query_tokens, chunk)
            if score > 0:
                scored_chunks.append((chunk, score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

    def save(self, file_path: str) -> None:
        data = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "total_docs": self.total_docs,
            "avg_dl": self.avg_dl,
            "doc_freqs": self.doc_freqs,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "text": c.text,
                    "source_url": c.source_url,
                    "metadata": c.metadata,
                }
                for c in self.chunks.values()
            ],
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Index file not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        self.chunk_size = data.get("chunk_size", 256)
        self.chunk_overlap = data.get("chunk_overlap", 32)
        self.total_docs = data.get("total_docs", 0)
        self.avg_dl = data.get("avg_dl", 0.0)
        self.doc_freqs = data.get("doc_freqs", {})

        self.chunks = {}
        for item in data.get("chunks", []):
            chunk = DocumentChunk(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                title=item.get("title", ""),
                text=item["text"],
                source_url=item.get("source_url"),
                metadata=item.get("metadata", {}),
                tokens=self.tokenize(item["text"]),
            )
            self.chunks[chunk.chunk_id] = chunk
