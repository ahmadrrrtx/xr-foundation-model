"""
Contamination control for XRFM.

Goal: prevent evaluation data leaking into training.

Architecture:
- Registry of protected evaluation datasets (benchmark data, validation, test)
- Comparison of training documents against protected content
- Simple n-gram overlap detection for Phase 1, pluggable for future sophisticated algorithms

The system creates architectural boundary so future training cannot accidentally ingest eval data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional

from xrfm.data.schema import Document


@dataclass
class ProtectedDataset:
    name: str
    description: str
    documents: List[Document] = field(default_factory=list)
    # Precomputed hashes for fast lookup
    content_hashes: Set[str] = field(default_factory=set)
    ngram_hashes: Set[str] = field(default_factory=set)
    version: str = "v1"

    def __post_init__(self):
        if not self.content_hashes and self.documents:
            self.content_hashes = {d.content_hash for d in self.documents}

    def add_document(self, doc: Document):
        self.documents.append(doc)
        self.content_hashes.add(doc.content_hash)


@dataclass
class ContaminationConfig:
    ngram_size: int = 13  # for n-gram overlap
    threshold: float = 0.8  # overlap threshold to flag
    check_exact_hash: bool = True
    check_ngram: bool = True


def _ngrams(text: str, n: int) -> Set[str]:
    words = text.split()
    if len(words) < n:
        return {text}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _hash_ngram(ngram: str) -> str:
    return hashlib.sha256(ngram.encode("utf-8")).hexdigest()[:16]


class ContaminationChecker:
    """
    Checks training docs against protected eval datasets.
    """

    def __init__(self, protected: List[ProtectedDataset], config: ContaminationConfig | None = None):
        self.protected = protected
        self.config = config or ContaminationConfig()
        # Build global lookup
        self._protected_hashes: Set[str] = set()
        self._protected_ngram_hashes: Set[str] = set()
        self._protected_by_name: Dict[str, ProtectedDataset] = {}

        for pd in protected:
            self._protected_by_name[pd.name] = pd
            self._protected_hashes.update(pd.content_hashes)
            if self.config.check_ngram:
                # Build ngram hashes if not already
                if not pd.ngram_hashes:
                    ngrams = set()
                    for doc in pd.documents:
                        ngrams.update(_ngrams(doc.text, self.config.ngram_size))
                    pd.ngram_hashes = {_hash_ngram(ng) for ng in ngrams}
                self._protected_ngram_hashes.update(pd.ngram_hashes)

    def check_document(self, doc: Document) -> Tuple[bool, Optional[str], float]:
        """
        Check single doc.
        Returns (is_contaminated, protected_dataset_name, score)
        """
        # Exact hash check
        if self.config.check_exact_hash:
            if doc.content_hash in self._protected_hashes:
                # Find which protected dataset
                for pd in self.protected:
                    if doc.content_hash in pd.content_hashes:
                        return True, pd.name, 1.0

        # N-gram overlap check
        if self.config.check_ngram:
            doc_ngrams = _ngrams(doc.text, self.config.ngram_size)
            if not doc_ngrams:
                return False, None, 0.0
            doc_ngram_hashes = {_hash_ngram(ng) for ng in doc_ngrams}
            overlap = len(doc_ngram_hashes & self._protected_ngram_hashes)
            ratio = overlap / len(doc_ngram_hashes) if doc_ngram_hashes else 0.0
            if ratio >= self.config.threshold:
                # Find which protected has most overlap (simplified)
                best_name = None
                best_overlap = 0
                for pd in self.protected:
                    inter = len(doc_ngram_hashes & pd.ngram_hashes)
                    if inter > best_overlap:
                        best_overlap = inter
                        best_name = pd.name
                return True, best_name, ratio

        return False, None, 0.0

    def check_documents(
        self, docs: List[Document]
    ) -> Tuple[List[Document], List[Tuple[Document, str, float]]]:
        """
        Check batch.
        Returns (clean, contaminated_with_info)
        """
        clean: List[Document] = []
        contaminated: List[Tuple[Document, str, float]] = []

        for doc in docs:
            is_contam, name, score = self.check_document(doc)
            if is_contam:
                contaminated.append((doc, name or "unknown", score))
            else:
                clean.append(doc)

        return clean, contaminated


# Registry for protected datasets (global)
class ContaminationRegistry:
    """Maintains registry of protected evaluation datasets."""

    def __init__(self):
        self.datasets: Dict[str, ProtectedDataset] = {}

    def register(self, dataset: ProtectedDataset):
        self.datasets[dataset.name] = dataset

    def get(self, name: str) -> Optional[ProtectedDataset]:
        return self.datasets.get(name)

    def list(self) -> List[str]:
        return list(self.datasets.keys())

    def checker(self, config: ContaminationConfig | None = None) -> ContaminationChecker:
        return ContaminationChecker(list(self.datasets.values()), config)


# Global instance
_global_registry = ContaminationRegistry()


def get_global_registry() -> ContaminationRegistry:
    return _global_registry
