"""
Contamination control and frozen evaluation protection for XRFM (`GATE-1`).

Prevents any evaluation item from leaking into Track A (`XRFM Native`) or
Track B (`XRFM CoderSec`) weight-training shards via:
1. Normalization-invariant SHA-256 exact document hash checks (collapsing
   whitespace, case-folding Latin characters, NFC Unicode normalization).
2. Normalization-invariant word n-gram overlap checks across all 7 target
   natural languages (`en`, `ur`, `ur-Latn`, `ar`, `hi`, `es`, `fr`) + `code`.
3. Automatic ingestion of `configs/eval/frozen_benchmarks_v1.json` (including
   checksum verification of local JSONL benchmarks such as
   `configs/eval/multicodersec_seed_v1.jsonl` and canary prompts for external
   suites).
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from xrfm.data.schema import Document


def _normalize_for_contamination(text: str) -> str:
    """Canonicalize text so whitespace/newline/case perturbations cannot evade contamination checks."""
    nfc = unicodedata.normalize("NFC", text or "")
    collapsed = " ".join(nfc.split())
    return collapsed.casefold()


def _normalized_sha256(text: str) -> str:
    return hashlib.sha256(_normalize_for_contamination(text).encode("utf-8")).hexdigest()


@dataclass
class ProtectedDataset:
    name: str
    description: str
    documents: list[Document] = field(default_factory=list)
    content_hashes: set[str] = field(default_factory=set)
    normalized_hashes: set[str] = field(default_factory=set)
    ngram_hashes: set[str] = field(default_factory=set)
    version: str = "v1"

    def __post_init__(self) -> None:
        if self.documents:
            if not self.content_hashes:
                self.content_hashes = {d.content_hash for d in self.documents}
            if not self.normalized_hashes:
                self.normalized_hashes = {_normalized_sha256(d.text) for d in self.documents}

    def add_document(self, doc: Document) -> None:
        self.documents.append(doc)
        self.content_hashes.add(doc.content_hash)
        self.normalized_hashes.add(_normalized_sha256(doc.text))


@dataclass
class ContaminationConfig:
    ngram_size: int = 8  # 8-word n-grams catch short multilingual & code prompts reliably
    threshold: float = 0.5  # overlap fraction to flag contamination
    check_exact_hash: bool = True
    check_ngram: bool = True
    manifest_path: str = "configs/eval/frozen_benchmarks_v1.json"


def _ngrams(text: str, n: int) -> set[str]:
    norm = _normalize_for_contamination(text)
    if not norm:
        return set()
    words = norm.split()
    if len(words) < n:
        return {norm}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _hash_ngram(ngram: str) -> str:
    return hashlib.sha256(ngram.encode("utf-8")).hexdigest()[:16]


class ContaminationChecker:
    """
    Checks training documents against protected evaluation datasets.
    """

    def __init__(
        self,
        protected: list[ProtectedDataset] | ContaminationRegistry,
        config: ContaminationConfig | None = None,
        *,
        ngram_size: int | None = None,
        overlap_threshold: float | None = None,
    ) -> None:
        if isinstance(protected, ContaminationRegistry):
            protected_list = list(protected.datasets.values())
        else:
            protected_list = list(protected)

        cfg = config or ContaminationConfig()
        if ngram_size is not None or overlap_threshold is not None:
            cfg = ContaminationConfig(
                ngram_size=ngram_size if ngram_size is not None else cfg.ngram_size,
                threshold=overlap_threshold if overlap_threshold is not None else cfg.threshold,
                check_exact_hash=cfg.check_exact_hash,
                check_ngram=cfg.check_ngram,
                manifest_path=cfg.manifest_path,
            )

        self.protected = protected_list
        self.config = cfg
        self._protected_hashes: set[str] = set()
        self._protected_norm_hashes: set[str] = set()
        self._protected_ngram_hashes: set[str] = set()
        self._protected_by_name: dict[str, ProtectedDataset] = {}

        for pd in self.protected:
            self._protected_by_name[pd.name] = pd
            self._protected_hashes.update(pd.content_hashes)
            self._protected_norm_hashes.update(pd.normalized_hashes)
            if self.config.check_ngram:
                ngrams: set[str] = set()
                for doc in pd.documents:
                    ngrams.update(_ngrams(doc.text, self.config.ngram_size))
                pd.ngram_hashes = {_hash_ngram(ng) for ng in ngrams}
                self._protected_ngram_hashes.update(pd.ngram_hashes)

    def check_document(self, doc: Document) -> tuple[bool, str | None, float]:
        """Return (is_contaminated, matched_protected_dataset_name, overlap_score)."""
        if self.config.check_exact_hash:
            if doc.content_hash in self._protected_hashes:
                for pd in self.protected:
                    if doc.content_hash in pd.content_hashes:
                        return True, pd.name, 1.0
            norm_h = _normalized_sha256(doc.text)
            if norm_h in self._protected_norm_hashes:
                for pd in self.protected:
                    if norm_h in pd.normalized_hashes:
                        return True, pd.name, 1.0

        if self.config.check_ngram:
            doc_ngrams = _ngrams(doc.text, self.config.ngram_size)
            if not doc_ngrams:
                return False, None, 0.0
            doc_ngram_hashes = {_hash_ngram(ng) for ng in doc_ngrams}
            overlap = len(doc_ngram_hashes & self._protected_ngram_hashes)
            ratio = overlap / len(doc_ngram_hashes) if doc_ngram_hashes else 0.0
            if ratio >= self.config.threshold:
                best_name: str | None = None
                best_overlap = 0
                for pd in self.protected:
                    inter = len(doc_ngram_hashes & pd.ngram_hashes)
                    if inter > best_overlap:
                        best_overlap = inter
                        best_name = pd.name
                return True, best_name, ratio

        return False, None, 0.0

    def check_documents(self, docs: list[Document]) -> tuple[list[Document], list[tuple[Document, str, float]]]:
        """Return (clean_docs, contaminated_with_info)."""
        clean: list[Document] = []
        contaminated: list[tuple[Document, str, float]] = []

        for doc in docs:
            is_contam, name, score = self.check_document(doc)
            if is_contam:
                contaminated.append((doc, name or "unknown", score))
            else:
                clean.append(doc)

        return clean, contaminated

    def filter_contaminated(self, docs: list[Document]) -> tuple[list[Document], list[Document], list[dict[str, Any]]]:
        """Convenience wrapper returning (clean_docs, contaminated_docs, match_records)."""
        clean, raw_matches = self.check_documents(docs)
        contam_docs = [m[0] for m in raw_matches]
        records = [
            {"document_id": m[0].document_id, "matched_benchmark": m[1], "score": round(m[2], 4)} for m in raw_matches
        ]
        return clean, contam_docs, records


class ContaminationRegistry:
    """Maintains registry of protected evaluation datasets."""

    def __init__(self) -> None:
        self.datasets: dict[str, ProtectedDataset] = {}

    def register(self, dataset: ProtectedDataset) -> None:
        self.datasets[dataset.name] = dataset

    def get(self, name: str) -> ProtectedDataset | None:
        return self.datasets.get(name)

    def list(self) -> builtins.list[str]:
        return list(self.datasets.keys())

    def checker(self, config: ContaminationConfig | None = None) -> ContaminationChecker:
        return ContaminationChecker(list(self.datasets.values()), config)


def load_frozen_benchmarks_into_registry(
    registry: ContaminationRegistry,
    manifest_path: str = "configs/eval/frozen_benchmarks_v1.json",
    verify_sha256: bool = True,
) -> dict[str, Any]:
    """
    Load all frozen evaluation benchmarks and canary prompts from ``manifest_path``
    into ``registry``, verifying SHA-256 checksums of local JSONL evaluation files.
    """
    if not os.path.isfile(manifest_path):
        return {"loaded_benchmarks": 0, "total_protected_items": 0}

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    total_items = 0
    loaded_suites = 0

    for entry in manifest.get("benchmarks", []):
        name = str(entry["name"])
        desc = str(entry.get("role", "frozen_eval"))
        docs: list[Document] = []

        local_path = entry.get("local_path")
        if local_path and os.path.isfile(local_path):
            expected_sha = entry.get("sha256")
            if verify_sha256 and expected_sha:
                with open(local_path, "rb") as bf:
                    actual_sha = hashlib.sha256(bf.read()).hexdigest()
                if actual_sha != expected_sha:
                    raise ValueError(
                        f"Frozen benchmark checksum mismatch for '{local_path}': "
                        f"expected {expected_sha}, got {actual_sha}"
                    )
            text_field = str(entry.get("text_field", "prompt"))
            with open(local_path, encoding="utf-8") as jf:
                for idx, line in enumerate(jf):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    row = json.loads(stripped)
                    prompt_text = str(row.get(text_field, ""))
                    if prompt_text:
                        docs.append(
                            Document(
                                text=prompt_text,
                                source=name,
                                document_id=str(row.get("id", f"{name}-{idx}")),
                                language=str(row.get("language", "unknown")),
                            )
                        )

        for idx, canary in enumerate(entry.get("canary_prompts", [])):
            if canary:
                docs.append(
                    Document(
                        text=str(canary),
                        source=name,
                        document_id=f"{name}-canary-{idx}",
                    )
                )

        pd = ProtectedDataset(
            name=name, description=desc, documents=docs, version=str(manifest.get("manifest_version", "1.0"))
        )
        registry.register(pd)
        loaded_suites += 1
        total_items += len(docs)

    return {
        "manifest_version": manifest.get("manifest_version", "1.0.0"),
        "loaded_benchmarks": loaded_suites,
        "total_protected_items": total_items,
    }


_global_registry = ContaminationRegistry()


def get_global_registry(autoload_frozen: bool = True) -> ContaminationRegistry:
    if autoload_frozen and not _global_registry.datasets:
        default_manifest = "configs/eval/frozen_benchmarks_v1.json"
        if os.path.isfile(default_manifest):
            load_frozen_benchmarks_into_registry(_global_registry, manifest_path=default_manifest)
    return _global_registry
