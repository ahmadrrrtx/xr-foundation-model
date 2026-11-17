"""
Deduplication for XRFM Phase 1 (`GATE-1`).

Implements:
1. Exact deduplication via content hash (SHA256), including stateful streaming mode.
2. Near deduplication via **Banded MinHash-LSH** (remediating Phase 0 Finding ``F-14``):
   - Partitions ``num_perm`` MinHash values into ``num_bands`` bands of ``rows_per_band``
     rows each (default: ``16 x 8 = 128`` permutations).
   - Hashes each band tuple into a hash bucket so candidate Jaccard verification runs
     ONLY against documents colliding in at least one LSH band ($O(N)$ expected cost
     instead of $O(N^2)$ all-pairs comparison).
"""

from __future__ import annotations

import hashlib
import struct
from collections import defaultdict
from dataclasses import dataclass, field

from xrfm.data.schema import Document

# ---------------------------------------------------------------------------
# Exact Deduplication
# ---------------------------------------------------------------------------


@dataclass
class ExactDedupConfig:
    enabled: bool = True
    keep: str = "first"  # first | last


@dataclass
class ExactDedupResult:
    unique_docs: list[Document]
    duplicate_docs: list[Document]
    stats: dict[str, int]


def deduplicate_exact(
    documents: list[Document],
    keep: str = "first",
    seen_hashes: set[str] | None = None,
) -> ExactDedupResult:
    """
    Remove exact duplicate documents based on content_hash (SHA256 of normalized text).

    Args:
        documents: Input documents (should be normalized before hashing).
        keep: 'first' keeps first occurrence.
        seen_hashes: Optional mutable set of hashes carried across streaming chunks.
    """
    seen: set[str] = seen_hashes if seen_hashes is not None else set()
    unique: list[Document] = []
    duplicates: list[Document] = []

    for doc in documents:
        h = doc.content_hash
        if h not in seen:
            seen.add(h)
            unique.append(doc)
        else:
            duplicates.append(doc)

    stats = {
        "input": len(documents),
        "unique": len(unique),
        "duplicates": len(duplicates),
    }
    return ExactDedupResult(unique_docs=unique, duplicate_docs=duplicates, stats=stats)


# ---------------------------------------------------------------------------
# Near Deduplication (Banded MinHash-LSH)
# ---------------------------------------------------------------------------


def _shingle(text: str, k: int = 5) -> set[bytes]:
    """Generate character k-gram shingles as UTF-8 bytes."""
    normalized = " ".join(text.lower().split())
    if len(normalized) < k:
        return {normalized.encode("utf-8")} if normalized else set()
    return {normalized[i : i + k].encode("utf-8") for i in range(len(normalized) - k + 1)}


_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _generate_hash_params(num_perm: int, seed: int = 42) -> list[tuple[int, int]]:
    """Deterministic hash function parameters (a, b) for universal hashing."""
    params: list[tuple[int, int]] = []
    for i in range(num_perm):
        h = hashlib.sha256(f"{seed}-{i}".encode()).digest()
        a = (struct.unpack("<Q", h[:8])[0] % (_MERSENNE_PRIME - 1)) + 1
        b = struct.unpack("<Q", h[8:16])[0] % _MERSENNE_PRIME
        params.append((a, b))
    return params


def compute_minhash(shingles: set[bytes], params: list[tuple[int, int]]) -> list[int]:
    """Compute MinHash signature of length len(params)."""
    if not shingles:
        return [_MAX_HASH] * len(params)

    shingle_hashes = [struct.unpack("<I", hashlib.md5(s, usedforsecurity=False).digest()[:4])[0] for s in shingles]

    sig: list[int] = []
    for a, b in params:
        min_val = _MERSENNE_PRIME
        for sh in shingle_hashes:
            val = (a * sh + b) % _MERSENNE_PRIME
            if val < min_val:
                min_val = val
        sig.append(min_val)
    return sig


def minhash_similarity(sig1: list[int], sig2: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2) or not sig1:
        return 0.0
    matches = sum(1 for x, y in zip(sig1, sig2) if x == y)
    return matches / len(sig1)


@dataclass
class NearDedupConfig:
    enabled: bool = True
    num_perm: int = 128
    shingle_size: int = 5
    threshold: float = 0.85  # Jaccard similarity threshold
    num_bands: int = 16  # Banded LSH: 16 bands x 8 rows = 128 perms (s-curve knee ~0.73)
    seed: int = 42


@dataclass
class MinHashLSHIndex:
    """Stateful Banded MinHash-LSH index supporting streaming / chunked near-deduplication."""

    num_perm: int = 128
    num_bands: int = 16
    rows_per_band: int = field(init=False)
    buckets: list[dict[bytes, list[int]]] = field(init=False)
    signatures: list[list[int]] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.num_bands <= 0 or self.num_perm <= 0:
            raise ValueError("num_perm and num_bands must be > 0")
        if self.num_perm % self.num_bands != 0:
            raise ValueError(f"num_perm ({self.num_perm}) must be divisible by num_bands ({self.num_bands})")
        self.rows_per_band = self.num_perm // self.num_bands
        self.buckets = [defaultdict(list) for _ in range(self.num_bands)]

    def _band_digest(self, sig: list[int], band_idx: int) -> bytes:
        start = band_idx * self.rows_per_band
        end = start + self.rows_per_band
        packed = struct.pack(f"<{self.rows_per_band}Q", *sig[start:end])
        return hashlib.blake2b(packed, digest_size=12).digest()

    def query_candidates(self, sig: list[int]) -> set[int]:
        """Return indices of previously indexed signatures sharing >= 1 LSH band."""
        candidates: set[int] = set()
        for b in range(self.num_bands):
            key = self._band_digest(sig, b)
            bucket = self.buckets[b].get(key)
            if bucket:
                candidates.update(bucket)
        return candidates

    def insert(self, doc_id: str, sig: list[int]) -> int:
        """Insert a document signature into all LSH bands and return its index."""
        idx = len(self.signatures)
        self.signatures.append(sig)
        self.doc_ids.append(doc_id)
        for b in range(self.num_bands):
            key = self._band_digest(sig, b)
            self.buckets[b][key].append(idx)
        return idx


@dataclass
class NearDedupResult:
    unique_docs: list[Document]
    duplicate_docs: list[tuple[Document, str, float]]  # (doc, matched_doc_id, similarity)
    stats: dict[str, int]


def deduplicate_near(
    documents: list[Document],
    config: NearDedupConfig | None = None,
    lsh_index: MinHashLSHIndex | None = None,
) -> NearDedupResult:
    """
    Remove near-duplicate documents using Banded MinHash-LSH ($O(N)$ expected scaling).
    """
    if config is None:
        config = NearDedupConfig()

    if not config.enabled or len(documents) == 0:
        return NearDedupResult(
            unique_docs=list(documents),
            duplicate_docs=[],
            stats={
                "input": len(documents),
                "unique": len(documents),
                "near_duplicates": 0,
                "candidate_pairs_checked": 0,
            },
        )

    # Adjust num_bands automatically if caller customized num_perm without matching num_bands
    num_bands = config.num_bands
    if config.num_perm % num_bands != 0:
        for candidate_b in (16, 8, 4, 2, 1):
            if config.num_perm % candidate_b == 0:
                num_bands = candidate_b
                break

    params = _generate_hash_params(config.num_perm, config.seed)
    index = lsh_index if lsh_index is not None else MinHashLSHIndex(num_perm=config.num_perm, num_bands=num_bands)

    unique: list[Document] = []
    duplicates: list[tuple[Document, str, float]] = []
    candidate_pairs_checked = 0

    for doc in documents:
        shingles = _shingle(doc.text, config.shingle_size)
        sig = compute_minhash(shingles, params)

        candidate_indices = index.query_candidates(sig)
        is_dup = False
        matched_id = ""
        matched_sim = 0.0

        for cand_idx in sorted(candidate_indices):
            candidate_pairs_checked += 1
            sim = minhash_similarity(sig, index.signatures[cand_idx])
            if sim >= config.threshold:
                is_dup = True
                matched_id = index.doc_ids[cand_idx]
                matched_sim = sim
                break

        if is_dup:
            duplicates.append((doc, matched_id, matched_sim))
        else:
            index.insert(doc.document_id, sig)
            unique.append(doc)

    stats = {
        "input": len(documents),
        "unique": len(unique),
        "near_duplicates": len(duplicates),
        "candidate_pairs_checked": candidate_pairs_checked,
    }
    return NearDedupResult(unique_docs=unique, duplicate_docs=duplicates, stats=stats)


def deduplicate_documents(
    documents: list[Document],
    exact_config: ExactDedupConfig | None = None,
    near_config: NearDedupConfig | None = None,
    do_exact: bool = True,
    do_near: bool = True,
) -> tuple[list[Document], dict[str, int], dict[str, object]]:
    """
    Combined exact SHA-256 + Banded MinHash-LSH near deduplication stage for DataPipeline.
    """
    ex_cfg = exact_config or ExactDedupConfig()
    nd_cfg = near_config or NearDedupConfig()

    current = list(documents)
    exact_dups = 0
    near_dups = 0
    candidate_pairs = 0
    details: dict[str, object] = {}

    if do_exact and ex_cfg.enabled:
        ex_res = deduplicate_exact(current, keep=ex_cfg.keep)
        current = ex_res.unique_docs
        exact_dups = ex_res.stats["duplicates"]
        details["exact"] = ex_res

    if do_near and nd_cfg.enabled:
        nd_res = deduplicate_near(current, config=nd_cfg)
        current = nd_res.unique_docs
        near_dups = nd_res.stats["near_duplicates"]
        candidate_pairs = nd_res.stats.get("candidate_pairs_checked", 0)
        details["near"] = nd_res

    stats = {
        "input": len(documents),
        "exact_duplicates": exact_dups,
        "near_duplicates": near_dups,
        "candidate_pairs_checked": candidate_pairs,
        "final_unique": len(current),
    }
    return current, stats, details
