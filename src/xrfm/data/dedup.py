"""
Deduplication for XRFM.

- Exact deduplication via SHA-256 content hash
- Near-duplicate detection via MinHash/LSH and SimHash

Design:
- Exact dedup deterministic, records statistics
- Near dedup pluggable interface
- Reports input, exact duplicates, near duplicates, final unique

References: Dolma uses Bloom-filter-based dedup, supports document/paragraph-level.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from xrfm.data.schema import Document


def compute_content_hash(text: str, algorithm: str = "sha256", normalize: bool = True) -> str:
    """
    Compute content hash for deduplication.
    Normalization applied before hashing: strip, lower? We use conservative.
    For exact dedup, we hash normalized text (strip whitespace).
    Document hash algorithm documented.
    """
    if normalize:
        # Normalize: strip leading/trailing whitespace, normalize newlines
        # Don't lower case to preserve case-sensitive duplicates as distinct? But spec says same content.
        # We use stripped version.
        text = text.strip()
    if algorithm == "sha256":
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    elif algorithm == "blake2b":
        return hashlib.blake2b(text.encode("utf-8")).hexdigest()
    else:
        raise ValueError(f"Unknown hash algorithm: {algorithm}")


@dataclass
class ExactDedupConfig:
    algorithm: str = "sha256"
    normalize_before_hash: bool = True
    keep_first: bool = True  # keep first occurrence


@dataclass
class ExactDedupResult:
    unique_docs: list[Document]
    duplicate_docs: list[tuple[Document, str]]  # (doc, canonical_id)
    hash_to_id: dict[str, str]
    stats: dict[str, int]


def deduplicate_exact(
    docs: list[Document],
    config: ExactDedupConfig | None = None,
) -> ExactDedupResult:
    """
    Deterministic exact deduplication via content hash.
    - Computes hash per doc
    - Tracks first occurrence
    - Returns unique + duplicates
    - Records statistics, does not silently drop without reporting
    """
    cfg = config or ExactDedupConfig()
    seen: dict[str, str] = {}  # hash -> canonical doc_id
    unique: list[Document] = []
    duplicates: list[tuple[Document, str]] = []

    for doc in docs:
        # Use doc.content_hash if already computed and normalization matches,
        # else recompute
        if cfg.normalize_before_hash:
            h = compute_content_hash(doc.text, algorithm=cfg.algorithm, normalize=True)
        else:
            h = doc.content_hash or compute_content_hash(doc.text, algorithm=cfg.algorithm, normalize=False)

        if h in seen:
            duplicates.append((doc, seen[h]))
        else:
            seen[h] = doc.document_id
            unique.append(doc)

    stats = {
        "input": len(docs),
        "unique": len(unique),
        "duplicates": len(duplicates),
    }

    return ExactDedupResult(
        unique_docs=unique,
        duplicate_docs=duplicates,
        hash_to_id=seen,
        stats=stats,
    )


# Near-duplicate detection


def _shingles(text: str, k: int = 5) -> set[str]:
    """k-word shingles."""
    words = text.split()
    if len(words) < k:
        return {text}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass
class NearDedupConfig:
    algorithm: str = "minhash"  # minhash | simhash | ngram_overlap
    threshold: float = 0.8  # similarity threshold
    num_perm: int = 128  # for minhash
    shingle_size: int = 5
    # LSH params
    lsh_threshold: float | None = None


@dataclass
class NearDedupResult:
    unique_docs: list[Document]
    near_duplicates: list[tuple[Document, str, float]]  # (doc, canonical_id, score)
    stats: dict[str, int]


class MinHash:
    """
    Simple MinHash implementation without external deps.
    For Phase 1, we implement a deterministic lightweight version.
    Production would use datasketch.
    """

    def __init__(self, num_perm: int = 128, seed: int = 42):
        self.num_perm = num_perm
        self.seed = seed
        # Generate random hash functions: (a*x + b) % p
        # Use deterministic pseudo-random
        import random

        rng = random.Random(seed)
        self.permutations = [(rng.randint(1, 1 << 32), rng.randint(0, 1 << 32)) for _ in range(num_perm)]
        self.prime = (1 << 61) - 1  # Mersenne prime

    def _hash_shingle(self, shingle: str) -> int:
        # Hash shingle to int
        return int(hashlib.sha256(shingle.encode("utf-8")).hexdigest()[:16], 16)

    def compute(self, shingles: set[str]) -> list[int]:
        if not shingles:
            return [self.prime] * self.num_perm
        hashes = [self._hash_shingle(s) for s in shingles]
        signature = []
        for a, b in self.permutations:
            min_hash = min((a * h + b) % self.prime for h in hashes)
            signature.append(min_hash)
        return signature

    def similarity(self, sig1: list[int], sig2: list[int]) -> float:
        if len(sig1) != len(sig2):
            raise ValueError("Signature length mismatch")
        matches = sum(1 for x, y in zip(sig1, sig2) if x == y)
        return matches / len(sig1)


def deduplicate_near(
    docs: list[Document],
    config: NearDedupConfig | None = None,
) -> NearDedupResult:
    """
    Near-duplicate detection.
    For Phase 1, we implement simple approach:
    - For minhash: compute signatures, compare pairwise with threshold (O(n^2) for small n, acceptable for golden)
    - For ngram_overlap: Jaccard over shingles
    Production would use LSH for scalability.

    Returns unique docs after near-dedup.
    """
    cfg = config or NearDedupConfig()
    threshold = cfg.threshold

    if not docs:
        return NearDedupResult(
            unique_docs=[], near_duplicates=[], stats={"input": 0, "unique": 0, "near_duplicates": 0}
        )

    # Precompute shingles and optionally minhash signatures
    shingles_list: list[set[str]] = [_shingles(d.text, k=cfg.shingle_size) for d in docs]
    signatures: list[list[int]] | None = None

    if cfg.algorithm == "minhash":
        mh = MinHash(num_perm=cfg.num_perm)
        signatures = [mh.compute(s) for s in shingles_list]

    unique: list[Document] = []
    near_dups: list[tuple[Document, str, float]] = []
    # Keep track of kept shingles/signatures for comparison
    kept_shingles: list[set[str]] = []
    kept_sigs: list[list[int]] = []
    kept_ids: list[str] = []

    for idx, doc in enumerate(docs):
        is_dup = False
        best_score = 0.0
        best_id = ""

        curr_shingles = shingles_list[idx]
        curr_sig = signatures[idx] if signatures else None

        # Compare against kept docs
        for k_idx, kept_id in enumerate(kept_ids):
            if cfg.algorithm == "minhash" and curr_sig is not None:
                sim = mh.similarity(curr_sig, kept_sigs[k_idx])
            else:
                # ngram_overlap / default
                sim = _jaccard(curr_shingles, kept_shingles[k_idx])

            if sim >= threshold:
                is_dup = True
                best_score = sim
                best_id = kept_id
                break
            if sim > best_score:
                best_score = sim
                best_id = kept_id

        if is_dup:
            near_dups.append((doc, best_id, best_score))
        else:
            unique.append(doc)
            kept_shingles.append(curr_shingles)
            if curr_sig is not None:
                kept_sigs.append(curr_sig)
            kept_ids.append(doc.document_id)

    stats = {
        "input": len(docs),
        "unique": len(unique),
        "near_duplicates": len(near_dups),
    }

    return NearDedupResult(unique_docs=unique, near_duplicates=near_dups, stats=stats)


def deduplicate_documents(
    docs: list[Document],
    exact_config: ExactDedupConfig | None = None,
    near_config: NearDedupConfig | None = None,
    do_exact: bool = True,
    do_near: bool = True,
) -> tuple[list[Document], dict[str, int], dict[str, list]]:
    """
    Combined exact + near dedup pipeline.
    Returns (final_unique, stats, details)
    Stats includes input, exact duplicates, near duplicates, final unique.
    """
    stats: dict[str, int] = {"input": len(docs)}
    details: dict[str, list] = {}

    current = docs

    if do_exact:
        exact_res = deduplicate_exact(current, exact_config)
        current = exact_res.unique_docs
        stats["exact_duplicates"] = exact_res.stats["duplicates"]
        stats["after_exact"] = len(current)
        details["exact_duplicates"] = [(d.document_id, canon) for d, canon in exact_res.duplicate_docs]
    else:
        stats["exact_duplicates"] = 0
        stats["after_exact"] = len(current)

    if do_near:
        near_res = deduplicate_near(current, near_config)
        current = near_res.unique_docs
        stats["near_duplicates"] = near_res.stats["near_duplicates"]
        stats["final_unique"] = len(current)
        details["near_duplicates"] = [(d.document_id, canon, score) for d, canon, score in near_res.near_duplicates]
    else:
        stats["near_duplicates"] = 0
        stats["final_unique"] = len(current)

    stats["output"] = len(current)

    return current, stats, details
