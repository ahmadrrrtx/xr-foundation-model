"""
Document-level data splitting for XRFM.

Fixes Phase 0 line-based splitting: atomic unit is document.

Requirements:
- deterministic
- seedable
- reproducible
- independent of execution order
- stable hashing strategy so adding workers doesn't change splits
- prove train ∩ val = ∅ etc.

We implement:
- Hash-based deterministic split (stable)
- Optional shuffled deterministic split via seeded RNG
- Document ID based, not line index
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set

from xrfm.data.schema import Document


@dataclass
class SplitConfig:
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 42
    method: str = "hash"  # hash | random
    # For hash method, we use document_id hash to assign split deterministically
    # For random method, shuffle with seed then sequential


def _validate_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if not (0.0 <= train <= 1.0 and 0.0 <= val <= 1.0 and 0.0 <= test <= 1.0):
        raise ValueError(f"Ratios must be in [0,1], got train={train} val={val} test={test}")
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to ~1.0, got {total}")


def _hash_doc_id(doc_id: str) -> int:
    """Stable hash of doc_id to int (0..2**64-1)."""
    h = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
    # Take first 16 hex chars = 64 bits
    return int(h[:16], 16)


def split_documents(
    docs: List[Document],
    config: SplitConfig | None = None,
) -> Dict[str, List[Document]]:
    """
    Split documents into train/val/test.

    Methods:
    - hash: deterministic by document_id hash, independent of order.
            For each doc, compute hash(doc_id) % 10000, assign by ratio thresholds.
            This is stable: adding docs doesn't change existing assignments except
            via ratio boundaries? Actually stable per doc, but we implement
            cumulative thresholds.
    - random: shuffle with seeded RNG, then sequential split (order-dependent but reproducible)

    Returns dict with keys train, val, test.
    """
    cfg = config or SplitConfig()
    _validate_ratios(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)

    if not docs:
        return {"train": [], "val": [], "test": []}

    if cfg.method == "hash":
        # Hash-based: deterministic, order-independent
        # Use 0-1 float from hash
        train_thr = cfg.train_ratio
        val_thr = cfg.train_ratio + cfg.val_ratio

        train: List[Document] = []
        val: List[Document] = []
        test: List[Document] = []

        for doc in docs:
            # Use document_id for stability; fallback to content_hash
            key = doc.document_id or doc.content_hash
            h_int = _hash_doc_id(key)
            # Normalize to [0,1)
            h_float = (h_int % 1000000) / 1000000.0

            if h_float < train_thr:
                train.append(doc)
            elif h_float < val_thr:
                val.append(doc)
            else:
                test.append(doc)

        # Edge case: ensure at least 1 train if possible
        if not train and docs:
            # Move one from val or test
            if val:
                train.append(val.pop(0))
            elif test:
                train.append(test.pop(0))
            else:
                train.append(docs[0])

        return {"train": train, "val": val, "test": test}

    elif cfg.method == "random":
        # Seeded shuffle then sequential
        rng = random.Random(cfg.seed)
        shuffled = list(docs)
        rng.shuffle(shuffled)

        n = len(shuffled)
        train_end = max(1, int(n * cfg.train_ratio)) if n > 0 else 0
        val_end = train_end + int(n * cfg.val_ratio)

        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]

        return {"train": train, "val": val, "test": test}

    else:
        raise ValueError(f"Unknown split method: {cfg.method}")


def verify_no_overlap(splits: Dict[str, List[Document]]) -> bool:
    """Verify train ∩ val = ∅ etc for document IDs."""
    train_ids = set(d.document_id for d in splits.get("train", []))
    val_ids = set(d.document_id for d in splits.get("val", []))
    test_ids = set(d.document_id for d in splits.get("test", []))

    return len(train_ids & val_ids) == 0 and len(train_ids & test_ids) == 0 and len(val_ids & test_ids) == 0


def split_reproducibility_check(docs: List[Document], config: SplitConfig) -> bool:
    """Check that same input + same seed = same split."""
    split1 = split_documents(docs, config)
    split2 = split_documents(docs, config)

    def ids(split_dict):
        return {k: sorted([d.document_id for d in v]) for k, v in split_dict.items()}

    return ids(split1) == ids(split2)
