"""
Dataset splitting for XRFM.

Split semantics (Phase 0 contract)
----------------------------------
* Splits operate on **documents/lines** (never mid-document character cuts).
* ``shuffle=False`` → deterministic *sequential* split: the first
  ``train_ratio`` of lines is train, the next ``val_ratio`` is val, the
  rest is test. This is the standard language-modeling convention
  (temporal/document-order holdout) and avoids shuffling-induced leakage
  for ordered corpora.
* ``shuffle=True`` → lines are shuffled with a **seeded** ``random.Random``
  instance local to the split (global RNG state is untouched), making the
  split reproducible for a given ``(lines, ratios, seed, dedup)`` input.
* The ``seed`` argument is always consumed when ``shuffle=True`` —
  Phase 0 fixed the previous implementation that accepted ``seed`` and
  silently ignored it.
* Exact-line dedup (``dedup=True``) runs *before* splitting so an identical
  line cannot appear in two splits.

Future-proofing: document-level splitting (multi-line documents with
boundaries) and MinHash/exact dedup can be layered on top of
:func:`split_documents` without changing the split-ratio contract.
"""

from __future__ import annotations

import random

__all__ = [
    "SplitRatiosError",
    "split_dataset_lines",
    "split_dataset",
    "normalize_text",
]


class SplitRatiosError(ValueError):
    """Raised when split ratios are invalid (range or sum)."""


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    for name, value in (("train_ratio", train_ratio), ("val_ratio", val_ratio), ("test_ratio", test_ratio)):
        if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
            raise SplitRatiosError(f"{name} must be in [0, 1], got {value!r}")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise SplitRatiosError(f"train/val/test ratios must sum to ~1.0, got {total}")


def split_dataset_lines(
    lines: list[str],
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    shuffle: bool = False,
    dedup: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """Split lines into (train, val, test) by document/line boundaries.

    Args:
        lines: Corpus lines (documents). Order defines the sequential split.
        train_ratio / val_ratio / test_ratio: Fractions summing to ~1.0.
        seed: Seed for the shuffle when ``shuffle=True`` (deterministic).
        shuffle: Shuffle line order (seeded) before splitting.
        dedup: Drop exact duplicate lines before splitting.

    Returns:
        ``(train_lines, val_lines, test_lines)``. Splits may be empty for
        tiny corpora (callers must handle empty splits).

    Raises:
        SplitRatiosError: invalid ratios.
    """
    _validate_ratios(train_ratio, val_ratio, test_ratio)

    if dedup:
        seen: set[str] = set()
        unique: list[str] = []
        for ln in lines:
            if ln not in seen:
                seen.add(ln)
                unique.append(ln)
        lines = unique

    if shuffle:
        # Local, seeded RNG: reproducible and does not mutate global state.
        rng = random.Random(seed)
        lines = list(lines)
        rng.shuffle(lines)

    n = len(lines)
    if n == 0:
        return [], [], []

    # Guarantee at least one training line for tiny corpora (a single-line
    # file would otherwise produce train_end == 0).
    train_end = min(n, max(1, int(n * train_ratio)))
    val_end = min(n, train_end + int(n * val_ratio))
    return lines[:train_end], lines[train_end:val_end], lines[val_end:]


def split_dataset(
    text: str,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    shuffle: bool = False,
    dedup: bool = True,
) -> tuple[str, str, str]:
    """Line-boundary split of raw text into train/val/test strings.

    Convenience wrapper around :func:`split_dataset_lines`; newlines are
    preserved inside each split.
    """
    tr, va, te = split_dataset_lines(
        text.splitlines(),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        shuffle=shuffle,
        dedup=dedup,
    )
    return "\n".join(tr), "\n".join(va), "\n".join(te)


def normalize_text(text: str) -> str:
    """Collapse irregular whitespace in text.

    NOTE: the dataset pipeline deliberately does *not* call this — it
    destroys newline/paragraph structure a language model needs. Retained
    for callers that explicitly want collapsed text.
    """
    import re

    text = text.replace("\r\n", "\n").replace("\t", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
