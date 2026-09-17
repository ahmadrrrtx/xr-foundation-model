"""
Sequence packing / chunking for XRFM.

Converts a token-id stream into fixed-length training examples. This module
owns **all** length/overlap validation so no invalid training example can be
produced silently:

* ``max_seq_len <= 0`` → :class:`PackingError` (was: silent misbehavior).
* ``overlap < 0`` or ``overlap >= max_seq_len`` → :class:`PackingError`.
  ``overlap == max_seq_len`` previously caused an **infinite loop** (stride
  of zero); it is now rejected up front.
* Empty input → empty output (a legitimate result, not an error).
* Short input → one short chunk (the dataset pads it; see
  ``xrfm.data.dataset``).

The stride is always ``max_seq_len - overlap >= 1``, so the loop makes
progress by construction and terminates.
"""

from __future__ import annotations

from collections.abc import Sequence

from xrfm.tokenization.interface import Tokenizer

__all__ = ["PackingError", "chunk_token_ids", "chunk_text"]


class PackingError(ValueError):
    """Raised for invalid packing/chunking parameters."""


def chunk_token_ids(ids: Sequence[int], max_seq_len: int, overlap: int = 0) -> list[list[int]]:
    """Chunk a token-id sequence into fixed-length windows.

    Args:
        ids: Token-id stream (any sequence of ints; may be empty).
        max_seq_len: Window length. Must be > 0.
        overlap: Number of tokens shared between consecutive windows.
            Must satisfy ``0 <= overlap < max_seq_len`` (stride >= 1).

    Returns:
        List of windows. The final window may be shorter than
        ``max_seq_len``; empty input yields ``[]``.

    Raises:
        PackingError: invalid ``max_seq_len`` or ``overlap``.
    """
    if not isinstance(max_seq_len, int) or isinstance(max_seq_len, bool) or max_seq_len <= 0:
        raise PackingError(f"max_seq_len must be a positive int, got {max_seq_len!r}")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise PackingError(f"overlap must be a non-negative int, got {overlap!r}")
    if overlap >= max_seq_len:
        raise PackingError(
            f"overlap ({overlap}) must be < max_seq_len ({max_seq_len}); "
            f"overlap >= max_seq_len would make no forward progress (infinite loop)"
        )

    ids = list(ids)
    if not ids:
        return []

    stride = max_seq_len - overlap
    return [ids[start : start + max_seq_len] for start in range(0, len(ids), stride)]


def chunk_text(
    text: str,
    max_seq_len: int,
    tokenizer: Tokenizer,
    overlap: int = 0,
) -> list[list[int]]:
    """Encode ``text`` with ``tokenizer`` and pack into fixed-length chunks.

    Raises:
        PackingError: invalid ``max_seq_len`` / ``overlap`` (validated before
            tokenization so bad parameters fail fast).
    """
    # Validate before encoding so parameter errors are cheap and early.
    chunk_token_ids([], max_seq_len=max_seq_len, overlap=overlap)
    token_ids = tokenizer.encode(text)
    return chunk_token_ids(token_ids, max_seq_len=max_seq_len, overlap=overlap)
