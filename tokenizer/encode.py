"""
Convenience encoding module for XRFM tokenizer.

Purpose: Provide a clean API for converting text to token IDs using any
`TokenizerInterface` implementation (currently BPE only).

This module is a thin wrapper around the tokenizer interface. It does not
contain core tokenization logic; that lives in `tokenizer/bpe.py` and
future algorithm modules.

Design principle: The dataset loader (`data/loader.py`, Phase 3) will import
from this module rather than from `bpe.py` directly. This ensures the loader
remains independent of the specific tokenizer algorithm.
"""

from typing import cast

from tokenizer.interface import TokenizerInterface


def encode_text(
    text: str | list[str],
    tokenizer: TokenizerInterface,
    add_special_tokens: bool = False,
) -> list[int] | list[list[int]]:
    """Encode text or list of texts using the provided tokenizer.

    Args:
        text: A single text string or a list of text strings.
        tokenizer: An instance implementing `TokenizerInterface`.
        add_special_tokens: If True, prepend special start/end tokens
            (reserved for future chat/template formats; basic BPE does not
            currently use this parameter, but the interface supports it).

    Returns:
        For single string: list of integer token IDs.
        For list of strings: list of lists (batch format).

    Raises:
        TypeError: If text type is unsupported.
        ValueError: If tokenizer encoding fails.
    """
    if isinstance(text, str):
        token_ids = tokenizer.encode(text)
        # Basic special token handling: reserved for future expansion.
        # Currently, the BPE tokenizer does not implement special tokens
        # by default, but the interface allows them.
        if add_special_tokens and hasattr(tokenizer, "special_tokens") and tokenizer.special_tokens:
            # For simplicity in v0.1, we do not modify token sequences here.
            # Future versions may prepend `<|startoftext|>` or similar.
            pass
        return token_ids
    elif isinstance(text, list) and all(isinstance(t, str) for t in text):
        out: list[list[int]] = [tokenizer.encode(t) for t in text]
        return out
    else:
        raise TypeError(f"encode_text expects str or List[str], got {type(text)}")


def decode_ids(
    token_ids: list[int] | list[list[int]],
    tokenizer: TokenizerInterface,
    skip_special_tokens: bool = False,
) -> str | list[str]:
    """Decode token IDs or batches of token IDs back to text.

    Args:
        token_ids: A single sequence of integers or a batch of sequences.
        tokenizer: An instance implementing `TokenizerInterface`.
        skip_special_tokens: If True, filter out special token IDs before
            decoding (reserved for future chat formats).

    Returns:
        Reconstructed text string(s).
    """
    if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], int):
        # Single sequence. The isinstance guard on the first element plus an
        # explicit cast narrows for mypy (container-level narrowing is not
        # inferred from element checks).
        single: list[int] = cast(list[int], token_ids)
        ids_to_decode: list[int] = single
        if skip_special_tokens and hasattr(tokenizer, "special_tokens"):
            special_ids = set(tokenizer.special_tokens.values())
            ids_to_decode = [t for t in single if t not in special_ids]
        return tokenizer.decode(ids_to_decode)
    elif isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
        # Batch of sequences.
        batch: list[list[int]] = cast(list[list[int]], token_ids)
        out: list[str] = [cast(str, decode_ids(s, tokenizer, skip_special_tokens=skip_special_tokens)) for s in batch]
        return out
    else:
        raise TypeError(f"decode_ids expects List[int] or List[List[int]], got {type(token_ids)}")
