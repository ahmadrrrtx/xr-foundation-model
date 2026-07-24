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

from typing import List, Union

from tokenizer.interface import TokenizerInterface
from tokenizer.bpe import BytePairEncoder


def encode_text(
    text: Union[str, List[str]],
    tokenizer: TokenizerInterface,
    add_special_tokens: bool = False,
) -> List[int]:
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
        return [tokenizer.encode(t) for t in text]
    else:
        raise TypeError(
            f"encode_text expects str or List[str], got {type(text)}"
        )


def decode_ids(
    token_ids: Union[List[int], List[List[int]]],
    tokenizer: TokenizerInterface,
    skip_special_tokens: bool = False,
) -> Union[str, List[str]]:
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
        # Single sequence.
        ids_to_decode = token_ids
        if skip_special_tokens and hasattr(tokenizer, "special_tokens"):
            special_ids = set(tokenizer.special_tokens.values())
            ids_to_decode = [token_id for token_id in token_ids if token_id not in special_ids]
        return tokenizer.decode(ids_to_decode)
    elif isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
        # Batch of sequences.
        return [
            decode_ids(seq, tokenizer, skip_special_tokens=skip_special_tokens)
            for seq in token_ids
        ]
    else:
        raise TypeError(
            f"decode_ids expects List[int] or List[List[int]], got {type(token_ids)}"
        )
