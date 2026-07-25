"""
Convenience decoding module for XRFM tokenizer.

Purpose: Provide a clean API for converting token IDs back to text using
any `TokenizerInterface` implementation.

Note: This module is intentionally thin. The core decode logic lives in
`tokenizer/bpe.py` (`BytePairEncoder.decode()`), and the stable interface
is defined in `tokenizer/interface.py`. This design ensures that the dataset
loader and evaluation pipeline can call decode functionality without depending
on the specific tokenizer algorithm.
"""

from tokenizer.encode import decode_ids
from tokenizer.interface import TokenizerInterface


def decode_text(
    token_ids: list[int] | list[list[int]],
    tokenizer: TokenizerInterface,
    skip_special_tokens: bool = False,
) -> str | list[str]:
    """Decode token IDs using the provided tokenizer instance.

    This is a thin wrapper around `tokenizer.decode()` that applies
    optional special-token filtering and supports both single sequences
    and batches.

    Args:
        token_ids: Integer token sequence or batch of sequences.
        tokenizer: Instance of any class implementing `TokenizerInterface`.
        skip_special_tokens: If True, exclude special token IDs.

    Returns:
        Reconstructed text (single string or list of strings).

    Design note: See `tokenizer/encode.py` for batch decoding logic.
    """
    return decode_ids(token_ids, tokenizer, skip_special_tokens=skip_special_tokens)
