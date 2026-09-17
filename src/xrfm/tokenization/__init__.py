"""XRFM tokenization subsystem.

Public surface:

    from xrfm.tokenization import Tokenizer, BPETokenizer, BytePairEncoder, encode_text, decode_text

``BPETokenizer`` is the canonical name of the byte-level BPE tokenizer;
``BytePairEncoder`` is the historical name (same class). The abstract
contract is :class:`Tokenizer` (historical name ``TokenizerInterface``).
"""

from xrfm.tokenization.bpe import BPETokenizer, BytePairEncoder
from xrfm.tokenization.decode import decode_ids, decode_text
from xrfm.tokenization.encode import encode_text
from xrfm.tokenization.interface import Tokenizer, TokenizerInterface

__all__ = [
    "BPETokenizer",
    "BytePairEncoder",
    "Tokenizer",
    "TokenizerInterface",
    "decode_ids",
    "decode_text",
    "encode_text",
]
