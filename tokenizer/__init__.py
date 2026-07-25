"""
XRFM Tokenizer Package.

Exports the stable tokenizer interface and original BPE implementation.
Future algorithms (SentencePiece, Unigram, TikToken-style) will be added
as additional modules without breaking existing imports.

Usage:
    from tokenizer import BytePairEncoder, TokenizerInterface, encode_text, decode_text

Design reference (conceptual, not copied):
- Sennrich et al. (2016) — Byte Pair Encoding for subword tokenization
- Raschka (2024) — Tokenizer pipeline design concepts
- OpenAI tiktoken — Vocabulary persistence patterns (concept only)
"""

from tokenizer.bpe import BytePairEncoder
from tokenizer.decode import decode_text
from tokenizer.encode import encode_text
from tokenizer.interface import TokenizerInterface

__all__ = [
    "TokenizerInterface",
    "BytePairEncoder",
    "encode_text",
    "decode_text",
]
