# Tokenizer Design — Phase 2 Planning

## Goal

Implement Byte Pair Encoding (BPE) from scratch for XRFM.

However, the tokenizer module must be designed so that future algorithms can be swapped without changing the dataset loader or training loop.

## Supported Algorithms (Future-Proof Interface)

- Byte Pair Encoding (BPE) — Phase 2 implementation
- SentencePiece (future — for multilingual)
- WordPiece (future — for encoder-style models)
- Unigram (future)
- TikToken-style (future — optimized BPE with special token handling)

## API Design (Stable Interface)

Every tokenizer must implement:

```python
class TokenizerInterface:
    def encode(text: str) -> List[int]: ...
    def decode(tokens: List[int]) -> str: ...
    def vocab_size(self) -> int: ...
    def save(path: str) -> None: ...
    def load(path: str) -> None: ...
```

The dataset loader (`data/loader.py`) will call `tokenizer.encode()` without knowing which algorithm is active. This ensures zero rewrites when upgrading from BPE to TikToken-style or SentencePiece.

## Conceptual References (Not Copied)

- Sennrich et al. (2016) — Byte Pair Encoding for subword tokenization
- Kudo (2018) — SentencePiece (future reference)
- Raschka (2024) — LLMs-from-Scratch (conceptual tokenizer pipeline reference)

Implementation will be original.
