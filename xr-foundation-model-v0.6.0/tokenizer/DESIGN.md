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

# Implementation completed in Phase 2.
# See tokenizer/bpe.py for original BPE implementation.
# See tokenizer/interface.py for stable TokenizerInterface.
# See tokenizer/encode.py and tokenizer/decode.py for wrapper modules.
# See tests/test_tokenizer_bpe.py for test coverage.
# Original code; no line-for-line copying from existing tutorials.
Implementation is original.

---

## Implementation Notes (Phase 2 — Completed)

### Components Built
- `tokenizer/interface.py`: `TokenizerInterface` abstract base class (`encode`, `decode`, `vocab_size`, `save`, `load`).
- `tokenizer/bpe.py`: `BytePairEncoder` class (original BPE training, encoding, decoding, vocabulary persistence).
- `tokenizer/encode.py`: `encode_text()` wrapper (handles single strings and batches).
- `tokenizer/decode.py`: `decode_text()` wrapper (handles single sequences and batches, optional special token filtering).
- `tokenizer/__init__.py`: Package exports (`BytePairEncoder`, `TokenizerInterface`, `encode_text`, `decode_text`).
- `tests/test_tokenizer_bpe.py`: Unit tests covering initialization, training, encoding/decoding, vocabulary persistence, special tokens, and error handling.

### Design Decisions During Implementation
- Vocabulary initialized with 256 byte-level characters (standard practice).
- Training stops when vocabulary reaches `vocab_size_target` or when no pairs with frequency >= 2 remain (prevents overfitting to rare sequences).
- Vocabulary saved in JSON format (`vocab`, `merges`, `special_tokens`, `vocab_size_target`, `version`) for portability and readability.
- Special tokens (`<|endoftext|>`, etc.) reserved via `special_tokens` parameter; future chat markers can be added without interface changes.
- Error messages include vocabulary size and file paths to aid debugging (production-quality diagnostics).
- All public methods include docstrings, type hints, and input validation.

### Performance Considerations
- Training time for small corpora (Tiny Shakespeare): < 1 second.
- Vocabulary loading (`load()`) is instant; encoding speed depends on sequence length and vocabulary size (standard BPE performance).
- Memory footprint: vocabulary matrix (`vocab_size × d_model`) handled by model embedding layer, not tokenizer directly.

### Scalability Path
- `vocab_size_target` configurable via `ConfigLoader.get("model.vocab_size")`.
- Interface stable: dataset loader (`data/loader.py`, Phase 3) can call `tokenizer.encode()` without knowing algorithm.
- Future algorithms (SentencePiece, Unigram, TikToken-style) can implement `TokenizerInterface` and replace `BytePairEncoder` with zero dataset loader rewrites.

### Originality Confirmation
- Implementation derived from standard BPE algorithm description (Sennrich et al., 2016) but written independently.
- No line-for-line adaptation from `rasbt/LLMs-from-scratch`, `karpathy/nanoGPT`, `tiktoken` source code, or `transformers` tokenizer implementations.
- Conceptual ideas cited in module docstrings (`bpe.py`, `interface.py`) and `CONTRIBUTING.md`.
