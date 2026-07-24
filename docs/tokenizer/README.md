# Tokenizer Module — XRFM

## Overview

The tokenizer converts raw text into integer token sequences and back.
This module implements the `TokenizerInterface` and provides the original
Byte Pair Encoding (`BytePairEncoder`) algorithm.

## Design Principles

- Stable interface (`tokenizer/interface.py`) allows future algorithm swaps.
- Config-driven (`ConfigLoader.get_model_config()` provides `vocab_size`).
- Original implementation (no line-for-line copying).
- Production quality: type hints, docstrings, validation, persistence.

## Components

- `tokenizer/interface.py` — Abstract base class (`TokenizerInterface`).
- `tokenizer/bpe.py` — `BytePairEncoder` (training, encoding, decoding, persistence).
- `tokenizer/encode.py` — Convenience wrapper (`encode_text()` for batches).
- `tokenizer/decode.py` — Convenience wrapper (`decode_text()` for batches).
- `tokenizer/__init__.py` — Package exports.

## Usage Example

```python
from tokenizer import BytePairEncoder, encode_text, decode_text
from xrfm.config.loader import ConfigLoader

config = ConfigLoader()
encoder = BytePairEncoder(vocab_size_target=config.get_model_config()["vocab_size"])
encoder.train("data/datasets/tiny_shakespeare.txt")

token_ids = encode_text("hello world", encoder)
text = decode_text(token_ids, encoder)
```

## Testing

Run: `python -m pytest tests/test_tokenizer_bpe.py -v`

## Performance Considerations

- Vocabulary training is a one-time preprocessing step.
- Encoding speed: linear in sequence length (with BPE merge iterations).
- Memory: vocabulary matrix handled by model embedding layer; tokenizer only stores vocabulary dictionary and merge rules.

## Future Improvements

- SentencePiece support (optional, post-v0.5.0).
- Special token expansion (chat markers, system prompts) — interface supports it.
- Vocabulary optimization (TikToken-style merge ordering).
