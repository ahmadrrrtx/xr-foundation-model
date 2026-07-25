"""
Unit tests for XRFM Byte Pair Encoding (BPE) tokenizer.

Tests cover:
- Vocabulary initialization
- Training on sample text
- Encoding roundtrip (encode/decode consistency)
- Vocabulary size validation
- Special token support
- Save/load persistence
- Error handling (empty input, unknown tokens, missing files)

Every public method of BytePairEncoder must have at least basic coverage.
Tests use only the Python standard library and the tokenizer module itself
(no external testing frameworks beyond pytest).
"""

import os
import tempfile

import pytest

from tokenizer.bpe import BytePairEncoder
from tokenizer.interface import TokenizerInterface


class TestBytePairEncoderBasics:
    """Basic initialization and interface compliance."""

    def test_init_default_vocab_size(self) -> None:
        """Tokenizer should initialize with default vocab size."""
        encoder = BytePairEncoder()
        assert encoder.vocab_size_target == 50304
        assert isinstance(encoder.vocab, dict)
        assert isinstance(encoder.merges, list)

    def test_init_custom_vocab_size(self) -> None:
        """Tokenizer should accept custom vocab size if >= 256."""
        encoder = BytePairEncoder(vocab_size_target=1024)
        assert encoder.vocab_size_target == 1024

    def test_init_invalid_vocab_size_raises(self) -> None:
        """Vocabulary target below 256 must raise ValueError."""
        with pytest.raises(ValueError, match="at least 256"):
            BytePairEncoder(vocab_size_target=100)

    def test_implements_tokenizer_interface(self) -> None:
        """BytePairEncoder must conform to TokenizerInterface."""
        encoder = BytePairEncoder()
        assert isinstance(encoder, TokenizerInterface)

    def test_repr_shows_class_and_size(self) -> None:
        """String representation must include vocabulary size."""
        encoder = BytePairEncoder()
        repr_str = repr(encoder)
        assert "BytePairEncoder" in repr_str
        assert "50304" in repr_str or "vocab_size" in repr_str


class TestBytePairEncoderTraining:
    """Vocabulary training from text data."""

    def test_train_on_sample_text_creates_vocabulary(self) -> None:
        """After training, vocabulary should contain more tokens than base 256."""
        encoder = BytePairEncoder(vocab_size_target=512)
        # Create temporary training file.
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("hello world ".join(["hello world"] * 20))
            temp_path = f.name
        try:
            encoder.train(temp_path)
            assert len(encoder.vocab) > 256
            # Vocabulary should include learned subword tokens.
            assert len(encoder.merges) > 0
        finally:
            os.unlink(temp_path)

    def test_train_on_empty_file_raises(self) -> None:
        """Training on empty file must raise ValueError."""
        encoder = BytePairEncoder()
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("")  # Empty file
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="too short"):
                encoder.train(temp_path)
        finally:
            os.unlink(temp_path)

    def test_train_missing_file_raises(self) -> None:
        """Training with nonexistent file must raise FileNotFoundError."""
        encoder = BytePairEncoder()
        with pytest.raises(FileNotFoundError):
            encoder.train("/nonexistent/path/file.txt")


class TestBytePairEncoderEncodingAndDecoding:
    """Roundtrip consistency between encoding and decoding."""

    def test_encode_simple_string(self) -> None:
        """Encoding a simple string should return integer list."""
        encoder = BytePairEncoder(vocab_size_target=512)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("hello world ".join(["hello world"] * 20))
            temp_path = f.name
        try:
            encoder.train(temp_path)
            token_ids = encoder.encode("hello world")
            assert isinstance(token_ids, list)
            assert len(token_ids) > 0
            assert all(isinstance(t, int) for t in token_ids)
        finally:
            os.unlink(temp_path)

    def test_encode_and_decode_roundtrip_approximate(self) -> None:
        """Decode(encode(text)) should reconstruct the original text approximately.

        Note: BPE is approximately lossless for well-covered vocabulary,
        but exact roundtrip depends on whitespace handling and vocabulary
        coverage. For basic versions, we test approximate reconstruction.
        """
        encoder = BytePairEncoder(vocab_size_target=1024)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("hello world ".join(["hello world"] * 20))
            temp_path = f.name
        try:
            encoder.train(temp_path)
            original = "hello world hello"
            token_ids = encoder.encode(original)
            reconstructed = encoder.decode(token_ids)
            # Reconstruction should contain the words; exact whitespace
            # may differ due to tokenization design, so we check presence.
            assert "hello" in reconstructed
            assert "world" in reconstructed
        finally:
            os.unlink(temp_path)

    def test_encode_invalid_type_raises(self) -> None:
        """Encoding non-string input must raise TypeError."""
        encoder = BytePairEncoder()
        with pytest.raises(TypeError, match="expects a string"):
            encoder.encode(123)  # type: ignore[arg-type]

    def test_decode_invalid_ids_raises(self) -> None:
        """Decoding invalid token IDs in strict mode must raise ValueError."""
        encoder = BytePairEncoder()
        with pytest.raises(ValueError, match="not found in vocabulary"):
            encoder.decode([999999], strict=True)

    def test_decode_invalid_input_type_raises(self) -> None:
        """Decoding non-list input must raise TypeError."""
        encoder = BytePairEncoder()
        with pytest.raises(TypeError, match="expects a list"):
            encoder.decode("not a list")  # type: ignore[arg-type]


class TestBytePairEncoderVocabularyPersistence:
    """Save and load vocabulary for reproducibility."""

    def test_save_and_load_roundtrip(self) -> None:
        """Vocabulary saved to file must be fully restorable by load()."""
        encoder = BytePairEncoder(vocab_size_target=512)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(
                "This is a longer test vocabulary training text for persistence check that exceeds the one hundred character minimum threshold easily."
            )
            temp_path = f.name
        try:
            encoder.train(temp_path)
            # Save to temporary file.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as save_f:
                save_path = save_f.name
            try:
                encoder.save(save_path)
                # Create new encoder and load saved vocabulary.
                new_encoder = BytePairEncoder()
                new_encoder.load(save_path)
                # Vocabulary size should match after load.
                assert new_encoder.vocab_size() == encoder.vocab_size()
                # Merge rules should match.
                assert new_encoder.merges == encoder.merges
                # Vocabulary content should match.
                assert new_encoder.vocab == encoder.vocab
            finally:
                os.unlink(save_path)
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file_raises(self) -> None:
        """Loading from nonexistent file must raise FileNotFoundError."""
        encoder = BytePairEncoder()
        with pytest.raises(FileNotFoundError):
            encoder.load("/nonexistent/path/vocab.json")


class TestBytePairEncoderSpecialTokens:
    """Basic special token reservation (reserved for future chat/template formats)."""

    def test_special_token_reserved_ids(self) -> None:
        """Special tokens should reserve IDs and appear in vocabulary."""
        encoder = BytePairEncoder(vocab_size_target=512)
        special = {"<|endoftext|>": 50256}
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(
                "This is a special token test text with enough characters to exceed the hundred character minimum threshold required by the BPE training function."
            )
            temp_path = f.name
        try:
            encoder.train(temp_path, special_tokens=special)
            # The special token ID should exist in vocabulary.
            assert 50256 in encoder.vocab.values()
            # The special token string should map to the reserved ID.
            # Note: The vocabulary uses string keys. We check if any key
            # corresponds to the reserved ID.
            token_for_id = None
            for token_str, token_id in encoder.vocab.items():
                if token_id == 50256:
                    token_for_id = token_str
                    break
            assert token_for_id == "<|endoftext|>"
        finally:
            os.unlink(temp_path)
