"""
Tokenizer interface definition for XRFM.

Purpose: Provide a stable abstract base class that all tokenizer algorithms
(BPE, SentencePiece, WordPiece, Unigram, future TikToken-style) must implement.
This ensures the dataset loader (`data/loader.py`, Phase 3) and training pipeline
can depend on tokenizer behavior without knowing which algorithm is active.

Conceptual reference: Tokenizer abstraction patterns common in open-source
LLM libraries (Hugging Face `tokenizers` library, `tiktoken` API design,
Sebastian Raschka's tokenizer chapter). Implementation is original.

Design principle (from TDR-002): The interface must remain stable across
minor versions (v0.2.0 through v2.0.0) to protect dataset loader and model
training code from rewrites.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
import os


class TokenizerInterface(ABC):
    """Abstract base class for all XRFM tokenizer algorithms.

    Every concrete tokenizer (BPE, SentencePiece, etc.) must implement these
    methods. The dataset loader (`data/loader.py`) will interact exclusively
    through this interface.

    Attributes:
        vocab (Dict[str, int]): Token string to integer mapping (optional,
            depends on algorithm design). Not required by interface.
    """

    def __init__(self) -> None:
        """Initialize tokenizer state. Subclasses should call super()."""
        super().__init__()

    @abstractmethod
    def encode(self, text: str, **kwargs) -> List[int]:
        """Convert a text string into a sequence of integer token IDs.

        Args:
            text: Input text string (may contain special tokens, whitespace,
                punctuation, and multilingual content).
            **kwargs: Additional encoding options (e.g., `add_special_tokens`,
                `truncate`, `max_length`). Subclasses define supported options.

        Returns:
            List of integer token IDs representing the input text.

        Raises:
            ValueError: If input text is empty or contains unsupported characters.
            TypeError: If input is not a string.

        Design note:
            Encoding must be deterministic: the same input text always produces
            the same token sequence (given the same vocabulary and merge rules).
        """
        ...

    @abstractmethod
    def decode(self, tokens: List[int], **kwargs) -> str:
        """Convert a sequence of integer token IDs back to a text string.

        Args:
            tokens: Sequence of integer token IDs produced by `encode()`.
            **kwargs: Additional decoding options (e.g., `skip_special_tokens`,
                `clean_up_tokenization_spaces`).

        Returns:
            Reconstructed text string. Note: this may not be exactly identical
            to the original input due to tokenization approximations (e.g.,
            whitespace normalization, special token removal).

        Raises:
            ValueError: If token IDs are outside the vocabulary range.
            TypeError: If input is not a list of integers.
        """
        ...

    @abstractmethod
    def vocab_size(self) -> int:
        """Return the size of the tokenizer vocabulary.

        This value must match the `vocab_size` parameter in the model
        configuration (`ConfigLoader.get_model_config()` returns this value
        to initialize the embedding layer: `nn.Embedding(vocab_size, ...)`).

        Returns:
            Integer representing the number of tokens in the vocabulary.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist tokenizer vocabulary and state to the file system.

        Args:
            path: File path for saved vocabulary (e.g., `tokenizer/vocab.json`,
                `tokenizer/merges.txt`). The exact format depends on the
                concrete tokenizer algorithm.

        Raises:
            FileNotFoundError: If the parent directory does not exist.
            OSError: If file write fails.

        Design note:
            The saved vocabulary must be loadable by `load()` without requiring
            retraining. For BPE, this typically includes the vocabulary dictionary
            and the ordered list of merge rules.
        """
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load tokenizer vocabulary and state from the file system.

        Args:
            path: File path from which to load vocabulary.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the loaded vocabulary format is invalid or corrupt.
        """
        ...

    def __repr__(self) -> str:
        """Provide a descriptive string representation of the tokenizer.

        Returns:
            A string describing the tokenizer class and vocabulary size.
        """
        return f"{self.__class__.__name__}(vocab_size={self.vocab_size()})"
