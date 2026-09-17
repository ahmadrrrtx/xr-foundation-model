"""
Tokenizer interface definition for XRFM.

Purpose: Provide a stable abstract base class that all tokenizer algorithms
(BPE, SentencePiece, WordPiece, Unigram, future TikToken-style) must implement.
This ensures the dataset loader, trainer, and inference engine can depend on
tokenizer behavior without knowing which algorithm is active.

The **tokenizer contract** (Phase 0):

* ``encode`` / ``decode`` are inverse for any Unicode text
  (``decode(encode(text)) == text`` for byte-level tokenizers).
* ``vocab_size()`` returns the *true* vocabulary size ``max(id) + 1`` and
  must equal the ``vocab_size`` the model is constructed with.
* Special-token ids are exposed explicitly via ``pad_token_id``,
  ``bos_token_id``, ``eos_token_id``, ``unk_token_id`` — each is
  ``int | None``. **Consumers (dataset packing, collation) must use these
  accessors and must not assume any hard-coded id** (e.g. ``pad_id = 0``).
  A tokenizer without a pad token reports ``None``, and callers either
  pass an explicit pad id or handle variable-length sequences.

Design principle (from TDR-002): the interface must remain stable across
minor versions to protect dataset loader and model training code from
rewrites. Phase 0 only *added* the special-token accessors; nothing was
removed or renamed.
"""

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Abstract base class for all XRFM tokenizer algorithms.

    Historical note: this class was named ``TokenizerInterface``; that name
    remains as an alias for backward compatibility.
    """

    def __init__(self) -> None:
        """Initialize tokenizer state. Subclasses should call super()."""
        super().__init__()

    # ------------------------------------------------------------------
    # Core contract
    # ------------------------------------------------------------------

    @abstractmethod
    def encode(self, text: str, **kwargs) -> list[int]:
        """Convert a text string into a sequence of integer token IDs.

        Must be deterministic: the same input text always produces the same
        token sequence (given the same vocabulary and merge rules).

        Raises:
            ValueError: If input text is empty or contains unsupported characters.
            TypeError: If input is not a string.
        """
        ...

    @abstractmethod
    def decode(self, tokens: list[int], **kwargs) -> str:
        """Convert a sequence of integer token IDs back to a text string.

        Raises:
            ValueError: If token IDs are outside the vocabulary range.
            TypeError: If input is not a list of integers.
        """
        ...

    @abstractmethod
    def vocab_size(self) -> int:
        """Return the size of the tokenizer vocabulary (``max(id) + 1``).

        This value must match the ``vocab_size`` the model is constructed
        with (``ModelConfig.vocab_size``).
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist tokenizer vocabulary and state to the file system."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load tokenizer vocabulary and state from the file system."""
        ...

    # ------------------------------------------------------------------
    # Special-token contract (Phase 0)
    # ------------------------------------------------------------------

    @property
    def pad_token_id(self) -> int | None:
        """Id of the padding token, or ``None`` if the tokenizer defines none."""
        return None

    @property
    def bos_token_id(self) -> int | None:
        """Id of the beginning-of-sequence token, or ``None``."""
        return None

    @property
    def eos_token_id(self) -> int | None:
        """Id of the end-of-sequence token, or ``None``."""
        return None

    @property
    def unk_token_id(self) -> int | None:
        """Id of the unknown-token, or ``None``."""
        return None

    def __repr__(self) -> str:
        """Provide a descriptive string representation of the tokenizer."""
        return f"{self.__class__.__name__}(vocab_size={self.vocab_size()})"


# Backward-compatible name (the historical name of this ABC).
TokenizerInterface = Tokenizer
