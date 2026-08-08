"""
Byte Pair Encoding (BPE) tokenizer for XRFM — v2 (byte-level).

Forensic-audit remediation (F-11, F-12, F-16):
- v1 was a *character*-level tokenizer over latin-1 characters: it destroyed
  all whitespace on encode (decode(encode(x)) != x), and raised ValueError on
  any character above U+00FF (no Chinese, Arabic, emoji, ...).
- v2 is a true **byte-level** tokenizer following the GPT-2/tiktoken design:
  * the base vocabulary is the 256 UTF-8 byte values (represented as
    latin-1-decoded strings, the tiktoken convention);
  * merges are learned over byte sequences;
  * whitespace (including newlines/tabs) is preserved exactly, so
    decode(encode(text)) == text for any Unicode text;
  * PAD/BOS/EOS/UNK special tokens are reserved at the top of the vocabulary
    (configurable), giving the training pipeline a stable padding id.

Public interface is unchanged (TokenizerInterface): train / encode / decode /
vocab_size / save / load. Original implementation (algorithm: Sennrich et al.
2016; byte-level design follows the GPT-2/tiktoken convention conceptually).
"""

import json
import os
import re
from collections import Counter

from tokenizer.interface import TokenizerInterface

_WS_RE = re.compile(r"\s+|\S+")


class BytePairEncoder(TokenizerInterface):
    """Byte-level Byte Pair Encoding tokenizer.

    Attributes:
        vocab (Dict[str, int]): token string -> id. Token strings are
            latin-1 views of UTF-8 byte sequences (tiktoken convention).
        merges (List[Tuple[str, str]]): ordered merge rules.
        vocab_size_target (int): target vocabulary size.
        special_tokens (Dict[str, int]): name -> id (e.g. "<|pad|>").
        pad_id / bos_id / eos_id / unk_id (int | None): resolved special ids.
    """

    def __init__(self, vocab_size_target: int = 50304) -> None:
        """Initialize with the 256 byte tokens as the base vocabulary.

        Args:
            vocab_size_target: Target number of tokens (>= 256).

        Raises:
            ValueError: If vocab_size_target < 256.
        """
        super().__init__()
        if vocab_size_target < 256:
            raise ValueError(f"vocab_size_target must be at least 256 to cover byte range, got {vocab_size_target}")
        self.vocab_size_target = vocab_size_target
        self.vocab: dict[str, int] = {}
        self.merges: list[tuple[str, str]] = []
        self.special_tokens: dict[str, int] = {}
        self.pad_id: int | None = None
        self.bos_id: int | None = None
        self.eos_id: int | None = None
        self.unk_id: int | None = None
        self._version = "xrfm-bpe-v2"
        # Base vocabulary: all 256 byte values as latin-1 strings.
        for i in range(256):
            self.vocab[bytes([i]).decode("latin-1")] = i

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _text_to_byte_tokens(text: str) -> list[str]:
        """Convert a text string into a list of single-byte token strings.

        Each UTF-8 byte is mapped to its latin-1 character, which keeps the
        byte stream intact and lossless.
        """
        return [bytes([b]).decode("latin-1") for b in text.encode("utf-8")]

    @staticmethod
    def _tokens_to_text(tokens: list[str]) -> str:
        """Inverse of _text_to_byte_tokens: latin-1 strings -> UTF-8 text."""
        return b"".join(t.encode("latin-1") for t in tokens).decode("utf-8", errors="replace")

    @staticmethod
    def _split_words(text: str) -> list[str]:
        """Split text into words that tile the input EXACTLY.

        Whitespace runs are attached as a prefix to the following word so the
        tokenizer can learn space-prefixed tokens (" the", "\\n") while every
        byte of the original text remains present in the stream.
        """
        parts = _WS_RE.findall(text)
        words: list[str] = []
        pending_ws: str | None = None
        for part in parts:
            if part.isspace():
                pending_ws = (pending_ws or "") + part
            else:
                words.append((pending_ws or "") + part)
                pending_ws = None
        if pending_ws:
            words.append(pending_ws)
        return words

    def _resolve_special_ids(self) -> None:
        """Set pad/bos/eos/unk ids from self.special_tokens (if present)."""
        self.pad_id = self.special_tokens.get("<|pad|>")
        self.bos_id = self.special_tokens.get("<|bos|>")
        self.eos_id = self.special_tokens.get("<|endoftext|>")
        if self.eos_id is None:
            self.eos_id = self.special_tokens.get("<|eos|>")
        self.unk_id = self.special_tokens.get("<|unk|>")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        text_path: str,
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        """Train byte-level BPE merges on a text file.

        Args:
            text_path: Path to training text file.
            special_tokens: Optional dict mapping special-token names to
                reserved integer ids (ids >= 256 recommended). If None,
                PAD/BOS/EOS/UNK are auto-reserved at the top of the target
                vocabulary.

        Raises:
            FileNotFoundError: If the text file does not exist.
            ValueError: If the text file is empty or too short.
        """
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Training text file not found: {text_path}")
        with open(text_path, encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
        self.train_on_text(raw_text, special_tokens=special_tokens)

    def train_on_text(
        self,
        raw_text: str,
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        """Train byte-level BPE merges on an in-memory text string.

        Args:
            raw_text: Training text.
            special_tokens: Optional dict mapping special-token names to
                reserved integer ids (ids >= 256 recommended). If None,
                PAD/BOS/EOS/UNK are auto-reserved at the top of the target
                vocabulary.

        Raises:
            ValueError: If the text is empty or too short.
        """
        if special_tokens is not None:
            self.special_tokens = dict(special_tokens)
        else:
            # Auto-reserve PAD/BOS/EOS/UNK at the top of the target vocab.
            top = self.vocab_size_target
            self.special_tokens = {
                "<|pad|>": top - 4,
                "<|bos|>": top - 3,
                "<|eos|>": top - 2,
                "<|unk|>": top - 1,
            }
        self._resolve_special_ids()

        if not raw_text or len(raw_text) < 100:
            raise ValueError(
                f"Training text too short ({len(raw_text)} chars); "
                "need at least 100 characters for meaningful BPE vocabulary."
            )

        # Words tile the text exactly; each word is a byte-token list.
        word_splits: list[list[str]] = []
        for word in self._split_words(raw_text):
            word_splits.append(self._text_to_byte_tokens(word))
        word_splits = [w for w in word_splits if w]
        if not word_splits:
            raise ValueError("Text preprocessing produced no words for training.")

        # Special tokens are reserved at the top of the target vocabulary but
        # are NOT inserted into self.vocab until after merging completes, so
        # learned merges take ids 256.. (target - n_specials - 1) and can never
        # overflow past the reserved ids.
        reserved_ids: set[int] = set(self.special_tokens.values())
        merge_limit = self.vocab_size_target - len(self.special_tokens)
        current_vocab_size = len(self.vocab)

        while current_vocab_size < merge_limit:
            pair_counts: Counter = Counter()
            for split in word_splits:
                for i in range(len(split) - 1):
                    pair_counts[(split[i], split[i + 1])] += 1

            if not pair_counts:
                break
            best_pair = max(pair_counts, key=lambda p: pair_counts[p])
            best_frequency = pair_counts[best_pair]
            if best_frequency < 2:
                break

            merged_str = best_pair[0] + best_pair[1]

            # Apply the merge to every word (always — even if the merged
            # string already exists, so training and encoding stay aligned).
            new_word_splits = []
            for split in word_splits:
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and split[i] == best_pair[0] and split[i + 1] == best_pair[1]:
                        new_split.append(merged_str)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                new_word_splits.append(new_split)
            word_splits = new_word_splits

            # Record the merge rule and add the token (if not already present
            # and not reserved for special tokens).
            if merged_str not in self.vocab:
                new_id = self._find_next_available_id(reserved_ids)
                self.vocab[merged_str] = new_id
                current_vocab_size += 1
            self.merges.append(best_pair)

        # Now that merging is complete, register the special tokens so that
        # vocab_size() covers their ids and decode() can render them.
        for name, sid in self.special_tokens.items():
            if name not in self.vocab:
                self.vocab[name] = sid
        self._resolve_special_ids()

    def _find_next_available_id(self, reserved_ids: set[int]) -> int:
        candidate = max(self.vocab.values()) + 1 if self.vocab else 0
        while candidate in reserved_ids or candidate in self.vocab.values():
            candidate += 1
        return candidate

    def _get_ranks(self) -> dict[tuple[str, str], int]:
        if not hasattr(self, "_merge_ranks_cache") or len(getattr(self, "_merge_ranks_cache", {})) != len(self.merges):
            self._merge_ranks_cache = {pair: rank for rank, pair in enumerate(self.merges)}
        return self._merge_ranks_cache

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str, **kwargs) -> list[int]:
        """Encode text into token ids using the learned byte-level BPE.

        Guarantees:
        - decode(encode(text)) == text for arbitrary Unicode input.
        - Deterministic for a fixed vocabulary/merge set.

        Args:
            text: Input text string.
            **kwargs: Reserved for extensions.

        Returns:
            List of integer token ids.

        Raises:
            TypeError: If input is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"encode() expects a string, got {type(text)}")

        ranks = self._get_ranks()
        token_ids: list[int] = []

        for word in self._split_words(text):
            parts = self._text_to_byte_tokens(word)
            # Repeatedly merge the lowest-rank adjacent pair (standard BPE).
            while len(parts) > 1:
                best_idx: int | None = None
                best_rank = float("inf")
                for i in range(len(parts) - 1):
                    r = ranks.get((parts[i], parts[i + 1]))
                    if r is not None and r < best_rank:
                        best_rank = r
                        best_idx = i
                if best_idx is None:
                    break
                merged = parts[best_idx] + parts[best_idx + 1]
                parts = parts[:best_idx] + [merged] + parts[best_idx + 2 :]

            for tok in parts:
                token_ids.append(self.vocab[tok])

        return token_ids

    def decode(self, tokens: list[int], strict: bool = False, **kwargs) -> str:
        """Decode token ids back to a text string.

        Args:
            tokens: Sequence of integer token ids.
            strict: If True, raise ValueError on unknown token ids.
            **kwargs: Reserved.

        Returns:
            Reconstructed text (lossless for ids produced by encode()).

        Raises:
            TypeError: If input is not a list of integers.
            ValueError: If strict and an id is not in the vocabulary.
        """
        if not isinstance(tokens, list):
            raise TypeError(f"decode() expects a list of integers, got {type(tokens)}")

        id_to_token: dict[int, str] = {v: k for k, v in self.vocab.items()}
        byte_parts: list[bytes] = []
        for token_id in tokens:
            if token_id in id_to_token:
                byte_parts.append(id_to_token[token_id].encode("latin-1"))
            else:
                if strict:
                    raise ValueError(
                        f"Token ID {token_id} not found in vocabulary. Vocabulary size: {self.vocab_size()}."
                    )
                # Non-strict: render unknown ids visibly but keep byte stream safe.
                byte_parts.append(f"<{token_id}>".encode())

        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self) -> int:
        """Return the number of entries in the vocabulary (max id + 1)."""
        if not self.vocab:
            return 0
        return max(self.vocab.values()) + 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist vocabulary, merges, and special-token ids to JSON.

        Args:
            path: Output file path.

        Raises:
            FileNotFoundError: If the parent directory does not exist.
            OSError: If the write fails.
        """
        parent_dir = os.path.dirname(path)
        if parent_dir and not os.path.exists(parent_dir):
            raise FileNotFoundError(f"Parent directory for tokenizer save does not exist: {parent_dir}")
        save_data = {
            "version": self._version,
            "vocab_size_target": self.vocab_size_target,
            "vocab": self.vocab,
            "merges": [{"part_a": a, "part_b": b} for a, b in self.merges],
            "special_tokens": self.special_tokens,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            raise OSError(f"Failed to save tokenizer to {path}: {exc}") from exc

    def load(self, path: str) -> None:
        """Restore a tokenizer previously saved with save().

        Args:
            path: Path to the JSON produced by save().

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the format is invalid.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer save file not found: {path}")
        try:
            with open(path, encoding="utf-8") as f:
                save_data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Failed to load tokenizer from {path}: invalid file format: {exc}") from exc

        required = {"vocab", "merges", "vocab_size_target"}
        missing = required - set(save_data.keys())
        if missing:
            raise ValueError(f"Tokenizer save file {path} is missing required fields: {missing}")

        self.vocab = {str(k): int(v) for k, v in save_data["vocab"].items()}
        self.merges = [(entry.get("part_a", ""), entry.get("part_b", "")) for entry in save_data.get("merges", [])]
        self.vocab_size_target = int(save_data.get("vocab_size_target", len(self.vocab)))
        self.special_tokens = {str(k): int(v) for k, v in save_data.get("special_tokens", {}).items()}
        self._version = str(save_data.get("version", "xrfm-bpe-v1"))
        # Keep special tokens decodable (they may not be in older vocab files).
        for name, sid in self.special_tokens.items():
            if name not in self.vocab:
                self.vocab[name] = sid
        self._resolve_special_ids()
