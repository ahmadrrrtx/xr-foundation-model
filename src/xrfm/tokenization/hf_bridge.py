"""
Hugging Face Rust `tokenizers` bridge for XRFM Track A (`XRFM Native`).

Provides:
1. ``HFTokenizerAdapter``: wraps a fast Rust ``tokenizers.Tokenizer`` instance
   while implementing 100% of XRFM's ``Tokenizer`` contract (``encode``,
   ``decode``, ``vocab_size``, ``save``, ``load``, ``pad_token_id``,
   ``bos_token_id``, ``eos_token_id``, ``unk_token_id``, and ``vocab`` dict for
   content-addressed ``tokenizer_fingerprint()``).
2. ``train_hf_candidate_tokenizer``: trains Byte-Level BPE or Byte-Fallback
   Unigram tokenizers in seconds on CPU across 7 natural languages + code,
   including Indic/Perso-Arabic grapheme-preserving regex pre-tokenization.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from xrfm.tokenization.interface import Tokenizer

DEFAULT_XRFM_SPECIAL_TOKENS: tuple[str, ...] = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|tool_call|>",
    "<|tool_response|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|cite_start|>",
    "<|cite_end|>",
)


class HFTokenizerAdapter(Tokenizer):
    """Adapter exposing a Rust ``tokenizers.Tokenizer`` via XRFM's ``Tokenizer`` ABC."""

    def __init__(self, backend: Any | None = None, version: str = "xrfm-hf-tok-v1") -> None:
        super().__init__()
        self._backend = backend
        self._version = version
        self.vocab: dict[str, int] = {}
        self.special_tokens: dict[str, int] = {}
        self._pad_id: int | None = None
        self._bos_id: int | None = None
        self._eos_id: int | None = None
        self._unk_id: int | None = None
        self._single_byte_ids: frozenset[int] = frozenset()
        if backend is not None:
            self._sync_metadata()

    def _sync_metadata(self) -> None:
        if self._backend is None:
            return
        self.vocab = dict(self._backend.get_vocab(with_added_tokens=True))
        added_decoder = self._backend.get_added_tokens_decoder()
        self.special_tokens = {tok_obj.content: int(tid) for tid, tok_obj in added_decoder.items() if tok_obj.special}
        self._pad_id = self.special_tokens.get("<|pad|>")
        self._bos_id = self.special_tokens.get("<|bos|>")
        self._eos_id = self.special_tokens.get("<|eos|>") or self.special_tokens.get("<|endoftext|>")
        self._unk_id = self.special_tokens.get("<|unk|>")

        # Identify single-byte fallback tokens so fertility evaluator can measure exact byte-fallback rate:
        # In ByteLevel BPE, base alphabet has 256 single-character byte symbols (ids before first merge,
        # specifically tokens whose decoded byte length is 1 and are not ASCII printable characters, or <0xXX> in Unigram).
        single_byte: set[int] = set()
        for tok_str, tid in self.vocab.items():
            if tid in self.special_tokens.values():
                continue
            if len(tok_str) == 6 and tok_str.startswith("<0x") and tok_str.endswith(">"):
                single_byte.add(tid)
            elif len(tok_str) == 1:
                # Check if decoding this single token yields an incomplete UTF-8 sequence (replacement char)
                decoded = self._backend.decode([tid], skip_special_tokens=False)
                if decoded == "\ufffd":
                    single_byte.add(tid)
        self._single_byte_ids = frozenset(single_byte)

    @property
    def single_byte_fallback_ids(self) -> frozenset[int]:
        return self._single_byte_ids

    @property
    def pad_token_id(self) -> int | None:
        return self._pad_id

    @property
    def bos_token_id(self) -> int | None:
        return self._bos_id

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_id

    @property
    def unk_token_id(self) -> int | None:
        return self._unk_id

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        if not isinstance(text, str):
            raise TypeError(f"encode() expects a string, got {type(text)}")
        if not text:
            return []
        if self._backend is None:
            raise RuntimeError("HFTokenizerAdapter has no loaded backend.")
        enc = self._backend.encode(text, add_special_tokens=False)
        return list(enc.ids)

    def decode(self, tokens: list[int], strict: bool = False, **kwargs: Any) -> str:
        if not isinstance(tokens, list):
            raise TypeError(f"decode() expects a list of integers, got {type(tokens)}")
        if not tokens:
            return ""
        if self._backend is None:
            raise RuntimeError("HFTokenizerAdapter has no loaded backend.")
        vsize = self.vocab_size()
        if strict:
            for tid in tokens:
                if tid < 0 or tid >= vsize:
                    raise ValueError(f"Token ID {tid} out of vocabulary range [0, {vsize})")
        return str(self._backend.decode(tokens, skip_special_tokens=False))

    def vocab_size(self) -> int:
        if not self.vocab:
            return 0
        return max(self.vocab.values()) + 1

    def save(self, path: str) -> None:
        if self._backend is None:
            raise RuntimeError("Cannot save uninitialized HFTokenizerAdapter.")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._backend.save(path)

    def load(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        from tokenizers import Tokenizer as RustTokenizer  # type: ignore[import-untyped]

        self._backend = RustTokenizer.from_file(path)
        self._sync_metadata()

    @classmethod
    def from_file(cls, path: str) -> HFTokenizerAdapter:
        inst = cls()
        inst.load(path)
        return inst

    @classmethod
    def pretrained_32k(cls) -> HFTokenizerAdapter:
        """Load the canonical Phase 2 winning 32,768-vocab multilingual+code tokenizer."""
        from importlib import resources

        tok_path = resources.files("xrfm.tokenization").joinpath("tokenizer_32k_multilingual.json")
        return cls.from_file(str(tok_path))


def train_hf_candidate_tokenizer(
    texts: Iterable[str],
    vocab_size: int = 32768,
    algorithm: str = "byte_bpe",
    script_protected: bool = True,
    special_tokens: list[str] | None = None,
    min_frequency: int = 2,
) -> HFTokenizerAdapter:
    """
    Train a candidate tokenizer (`byte_bpe` or `unigram`) using Hugging Face `tokenizers`.

    Args:
        texts: Iterable of UTF-8 training documents.
        vocab_size: Target vocabulary size (e.g. 32768 or 49152).
        algorithm: 'byte_bpe' or 'unigram'.
        script_protected: When True, uses Unicode letter+combining-mark (`\\p{L}\\p{M}*`),
            single-digit splitting (`\\p{N}`), and indentation-preserving pre-tokenization
            so Devanagari matras, Arabic/Urdu harakat, and Python 4-space indents stay intact.
        special_tokens: Reserved special tokens (defaults to ``DEFAULT_XRFM_SPECIAL_TOKENS``).
        min_frequency: Minimum pair frequency for BPE merges.
    """
    from tokenizers import Regex, decoders, models, normalizers, pre_tokenizers, processors, trainers
    from tokenizers import Tokenizer as RustTokenizer

    specials = list(special_tokens) if special_tokens is not None else list(DEFAULT_XRFM_SPECIAL_TOKENS)

    if algorithm == "byte_bpe":
        rust_tok = RustTokenizer(models.BPE(byte_fallback=False))
        rust_tok.normalizer = normalizers.NFC()

        if script_protected:
            # GPT-4o / Qwen-inspired multilingual + code regex:
            # - Preserves contraction suffixes
            # - Groups Unicode letters WITH combining marks (\p{L}\p{M}*) for Hindi Devanagari & Urdu/Arabic!
            # - Splits individual digits (\p{N}) for arithmetic fidelity
            # - Preserves indentation runs and newlines cleanly
            pattern = Regex(
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{M}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            )
            rust_tok.pre_tokenizer = pre_tokenizers.Sequence(
                [
                    pre_tokenizers.Split(pattern=pattern, behavior="isolated", invert=False),
                    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
                ]
            )
        else:
            rust_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)

        rust_tok.post_processor = processors.ByteLevel(trim_offsets=False)
        rust_tok.decoder = decoders.ByteLevel()

        bpe_trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        rust_tok.train_from_iterator(texts, trainer=bpe_trainer)

    elif algorithm == "unigram":
        rust_tok = RustTokenizer(models.Unigram(byte_fallback=True))
        rust_tok.normalizer = normalizers.NFC()
        rust_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        rust_tok.post_processor = processors.ByteLevel(trim_offsets=False)
        rust_tok.decoder = decoders.ByteLevel()

        uni_trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=specials,
            unk_token="<|unk|>",
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        rust_tok.train_from_iterator(texts, trainer=uni_trainer)
    else:
        raise ValueError(f"Unsupported algorithm='{algorithm}'. Expected 'byte_bpe' or 'unigram'.")

    return HFTokenizerAdapter(backend=rust_tok, version=f"xrfm-{algorithm}-{vocab_size}")
