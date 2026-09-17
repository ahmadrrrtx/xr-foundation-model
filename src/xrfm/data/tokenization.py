"""
Tokenization pipeline integration for XRFM.

Records tokenizer identity, version, config, hash, special token IDs.
Reproducible from raw source + processing config + tokenizer version.

Integrates Phase 0 tokenizer contract.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from xrfm.data.schema import Document


def _hash_tokenizer_config(tokenizer) -> str:
    """Hash tokenizer identity for reproducibility."""
    # Try to get vocab or config
    try:
        vocab_size = tokenizer.vocab_size()
    except Exception:
        vocab_size = 0

    # Try to get special tokens
    try:
        pad_id = getattr(tokenizer, "pad_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        bos_id = getattr(tokenizer, "bos_token_id", None)
        unk_id = getattr(tokenizer, "unk_token_id", None)
    except Exception:
        pad_id = eos_id = bos_id = unk_id = None

    payload = json.dumps(
        {
            "vocab_size": vocab_size,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
            "bos_token_id": bos_id,
            "unk_token_id": unk_id,
            "class": type(tokenizer).__name__,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class TokenizerInfo:
    name: str
    version: str
    hash: str
    vocab_size: int
    pad_token_id: Optional[int]
    eos_token_id: Optional[int]
    bos_token_id: Optional[int]
    unk_token_id: Optional[int]
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tokenizer(cls, tokenizer, name: str = "xrfm-bpe", version: str = "v1") -> TokenizerInfo:
        vocab_size = tokenizer.vocab_size() if hasattr(tokenizer, "vocab_size") else 0
        return cls(
            name=name,
            version=version,
            hash=_hash_tokenizer_config(tokenizer),
            vocab_size=vocab_size,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            bos_token_id=getattr(tokenizer, "bos_token_id", None),
            unk_token_id=getattr(tokenizer, "unk_token_id", None),
            config={"class": type(tokenizer).__name__},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "hash": self.hash,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "unk_token_id": self.unk_token_id,
            "config": self.config,
        }


@dataclass
class TokenizedDocument:
    document_id: str
    source: str
    tokens: List[int]
    token_count: int
    tokenizer_hash: str
    content_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source": self.source,
            "tokens": self.tokens,
            "token_count": self.token_count,
            "tokenizer_hash": self.tokenizer_hash,
            "content_hash": self.content_hash,
        }


@dataclass
class TokenizationConfig:
    add_eos: bool = True
    add_bos: bool = False
    # Whether to insert EOS between documents when packing later
    # This config is for per-doc tokenization
    max_length: Optional[int] = None  # truncate if needed


def tokenize_documents(
    docs: List[Document],
    tokenizer,
    config: TokenizationConfig | None = None,
) -> Tuple[List[TokenizedDocument], TokenizerInfo, Dict[str, int]]:
    """
    Tokenize list of documents, recording tokenizer identity.
    Returns (tokenized_docs, tokenizer_info, stats)
    Stats: total tokens, avg, etc.
    """
    cfg = config or TokenizationConfig()
    info = TokenizerInfo.from_tokenizer(tokenizer)

    tokenized: List[TokenizedDocument] = []
    total_tokens = 0

    for doc in docs:
        try:
            tokens = tokenizer.encode(doc.text)
        except Exception as e:
            # If tokenizer fails, skip? For Phase 1, we record 0 and continue
            # But better to raise
            raise ValueError(f"Tokenization failed for doc {doc.document_id}: {e}") from e

        # Apply BOS/EOS
        if cfg.add_bos and info.bos_token_id is not None:
            tokens = [info.bos_token_id] + tokens
        if cfg.add_eos and info.eos_token_id is not None:
            tokens = tokens + [info.eos_token_id]

        # Truncate if needed
        if cfg.max_length and len(tokens) > cfg.max_length:
            tokens = tokens[: cfg.max_length]

        td = TokenizedDocument(
            document_id=doc.document_id,
            source=doc.source,
            tokens=tokens,
            token_count=len(tokens),
            tokenizer_hash=info.hash,
            content_hash=doc.content_hash,
        )
        tokenized.append(td)
        total_tokens += len(tokens)

    stats = {
        "total_documents": len(docs),
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": total_tokens // max(len(docs), 1),
    }

    return tokenized, info, stats
