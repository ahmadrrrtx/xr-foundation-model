"""
Sequence packing for XRFM Phase 1.

Goal:
documents -> tokens -> continuous token stream -> fixed-length sequences

Avoid unnecessary padding, support sequence_length, deterministic.

Documents:
- whether EOS is inserted
- whether documents may cross sequence boundaries
- whether document boundaries are preserved
- whether attention masking is required
- how leftovers are handled

We implement packing that concatenates token streams and chunks into fixed length,
with optional EOS insertion and tracking of document boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from xrfm.data.tokenization import TokenizedDocument


@dataclass
class PackingConfig:
    sequence_length: int = 2048
    add_eos_between_docs: bool = True
    eos_token_id: Optional[int] = None
    # If True, documents may cross sequence boundaries (efficient)
    # If False, each sequence contains at most one document (wasteful but preserves boundaries)
    allow_cross_document: bool = True
    # How to handle leftovers: drop, pad, keep
    leftover_handling: str = "drop"  # drop | pad | keep
    pad_token_id: int = 0
    # Whether to preserve document boundaries via attention mask (future)
    preserve_boundaries: bool = False


@dataclass
class PackedSequence:
    input_ids: List[int]
    # Optional: document boundaries within sequence (list of start indices)
    doc_boundaries: List[int] = field(default_factory=list)
    # Optional: source tracking
    sources: List[str] = field(default_factory=list)


@dataclass
class PackingResult:
    sequences: List[PackedSequence]
    total_tokens: int
    total_sequences: int
    discarded_tokens: int
    padded_tokens: int
    # For reporting
    tokens_per_source: Dict[str, int] = field(default_factory=dict)


def pack_tokenized_documents(
    tokenized_docs: List[TokenizedDocument],
    config: PackingConfig,
) -> PackingResult:
    """
    Pack tokenized documents into fixed-length sequences.

    Steps:
    1. Build continuous token stream by concatenating docs, optionally inserting EOS.
    2. Chunk into sequence_length pieces.
    3. Handle leftovers per config.
    4. Track discarded/padded counts.

    Deterministic.
    """
    if config.sequence_length <= 0:
        raise ValueError(f"sequence_length must be >0, got {config.sequence_length}")

    # Build stream
    stream: List[int] = []
    # Track doc boundaries in stream
    stream_boundaries: List[Tuple[int, str]] = []  # (position, source)

    tokens_per_source: Dict[str, int] = {}

    for doc in tokenized_docs:
        start_pos = len(stream)
        stream_boundaries.append((start_pos, doc.source))
        stream.extend(doc.tokens)
        tokens_per_source[doc.source] = tokens_per_source.get(doc.source, 0) + doc.token_count

        if config.add_eos_between_docs and config.eos_token_id is not None:
            # Avoid double EOS if doc already ends with EOS
            if not doc.tokens or doc.tokens[-1] != config.eos_token_id:
                stream.append(config.eos_token_id)
                tokens_per_source[doc.source] += 1

    total_tokens = len(stream)
    seq_len = config.sequence_length

    sequences: List[PackedSequence] = []
    discarded = 0
    padded = 0

    # Chunk
    if config.allow_cross_document:
        # Simple: split stream into chunks
        num_full = total_tokens // seq_len
        for i in range(num_full):
            start = i * seq_len
            end = start + seq_len
            chunk = stream[start:end]
            # Find doc boundaries within this chunk
            boundaries = []
            sources_in_chunk = []
            for b_pos, src in stream_boundaries:
                if start <= b_pos < end:
                    boundaries.append(b_pos - start)
                    sources_in_chunk.append(src)
            sequences.append(PackedSequence(input_ids=chunk, doc_boundaries=boundaries, sources=sources_in_chunk))

        leftover = total_tokens % seq_len
        if leftover > 0:
            if config.leftover_handling == "keep":
                # Keep as short sequence? But training expects fixed length, so we pad if needed?
                # For keep, we produce a sequence with leftover tokens + padding
                chunk = stream[num_full * seq_len :]
                # Pad
                pad_needed = seq_len - len(chunk)
                chunk_padded = chunk + [config.pad_token_id] * pad_needed
                padded += pad_needed
                boundaries = []
                sources_in_chunk = []
                start = num_full * seq_len
                for b_pos, src in stream_boundaries:
                    if start <= b_pos < total_tokens:
                        boundaries.append(b_pos - start)
                        sources_in_chunk.append(src)
                sequences.append(PackedSequence(input_ids=chunk_padded, doc_boundaries=boundaries, sources=sources_in_chunk))
            elif config.leftover_handling == "pad":
                # Same as keep, but always pad
                chunk = stream[num_full * seq_len :]
                pad_needed = seq_len - len(chunk)
                chunk_padded = chunk + [config.pad_token_id] * pad_needed
                padded += pad_needed
                sequences.append(PackedSequence(input_ids=chunk_padded, doc_boundaries=[], sources=[]))
            else:  # drop
                discarded = leftover
    else:
        # Preserve document boundaries: each sequence contains at most one doc, no cross-doc
        # This is less efficient, but documents are kept separate
        for doc in tokenized_docs:
            tokens = doc.tokens
            # If doc longer than seq_len, split it
            for start in range(0, len(tokens), seq_len):
                chunk = tokens[start : start + seq_len]
                if len(chunk) < seq_len:
                    if config.leftover_handling == "drop":
                        discarded += len(chunk)
                        continue
                    else:
                        pad_needed = seq_len - len(chunk)
                        chunk = chunk + [config.pad_token_id] * pad_needed
                        padded += pad_needed
                sequences.append(PackedSequence(input_ids=chunk, doc_boundaries=[0], sources=[doc.source]))

    return PackingResult(
        sequences=sequences,
        total_tokens=total_tokens,
        total_sequences=len(sequences),
        discarded_tokens=discarded,
        padded_tokens=padded,
        tokens_per_source=tokens_per_source,
    )
