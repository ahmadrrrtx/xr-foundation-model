"""
Deterministic dataset sharding for XRFM.

A distributed training run should be able to say world_size=8 rank=3 and know exactly
which shards/samples it is responsible for.

Must be impossible for workers to accidentally process identical data because of naive IterableDataset replication.

PyTorch warns that iterable datasets are replicated across workers and need worker-aware partitioning.

We implement:
- Shard writer: splits sequences into N shards, each shard is a binary file
- Shard manifest: lists shards, counts, checksums
- Deterministic sharding logic for distributed reading
- Tests for 1,2,4,8 workers with no duplicates

Format choice:
- Each shard: numpy .npy file of shape (num_sequences, seq_len) dtype determined by vocab size
- For simplicity and portability, we use uint32 for vocab < 2^32, else uint16 if vocab < 65536
- Alternative: binary packed .bin files, but .npy is simple, memory-mappable, and torch-compatible
- We also support JSONL for debugging

We document decision in ADR.

Memory safety: streaming write, not loading all sequences into RAM at once if possible.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator

import numpy as np

from xrfm.data.packing_extended import PackedSequence


@dataclass
class ShardingConfig:
    num_shards: int = 8
    sequences_per_shard: Optional[int] = None  # if set, overrides num_shards logic
    dtype: str = "auto"  # auto | uint16 | uint32 | int32
    output_dir: str = "processed/tokens"
    shard_prefix: str = "shard"
    # For deterministic assignment
    seed: int = 42


@dataclass
class ShardInfo:
    shard_id: int
    path: str
    num_sequences: int
    num_tokens: int
    sha256: str
    dtype: str
    sequence_length: int


def _determine_dtype(vocab_size: int, requested: str = "auto") -> np.dtype:
    if requested != "auto":
        return np.dtype(requested)
    # Auto: if vocab < 65535 use uint16 else uint32
    if vocab_size < 65535:
        return np.dtype("uint16")
    else:
        return np.dtype("uint32")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_shards(
    sequences: List[PackedSequence],
    config: ShardingConfig,
    vocab_size: int = 50304,
    sequence_length: int = 2048,
) -> List[ShardInfo]:
    """
    Write sequences into sharded files.

    Deterministic: sequences are assigned round-robin or contiguous?
    We use contiguous chunks for simplicity and deterministic reading.

    Returns list of ShardInfo.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    dtype = _determine_dtype(vocab_size, config.dtype)

    # Determine sharding
    total = len(sequences)
    if total == 0:
        return []

    if config.sequences_per_shard:
        num_shards = (total + config.sequences_per_shard - 1) // config.sequences_per_shard
        per_shard = config.sequences_per_shard
    else:
        num_shards = config.num_shards
        per_shard = (total + num_shards - 1) // num_shards

    shard_infos: List[ShardInfo] = []

    for shard_id in range(num_shards):
        start = shard_id * per_shard
        end = min(start + per_shard, total)
        if start >= total:
            break
        shard_seqs = sequences[start:end]

        # Convert to numpy array
        arr = np.array([s.input_ids for s in shard_seqs], dtype=dtype)

        shard_name = f"{config.shard_prefix}-{shard_id:05d}.npy"
        shard_path = os.path.join(config.output_dir, shard_name)

        # Save as .npy (includes header)
        np.save(shard_path, arr)

        sha = _sha256_file(shard_path)

        info = ShardInfo(
            shard_id=shard_id,
            path=shard_path,
            num_sequences=len(shard_seqs),
            num_tokens=len(shard_seqs) * sequence_length,
            sha256=sha,
            dtype=str(dtype),
            sequence_length=sequence_length,
        )
        shard_infos.append(info)

    return shard_infos


def assign_shards_to_rank(
    shard_infos: List[ShardInfo],
    world_size: int,
    rank: int,
) -> List[ShardInfo]:
    """
    Deterministic shard assignment for distributed training.
    Each rank gets a disjoint subset.

    Simple round-robin: shard_id % world_size == rank
    Ensures no duplicates, full coverage when combined.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be >0, got {world_size}")
    if not (0 <= rank < world_size):
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    return [s for s in shard_infos if s.shard_id % world_size == rank]


def assign_samples_to_worker(
    num_samples: int,
    num_workers: int,
    worker_id: int,
) -> List[int]:
    """
    Assign sample indices to DataLoader workers.
    Prevents duplicate sample consumption when using IterableDataset.

    PyTorch's DataLoader with num_workers>0 replicates dataset; each worker must process disjoint subset.
    We use contiguous split for determinism.
    """
    if num_workers <= 0:
        raise ValueError("num_workers must be >0")
    if not (0 <= worker_id < num_workers):
        raise ValueError(f"worker_id must be in [0, {num_workers})")

    per_worker = (num_samples + num_workers - 1) // num_workers
    start = worker_id * per_worker
    end = min(start + per_worker, num_samples)
    return list(range(start, end))


class ShardedTokenDataset:
    """
    Map-style dataset that reads sharded token files.

    Supports:
    - Deterministic ordering
    - Memory-mapped reading (optional)
    - Distributed sharding via rank/world_size
    - Worker-aware partitioning for DataLoader workers
    """

    def __init__(
        self,
        shard_infos: List[ShardInfo],
        rank: int = 0,
        world_size: int = 1,
        mmap: bool = True,
    ):
        self.all_shards = shard_infos
        self.rank = rank
        self.world_size = world_size
        self.mmap = mmap

        # Assign shards to this rank
        self.shards = assign_shards_to_rank(shard_infos, world_size, rank)

        # Precompute total sequences and offsets
        self._shard_arrays: List[np.ndarray] = []
        self._shard_offsets: List[int] = []
        self._total = 0

        for info in self.shards:
            # Lazy load? For now load metadata only, actual arrays loaded on demand
            # But we need total count
            self._shard_offsets.append(self._total)
            self._total += info.num_sequences

        # For map-style, we will load arrays lazily and cache
        self._loaded: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self._total

    def _load_shard(self, shard_idx: int) -> np.ndarray:
        if shard_idx in self._loaded:
            return self._loaded[shard_idx]
        info = self.shards[shard_idx]
        if self.mmap:
            arr = np.load(info.path, mmap_mode="r")
        else:
            arr = np.load(info.path)
        self._loaded[shard_idx] = arr
        return arr

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self._total:
            raise IndexError(f"Index {idx} out of range for dataset of size {self._total}")

        # Find which shard contains idx
        # Since offsets are sorted, we can linear scan (num shards small) or binary search
        # Linear for simplicity
        shard_idx = 0
        for i, offset in enumerate(self._shard_offsets):
            next_offset = self._shard_offsets[i + 1] if i + 1 < len(self._shard_offsets) else self._total
            if offset <= idx < next_offset:
                shard_idx = i
                break

        local_idx = idx - self._shard_offsets[shard_idx]
        arr = self._load_shard(shard_idx)
        return arr[local_idx]

    def get_worker_partition(self, worker_id: int, num_workers: int) -> List[int]:
        """Get sample indices for a specific worker."""
        return assign_samples_to_worker(len(self), num_workers, worker_id)

    def verify_no_duplicate_across_ranks(self, world_size: int) -> bool:
        """Verify that shards assigned to different ranks are disjoint and cover all."""
        all_assigned: List[int] = []
        for rank in range(world_size):
            assigned = assign_shards_to_rank(self.all_shards, world_size, rank)
            all_assigned.extend([s.shard_id for s in assigned])

        # Check no duplicates
        if len(all_assigned) != len(set(all_assigned)):
            return False
        # Check coverage
        expected = set(s.shard_id for s in self.all_shards)
        if set(all_assigned) != expected:
            return False
        return True


def verify_worker_no_duplicates(num_samples: int, num_workers: int) -> bool:
    """Test that worker partitions have no duplicates and full coverage."""
    all_indices: List[int] = []
    for wid in range(num_workers):
        part = assign_samples_to_worker(num_samples, num_workers, wid)
        all_indices.extend(part)

    # Check no duplicates
    if len(all_indices) != len(set(all_indices)):
        return False
    # Check coverage
    if set(all_indices) != set(range(num_samples)):
        return False
    # Check no overlap between workers
    partitions = [set(assign_samples_to_worker(num_samples, num_workers, wid)) for wid in range(num_workers)]
    for i in range(num_workers):
        for j in range(i + 1, num_workers):
            if partitions[i] & partitions[j]:
                return False
    return True
