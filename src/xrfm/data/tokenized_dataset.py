"""
Tokenized dataset for training from sharded files.

Implements map-style and iterable-style datasets that read sharded token files deterministically.

Supports:
- Loading from manifest
- Rank/world_size sharding
- Worker-aware partitioning
- One training step verification
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Iterator, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

import numpy as np

from xrfm.data.sharding import ShardInfo, assign_shards_to_rank, assign_samples_to_worker
from xrfm.data.manifest_v2 import DatasetManifestV2


class ShardedTokenDatasetMap(Dataset):
    """
    Map-style dataset over sharded token files.
    Each item is (input_ids, targets) for next-token training.

    Compatible with XRFM Trainer.
    """

    def __init__(
        self,
        manifest_path: str | None = None,
        shard_infos: List[ShardInfo] | None = None,
        rank: int = 0,
        world_size: int = 1,
        ignore_index: int = -100,
        mmap: bool = True,
    ):
        if manifest_path:
            manifest = DatasetManifestV2.load(manifest_path)
            # Reconstruct ShardInfo from manifest
            shard_infos = [
                ShardInfo(
                    shard_id=s.shard_id,
                    path=s.path,
                    num_sequences=s.num_sequences,
                    num_tokens=s.num_tokens,
                    sha256=s.sha256,
                    dtype=s.dtype,
                    sequence_length=s.num_tokens // max(s.num_sequences, 1) if s.num_sequences else 0,
                )
                for s in manifest.shards
            ]
            self.sequence_length = manifest.sequence_length
        else:
            self.sequence_length = 0

        if not shard_infos:
            raise ValueError("Must provide manifest_path or shard_infos")

        self.shard_infos = shard_infos
        self.rank = rank
        self.world_size = world_size
        self.ignore_index = ignore_index
        self.mmap = mmap

        # Assign shards to rank
        self.assigned_shards = assign_shards_to_rank(shard_infos, world_size, rank)

        # Compute offsets
        self._shard_arrays: dict[int, np.ndarray] = {}
        self._offsets: List[int] = []
        self._total = 0
        for info in self.assigned_shards:
            self._offsets.append(self._total)
            self._total += info.num_sequences
            if self.sequence_length == 0:
                self.sequence_length = info.sequence_length or 2048

    def __len__(self) -> int:
        return self._total

    def _load_shard(self, shard_idx: int) -> np.ndarray:
        if shard_idx in self._shard_arrays:
            return self._shard_arrays[shard_idx]
        info = self.assigned_shards[shard_idx]
        if self.mmap:
            arr = np.load(info.path, mmap_mode="r")
        else:
            arr = np.load(info.path)
        self._shard_arrays[shard_idx] = arr
        return arr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._total:
            raise IndexError(f"Index {idx} out of range {self._total}")

        # Find shard
        shard_idx = 0
        for i, offset in enumerate(self._offsets):
            next_offset = self._offsets[i + 1] if i + 1 < len(self._offsets) else self._total
            if offset <= idx < next_offset:
                shard_idx = i
                break

        local_idx = idx - self._offsets[shard_idx]
        arr = self._load_shard(shard_idx)
        input_ids = arr[local_idx]  # shape (seq_len,)

        # For training, input is input_ids, target is next token
        # input_ids: [0..L-1], targets: [1..L-1, ignore_index]
        # Convert to torch
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        target_list = list(input_ids[1:]) + [self.ignore_index]
        target_tensor = torch.tensor(target_list, dtype=torch.long)

        return input_tensor, target_tensor


class ShardedTokenDatasetIterable(IterableDataset):
    """
    Iterable dataset that correctly handles worker partitioning.

    PyTorch's IterableDataset is replicated across workers; we must explicitly partition.

    Implements worker-aware sharding to prevent duplicate samples.
    """

    def __init__(
        self,
        manifest_path: str | None = None,
        shard_infos: List[ShardInfo] | None = None,
        rank: int = 0,
        world_size: int = 1,
        ignore_index: int = -100,
        shuffle: bool = False,
        seed: int = 42,
    ):
        if manifest_path:
            manifest = DatasetManifestV2.load(manifest_path)
            shard_infos = [
                ShardInfo(
                    shard_id=s.shard_id,
                    path=s.path,
                    num_sequences=s.num_sequences,
                    num_tokens=s.num_tokens,
                    sha256=s.sha256,
                    dtype=s.dtype,
                    sequence_length=manifest.sequence_length,
                )
                for s in manifest.shards
            ]
            self.sequence_length = manifest.sequence_length
        else:
            self.sequence_length = 0

        if not shard_infos:
            raise ValueError("Must provide manifest_path or shard_infos")

        self.shard_infos = shard_infos
        self.rank = rank
        self.world_size = world_size
        self.ignore_index = ignore_index
        self.shuffle = shuffle
        self.seed = seed

        # Assign shards to rank
        self.assigned_shards = assign_shards_to_rank(shard_infos, world_size, rank)
        self._total = sum(s.num_sequences for s in self.assigned_shards)

    def __len__(self) -> int:
        # For IterableDataset, __len__ not strictly required but useful
        return self._total

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        if worker_info is None:
            # Single worker: iterate over all assigned samples
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Determine which sample indices this worker should handle
        # We need to map global sample index to shard + local index
        # Build list of all sample indices for this rank
        all_indices = list(range(self._total))

        # Partition among workers
        worker_indices = assign_samples_to_worker(len(all_indices), num_workers, worker_id)
        # Map to actual global indices
        global_indices_for_worker = [all_indices[i] for i in worker_indices]

        # Optional shuffle (deterministic per worker)
        if self.shuffle:
            import random

            rng = random.Random(self.seed + worker_id)
            rng.shuffle(global_indices_for_worker)

        # Now iterate
        # Precompute offsets
        offsets: List[int] = []
        total = 0
        for info in self.assigned_shards:
            offsets.append(total)
            total += info.num_sequences

        # For efficiency, load shards as needed
        # We'll keep a cache of loaded shard arrays
        loaded: dict[int, np.ndarray] = {}

        def load_shard(shard_idx: int) -> np.ndarray:
            if shard_idx not in loaded:
                info = self.assigned_shards[shard_idx]
                loaded[shard_idx] = np.load(info.path)
            return loaded[shard_idx]

        for global_idx in global_indices_for_worker:
            # Find shard
            shard_idx = 0
            for i, offset in enumerate(offsets):
                next_offset = offsets[i + 1] if i + 1 < len(offsets) else total
                if offset <= global_idx < next_offset:
                    shard_idx = i
                    break
            local_idx = global_idx - offsets[shard_idx]
            arr = load_shard(shard_idx)
            input_ids = arr[local_idx]
            input_tensor = torch.tensor(input_ids, dtype=torch.long)
            target_list = list(input_ids[1:]) + [self.ignore_index]
            target_tensor = torch.tensor(target_list, dtype=torch.long)
            yield input_tensor, target_tensor
