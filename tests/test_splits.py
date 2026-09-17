"""Phase 0 dataset-split tests: determinism, seed consumption, semantics."""

import random

import pytest

from xrfm.data import split_dataset, split_dataset_lines
from xrfm.data.splits import SplitRatiosError

LINES = [f"line {i:03d}" for i in range(100)]


class TestSplitSemantics:
    def test_sequential_split_is_default_and_ordered(self):
        tr, va, te = split_dataset_lines(LINES, shuffle=False)
        assert tr[0] == "line 000"
        assert va[0] == "line 090"
        assert te[0] == "line 095"
        assert len(tr) == 90 and len(va) == 5 and len(te) == 5

    def test_split_text_preserves_content(self):
        text = "\n".join(LINES)
        tr, va, te = split_dataset(text, shuffle=False)
        assert "\n".join(LINES[:90]) == tr

    def test_ratios_must_sum_to_one(self):
        with pytest.raises(SplitRatiosError, match="sum"):
            split_dataset_lines(LINES, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_ratio_bounds(self):
        with pytest.raises(SplitRatiosError):
            split_dataset_lines(LINES, train_ratio=1.2, val_ratio=-0.1, test_ratio=-0.1)

    def test_empty_input(self):
        assert split_dataset_lines([]) == ([], [], [])


class TestSeedBehavior:
    """Phase 0 #8: the seed argument must actually be used."""

    def test_seeded_shuffle_is_deterministic(self):
        a = split_dataset_lines(LINES, shuffle=True, seed=42)
        b = split_dataset_lines(LINES, shuffle=True, seed=42)
        assert a == b

    def test_different_seeds_usually_differ(self):
        a = split_dataset_lines(LINES, shuffle=True, seed=1)
        b = split_dataset_lines(LINES, shuffle=True, seed=2)
        assert a != b

    def test_seed_changes_partition_membership(self):
        """With shuffle, lines can move between splits vs sequential order."""
        seq_tr, _, _ = split_dataset_lines(LINES, shuffle=False)
        shu_tr, _, _ = split_dataset_lines(LINES, shuffle=True, seed=7)
        assert "line 000" in seq_tr
        assert shu_tr != seq_tr

    def test_shuffle_does_not_mutate_global_rng(self):
        rng = random.Random(0)
        expected = [rng.random() for _ in range(10)]
        random.seed(0)
        split_dataset_lines(LINES, shuffle=True, seed=123)
        assert [random.random() for _ in range(10)] == expected

    def test_dedup_prevents_cross_split_leakage(self):
        lines = ["dup"] * 10 + [f"uniq{i}" for i in range(10)]
        tr, va, te = split_dataset_lines(lines, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05, dedup=True)
        all_splits = set(tr) | set(va) | set(te)
        assert "dup" in all_splits
        # 'dup' appears in exactly one split
        count = (1 if "dup" in tr else 0) + (1 if "dup" in va else 0) + (1 if "dup" in te else 0)
        assert count == 1
