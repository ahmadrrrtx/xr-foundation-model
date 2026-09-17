"""Phase 0 chunking edge-case tests (spec #9).

Every case listed in the Phase 0 requirements:
    overlap = 0, 1, max_seq_len-1, max_seq_len, > max_seq_len,
    negative overlap, empty input, very short input, exact-length input,
    max_seq_len <= 0.
"""

import pytest

from xrfm.data import PackingError, chunk_token_ids

IDS = list(range(100))


class TestValidChunking:
    def test_overlap_zero_non_overlapping_tiles(self):
        chunks = chunk_token_ids(IDS, max_seq_len=10, overlap=0)
        assert chunks[0] == IDS[0:10]
        assert chunks[1] == IDS[10:20]
        assert sum(len(c) for c in chunks) == 100

    def test_overlap_one_shares_one_token(self):
        chunks = chunk_token_ids(IDS, max_seq_len=10, overlap=1)
        assert chunks[0][-1] == chunks[1][0]
        assert chunks[0] == IDS[0:10]
        assert chunks[1] == IDS[9:19]

    def test_overlap_max_minus_one(self):
        # Stride-1 sliding window: one window per start position (documented
        # contract — trailing shorter windows are included; the dataset
        # masks their padding via -100).
        chunks = chunk_token_ids(list(range(5)), max_seq_len=4, overlap=3)
        assert chunks == [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4], [3, 4], [4]]
        assert chunks[1][:3] == chunks[0][1:] == [1, 2, 3]  # 3 shared tokens

    def test_short_input_single_short_chunk(self):
        chunks = chunk_token_ids([1, 2, 3], max_seq_len=10, overlap=0)
        assert chunks == [[1, 2, 3]]

    def test_exact_length_single_chunk(self):
        chunks = chunk_token_ids(list(range(10)), max_seq_len=10, overlap=0)
        assert chunks == [list(range(10))]

    def test_empty_input_returns_empty_list(self):
        assert chunk_token_ids([], max_seq_len=8, overlap=0) == []


class TestInvalidChunking:
    """These previously caused infinite loops or silent corruption (D5)."""

    def test_overlap_equal_to_max_seq_len_rejected(self):
        with pytest.raises(PackingError, match="overlap"):
            chunk_token_ids(IDS, max_seq_len=10, overlap=10)

    def test_overlap_greater_than_max_seq_len_rejected(self):
        with pytest.raises(PackingError, match="overlap"):
            chunk_token_ids(IDS, max_seq_len=10, overlap=25)

    def test_negative_overlap_rejected(self):
        with pytest.raises(PackingError, match="overlap"):
            chunk_token_ids(IDS, max_seq_len=10, overlap=-1)

    def test_zero_max_seq_len_rejected(self):
        with pytest.raises(PackingError, match="max_seq_len"):
            chunk_token_ids(IDS, max_seq_len=0)

    def test_negative_max_seq_len_rejected(self):
        with pytest.raises(PackingError, match="max_seq_len"):
            chunk_token_ids(IDS, max_seq_len=-8)

    def test_boolean_args_rejected(self):
        with pytest.raises(PackingError):
            chunk_token_ids(IDS, max_seq_len=True, overlap=False)

    def test_validation_happens_before_tokenization(self):
        """chunk_text with bad params must fail fast, not encode first."""

        class Boom:
            def encode(self, text):
                raise AssertionError("tokenizer must not be called for invalid params")

        from xrfm.data import chunk_text

        with pytest.raises(PackingError):
            chunk_text("some text", max_seq_len=4, tokenizer=Boom(), overlap=4)
