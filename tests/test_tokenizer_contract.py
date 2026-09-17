"""Phase 0 tokenizer-contract tests: special tokens, round-trip, unicode."""

import pytest

from xrfm.tokenization import BPETokenizer, encode_text


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.pretrained()


class TestRoundTrip:
    def test_ascii(self, tok):
        s = "Hello, world! 123"
        assert tok.decode(tok.encode(s)) == s

    def test_unicode(self, tok):
        s = "héllo wörld — 你好世界 مرحبا ✓emoji🚀"
        assert tok.decode(tok.encode(s)) == s

    def test_whitespace_preserved(self, tok):
        s = "a  b\t\tc\n\nd \n e"
        assert tok.decode(tok.encode(s)) == s

    def test_empty_string(self, tok):
        assert tok.encode("") == []
        assert tok.decode([]) == ""

    def test_deterministic(self, tok):
        s = "the same text twice"
        assert tok.encode(s) == tok.encode(s)


class TestSpecialTokenContract:
    """Phase 0 #10: explicit contract accessors, never hard-coded ids."""

    def test_pretrained_defines_all_specials(self, tok):
        for name in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"):
            value = getattr(tok, name)
            assert isinstance(value, int), f"{name} should be an int on the pretrained tokenizer"

    def test_pad_is_not_hardcoded_zero(self, tok):
        assert tok.pad_token_id != 0

    def test_contract_ids_in_vocab_range(self, tok):
        v = tok.vocab_size()
        for name in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"):
            assert 0 <= getattr(tok, name) < v

    def test_fresh_untrained_tokenizer_reports_none(self):
        fresh = BPETokenizer()
        assert fresh.pad_token_id is None
        assert fresh.eos_token_id is None

    def test_trained_tokenizer_reserves_specials(self, tmp_path):
        t = BPETokenizer(vocab_size_target=512)
        t.train_on_text("training text for the tokenizer. " * 20)
        assert t.pad_token_id is not None
        assert t.vocab_size() <= 512

    def test_encode_text_helper(self, tok):
        ids = encode_text("hello", tok)
        assert ids == tok.encode("hello")


class TestVocabSizeContract:
    def test_matches_max_id_plus_one(self, tok):
        ids = tok.encode("arbitrary text with symbols !@# %^&")
        assert max(ids) < tok.vocab_size()

    def test_matches_model_vocab(self):
        from xrfm.models import XRFMModel

        t = BPETokenizer.pretrained()
        model = XRFMModel(vocab_size=t.vocab_size())
        assert model.embedding.vocab_size == t.vocab_size()

    def test_interface_repr(self, tok):
        assert str(tok.vocab_size()) in repr(tok)
