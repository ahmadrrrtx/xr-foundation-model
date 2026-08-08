"""
Tests for Phase 31 inference controls: repetition penalty, EOS handling,
stop sequences, and max generation tokens.
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from inference.engine import GenerationEngine  # noqa: E402
from inference.sampling import apply_repetition_penalty, sample_token  # noqa: E402
from model.gpt import GPTModel  # noqa: E402


def _make_engine():
    m = GPTModel("config/tiny.yaml", vocab_size=512)
    m.eval()
    return GenerationEngine(m)


class TestRepetitionPenalty:
    def test_penalty_reduces_repeated_token_logit(self):
        logits = torch.full((1, 10), 1.0)
        logits[0, 3] = 5.0  # token 3 is very likely
        seen = torch.tensor([[3, 7]])  # token 3 already generated
        out = apply_repetition_penalty(logits, penalty=1.5, seen_ids=seen)
        assert out[0, 3].item() < 5.0, "repeated token logit must be reduced"
        assert abs(out[0, 7].item() - 1.0 / 1.5) < 1e-6  # positive -> divide
        assert abs(out[0, 4].item() - 1.0) < 1e-6  # unseen unchanged

    def test_penalty_one_is_noop(self):
        logits = torch.randn(1, 10)
        seen = torch.tensor([[0, 1, 2]])
        out = apply_repetition_penalty(logits.clone(), penalty=1.0, seen_ids=seen)
        assert torch.equal(out, logits)

    def test_sample_token_requires_seen_ids(self):
        logits = torch.randn(1, 20)
        with pytest.raises(ValueError, match="seen_ids"):
            sample_token(logits, temperature=0.8, repetition_penalty=1.5)

    def test_generate_with_penalty_runs(self):
        eng = _make_engine()
        prompt = torch.tensor([5, 9, 12])
        with torch.no_grad():
            out = eng.generate(
                prompt,
                max_new_tokens=20,
                temperature=0.8,
                repetition_penalty=1.5,
            )
        assert out.dim() == 1
        assert out.shape[0] == 3 + 20
        assert torch.isfinite(out).all()


class TestEOSAndStopSequences:
    def test_stop_token_id_stops(self):
        eng = _make_engine()
        prompt = torch.tensor([3, 4])
        with torch.no_grad():
            out = eng.generate(prompt, max_new_tokens=50, temperature=0.8, stop_token_id=99)
        # If token 99 was generated it must be the last token; else length cap.
        assert out.shape[0] <= 2 + 50
        if (out == 99).any():
            assert out[-1].item() == 99

    def test_stop_sequence_truncates(self):
        eng = _make_engine()
        prompt = torch.tensor([3, 4])
        with torch.no_grad():
            out = eng.generate(
                prompt,
                max_new_tokens=50,
                temperature=0.8,
                stop_sequences=["ZZZ"],
                decode_fn=lambda ids: "generated ... ZZZ",  # always contains the stop string
            )
        assert out.shape[0] == 3, "should stop immediately after first token when stop seq present"

    def test_stop_sequences_requires_decode_fn(self):
        eng = _make_engine()
        with pytest.raises(ValueError, match="decode_fn"):
            eng.generate(torch.tensor([3]), max_new_tokens=5, stop_sequences=["x"])

    def test_max_new_tokens_enforced(self):
        eng = _make_engine()
        prompt = torch.tensor([3, 4, 5])
        with torch.no_grad():
            out = eng.generate(prompt, max_new_tokens=7, temperature=0.0)
        assert out.shape[0] == 3 + 7

    def test_invalid_repetition_penalty(self):
        eng = _make_engine()
        with pytest.raises(ValueError, match="repetition_penalty"):
            eng.generate(torch.tensor([3]), max_new_tokens=5, repetition_penalty=0.5)
