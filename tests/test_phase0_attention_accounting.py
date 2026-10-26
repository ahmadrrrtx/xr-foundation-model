"""Strict Phase 0 causality, cache-parity, and generation-accounting evidence."""

from __future__ import annotations

import pytest
import torch

from xrfm.config.schema import ModelConfig
from xrfm.inference.engine import GenerationEngine
from xrfm.models.gpt import GPTModel


def _model(max_seq_len: int = 16) -> GPTModel:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=64,
        d_model=16,
        n_layers=2,
        n_heads=4,
        d_ff=32,
        max_seq_len=max_seq_len,
        dropout=0.0,
        use_rope=True,
    )
    return GPTModel(config).eval()


def test_future_token_changes_cannot_affect_past_logits() -> None:
    model = _model()
    original = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed_future = torch.tensor([[1, 2, 3, 40, 41, 42]])
    with torch.no_grad():
        original_logits, _ = model(original)
        changed_logits, _ = model(changed_future)
    torch.testing.assert_close(original_logits[:, :3], changed_logits[:, :3], rtol=1e-5, atol=1e-6)


def test_incremental_kv_cache_matches_full_forward_numerically() -> None:
    model = _model()
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    with torch.no_grad():
        full_logits, _ = model(prompt)
        incremental_logits, cache = model(prompt[:, :3], use_cache=True)
        torch.testing.assert_close(incremental_logits, full_logits[:, :3], rtol=1e-4, atol=1e-5)
        for position in range(3, prompt.shape[1]):
            incremental_logits, cache = model(prompt[:, position : position + 1], past_key_values=cache, use_cache=True)
            torch.testing.assert_close(incremental_logits[:, -1], full_logits[:, position], rtol=1e-4, atol=1e-5)


def test_cached_and_uncached_greedy_generation_are_identical() -> None:
    engine = GenerationEngine(_model())
    prompt = torch.tensor([[1, 2, 3, 4]])
    cached = engine.generate(prompt, max_new_tokens=4, temperature=0)
    uncached = engine.generate_batch(prompt, max_new_tokens=4, temperature=0)
    assert torch.equal(cached, uncached.squeeze(0))


def test_stop_token_is_counted_and_terminal_metadata_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = GenerationEngine(_model())
    sampled = iter([7, 8, 9])

    def fake_sample(logits: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return torch.tensor([[next(sampled)]], device=logits.device)

    monkeypatch.setattr("xrfm.inference.engine.sample_token", fake_sample)
    events = list(engine.stream(torch.tensor([[1, 2]]), max_new_tokens=5, stop_token_id=8))
    assert [event.token_id for event in events if event.token_id is not None] == [7, 8]
    final = events[-1].result
    assert final is not None
    assert final.token_ids.tolist() == [1, 2, 7, 8]
    assert final.completion_token_ids.tolist() == [7, 8]
    assert (final.prompt_tokens, final.effective_prompt_tokens, final.completion_tokens) == (2, 2, 2)
    assert final.truncated_prompt_tokens == 0
    assert final.finish_reason == "stop"
    assert final.stop_sequence is None


def test_stop_string_is_counted_and_identified(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = GenerationEngine(_model())
    sampled = iter([5, 6, 7, 8])

    def fake_sample(logits: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return torch.tensor([[next(sampled)]], device=logits.device)

    monkeypatch.setattr("xrfm.inference.engine.sample_token", fake_sample)
    decoding = {5: "A", 6: "B", 7: "C", 8: "D"}
    final = engine.generate_with_metadata(
        torch.tensor([[1, 2]]),
        max_new_tokens=5,
        stop_sequences=["BC"],
        decode_fn=lambda ids: "".join(decoding[token] for token in ids),
    )
    assert final.completion_token_ids.tolist() == [5, 6, 7]
    assert final.completion_tokens == 3
    assert final.finish_reason == "stop"
    assert final.stop_sequence == "BC"


def test_truncation_preserves_original_effective_and_completion_accounting() -> None:
    engine = GenerationEngine(_model(max_seq_len=4))
    final = engine.generate_with_metadata(torch.tensor([[1, 2, 3, 4, 5, 6]]), max_new_tokens=10, temperature=0)
    assert final.prompt_tokens == 6
    assert final.effective_prompt_tokens == 3
    assert final.truncated_prompt_tokens == 3
    assert final.completion_tokens == 1
    assert final.token_ids[:3].tolist() == [4, 5, 6]
    assert final.completion_token_ids.shape == (1,)
    assert final.finish_reason == "length"
