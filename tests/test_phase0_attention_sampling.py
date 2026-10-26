"""Regression tests for composed attention masks and sampling filters."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import xrfm.optimization.flash_attention as flash_attention
from xrfm.inference.sampling import sample_token
from xrfm.models.attention.multi_head import MultiHeadAttention


def test_explicit_padding_mask_composes_with_causality(monkeypatch: pytest.MonkeyPatch) -> None:
    attn = MultiHeadAttention(d_model=8, n_heads=2, dropout=0.0, use_rope=False)
    attn.eval()
    x = torch.randn(1, 3, 8)
    # Batch padding mask: the final key is padded. Causality must still hide
    # key 1 from query 0 even though the explicit mask marks it as valid.
    mask = torch.tensor([[1, 1, 0]], dtype=torch.float32)
    captured: dict[str, torch.Tensor] = {}

    def capture_mask(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert mask is not None
        captured["mask"] = mask.detach().clone()
        return torch.zeros_like(v)

    monkeypatch.setattr(flash_attention, "flash_attention_forward", capture_mask)
    attn(x, mask=mask)
    additive = torch.broadcast_to(captured["mask"], (1, 1, 3, 3))
    assert torch.isneginf(additive[0, 0, 0, 1])
    assert torch.isneginf(additive[0, 0, 1, 2])
    assert torch.all(torch.isneginf(additive[..., 2]))


def test_top_k_and_top_p_are_composed_before_one_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, torch.Tensor] = {}

    def capture_distribution(probs: torch.Tensor, num_samples: int) -> torch.Tensor:
        captured["probs"] = probs.detach().clone()
        return torch.argmax(probs, dim=-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", capture_distribution)
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    token = sample_token(logits, temperature=1.0, top_k=3, top_p=0.5)

    probs = captured["probs"]
    assert token.item() == 0
    assert probs[0, 0].item() == pytest.approx(1.0)
    assert torch.count_nonzero(probs).item() == 1


def test_invalid_top_p_is_rejected_even_when_top_k_is_set() -> None:
    with pytest.raises(ValueError, match="top_p"):
        sample_token(torch.tensor([[1.0, 0.5, 0.0]]), top_k=2, top_p=0.0)
