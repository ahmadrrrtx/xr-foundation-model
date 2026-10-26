"""Composable greedy, temperature, top-k, top-p, and repetition sampling."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    """Select the highest-logit token for each batch row."""
    return logits.argmax(dim=-1, keepdim=True)


def _validate_temperature(temperature: float) -> None:
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")


def _apply_top_k(scaled: torch.Tensor, top_k: int) -> torch.Tensor:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError(f"top_k must be a positive int, got {top_k!r}")
    k = min(top_k, scaled.shape[-1])
    topk_vals, topk_idx = torch.topk(scaled, k=k, dim=-1)
    del topk_vals
    keep = torch.zeros_like(scaled, dtype=torch.bool)
    keep.scatter_(-1, topk_idx, True)
    return scaled.masked_fill(~keep, float("-inf"))


def _apply_top_p(scaled: torch.Tensor, top_p: float) -> torch.Tensor:
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Keep the first token that crosses the threshold, not just tokens below it.
    remove_sorted = cumulative > top_p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False
    remove = torch.zeros_like(scaled, dtype=torch.bool)
    remove.scatter_(-1, sorted_indices, remove_sorted)
    return scaled.masked_fill(remove, float("-inf"))


def _sample_filtered(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Apply every requested filter to one distribution, then draw once."""
    _validate_temperature(temperature)
    if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
        raise ValueError(f"top_k must be a positive int, got {top_k!r}")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if temperature == 0:
        return sample_greedy(logits)

    scaled = logits / temperature
    if top_k is not None:
        scaled = _apply_top_k(scaled, top_k)
    if top_p is not None:
        # Nucleus probabilities are computed *after* top-k filtering. This is
        # the composition contract; filtering one distribution avoids the old
        # behavior where top-p was silently ignored whenever top-k was set.
        scaled = _apply_top_p(scaled, top_p)
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def sample_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Sample from temperature-scaled logits; zero temperature is greedy."""
    return _sample_filtered(logits, temperature=temperature)


def sample_top_k(logits: torch.Tensor, top_k: int = 50, temperature: float = 1.0) -> torch.Tensor:
    """Sample from the top-k distribution."""
    return _sample_filtered(logits, temperature=temperature, top_k=top_k)


def sample_top_p(logits: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """Sample from the smallest nucleus whose mass reaches ``top_p``."""
    return _sample_filtered(logits, temperature=temperature, top_p=top_p)


def apply_repetition_penalty(
    logits: torch.Tensor,
    penalty: float,
    seen_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply the CTRL-style repetition penalty in place and return logits."""
    if penalty <= 1.0 or seen_ids.numel() == 0:
        return logits
    for b in range(logits.shape[0]):
        unique = torch.unique(seen_ids[b])
        row = logits[b]
        row[unique] = torch.where(row[unique] > 0, row[unique] / penalty, row[unique] * penalty)
    return logits


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    seen_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample once after repetition, temperature, top-k, and top-p composition."""
    if repetition_penalty < 1.0:
        raise ValueError(f"repetition_penalty must be >= 1.0, got {repetition_penalty}")
    _validate_temperature(temperature)
    # Validate filters even for greedy decoding; silently ignoring an invalid
    # request makes API behavior depend on an unrelated parameter.
    if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
        raise ValueError(f"top_k must be a positive int, got {top_k!r}")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    working = logits.clone()
    if repetition_penalty > 1.0:
        if seen_ids is None:
            raise ValueError("seen_ids required when repetition_penalty > 1.0")
        working = apply_repetition_penalty(working, repetition_penalty, seen_ids)
    return _sample_filtered(working, temperature=temperature, top_k=top_k, top_p=top_p)
