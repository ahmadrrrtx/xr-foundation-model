"""
Sampling strategies for XRFM inference engine (v0.6.0).

Supports greedy, temperature, top-k, and top-p (nucleus) sampling.
All functions operate on logits tensors of shape (batch, vocab_size).

Conceptual references (not copied):
- Holtzman et al. (2020) — Top-p / Nucleus Sampling (arXiv:1904.09751)
- Fan et al. (2018) — Top-k sampling
- Brown et al. (2020) — GPT-3 generation strategies

Implementation is original.
"""

import torch
import torch.nn.functional as F


def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    """Select the most probable next token (greedy decoding).

    Args:
        logits: Raw logits (batch, vocab_size).

    Returns:
        Token IDs (batch, 1).
    """
    return logits.argmax(dim=-1, keepdim=True)


def sample_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Sample with temperature scaling.

    Higher temperature (>1.0) → more uniform/random.
    Lower temperature (<1.0) → sharper/more deterministic.
    Temperature = 0 → equivalent to greedy.

    Args:
        logits: Raw logits (batch, vocab_size).
        temperature: Scaling factor. Must be >= 0.

    Returns:
        Token IDs (batch, 1).
    """
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if temperature == 0:
        return sample_greedy(logits)

    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def sample_top_k(logits: torch.Tensor, top_k: int = 50, temperature: float = 1.0) -> torch.Tensor:
    """Sample from the top-k most probable tokens.

    Args:
        logits: Raw logits (batch, vocab_size).
        top_k: Number of highest-probability tokens to keep. Must be > 0.
        temperature: Temperature for scaling before top-k filtering.

    Returns:
        Token IDs (batch, 1).
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")

    if temperature == 0:
        return sample_greedy(logits)

    # Scale by temperature
    scaled = logits / temperature

    # Get top-k values and indices; mask everything else
    top_k = min(top_k, scaled.shape[-1])
    topk_vals, topk_idx = torch.topk(scaled, k=top_k, dim=-1)
    mask = torch.ones_like(scaled, dtype=torch.bool)
    for b in range(scaled.shape[0]):
        mask[b, topk_idx[b]] = False
    scaled = scaled.masked_fill(mask, float("-inf"))

    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def sample_top_p(
    logits: torch.Tensor,
    top_p: float = 0.9,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample using nucleus (top-p) sampling.

    Keeps the smallest set of tokens whose cumulative probability
    exceeds top_p, then samples from that set.

    Args:
        logits: Raw logits (batch, vocab_size).
        top_p: Cumulative probability threshold in (0, 1].
        temperature: Temperature for scaling before filtering.

    Returns:
        Token IDs (batch, 1).
    """
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")

    if temperature == 0:
        return sample_greedy(logits)

    logits.shape[0]

    # Scale by temperature
    scaled = logits / temperature

    # Sort in descending order
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)

    # Cumulative probabilities
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above top_p
    # Shift by 1 to include the token that crosses the threshold
    sorted_mask = cumsum > top_p
    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
    sorted_mask[:, 0] = False

    # Scatter mask back to original indices
    mask = torch.zeros_like(scaled, dtype=torch.bool)
    mask = mask.scatter(1, sorted_indices, sorted_mask)
    scaled = scaled.masked_fill(mask, float("-inf"))

    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def apply_repetition_penalty(
    logits: torch.Tensor,
    penalty: float,
    seen_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply a repetition penalty to logits of previously generated tokens.

    Standard CTRL-style penalty (Keskar et al., 2019): for each token id
    already present in the sequence, its logit is divided by ``penalty``
    (penalty > 1 suppresses repetition) — equivalently, logits are pushed
    away from 0 for repeated tokens.

    Args:
        logits: Raw logits (batch, vocab_size).
        penalty: Penalty factor (> 1.0 suppresses repeats; 1.0 = no-op).
        seen_ids: Token ids already generated, shape (batch, seen_len).

    Returns:
        Modified logits (same shape).
    """
    if penalty <= 1.0:
        return logits
    if seen_ids.numel() == 0:
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
    """Sample next token with combined strategies.

    Priority: repetition penalty > temperature=0 (greedy) > top_k > top_p > temperature.

    Args:
        logits: Raw logits (batch, vocab_size).
        temperature: 0 for greedy, > 0 for sampling temperature.
        top_k: If set, apply top-k filtering.
        top_p: If set, apply top-p (nucleus) filtering.
        repetition_penalty: Penalty > 1.0 for previously generated tokens
            (Phase 31 hardening). 1.0 = disabled.
        seen_ids: Token ids generated so far (batch, seen_len); required
            when repetition_penalty > 1.0.

    Returns:
        Token IDs (batch, 1).
    """
    if repetition_penalty > 1.0:
        if seen_ids is None:
            raise ValueError("seen_ids required when repetition_penalty > 1.0")
        logits = apply_repetition_penalty(logits, repetition_penalty, seen_ids)

    # Greedy when temperature is 0
    if temperature == 0:
        return sample_greedy(logits)

    # Apply top-k first (more restrictive), then top-p
    if top_k is not None and top_k > 0:
        return sample_top_k(logits, top_k=top_k, temperature=temperature)

    if top_p is not None and top_p > 0:
        return sample_top_p(logits, top_p=top_p, temperature=temperature)

    # Pure temperature sampling
    return sample_temperature(logits, temperature=temperature)
