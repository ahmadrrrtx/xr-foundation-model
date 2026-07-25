"""
Perplexity evaluation for XRFM (v0.7.0).

Computes intrinsic language model quality via token-level perplexity:
    PPL = exp(average cross-entropy loss over all predicted tokens)

Uses next-token prediction with shifted targets, strided evaluation
for long sequences, and proper aggregation across batches.

Conceptual references (not copied):
- Jelinek et al. (1977) — Perplexity as exponential of cross-entropy
- Brown et al. (1992) — Class-based n-gram models
- Hoffmann et al. (2022) — Chinchilla evaluation methodology
- HuggingFace evaluate — perplexity implementation pattern

Implementation is original.
"""

import logging
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.gpt import GPTModel

logger = logging.getLogger("xrfm.evaluation")


def compute_perplexity(
    model: GPTModel,
    dataloader: DataLoader,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Compute perplexity over a dataset using next-token prediction.

    For each batch of token IDs (input_ids, target_ids), the model
    predicts next-token logits. Cross-entropy loss is accumulated
    across all tokens, then exponentiated to produce perplexity.

    Mathematical definition:
        PPL = exp( -1/N * sum_i log P(token_i | context_i) )
        PPL = exp( cross_entropy_loss )

    Lower perplexity = better model. Random baseline: PPL ≈ vocab_size.

    Args:
        model: GPTModel to evaluate.
        dataloader: DataLoader yielding (input_ids, target_ids) batches
            where target_ids are shifted by 1 for next-token prediction.
        max_batches: If set, only evaluate this many batches (for quick checks).

    Returns:
        Dict with:
            "perplexity": float — perplexity score
            "loss": float — average cross-entropy loss per token
            "total_tokens": int — number of tokens evaluated
            "total_batches": int — number of batches processed

    Raises:
        ValueError: If dataloader is empty or yields malformed batches.
        RuntimeError: If any batch produces NaN/Inf logits.
    """
    model.eval()
    total_loss: float = 0.0
    total_tokens: int = 0
    batch_count: int = 0

    with torch.no_grad():
        for batch_input_ids, batch_target_ids in dataloader:
            # Validate batch
            if batch_input_ids.dim() != 2:
                raise ValueError(
                    f"Expected 2D input batch, got {batch_input_ids.dim()}D. "
                    f"Check DataLoader configuration."
                )
            if batch_target_ids.dim() != 2 or batch_target_ids.shape != batch_input_ids.shape:
                raise ValueError(
                    f"Target shape {batch_target_ids.shape} must match "
                    f"input shape {batch_input_ids.shape}"
                )

            batch_size, seq_len = batch_input_ids.shape
            num_tokens = batch_size * seq_len

            # Forward pass
            logits, _ = model(batch_input_ids)

            # Validate logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                raise RuntimeError(
                    f"NaN or Inf in model logits at batch {batch_count}. "
                    f"Check model weights and numerical stability."
                )

            # Cross-entropy loss: logits predict target_ids at each position
            # logits shape: (batch, seq, vocab_size) → (batch*seq, vocab_size)
            # targets shape: (batch, seq) → (batch*seq,)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                batch_target_ids.view(-1),
                reduction="sum",  # Accumulate total NLL across all tokens
            )

            total_loss += loss.item()
            total_tokens += num_tokens
            batch_count += 1

            if max_batches is not None and batch_count >= max_batches:
                break

    if total_tokens == 0:
        raise ValueError(
            "No tokens evaluated. Dataloader may be empty. "
            "Check dataset and batch configuration."
        )

    # Average negative log-likelihood per token
    avg_loss = total_loss / total_tokens
    # Perplexity = exp(average cross-entropy)
    perplexity = math.exp(avg_loss)

    logger.info(
        "Perplexity evaluation: PPL=%.4f, loss=%.4f, tokens=%d, batches=%d",
        perplexity,
        avg_loss,
        total_tokens,
        batch_count,
    )

    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "total_tokens": total_tokens,
        "total_batches": batch_count,
    }


def compute_perplexity_strided(
    model: GPTModel,
    token_ids: torch.Tensor,
    stride: int = 512,
    max_seq_len: int = 1024,
) -> dict[str, float]:
    """Compute perplexity with strided sliding window over a long sequence.

    For texts longer than max_seq_len, slides a window of size max_seq_len
    with stride across the token sequence. Each window is evaluated as a
    single batch. Overlapping regions are counted once per total context.

    This matches the evaluation methodology used in Chinchilla, Llama,
    and HuggingFace evaluate — it prevents information leakage between
    windows by never predicting tokens that appeared as context in the
    same window.

    Args:
        model: GPTModel to evaluate.
        token_ids: 1D tensor of token IDs for the full text.
        stride: Step size between windows. Smaller = more overlap, slower.
        max_seq_len: Window size. Must be <= model.max_seq_len.

    Returns:
        Dict with perplexity, loss, total_tokens, total_windows.
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    if max_seq_len > model.max_seq_len:
        raise ValueError(
            f"max_seq_len ({max_seq_len}) exceeds model.max_seq_len " f"({model.max_seq_len})"
        )

    model.eval()
    total_nll: float = 0.0
    total_predicted_tokens: int = 0
    seq_length = len(token_ids)
    window_count: int = 0

    prev_end_loc = 0
    with torch.no_grad():
        for begin_loc in range(0, seq_length, stride):
            end_loc = min(begin_loc + max_seq_len, seq_length)
            window_count += 1

            # Window of token IDs
            window_ids = token_ids[begin_loc:end_loc].unsqueeze(0)  # (1, window_len)

            # Forward pass
            logits, _ = model(window_ids)

            # Shift for next-token prediction
            # logits[:, t, :] predicts token at position t+1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = window_ids[:, 1:].contiguous()

            # Only count loss for tokens beyond the previous window
            # to avoid double-counting overlapped tokens
            trg_len = shift_labels.shape[1]
            if prev_end_loc > begin_loc:
                # Overlap region: skip already-evaluated prefix
                overlap = prev_end_loc - begin_loc
                shift_logits = shift_logits[:, overlap:, :]
                shift_labels = shift_labels[:, overlap:]
                trg_len = shift_labels.shape[1]

            if trg_len <= 0:
                prev_end_loc = end_loc
                continue

            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="sum",
            )

            total_nll += loss.item()
            total_predicted_tokens += trg_len
            prev_end_loc = end_loc

    if total_predicted_tokens == 0:
        return {
            "perplexity": float("inf"),
            "loss": float("inf"),
            "total_tokens": 0,
            "total_windows": window_count,
        }

    avg_loss = total_nll / total_predicted_tokens
    perplexity = math.exp(avg_loss)

    logger.info(
        "Strided PPL: %.4f, loss=%.4f, tokens=%d, stride=%d, windows=%d",
        perplexity,
        avg_loss,
        total_predicted_tokens,
        stride,
        window_count,
    )

    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "total_tokens": total_predicted_tokens,
        "total_windows": window_count,
    }


def evaluate_checkpoint(
    model: GPTModel,
    dataloader: DataLoader,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Convenience wrapper for full checkpoint evaluation.

    Returns perplexity and additional diagnostics.

    Args:
        model: GPTModel with loaded checkpoint weights.
        dataloader: Validation/test DataLoader.
        max_batches: Optional batch limit.

    Returns:
        Dict with perplexity, loss, tokens_per_second estimate,
        total_tokens, and total_batches.
    """
    import time

    start = time.perf_counter()
    results = compute_perplexity(model, dataloader, max_batches=max_batches)
    elapsed = time.perf_counter() - start

    tokens_per_sec = results["total_tokens"] / elapsed if elapsed > 0 else 0.0

    return {
        **results,
        "eval_time_seconds": elapsed,
        "tokens_per_second": tokens_per_sec,
    }
