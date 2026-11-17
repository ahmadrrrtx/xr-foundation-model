"""
Perplexity and Bits-Per-Byte (BPB) evaluation for XRFM (v0.7.0+).

Computes intrinsic language model quality via:
1. Token-level perplexity: ``PPL = exp(average cross-entropy loss over non-ignored tokens)``
2. Tokenizer-agnostic Bits-Per-Byte (``BPB``):
       ``BPB = total_nll_nats / (total_utf8_bytes * ln(2))``
   Essential for fair cross-tokenizer and multilingual evaluation across
   English, Urdu, Roman Urdu, Arabic, Hindi, Spanish, French, and code.

Conceptual references (not copied):
- Jelinek et al. (1977) — Perplexity as exponential of cross-entropy
- Hoffmann et al. (2022) — Chinchilla evaluation methodology
- Gao et al. (2020) — Pile Bits-Per-Byte (BPB) normalization

Implementation is original.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from xrfm.models.gpt import GPTModel

logger = logging.getLogger("xrfm.evaluation")


def _get_model_device(model: nn.Module) -> torch.device:
    """Return the device hosting ``model``'s parameters (defaults to CPU)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _unpack_eval_batch(
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Accept ``(input_ids, target_ids)`` or ``(input_ids, target_ids, mask)``."""
    if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
        raise ValueError("Evaluation dataloader must yield (input_ids, target_ids) or (input_ids, target_ids, mask).")
    inp, tgt = batch[0], batch[1]
    mask = batch[2] if len(batch) == 3 else None
    return inp, tgt, mask


def compute_perplexity(
    model: GPTModel,
    dataloader: DataLoader,
    max_batches: int | None = None,
    ignore_index: int = -100,
) -> dict[str, float]:
    """Compute perplexity over a dataset using next-token prediction.

    Automatically moves CPU batches to the model's device and supports both
    2-tuple ``(input_ids, target_ids)`` and 3-tuple ``(input_ids, target_ids, mask)``
    dataloaders.
    """
    model.eval()
    device = _get_model_device(model)
    total_loss: float = 0.0
    total_tokens: int = 0
    batch_count: int = 0

    with torch.no_grad():
        for raw_batch in dataloader:
            batch_input_ids, batch_target_ids, batch_mask = _unpack_eval_batch(raw_batch)

            if batch_input_ids.dim() != 2:
                raise ValueError(
                    f"Expected 2D input batch, got {batch_input_ids.dim()}D. Check DataLoader configuration."
                )
            if batch_target_ids.dim() != 2 or batch_target_ids.shape != batch_input_ids.shape:
                raise ValueError(
                    f"Target shape {batch_target_ids.shape} must match input shape {batch_input_ids.shape}"
                )

            valid = int((batch_target_ids != ignore_index).sum().item())
            if valid == 0:
                continue

            if batch_input_ids.device != device:
                batch_input_ids = batch_input_ids.to(device)
            if batch_mask is not None and batch_mask.device != device:
                batch_mask = batch_mask.to(device)

            logits, _ = model(batch_input_ids, mask=batch_mask)
            if batch_target_ids.device != logits.device:
                batch_target_ids = batch_target_ids.to(logits.device)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                raise RuntimeError(
                    f"NaN or Inf in model logits at batch {batch_count}. Check model weights and numerical stability."
                )

            loss = F.cross_entropy(
                logits.float().view(-1, logits.shape[-1]),
                batch_target_ids.view(-1),
                reduction="sum",
                ignore_index=ignore_index,
            )

            total_loss += float(loss.item())
            total_tokens += valid
            batch_count += 1

            if max_batches is not None and batch_count >= max_batches:
                break

    if total_tokens == 0:
        raise ValueError("No tokens evaluated. Dataloader may be empty. Check dataset and batch configuration.")

    avg_loss = total_loss / total_tokens
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
    """Compute perplexity with strided sliding window over a long sequence."""
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    if max_seq_len > model.max_seq_len:
        raise ValueError(f"max_seq_len ({max_seq_len}) exceeds model.max_seq_len ({model.max_seq_len})")

    model.eval()
    device = _get_model_device(model)
    total_nll: float = 0.0
    total_predicted_tokens: int = 0
    seq_length = len(token_ids)
    window_count: int = 0

    prev_end_loc = 0
    with torch.no_grad():
        for begin_loc in range(0, seq_length, stride):
            end_loc = min(begin_loc + max_seq_len, seq_length)
            window_count += 1

            window_ids = token_ids[begin_loc:end_loc].unsqueeze(0)
            if window_ids.device != device:
                window_ids = window_ids.to(device)

            logits, _ = model(window_ids)
            labels_src = (
                window_ids
                if window_ids.device == logits.device
                else token_ids[begin_loc:end_loc].unsqueeze(0).to(logits.device)
            )

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels_src[:, 1:].contiguous()

            trg_len = shift_labels.shape[1]
            if prev_end_loc > begin_loc:
                overlap = prev_end_loc - begin_loc
                shift_logits = shift_logits[:, overlap:, :]
                shift_labels = shift_labels[:, overlap:]
                trg_len = shift_labels.shape[1]

            if trg_len <= 0:
                prev_end_loc = end_loc
                continue

            loss = F.cross_entropy(
                shift_logits.float().reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="sum",
            )

            total_nll += float(loss.item())
            total_predicted_tokens += trg_len
            prev_end_loc = end_loc
            if end_loc == seq_length:
                break

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


def compute_bits_per_byte(
    model: GPTModel,
    texts: Iterable[str],
    tokenizer: Any,
    stride: int | None = None,
    max_seq_len: int | None = None,
) -> dict[str, float]:
    """Compute tokenizer-independent Bits-Per-Byte (BPB) over raw UTF-8 texts.

    Formula:
        ``BPB = sum_nll_nats / (total_utf8_bytes * ln(2))``

    When ``bos_token_id`` is available on ``tokenizer``, it is prepended as
    conditioning context so all ``K`` content tokens of each string are scored
    fairly against ``len(text.encode('utf-8'))`` bytes. Otherwise ``K - 1``
    continuation tokens are scored.

    Args:
        model: ``XRFMModel`` / ``GPTModel`` to evaluate.
        texts: Non-empty iterable of evaluation strings (any Unicode script).
        tokenizer: Tokenizer exposing ``encode(str) -> list[int]`` and optional
            ``bos_token_id`` / ``bos_id``.
        stride: Sliding window stride (defaults to ``max_seq_len // 2``).
        max_seq_len: Sliding window width (defaults to ``model.max_seq_len``).

    Returns:
        Dict with ``bpb``, ``total_nll_nats``, ``avg_token_loss``,
        ``total_bytes``, ``total_tokens``, ``bytes_per_token``.
    """
    resolved_max_len = model.max_seq_len if max_seq_len is None else max_seq_len
    if resolved_max_len <= 1 or resolved_max_len > model.max_seq_len:
        raise ValueError(f"max_seq_len ({resolved_max_len}) must be in [2, model.max_seq_len={model.max_seq_len}]")
    resolved_stride = max(1, resolved_max_len // 2) if stride is None else stride
    if resolved_stride <= 0:
        raise ValueError(f"stride must be positive, got {resolved_stride}")

    bos_id = getattr(tokenizer, "bos_token_id", getattr(tokenizer, "bos_id", None))

    model.eval()
    device = _get_model_device(model)
    total_nll_nats: float = 0.0
    total_bytes: int = 0
    total_scored_tokens: int = 0

    with torch.no_grad():
        for text in texts:
            if not text:
                continue
            utf8_bytes = len(text.encode("utf-8"))
            ids = list(tokenizer.encode(text))
            if not ids:
                continue
            if bos_id is not None and 0 <= int(bos_id) < model.embedding.vocab_size:
                seq = [int(bos_id)] + ids
            elif len(ids) >= 2:
                seq = ids
            else:
                continue

            token_tensor = torch.tensor(seq, dtype=torch.long)
            seq_length = len(token_tensor)
            prev_end_loc = 0
            for begin_loc in range(0, seq_length, resolved_stride):
                end_loc = min(begin_loc + resolved_max_len, seq_length)
                window_ids = token_tensor[begin_loc:end_loc].unsqueeze(0)
                if window_ids.device != device:
                    window_ids = window_ids.to(device)
                logits, _ = model(window_ids)
                labels_src = (
                    window_ids
                    if window_ids.device == logits.device
                    else token_tensor[begin_loc:end_loc].unsqueeze(0).to(logits.device)
                )

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels_src[:, 1:].contiguous()
                if prev_end_loc > begin_loc:
                    overlap = prev_end_loc - begin_loc
                    shift_logits = shift_logits[:, overlap:, :]
                    shift_labels = shift_labels[:, overlap:]

                trg_len = int(shift_labels.shape[1])
                if trg_len > 0:
                    loss = F.cross_entropy(
                        shift_logits.float().reshape(-1, shift_logits.shape[-1]),
                        shift_labels.reshape(-1),
                        reduction="sum",
                    )
                    total_nll_nats += float(loss.item())
                    total_scored_tokens += trg_len

                prev_end_loc = end_loc
                if end_loc == seq_length:
                    break

            total_bytes += utf8_bytes

    if total_bytes == 0 or total_scored_tokens == 0:
        raise ValueError("No non-empty texts with scorable tokens provided to compute_bits_per_byte.")

    bpb = total_nll_nats / (total_bytes * math.log(2.0))
    avg_token_loss = total_nll_nats / total_scored_tokens
    return {
        "bpb": bpb,
        "total_nll_nats": total_nll_nats,
        "avg_token_loss": avg_token_loss,
        "total_bytes": total_bytes,
        "total_tokens": total_scored_tokens,
        "bytes_per_token": total_bytes / total_scored_tokens,
    }


def evaluate_checkpoint(
    model: GPTModel,
    dataloader: DataLoader,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Convenience wrapper for full checkpoint evaluation."""
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
