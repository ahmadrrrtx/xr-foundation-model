"""
FlashAttention integration for XRFM (v0.9.0).

Wraps PyTorch's torch.nn.functional.scaled_dot_product_attention
as a drop-in replacement for the manual matmul+softmax attention in
MultiHeadAttention. PyTorch automatically dispatches to the optimal
backend: FlashAttention-2 on CUDA, MemoryEfficientAttention, or
fallback math implementation.

Conceptual references (not copied):
- Dao et al. (2022) — FlashAttention: Fast and Memory-Efficient Exact Attention
- Dao (2023) — FlashAttention-2: Faster Attention with Better Parallelism
- PyTorch 2.0+ — torch.nn.functional.scaled_dot_product_attention

Performance: 2-4x faster than manual attention for sequences > 512 tokens.
Memory: O(seq_len) instead of O(seq_len^2) for attention matrix.

Implementation is original.
"""

import math

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def is_flash_attention_available() -> bool:
    """Check if FlashAttention backend is available on current hardware."""
    try:
        return torch.backends.cuda.flash_sdp_enabled()
    except (AttributeError, RuntimeError):
        return torch.cuda.is_available()


def is_mem_efficient_available() -> bool:
    """Check if MemoryEfficientAttention backend is available."""
    try:
        return torch.backends.cuda.mem_efficient_sdp_enabled()
    except (AttributeError, RuntimeError):
        return False


def get_available_backend() -> str:
    """Return the best available SDP backend.

    Priority: flash > mem_efficient > math (always available).
    """
    if is_flash_attention_available():
        return "flash"
    if is_mem_efficient_available():
        return "mem_efficient"
    return "math"


# ---------------------------------------------------------------------------
# Core attention functions
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: float | None = None,
    force_backend: str | None = None,
) -> torch.Tensor:
    """Compute attention using the fastest available backend.

    Drop-in replacement for manual Q@K^T + softmax + @V.

    When force_backend is None, PyTorch auto-selects the optimal
    kernel based on input shapes, dtypes, and hardware.

    Args:
        query: (batch, n_heads, seq_q, d_head)
        key: (batch, n_heads, seq_kv, d_head)
        value: (batch, n_heads, seq_kv, d_head)
        attn_mask: Optional attention mask (additive, broadcastable).
        dropout_p: Dropout probability (only active during training).
        is_causal: If True, applies causal mask (lower triangular).
        scale: Scaling factor (default: 1/sqrt(d_head)).
        force_backend: "flash", "mem_efficient", "math", or None (auto).

    Returns:
        Attention output (batch, n_heads, seq_q, d_head).
    """
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    if force_backend is not None:
        enable_flash = force_backend == "flash"
        enable_mem = force_backend == "mem_efficient"
        enable_math = force_backend == "math"
        with torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash,
            enable_mem_efficient=enable_mem,
            enable_math=enable_math,
        ):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal and attn_mask is None,
                scale=scale,
            )
    else:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal and attn_mask is None,
            scale=scale,
        )


# ---------------------------------------------------------------------------
# Attention wrapper — drop-in for MultiHeadAttention internals
# ---------------------------------------------------------------------------


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    force_backend: str | None = None,
) -> torch.Tensor:
    """Optimized attention forward pass.

    Drop this in place of the manual matmul+softmax+matmul sequence
    in MultiHeadAttention.forward().

    Args:
        q: Query (batch, n_heads, seq, d_head) — already RoPE'd.
        k: Key (batch, n_heads, seq_kv, d_head).
        v: Value (batch, n_heads, seq_kv, d_head).
        mask: Optional attention mask.
        dropout_p: Dropout rate (0 during eval if desired).
        training: Whether in training mode (dropout active).
        force_backend: Override backend selection.

    Returns:
        Attention output (batch, n_heads, seq_q, d_head).
    """
    use_dropout = dropout_p if training else 0.0

    # If we have an explicit mask, don't pass is_causal
    # (PyTorch SDP handles causal only when no mask is given)
    if mask is not None:
        return scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=use_dropout,
            is_causal=False,
            force_backend=force_backend,
        )
    else:
        return scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=use_dropout,
            is_causal=True,
            force_backend=force_backend,
        )


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------


def benchmark_attention(
    batch_size: int = 2,
    n_heads: int = 8,
    seq_len: int = 1024,
    d_head: int = 64,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    num_warmup: int = 5,
    num_trials: int = 20,
) -> dict:
    """Benchmark FlashAttention vs manual attention.

    Returns timing and speedup for each backend.
    """
    q = torch.randn(batch_size, n_heads, seq_len, d_head, dtype=dtype, device=device)
    k = torch.randn(batch_size, n_heads, seq_len, d_head, dtype=dtype, device=device)
    v = torch.randn(batch_size, n_heads, seq_len, d_head, dtype=dtype, device=device)
    scale = 1.0 / math.sqrt(d_head)

    results = {}

    # Manual attention baseline
    def _manual():
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    # Warmup
    for _ in range(num_warmup):
        _manual()

    if device == "cuda":
        torch.cuda.synchronize()
    torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    torch.cuda.Event(enable_timing=True) if device == "cuda" else None

    import time

    t0 = time.perf_counter()
    for _ in range(num_trials):
        _manual()
    t1 = time.perf_counter()
    manual_time = (t1 - t0) / num_trials * 1000  # ms
    results["manual_ms"] = manual_time

    # FlashAttention (auto)
    for _ in range(num_warmup):
        scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)

    t0 = time.perf_counter()
    for _ in range(num_trials):
        scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)
    t1 = time.perf_counter()
    sdpa_time = (t1 - t0) / num_trials * 1000
    results["sdpa_auto_ms"] = sdpa_time
    results["speedup_vs_manual"] = manual_time / sdpa_time if sdpa_time > 0 else 0

    return results
