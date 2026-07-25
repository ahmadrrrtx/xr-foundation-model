"""
Embedding layer for XR Foundation Model (XRFM).

Purpose: Map integer token IDs (from tokenizer) to dense continuous vectors.
This is the first layer of the decoder-only GPT architecture.

Conceptual references (NOT copied):
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need.
  (Embedding layer design concept — standard practice; implementation original.)
- Llama 3 Technical Report (Meta, 2024) — Embedding design, vocabulary scaling,
  weight tying practices (conceptual reference only; no code copied).
- DeepSeek-V3 Technical Report (DeepSeek-AI, 2024) — Vocabulary and embedding design.

Implementation is original. The embedding layer uses Xavier (Glorot) initialization
and supports weight tying with the output projection layer (`lm_head`).

Design principles (from Phase 4 architecture freeze):
- Config-driven (`vocab_size`, `d_model` from `ConfigLoader`).
- Original code (no copied implementations from PyTorch native modules or other repositories).
- Stable interface (`nn.Module` subclass with `forward` method compatible with standard PyTorch training loops).
- Weight tying (`self.weight` shared with `lm_head` by default; configurable via model constructor).
- Numerical stability (Xavier initialization prevents gradient explosion/vanishing).
"""

import torch
import torch.nn as nn


class XRFMEmbedding(nn.Module):
    """Original embedding layer for XRFM decoder-only transformer.

    This module maps token IDs to dense vectors of dimension `d_model`.
    It supports weight tying with the output projection layer (standard
    practice in modern LLMs) and uses Xavier initialization for stability.

    Attributes:
        embedding (nn.Embedding): The core embedding matrix.
        vocab_size (int): Size of the vocabulary (from tokenizer).
        d_model (int): Dimension of embedding vectors (model hidden size).
        weight_tied (bool): Whether the embedding weight is shared with
            the output projection layer (`lm_head`). When True, the model
            constructor should assign `lm_head.weight = self.embedding.weight`.

    Design note: Weight tying reduces parameter count by `vocab_size * d_model`
    and improves training stability by ensuring the embedding and projection
    layers learn consistent representations. This is the default behavior for
    modern decoder-only models (GPT family, Llama, Mistral, DeepSeek, Qwen).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        weight_tied: bool = True,
        padding_idx: int | None = None,
    ) -> None:
        """Initialize the embedding layer.

        Args:
            vocab_size: Size of the tokenizer vocabulary (`ConfigLoader.get_model_config()["vocab_size"]`).
            d_model: Dimension of the embedding vectors (`ConfigLoader.get_model_config()["d_model"]`).
            weight_tied: Whether to design for weight tying with output projection.
                When True, the embedding weight matrix (`self.weight`) can be
                assigned to `lm_head.weight` by the model constructor (`GPTModel`).
                Default: True (standard modern practice).
            padding_idx: Optional index for padding tokens. If set, embeddings
                for this index are initialized to zero and not updated during
                training (useful for batched sequences with variable length).

        Raises:
            ValueError: If `vocab_size` or `d_model` is non-positive.
        """
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be positive, got {vocab_size}. Check tokenizer vocabulary or ConfigLoader settings."
            )
        if d_model <= 0:
            raise ValueError(
                f"d_model must be positive, got {d_model}. Check ConfigLoader model settings."
            )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight_tied = weight_tied
        self.padding_idx = padding_idx

        # Initialize embedding matrix with Xavier (Glorot) initialization.
        # Xavier scaling: std = sqrt(2 / (fan_in + fan_out)).
        # For embeddings, fan_in = vocab_size, fan_out = d_model.
        # This ensures initial embeddings have appropriate variance to prevent
        # gradient explosion or vanishing at the start of training.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        # Apply Xavier initialization to the embedding weight matrix.
        # Note: `nn.Embedding` initializes with N(0, 1) by default; we override
        # with Xavier scaling for numerical stability.
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier (Glorot) initialization to embedding weights.

        This is a private helper method called during initialization.
        It scales weights by `sqrt(2 / (vocab_size + d_model))`, which is
        the standard Xavier uniform initialization for rectangular matrices.
        """
        # Xavier initialization formula for uniform distribution:
        # limit = sqrt(6 / (fan_in + fan_out))
        # We use the standard PyTorch Xavier uniform initialization.
        nn.init.xavier_uniform_(self.embedding.weight)
        # If padding_idx is set, reset padding embeddings to zero (standard practice
        # for padding tokens, ensuring they don't contribute to attention or loss).
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Convert token IDs to dense embedding vectors.

        Args:
            input_ids: Tensor of token IDs with shape `(batch_size, sequence_length)`
                or `(sequence_length,)` for single sequences.

        Returns:
            Embedding tensor with shape `(batch_size, sequence_length, d_model)`
            or `(sequence_length, d_model)` for single sequences.

        Raises:
            IndexError: If any token ID exceeds vocabulary size (indicating
                tokenizer/model vocabulary mismatch).
        """
        # Input validation: check for out-of-range token IDs before embedding lookup.
        # This provides early, clear error messages (rather than PyTorch's
        # generic index error) to help debug vocabulary mismatches.
        if (input_ids >= self.vocab_size).any():
            max_id = int(input_ids.max().item())
            raise IndexError(
                f"Token ID {max_id} exceeds vocabulary size ({self.vocab_size}). "
                f"This indicates a mismatch between tokenizer vocabulary and model embedding layer. "
                f"Check tokenizer training (vocab_size_target) and ConfigLoader settings (model.vocab_size)."
            )
        # Standard embedding lookup: mapping integer IDs to dense vectors.
        return self.embedding(input_ids)
