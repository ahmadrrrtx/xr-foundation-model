"""
Inference engine for XRFM (v0.6.0).

Provides autoregressive token generation with KV cache, temperature,
top-k, and top-p (nucleus) sampling.

Conceptual references (not copied):
- Brown et al. (2020) — GPT-3 generation
- Meta AI (2024) — Llama 3 inference design
- HuggingFace transformers — generate() API pattern

Implementation is original.
"""

import logging
from typing import Optional, List

import torch

from model.gpt import GPTModel
from inference.kv_cache import KVCache
from inference.sampling import sample_token

logger = logging.getLogger("xrfm.inference")


class GenerationEngine:
    """Autoregressive text generation with KV cache.

    Wraps a GPTModel instance and handles the full generation loop:
    prompt encoding → KV-cached forward passes → token sampling →
    sequence assembly.

    Attributes:
        model: The GPTModel used for generation.
        max_seq_len: Maximum total sequence length (prompt + generated).
    """

    def __init__(self, model: GPTModel) -> None:
        if not isinstance(model, GPTModel):
            raise TypeError(
                f"model must be GPTModel, got {type(model).__name__}"
            )
        self.model = model
        self.max_seq_len = model.max_seq_len

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively with KV cache.

        The prompt is processed in a single forward pass (caching K/V).
        Subsequent tokens are generated one at a time, each requiring
        only a single-token forward pass with cache reuse.

        Args:
            input_ids: Prompt token IDs (seq,) or (1, seq).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: 0 = greedy; > 0 = sampling temperature.
            top_k: If set, restrict to top-k tokens.
            top_p: If set, use nucleus (top-p) sampling.
            stop_token_id: Optional token ID to stop generation.

        Returns:
            Full token sequence including prompt: (total_seq_len,).

        Raises:
            ValueError: If max_new_tokens <= 0 or temperature < 0.
        """
        if max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be > 0, got {max_new_tokens}"
            )
        if temperature < 0:
            raise ValueError(
                f"temperature must be >= 0, got {temperature}"
            )

        self.model.eval()

        # Ensure batched: (seq,) -> (1, seq)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size, prompt_len = input_ids.shape
        if batch_size != 1:
            raise ValueError(
                f"generate() expects batch_size=1, got {batch_size}"
            )

        # Truncate prompt if exceeds max_seq_len
        if prompt_len > self.max_seq_len:
            logger.warning(
                "Prompt length (%d) exceeds max_seq_len (%d). Truncating.",
                prompt_len, self.max_seq_len,
            )
            input_ids = input_ids[:, -self.max_seq_len:]
            prompt_len = input_ids.shape[1]

        generated = input_ids.clone()
        past_key_values: Optional[List] = None

        # Process prompt: full forward pass with caching
        logits, past_key_values = self.model(
            input_ids, use_cache=True, past_key_values=None
        )
        # logits: (1, prompt_len, vocab_size)
        next_token_logits = logits[:, -1, :]

        # Sample first token
        next_token = sample_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated = torch.cat([generated, next_token], dim=1)

        # Autoregressive loop: one token at a time with KV cache
        for step in range(1, max_new_tokens):
            # Stop if we hit the stop token
            if stop_token_id is not None and next_token.item() == stop_token_id:
                break

            # Stop if we exceed max_seq_len
            if generated.shape[1] >= self.max_seq_len:
                logger.warning(
                    "Reached max_seq_len (%d). Stopping generation.",
                    self.max_seq_len,
                )
                break

            # Single-token forward with cache
            current_input = next_token  # (1, 1)
            logits, past_key_values = self.model(
                current_input,
                use_cache=True,
                past_key_values=past_key_values,
            )
            # logits: (1, 1, vocab_size)
            next_token_logits = logits[:, -1, :]

            # Sample next token
            next_token = sample_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated = torch.cat([generated, next_token], dim=1)

        return generated.squeeze(0)

    def generate_batch(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate for a batch of prompts (without KV cache, for simplicity).

        Processes all tokens in a single forward pass per step.
        This is less efficient than per-sequence caching but works for
        batch evaluation scenarios.

        Args:
            input_ids: Prompt token IDs (batch, seq).
            max_new_tokens: Max tokens to generate per sequence.
            temperature: Sampling temperature (0 = greedy).
            top_k: Optional top-k filter.
            top_p: Optional top-p filter.

        Returns:
            Full sequences: (batch, seq + max_new_tokens).
        """
        if max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be > 0, got {max_new_tokens}"
            )

        self.model.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            seq_input = generated[:, -self.max_seq_len:]

            logits, _ = self.model(seq_input, use_cache=False)
            next_token_logits = logits[:, -1, :]

            next_token = sample_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated = torch.cat([generated, next_token], dim=1)

        return generated
