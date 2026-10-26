"""KV-cached autoregressive generation with explicit accounting and streaming."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import torch

from xrfm.inference.sampling import sample_token
from xrfm.models.gpt import GPTModel

logger = logging.getLogger("xrfm.inference")


@dataclass
class GenerationMetadata:
    """Complete token/accounting metadata for one engine generation attempt."""

    token_ids: torch.Tensor
    prompt_tokens: int
    effective_prompt_tokens: int
    completion_tokens: int
    truncated_prompt_tokens: int
    finish_reason: str
    stop_sequence: str | None = None

    @property
    def completion_token_ids(self) -> torch.Tensor:
        return self.token_ids[self.effective_prompt_tokens :]


@dataclass
class GenerationEvent:
    """One token event or the terminal metadata event from :meth:`stream`."""

    token_id: int | None = None
    result: GenerationMetadata | None = None


class GenerationEngine:
    """Autoregressive text generation with a single shared event pipeline."""

    def __init__(self, model: GPTModel, compile_model: bool = False) -> None:
        if not isinstance(model, GPTModel):
            raise TypeError(f"model must be GPTModel, got {type(model).__name__}")
        self.model = model
        self.max_seq_len = model.max_seq_len
        self.compiled = False
        self.last_result: GenerationMetadata | None = None
        if compile_model and hasattr(torch, "compile"):
            try:
                compiled = cast(GPTModel, torch.compile(model, mode="reduce-overhead"))
                self.model = compiled
                self.compiled = True
                logger.info("Compiled model with torch.compile(mode='reduce-overhead')")
            except Exception as exc:  # noqa: BLE001
                logger.warning("torch.compile failed: %s. Falling back to eager model.", exc)

    @torch.no_grad()
    def stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_token_id: int | None = None,
        repetition_penalty: float = 1.0,
        stop_sequences: list[str] | None = None,
        decode_fn: Callable[[list[int]], str] | None = None,
    ) -> Iterator[GenerationEvent]:
        """Yield each sampled token, then one terminal accounting event.

        Prompt truncation retains room for the first generated token whenever
        the context has at least two positions. Stop strings are checked only
        in generated text, never in the prompt. Generated stop tokens/strings
        count toward usage; the API may hide stop-string text from its response.
        """
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {repetition_penalty}")
        if stop_sequences and decode_fn is None:
            raise ValueError("decode_fn is required when stop_sequences is set")
        if stop_sequences and any(not isinstance(value, str) or not value for value in stop_sequences):
            raise ValueError("stop sequences must be non-empty strings")

        self.model.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError("stream() expects input_ids with shape (seq,) or (1, seq)")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one prompt token")

        original_prompt_len = int(input_ids.shape[1])
        # Keep one context slot for generation. Otherwise a long prompt would
        # make the old API report a completion while slicing away every token.
        prompt_limit = max(1, self.max_seq_len - 1)
        if original_prompt_len > prompt_limit:
            logger.warning(
                "Prompt length %d exceeds generation prompt budget %d; truncating", original_prompt_len, prompt_limit
            )
            input_ids = input_ids[:, -prompt_limit:]
        effective_prompt_len = int(input_ids.shape[1])
        generated = input_ids.clone()
        completion_count = 0
        finish_reason = "length"
        matched_stop: str | None = None
        past_key_values: list | None = None
        next_token: torch.Tensor | None = None

        def result() -> GenerationMetadata:
            return GenerationMetadata(
                token_ids=generated.squeeze(0),
                prompt_tokens=original_prompt_len,
                effective_prompt_tokens=effective_prompt_len,
                completion_tokens=completion_count,
                truncated_prompt_tokens=original_prompt_len - effective_prompt_len,
                finish_reason=finish_reason,
                stop_sequence=matched_stop,
            )

        if effective_prompt_len >= self.max_seq_len:
            final = result()
            self.last_result = final
            yield GenerationEvent(result=final)
            return

        logits, past_key_values = self.model(input_ids, use_cache=True, past_key_values=None)
        for step in range(max_new_tokens):
            if step > 0:
                if generated.shape[1] >= self.max_seq_len:
                    finish_reason = "length"
                    break
                assert next_token is not None
                logits, past_key_values = self.model(
                    next_token,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            next_token = sample_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seen_ids=generated,
            )
            generated = torch.cat([generated, next_token], dim=1)
            completion_count += 1
            yield GenerationEvent(token_id=int(next_token.item()))

            if stop_token_id is not None and int(next_token.item()) == stop_token_id:
                finish_reason = "stop"
                break
            if stop_sequences and decode_fn is not None:
                decoded = decode_fn(generated[0, effective_prompt_len:].tolist())
                matched_stop = next((value for value in stop_sequences if value in decoded), None)
                if matched_stop is not None:
                    finish_reason = "stop"
                    break

        final = result()
        self.last_result = final
        yield GenerationEvent(result=final)

    def generate_with_metadata(self, input_ids: torch.Tensor, **kwargs: Any) -> GenerationMetadata:
        """Run generation and return its terminal accounting metadata."""
        final: GenerationMetadata | None = None
        for event in self.stream(input_ids, **kwargs):
            if event.result is not None:
                final = event.result
        if final is None:  # pragma: no cover - defensive generator contract
            raise RuntimeError("generation ended without a terminal result")
        return final

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_token_id: int | None = None,
        repetition_penalty: float = 1.0,
        stop_sequences: list[str] | None = None,
        decode_fn: Callable[[list[int]], str] | None = None,
    ) -> torch.Tensor:
        """Return the full effective prompt plus generated token sequence."""
        return self.generate_with_metadata(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_token_id=stop_token_id,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences,
            decode_fn=decode_fn,
        ).token_ids

    def generate_batch(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Generate for a batch without KV caching (evaluation helper)."""
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")
        self.model.eval()
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            seq_input = generated[:, -self.max_seq_len :]
            logits, _ = self.model(seq_input, use_cache=False)
            next_token = sample_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated = torch.cat([generated, next_token], dim=1)
        return generated
