"""
Functional text-generation API for XRFM.

This is the stable boundary between *model code*, *generation logic*, and
*product layers* (API server / web UI call this; they never reimplement
sampling):

    from xrfm import XRFMModel, generate
    from xrfm.tokenization import BPETokenizer

    model = XRFMModel("config/tiny.yaml")
    tok = BPETokenizer.pretrained()
    print(generate(model, "Once upon a time", tokenizer=tok, max_new_tokens=64))

Token-level control (KV cache reuse across calls, stop sequences, batch
handling) remains available via :class:`xrfm.inference.engine.GenerationEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from xrfm.inference.engine import GenerationEngine
from xrfm.models.gpt import XRFMModel
from xrfm.tokenization.interface import Tokenizer

__all__ = ["GenerationResult", "generate"]


@dataclass
class GenerationResult:
    """Outcome of a :func:`generate` call.

    Attributes:
        text: Decoded continuation only (prompt excluded) when the prompt
            was text; empty string for token prompts unless ``decode`` given.
        token_ids: Generated token ids (continuation only).
        prompt_token_ids: Tokenized prompt (as used by the model).
    """

    text: str
    token_ids: list[int]
    prompt_token_ids: list[int]

    def __iter__(self):  # allow tuple-style unpacking: text, ids = generate(...)
        return iter((self.text, self.token_ids))


def generate(
    model: XRFMModel,
    prompt: str | torch.Tensor | list[int],
    tokenizer: Tokenizer | None = None,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    repetition_penalty: float = 1.0,
    decode: bool = True,
) -> GenerationResult:
    """Generate a continuation for ``prompt`` (text in → text out).

    Args:
        model: Trained :class:`xrfm.models.XRFMModel`.
        prompt: Text string (requires ``tokenizer``), a token-id tensor
            ``(seq,)`` / ``(1, seq)``, or a list of token ids.
        tokenizer: Required when ``prompt`` is a ``str``; also used to
            decode the output.
        max_new_tokens: Upper bound on generated tokens (> 0).
        temperature: Sampling temperature; ``0`` = greedy.
        top_k: Optional top-k filtering.
        top_p: Optional nucleus (top-p) filtering.
        eos_token_id: Stop when this id is sampled (defaults to the
            tokenizer contract ``eos_token_id`` when available).
        repetition_penalty: >= 1.0; penalizes repeated tokens.
        decode: If False, skip text decoding (``result.text`` will be
            ``""`` unless a tokenizer is available).

    Returns:
        :class:`GenerationResult` with the continuation text and token ids.

    Raises:
        ValueError: invalid arguments (bad temperature, text prompt without
            tokenizer, etc.).
    """
    engine = GenerationEngine(model)

    prompt_ids: torch.Tensor
    if isinstance(prompt, str):
        if tokenizer is None:
            raise ValueError("a tokenizer is required when prompt is a string")
        prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    elif isinstance(prompt, torch.Tensor):
        prompt_ids = prompt if prompt.dim() == 2 else prompt.unsqueeze(0)
    elif isinstance(prompt, list):
        prompt_ids = torch.tensor([prompt], dtype=torch.long)
    else:
        raise TypeError(f"prompt must be str, torch.Tensor, or list[int]; got {type(prompt).__name__}")

    if eos_token_id is None and tokenizer is not None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    out_ids = engine.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_token_id=eos_token_id,
        repetition_penalty=repetition_penalty,
    )

    prompt_len = prompt_ids.shape[1]
    new_ids = out_ids[prompt_len:].tolist()

    text = ""
    if decode:
        if tokenizer is not None:
            text = tokenizer.decode(new_ids)
        else:
            raise ValueError("decode=True requires a tokenizer; pass tokenizer=... or decode=False")

    return GenerationResult(text=text, token_ids=new_ids, prompt_token_ids=prompt_ids[0].tolist())
