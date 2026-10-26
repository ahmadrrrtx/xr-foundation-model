"""Synchronous and genuinely incremental text completion endpoints."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import torch
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.schemas import CompletionChoice, CompletionRequest, CompletionResponse, CompletionUsage
from api.state import APIState, require_ready

router = APIRouter()


def _usage(result: Any) -> CompletionUsage:
    return CompletionUsage(
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.prompt_tokens + result.completion_tokens,
    )


def _stream_safe_text(text: str, stop_sequences: list[str] | None, terminal: bool = False) -> str:
    """Exclude complete stops and hold back suffixes that may become one."""
    if not stop_sequences:
        return text
    matches = [text.find(value) for value in stop_sequences if value in text]
    if matches:
        return text[: min(matches)]
    if terminal:
        return text
    holdback = 0
    for stop in stop_sequences:
        for length in range(1, min(len(stop) - 1, len(text)) + 1):
            if stop.startswith(text[-length:]):
                holdback = max(holdback, length)
    return text[:-holdback] if holdback else text


def _complete(req: CompletionRequest, state: APIState) -> CompletionResponse:
    input_ids = torch.tensor([state.tokenizer.encode(req.prompt)], dtype=torch.long)
    with state.generation_lock:
        result = state.engine.generate_with_metadata(
            input_ids,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            stop_token_id=state.tokenizer.eos_id,
            stop_sequences=req.stop,
            decode_fn=state.tokenizer.decode,
        )
    completion_ids = result.completion_token_ids.tolist()
    # EOS is an accounting token, not user-visible text. Stop strings are
    # likewise excluded from the returned text but retain their token usage.
    if completion_ids and state.tokenizer.eos_id is not None and completion_ids[-1] == state.tokenizer.eos_id:
        completion_ids = completion_ids[:-1]
    text = state.tokenizer.decode(completion_ids)
    if result.stop_sequence:
        text = text.split(result.stop_sequence, 1)[0]
    return CompletionResponse(
        id=f"xrfm-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=state.model_id,
        choices=[CompletionChoice(text=text, finish_reason=result.finish_reason)],
        usage=_usage(result),
    )


@router.post("/v1/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest, request: Request) -> CompletionResponse:
    state = require_ready(request)
    if req.stream:
        raise HTTPException(status_code=409, detail="stream=true requires /v1/completions/stream")
    try:
        return await asyncio.to_thread(_complete, req, state)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Generation failed: {type(exc).__name__}") from exc


@router.post("/v1/completions/stream")
async def completions_stream(req: CompletionRequest, request: Request) -> StreamingResponse:
    state = require_ready(request)
    try:
        input_ids = torch.tensor([state.tokenizer.encode(req.prompt)], dtype=torch.long)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Tokenization failed: {exc}") from exc

    async def generate_stream() -> AsyncIterator[str]:
        emitted_text = ""
        completion_ids: list[int] = []
        with state.generation_lock:
            events = state.engine.stream(
                input_ids,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                stop_token_id=state.tokenizer.eos_id,
                stop_sequences=req.stop,
                decode_fn=state.tokenizer.decode,
            )
            try:
                for event in events:
                    if await request.is_disconnected():
                        events.close()
                        return
                    if event.token_id is not None:
                        completion_ids.append(event.token_id)
                        if event.token_id == state.tokenizer.eos_id:
                            continue
                        decoded = state.tokenizer.decode(completion_ids)
                        visible = _stream_safe_text(decoded, req.stop)
                        if visible.startswith(emitted_text):
                            delta = visible[len(emitted_text) :]
                            emitted_text = visible
                            if delta:
                                yield f"data: {json.dumps({'token': {'text': delta}}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0)
                    elif event.result is not None:
                        result = event.result
                        final_ids = completion_ids
                        if final_ids and final_ids[-1] == state.tokenizer.eos_id:
                            final_ids = final_ids[:-1]
                        final_text = _stream_safe_text(state.tokenizer.decode(final_ids), req.stop, terminal=True)
                        if final_text.startswith(emitted_text):
                            delta = final_text[len(emitted_text) :]
                            emitted_text = final_text
                            if delta:
                                yield f"data: {json.dumps({'token': {'text': delta}}, ensure_ascii=False)}\n\n"
                        payload = {
                            "done": True,
                            "finish_reason": result.finish_reason,
                            "usage": _usage(result).model_dump(),
                        }
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except ValueError as exc:
                yield f"event: error\ndata: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
