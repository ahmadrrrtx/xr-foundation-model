"""Tokenize/detokenize endpoints backed by the ready lifespan tokenizer."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.schemas import DetokenizeRequest, DetokenizeResponse, TokenizeRequest, TokenizeResponse
from api.state import require_ready

router = APIRouter()


@router.post("/v1/tokenize", response_model=TokenizeResponse)
async def tokenize(req: TokenizeRequest, request: Request) -> TokenizeResponse:
    state = require_ready(request)
    try:
        tokens = state.tokenizer.encode(req.text)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Tokenization failed: {type(exc).__name__}") from exc
    return TokenizeResponse(tokens=tokens, count=len(tokens))


@router.post("/v1/detokenize", response_model=DetokenizeResponse)
async def detokenize(req: DetokenizeRequest, request: Request) -> DetokenizeResponse:
    state = require_ready(request)
    try:
        text = state.tokenizer.decode(req.tokens)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Detokenization failed: {type(exc).__name__}") from exc
    return DetokenizeResponse(text=text)
