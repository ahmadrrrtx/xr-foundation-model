"""Tokenize/detokenize endpoints backed by the ready lifespan tokenizer."""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.schemas import DetokenizeRequest, DetokenizeResponse, TokenizeRequest, TokenizeResponse
from api.state import require_ready

router = APIRouter()


@router.post("/v1/tokenize", response_model=TokenizeResponse)
async def tokenize(req: TokenizeRequest, request: Request) -> TokenizeResponse:
    state = require_ready(request)
    tokens = state.tokenizer.encode(req.text)
    return TokenizeResponse(tokens=tokens, count=len(tokens))


@router.post("/v1/detokenize", response_model=DetokenizeResponse)
async def detokenize(req: DetokenizeRequest, request: Request) -> DetokenizeResponse:
    state = require_ready(request)
    return DetokenizeResponse(text=state.tokenizer.decode(req.tokens))
