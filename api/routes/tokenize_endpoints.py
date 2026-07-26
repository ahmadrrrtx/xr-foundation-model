"""Tokenize/detokenize endpoints."""
from fastapi import APIRouter, HTTPException
from api.main import _tokenizer, _model_loaded
from api.schemas import TokenizeRequest, TokenizeResponse, DetokenizeRequest, DetokenizeResponse

router = APIRouter()

@router.post("/v1/tokenize", response_model=TokenizeResponse)
async def tokenize(req: TokenizeRequest):
    if not _model_loaded or _tokenizer is None:
        raise HTTPException(503, "Model not loaded")
    tokens = _tokenizer.encode(req.text)
    return TokenizeResponse(tokens=tokens, count=len(tokens))

@router.post("/v1/detokenize", response_model=DetokenizeResponse)
async def detokenize(req: DetokenizeRequest):
    if not _model_loaded or _tokenizer is None:
        raise HTTPException(503, "Model not loaded")
    text = _tokenizer.decode(req.tokens)
    return DetokenizeResponse(text=text)
