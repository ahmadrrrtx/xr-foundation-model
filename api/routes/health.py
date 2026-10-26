"""Liveness, readiness, and model discovery endpoints."""

from __future__ import annotations

import torch
from fastapi import APIRouter, HTTPException, Request

from api.schemas import HealthResponse, ModelInfo
from api.state import get_state, require_ready
from xrfm import __version__

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    state = get_state(request)
    return HealthResponse(
        status="healthy",
        version=__version__,
        model_loaded=state.model_loaded,
        gpu_available=torch.cuda.is_available(),
    )


@router.get("/health/ready")
async def readiness(request: Request) -> dict[str, str]:
    state = get_state(request)
    if not state.model_loaded or state.model is None or state.engine is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail=state.startup_error or "Model/tokenizer/engine are not ready")
    return {"status": "ready"}


@router.get("/v1/models", response_model=list[ModelInfo])
async def list_models(request: Request) -> list[ModelInfo]:
    state = require_ready(request)
    return [
        ModelInfo(
            id=state.model_id,
            vocab_size=state.model.embedding.vocab_size,
            max_seq_len=state.model.max_seq_len,
            parameter_count=state.model.parameter_count(),
        )
    ]
