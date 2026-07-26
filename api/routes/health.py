"""Health check endpoints."""

import torch
from fastapi import APIRouter

from api.main import _model, _model_loaded
from api.schemas import HealthResponse, ModelInfo
from xrfm import __version__

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if _model_loaded else "loading",
        version=__version__,
        model_loaded=_model_loaded,
        gpu_available=torch.cuda.is_available(),
    )


@router.get("/health/ready")
async def readiness():
    if not _model_loaded:
        return {"status": "not ready", "detail": "Model not loaded"}
    return {"status": "ready"}


@router.get("/v1/models", response_model=list[ModelInfo])
async def list_models():
    if not _model_loaded or _model is None:
        return []
    return [
        ModelInfo(
            id="xrfm-default",
            vocab_size=_model.embedding.vocab_size,
            max_seq_len=_model.max_seq_len,
            parameter_count=_model.parameter_count(),
        )
    ]
