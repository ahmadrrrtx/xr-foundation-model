"""Metrics endpoint."""

import time

import torch
from fastapi import APIRouter

from api.main import _model, _model_loaded, _startup_time

router = APIRouter()


@router.get("/metrics")
async def metrics():
    uptime = time.time() - _startup_time
    return {
        "uptime_seconds": round(uptime, 1),
        "model_loaded": _model_loaded,
        "parameters": _model.parameter_count() if _model else 0,
        "gpu_available": torch.cuda.is_available(),
        "version": "1.0.0",
    }
