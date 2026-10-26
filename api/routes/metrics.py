"""Metrics derived from the current application lifecycle state."""

import time

import torch
from fastapi import APIRouter, Request

from api.state import get_state
from xrfm import __version__

router = APIRouter()


@router.get("/metrics")
async def metrics(request: Request) -> dict[str, object]:
    state = get_state(request)
    return {
        "uptime_seconds": round(time.time() - state.started_at, 1),
        "model_loaded": state.model_loaded,
        "parameters": state.model.parameter_count() if state.model is not None else 0,
        "gpu_available": torch.cuda.is_available(),
        "version": __version__,
    }
