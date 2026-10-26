"""XRFM FastAPI application with an explicit, checkpoint-backed lifespan."""

from __future__ import annotations

import glob
import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.agent_config import validate_agent_configuration
from api.protection import ProtectionConfig, RequestProtectionMiddleware, RequestProtector
from api.state import APIState
from xrfm import __version__
from xrfm.agent.control import AgentControl
from xrfm.config.schema import ModelConfig
from xrfm.inference.engine import GenerationEngine
from xrfm.models.gpt import GPTModel
from xrfm.tokenization import BPETokenizer

# Compatibility snapshots for older integrations. Routes never read these;
# app.state.xrfm is the sole lifecycle authority.
_model: GPTModel | None = None
_engine: GenerationEngine | None = None
_tokenizer: Any = None
_startup_time = 0.0
_model_loaded = False

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":"%(message)s"}',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("xrfm.api")


def _load_tokenizer() -> Any:
    configured = os.getenv("XRFM_TOKENIZER_PATH")
    vocab_path = configured or os.path.join("tokenizer", "vocab.json")
    if os.path.exists(vocab_path):
        tokenizer = BPETokenizer()
        tokenizer.load(vocab_path)
        logger.info("Loaded tokenizer from %s (vocab=%d)", vocab_path, tokenizer.vocab_size())
        return tokenizer
    if configured:
        raise FileNotFoundError(f"configured tokenizer does not exist: {configured}")
    tokenizer = BPETokenizer.pretrained()
    logger.info("Loaded packaged tokenizer (vocab=%d)", tokenizer.vocab_size())
    return tokenizer


def _checkpoint_candidates() -> list[str]:
    configured = os.getenv("XRFM_CHECKPOINT_PATH")
    if configured:
        return [configured]

    def sort_key(path: str) -> tuple[int, float]:
        match = re.search(r"checkpoint_step_(\d+)\.pt$", path)
        return (int(match.group(1)) if match else -1, os.path.getmtime(path))

    return sorted(glob.glob(os.path.join("checkpoints", "checkpoint_step_*.pt")), key=sort_key, reverse=True)


def _load_checkpoint(tokenizer: Any, checkpoint_path: str) -> tuple[GPTModel, GenerationEngine]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    extra = payload.get("extra", {}) or {}
    model_data = extra.get("model_config")
    if model_data is None and isinstance(extra.get("config"), dict):
        model_data = extra["config"].get("model")
    if model_data is not None:
        model_config = ModelConfig.from_dict(dict(model_data))
        if model_config.vocab_size != tokenizer.vocab_size():
            raise ValueError(
                f"checkpoint vocab_size={model_config.vocab_size} != tokenizer vocab_size={tokenizer.vocab_size()}"
            )
        model = GPTModel(model_config)
    else:
        # Older checkpoints have no architecture metadata; strict state-dict
        # loading below still prevents silently serving an incompatible model.
        model = GPTModel(vocab_size=tokenizer.vocab_size())
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device_name = os.getenv("XRFM_API_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model.to(torch.device(device_name))
    model.eval()
    return model, GenerationEngine(model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Own one complete model/tokenizer/engine lifecycle per application."""
    global _model, _engine, _tokenizer, _startup_time, _model_loaded
    state = APIState(started_at=time.time())
    state.agent_control = AgentControl()
    state.protection = RequestProtector(ProtectionConfig.from_env())
    app.state.xrfm = state
    _model = None
    _engine = None
    _tokenizer = None
    _model_loaded = False
    _startup_time = state.started_at
    # Configuration errors are startup failures, not a public demo fallback.
    validate_agent_configuration()

    try:
        state.tokenizer = _load_tokenizer()
        candidates = _checkpoint_candidates()
        if not candidates:
            raise FileNotFoundError("no compatible checkpoint configured (set XRFM_CHECKPOINT_PATH)")
        errors: list[str] = []
        for candidate in candidates:
            try:
                state.model, state.engine = _load_checkpoint(state.tokenizer, candidate)
                state.checkpoint_path = candidate
                state.model_loaded = True
                break
            except Exception as exc:  # noqa: BLE001 - try the next local candidate
                errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
                logger.warning("Checkpoint rejected: %s", errors[-1])
                if os.getenv("XRFM_CHECKPOINT_PATH"):
                    break
        if not state.model_loaded:
            raise RuntimeError("; ".join(errors) or "no usable checkpoint")
        _model, _engine, _tokenizer = state.model, state.engine, state.tokenizer
        _model_loaded = True
        logger.info(
            "Model ready in %.2fs (params=%d, checkpoint=%s)",
            time.time() - state.started_at,
            state.model.parameter_count(),
            state.checkpoint_path,
        )
    except Exception as exc:  # readiness, not liveness, reports model failure
        state.startup_error = str(exc)
        state.model_loaded = False
        state.model = None
        state.engine = None
        logger.error("XRFM model is not ready: %s", exc)

    try:
        yield
    finally:
        if state.agent_control is not None:
            state.agent_control.close()
            state.agent_control = None
        state.protection = None
        state.model = None
        state.engine = None
        state.tokenizer = None
        state.model_loaded = False
        _model = None
        _engine = None
        _tokenizer = None
        _model_loaded = False
        logger.info("XRFM API lifecycle closed")


app = FastAPI(title="XR Foundation Model API", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-XRFM-Agent-Key"],
)
app.add_middleware(RequestProtectionMiddleware)


@app.middleware("http")
async def security_headers(request: Request, call_next: Any):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


@app.middleware("http")
async def request_timing(request: Request, call_next: Any):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time-Ms"] = str(round(elapsed * 1000, 2))
    logger.info("%s %s %d %.2fms", request.method, request.url.path, response.status_code, elapsed * 1000)
    return response


from api.routes import agent, completions, health, metrics, search, tokenize_endpoints  # noqa: E402

app.include_router(health.router, tags=["Health"])
app.include_router(completions.router, tags=["Completions"])
app.include_router(tokenize_endpoints.router, tags=["Tokenize"])
app.include_router(metrics.router, tags=["Metrics"])
app.include_router(search.router, tags=["Search Engine"])
app.include_router(agent.router, tags=["Agent Runtime"])

try:
    app.mount("/ui", StaticFiles(directory="webui", html=True), name="webui")
except Exception:  # pragma: no cover - optional repository UI
    pass
