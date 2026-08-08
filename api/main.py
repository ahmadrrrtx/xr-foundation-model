"""XRFM FastAPI application (v1.0.0).

Production API server with: health checks, model info, text generation
(sync + streaming), tokenization, structured logging, rate limiting,
security headers, and CORS.
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

# Ensure repository root is on sys.path for Windows & cross-platform imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Model and engine (lazy loaded in lifespan)
_model = None
_engine = None
_tokenizer = None
_startup_time: float = 0.0
_model_loaded: bool = False

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":"%(message)s"}',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("xrfm.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global _model, _engine, _tokenizer, _startup_time, _model_loaded
    logger.info("Loading XRFM model...")
    _startup_time = time.time()
    _checkpoint_loaded = False

    try:
        import glob

        from inference.engine import GenerationEngine
        from model.gpt import GPTModel
        from tokenizer.bpe import BytePairEncoder

        # Tokenizer from disk if present (coherent vocab), else a fresh BPE.
        _tokenizer = BytePairEncoder()
        vocab_path = os.path.join("tokenizer", "vocab.json")
        if os.path.exists(vocab_path):
            _tokenizer.load(vocab_path)
            logger.info("Loaded tokenizer from %s (vocab=%d)", vocab_path, _tokenizer.vocab_size())

        # Model built with the tokenizer's actual vocabulary size (F-13).
        _model = GPTModel(vocab_size=_tokenizer.vocab_size())

        # Load the most recent checkpoint if present (F-42: previously the API
        # served a freshly-random model and never touched the committed ckpt).
        ckpts = sorted(glob.glob(os.path.join("checkpoints", "checkpoint_step_*.pt")))
        if ckpts:
            latest = ckpts[-1]
            try:
                import torch

                sd = torch.load(latest, map_location="cpu", weights_only=True)["model_state_dict"]
                _model.load_state_dict(sd, strict=True)
                _checkpoint_loaded = True
                logger.info("Loaded checkpoint weights: %s", latest)
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not load checkpoint %s: %s", latest, e)

        _engine = GenerationEngine(_model)
        _model_loaded = True

        elapsed = time.time() - _startup_time
        logger.info(
            "Model loaded in %.2fs (params=%d, checkpoint_loaded=%s)",
            elapsed,
            _model.parameter_count(),
            _checkpoint_loaded,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to load model: %s", e)
        _model_loaded = False

    yield  # Server runs here

    logger.info("Shutting down XRFM API")
    _model = None
    _engine = None


app = FastAPI(
    title="XR Foundation Model API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# Security headers
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


# Request timing
@app.middleware("http")
async def request_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time-Ms"] = str(round(elapsed * 1000, 2))
    logger.info("%s %s %d %.2fms", request.method, request.url.path, response.status_code, elapsed * 1000)
    return response


# Import and register routes
# Forensic-audit fix (F-41): `search_routes` never existed; the API could not
# import. The search routes now live in `api/routes/search.py`.
from api.routes import completions, health, metrics, search, tokenize_endpoints  # noqa: E402

# The routes import globals from api.main, creating a module cycle; mypy
# cannot resolve `router` types through it, so the cycle is documented and
# the includes are typed as Any (justified exclusion, Phase 29).
app.include_router(health.router, tags=["Health"])  # type: ignore[has-type]
app.include_router(completions.router, tags=["Completions"])  # type: ignore[has-type]
app.include_router(tokenize_endpoints.router, tags=["Tokenize"])  # type: ignore[has-type]
app.include_router(metrics.router, tags=["Metrics"])  # type: ignore[has-type]
app.include_router(search.router, tags=["Search Engine"])  # type: ignore[has-type]

# Mount web UI
try:
    app.mount("/ui", StaticFiles(directory="webui", html=True), name="webui")
except Exception:
    pass  # webui dir may not exist
