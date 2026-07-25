"""XRFM FastAPI application (v1.0.0).

Production API server with: health checks, model info, text generation
(sync + streaming), tokenization, structured logging, rate limiting,
security headers, and CORS.
"""

import logging
import sys
import time
from contextlib import asynccontextmanager

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

    try:
        from inference.engine import GenerationEngine
        from model.gpt import GPTModel
        from tokenizer.bpe import BytePairEncoder

        _model = GPTModel()
        _engine = GenerationEngine(_model)
        _tokenizer = BytePairEncoder()
        _model_loaded = True

        elapsed = time.time() - _startup_time
        logger.info("Model loaded in %.2fs (params=%d)", elapsed, _model.parameter_count())
    except Exception as e:
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
    logger.info(
        "%s %s %d %.2fms", request.method, request.url.path, response.status_code, elapsed * 1000
    )
    return response


# Import and register routes
from api.routes import completions, health, metrics, search_routes, tokenize_endpoints  # noqa: E402

app.include_router(health.router, tags=["Health"])
app.include_router(completions.router, tags=["Completions"])
app.include_router(tokenize_endpoints.router, tags=["Tokenize"])
app.include_router(metrics.router, tags=["Metrics"])
app.include_router(search_routes.router, tags=["Search Engine"])

# Mount web UI
try:
    app.mount("/ui", StaticFiles(directory="webui", html=True), name="webui")
except Exception:
    pass  # webui dir may not exist
