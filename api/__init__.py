"""XRFM Production API (v1.0.0).

Provides FastAPI server with:
- Health/readiness checks
- Text generation (sync + streaming SSE)
- Tokenization
- Metrics
- Web UI

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
Production:
    gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
"""
