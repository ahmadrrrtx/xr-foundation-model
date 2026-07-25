# XRFM Phase 10 — Production Deployment: Research Report

**Date:** 2026-07-25
**Engineer:** Principal AI Infrastructure Engineer
**Status:** RESEARCH COMPLETE — Ready for Implementation

---

## 1. FastAPI Production Stack

### Industry Standard (8+ sources confirmed)

```
Internet → Nginx/Traefik (TLS, rate-limit, buffering)
         → Gunicorn (process manager, 2N+1 workers)
         → Uvicorn Workers (async event loop)
         → FastAPI → Model Service
```

### Key Findings
- Gunicorn + Uvicorn is THE production pattern (not uvicorn alone)
- Workers: (2 × cores) + 1 for I/O, match cores for ML inference
- `preload_app=True` saves memory via copy-on-write for model
- `max_requests=1000-5000` recycles workers (prevents memory leaks)
- Streaming needs `proxy_buffering off` in reverse proxy
- ASGI lifespan for model loading: load once at startup

### XRFM Classification
| Component | Class |
|---|---|
| FastAPI + Uvicorn | CORE |
| Gunicorn process manager | CORE |
| ASGI lifespan for model | CORE |
| Nginx reverse proxy | OPTIONAL |
| Kubernetes | RESEARCH-ONLY |

## 2. API Design (OpenAI-compatible)

### Endpoints
- GET /health, /health/ready — liveness + readiness
- GET /v1/models — model info
- POST /v1/completions — text generation (sync + streaming SSE)
- POST /v1/tokenize — encode/decode
- GET /metrics — model metrics

All classified CORE.

## 3. vLLM Assessment
vLLM is industry standard for production serving. XRFM custom architecture means direct integration requires weight export path.
- Custom inference engine: CORE (works now)
- vLLM weight export: OPTIONAL
- PagedAttention/Continuous batching: RESEARCH-ONLY

## 4. Docker: Multi-Stage Builds
- Base: python:3.11-slim (CPU), nvidia/cuda:12.x-runtime (GPU)
- Multi-stage: builder installs deps, runtime copies only needed files
- Model weights as volumes, not baked into image
- Target: ~1.5GB CPU, ~5GB GPU (+ model volume)

All classified CORE.

## 5. GitHub Actions CI/CD
- ci.yml: lint (ruff) + test (pytest matrix 3.10-3.12) on push/PR
- cd.yml: Docker build + push to GHCR on main
- release.yml: publish on semver tag
- Trivy vulnerability scanning

All classified CORE.

## 6. Security (Defense-in-Depth)
- CORS with explicit allowlist
- Rate limiting (slowapi/get_remote_address)
- Security headers (X-Content-Type-Options, HSTS, X-Frame-Options)
- API key auth via dependency injection
- Pydantic input validation + request size limits
- Dependency scanning in CI

All classified CORE.

## 7. Monitoring
- Structured JSON logging to stdout
- Request timing middleware
- Health checks (liveness + readiness)
- Token throughput metrics
- Prometheus/Grafana: OPTIONAL
- OpenTelemetry: RESEARCH-ONLY

## 8. Web UI
- Single static HTML page (no React, no npm)
- SSE for streaming token display
- Dark/light mode, settings panel, localStorage history
- Classification: CORE

## 9. Architecture Validation: NO BREAKING CHANGES
All existing modules (GPTModel, GenerationEngine, Tokenizer, ConfigLoader) are compatible. Phase 10 adds api/, deployment/, webui/, .github/ without modifying any existing source.

## 10. References
1. FastAPI Deployment 2026 — zestminds.com
2. FastAPI for Production — dsinnovators.com
3. LLMs with FastAPI — zignuts.com
4. FastAPI Production Setup — markaicode.com
5. FastAPI + Gunicorn — oneuptime.com
6. GitHub Actions CI/CD 2026 — devops.gheware.com
7. GitHub Actions Docker Build — tutorials.technology
8. Docker LLM Workloads — latitude.so
9. vLLM Architecture — markaicode.com / runpod.io
10. vLLM 2026 — futureagi.com
11. LLM Serving Comparison — arxiv 2511.17593
12. Docker Multi-Stage — dzone.com
13. FastAPI Security — shipsafer.app
14. FastAPI Rate Limiting — oneuptime.com
