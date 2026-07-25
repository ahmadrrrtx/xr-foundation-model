# XRFM Deployment Guide — v1.0.0

## Quick Start

```bash
# CPU
docker compose -f deployment/docker-compose.yml up -d

# GPU
docker compose -f deployment/docker-compose.yml --profile gpu up -d

# Or directly
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Liveness check |
| GET | /health/ready | Readiness |
| GET | /v1/models | Model info |
| POST | /v1/completions | Text generation |
| POST | /v1/completions/stream | Streaming SSE |
| POST | /v1/tokenize | Encode text |
| POST | /v1/detokenize | Decode tokens |
| GET | /metrics | Server metrics |
| GET | /ui | Web chat interface |

## Docker Images

- CPU: `docker build -f deployment/Dockerfile -t xrfm .`
- GPU: `docker build -f deployment/Dockerfile.gpu -t xrfm:gpu .`
