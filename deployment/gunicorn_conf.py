"""Gunicorn configuration for XRFM production."""
import multiprocessing, os

bind = os.getenv("BIND", "0.0.0.0:8000")
workers = int(os.getenv("WORKERS", (multiprocessing.cpu_count() * 2) + 1))
worker_class = "uvicorn.workers.UvicornWorker"
timeout = int(os.getenv("TIMEOUT", 300))
graceful_timeout = 30
max_requests = int(os.getenv("MAX_REQUESTS", 1000))
max_requests_jitter = 50
preload_app = True
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")
