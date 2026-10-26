"""Per-application request size, rate, and concurrency protection."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_GUARDED_PATHS = {
    "/v1/completions",
    "/v1/completions/stream",
    "/v1/tokenize",
    "/v1/detokenize",
    "/v1/agent/run",
    "/v1/agent/stream",
}


def _positive_int(name: str, default: int, maximum: int) -> int:
    raw = os.getenv(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if not 0 < value <= maximum:
        raise RuntimeError(f"{name} must be in [1, {maximum}]")
    return value


def _positive_float(name: str, default: float, maximum: float) -> float:
    raw = os.getenv(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or not 0 < value <= maximum:
        raise RuntimeError(f"{name} must be finite and in (0, {maximum}]")
    return value


@dataclass(frozen=True)
class ProtectionConfig:
    max_body_bytes: int = 1_048_576
    rate_limit_requests: int = 120
    rate_window_seconds: float = 60.0
    max_concurrent: int = 8
    max_tracked_clients: int = 10_000

    @classmethod
    def from_env(cls) -> ProtectionConfig:
        return cls(
            max_body_bytes=_positive_int("XRFM_API_MAX_BODY_BYTES", 1_048_576, 67_108_864),
            rate_limit_requests=_positive_int("XRFM_API_RATE_LIMIT_REQUESTS", 120, 100_000),
            rate_window_seconds=_positive_float("XRFM_API_RATE_WINDOW_SECONDS", 60.0, 3_600.0),
            max_concurrent=_positive_int("XRFM_API_MAX_CONCURRENT", 8, 1_024),
        )


class RequestProtector:
    """Thread-safe admission state for one FastAPI application lifecycle."""

    def __init__(self, config: ProtectionConfig) -> None:
        self.config = config
        self._capacity = threading.BoundedSemaphore(config.max_concurrent)
        self._rate_lock = threading.Lock()
        self._clients: dict[str, tuple[float, int]] = {}

    def _rate_allow(self, client: str, now: float) -> tuple[bool, int]:
        window = self.config.rate_window_seconds
        with self._rate_lock:
            current = self._clients.get(client)
            if current is not None and now - current[0] < window:
                started, count = current
                if count >= self.config.rate_limit_requests:
                    return False, max(1, math.ceil(started + window - now))
                self._clients[client] = (started, count + 1)
                return True, 0

            if len(self._clients) >= self.config.max_tracked_clients:
                expired = [key for key, (started, _) in self._clients.items() if now - started >= window]
                for key in expired:
                    del self._clients[key]
                if len(self._clients) >= self.config.max_tracked_clients:
                    return False, max(1, math.ceil(window))
            self._clients[client] = (now, 1)
            return True, 0

    def acquire(self, client: str) -> tuple[str | None, int]:
        """Acquire capacity before charging rate quota; reject without waiting."""
        if not self._capacity.acquire(blocking=False):
            return "concurrent request capacity exceeded", 1
        try:
            allowed, retry_after = self._rate_allow(client, time.monotonic())
        except Exception:
            self._capacity.release()
            raise
        if not allowed:
            self._capacity.release()
            return "request rate limit or tracked-client capacity exceeded", retry_after
        return None, 0

    def release(self) -> None:
        self._capacity.release()


async def _send_json(send: Send, status: int, detail: str, headers: dict[str, str] | None = None) -> None:
    body = json.dumps({"detail": detail}, ensure_ascii=False).encode("utf-8")
    response_headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    if headers:
        response_headers.extend((key.lower().encode(), value.encode()) for key, value in headers.items())
    await send({"type": "http.response.start", "status": status, "headers": response_headers})
    await send({"type": "http.response.body", "body": body})


class RequestProtectionMiddleware:
    """Enforce real body bytes and hold admission through streaming completion."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        app_state = getattr(scope["app"].state, "xrfm", None)
        protector: RequestProtector | None = getattr(app_state, "protection", None)
        guarded = scope["path"] in _GUARDED_PATHS
        if protector is None:
            if guarded:
                await _send_json(send, 503, "API request protection is not ready")
                return
            await self.app(scope, receive, send)
            return

        content_length = Headers(scope=scope).get("content-length")
        if content_length is not None:
            try:
                declared_bytes = int(content_length)
            except ValueError:
                await _send_json(send, 400, "invalid Content-Length")
                return
            if declared_bytes < 0:
                await _send_json(send, 400, "invalid Content-Length")
                return
            if declared_bytes > protector.config.max_body_bytes:
                await _send_json(send, 413, "request body exceeds configured byte limit")
                return

        acquired = False
        try:
            if guarded:
                client: Any = scope.get("client")
                client_host = client[0] if client else "unknown"
                reason, retry_after = protector.acquire(client_host)
                if reason is not None:
                    await _send_json(send, 429, reason, {"Retry-After": str(retry_after)})
                    return
                acquired = True

            # Read at most limit+1 bytes before FastAPI parses JSON. Replaying
            # the bounded body avoids framework-specific exception mapping and
            # keeps chunked requests subject to exactly the same 413 contract.
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                if message["type"] != "http.request":
                    continue
                body.extend(message.get("body", b""))
                if len(body) > protector.config.max_body_bytes:
                    await _send_json(send, 413, "request body exceeds configured byte limit")
                    return
                if not message.get("more_body", False):
                    break

            replayed = False

            async def replay_receive() -> Message:
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, replay_receive, send)
        finally:
            if acquired:
                protector.release()
