"""Provider redirect and real-socket cancellation contracts for Phase 0."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from xrfm.agent import BackendCancelledError, BackendExecutor, OpenAICompatibleBackend
from xrfm.agent.backends import BackendProtocolError


class _QuietHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return None


class _CaptureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, handler: type[BaseHTTPRequestHandler]) -> None:
        super().__init__(("127.0.0.1", 0), handler)
        self.redirect_target = ""
        self.received_authorizations: list[str | None] = []
        self.request_started = threading.Event()
        self.disconnect_seen = threading.Event()


@contextmanager
def _running_server(handler: type[BaseHTTPRequestHandler]) -> Iterator[_CaptureServer]:
    server = _CaptureServer(handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class _RedirectHandler(_QuietHandler):
    def do_POST(self) -> None:
        self.send_response(302)
        self.send_header("Location", self.server.redirect_target)  # type: ignore[attr-defined]
        self.send_header("Content-Length", "0")
        self.end_headers()


class _CaptureHandler(_QuietHandler):
    def do_GET(self) -> None:
        self.server.received_authorizations.append(self.headers.get("Authorization"))  # type: ignore[attr-defined]
        body = json.dumps({"choices": [{"message": {"content": "redirect followed"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _DisconnectHandler(_QuietHandler):
    def do_POST(self) -> None:
        server = self.server  # type: ignore[assignment]
        content_length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(content_length)
        server.request_started.set()
        self.connection.settimeout(5)
        try:
            if self.connection.recv(1) == b"":
                server.disconnect_seen.set()
        except (ConnectionResetError, TimeoutError):
            server.disconnect_seen.set()
        self.close_connection = True


def test_provider_redirect_cannot_receive_credentials() -> None:
    with _running_server(_RedirectHandler) as redirector, _running_server(_CaptureHandler) as receiver:
        redirector.redirect_target = f"http://127.0.0.1:{receiver.server_port}/credential-canary"
        backend = OpenAICompatibleBackend(
            f"http://127.0.0.1:{redirector.server_port}",
            "test-model",
            api_key="test-provider-canary",
        )
        error: Exception | None = None
        try:
            backend.chat([], [])
        except Exception as exc:
            error = exc
    assert receiver.received_authorizations == []
    assert isinstance(error, BackendProtocolError)
    assert "redirect" in str(error).lower()


def test_cancelling_provider_process_closes_real_socket() -> None:
    with _running_server(_DisconnectHandler) as server:
        backend = OpenAICompatibleBackend(f"http://127.0.0.1:{server.server_port}", "test-model", timeout=10)
        cancel_event = threading.Event()
        errors: list[Exception] = []

        def execute() -> None:
            try:
                BackendExecutor().execute(backend, [], [], timeout=5, cancel_event=cancel_event)
            except Exception as exc:
                errors.append(exc)

        started = time.monotonic()
        thread = threading.Thread(target=execute)
        thread.start()
        assert server.request_started.wait(3), "provider request did not reach the loopback server"
        cancel_event.set()
        thread.join(timeout=3)
        assert not thread.is_alive(), "backend cancellation did not return"
        assert time.monotonic() - started < 4
        assert len(errors) == 1 and isinstance(errors[0], BackendCancelledError)
        assert server.disconnect_seen.wait(3), "provider socket remained open after process cancellation"
