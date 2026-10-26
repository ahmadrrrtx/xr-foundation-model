"""Phase 0 API status, payload, rate, and concurrency contract regressions."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_phase0_api import configure_service, make_tiny_service_artifacts


def test_completion_stream_flag_conflicts_with_unary_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tokenizer_path, checkpoint_path, _ = make_tiny_service_artifacts(tmp_path)
    configure_service(monkeypatch, tokenizer_path, checkpoint_path)
    from api.main import app

    with TestClient(app) as client:
        response = client.post("/v1/completions", json={"prompt": "hello", "stream": True})
    assert response.status_code == 409
    assert "/v1/completions/stream" in response.text


def test_tokenize_value_errors_are_bad_requests(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tokenizer_path, checkpoint_path, _ = make_tiny_service_artifacts(tmp_path)
    configure_service(monkeypatch, tokenizer_path, checkpoint_path)
    from api.main import app

    with TestClient(app) as client:

        def invalid_input(text: str) -> list[int]:
            raise ValueError("invalid tokenizer input")

        monkeypatch.setattr(app.state.xrfm.tokenizer, "encode", invalid_input)
        response = client.post("/v1/tokenize", json={"text": "hello"})
    assert response.status_code == 400
    assert "invalid tokenizer input" in response.text


def test_unexpected_generation_failure_is_sanitized_500(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tokenizer_path, checkpoint_path, _ = make_tiny_service_artifacts(tmp_path)
    configure_service(monkeypatch, tokenizer_path, checkpoint_path)
    from api.main import app

    with TestClient(app) as client:

        def crash(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("sensitive internal detail")

        monkeypatch.setattr(app.state.xrfm.engine, "generate_with_metadata", crash)
        response = client.post("/v1/completions", json={"prompt": "hello"})
    assert response.status_code == 500
    assert "RuntimeError" in response.text
    assert "sensitive internal detail" not in response.text


def test_request_body_limit_returns_413(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tokenizer_path, checkpoint_path, _ = make_tiny_service_artifacts(tmp_path)
    configure_service(monkeypatch, tokenizer_path, checkpoint_path)
    monkeypatch.setenv("XRFM_API_MAX_BODY_BYTES", "128")
    from api.main import app

    with TestClient(app) as client:
        response = client.post("/v1/tokenize", json={"text": "x" * 512})
    assert response.status_code == 413


def test_rate_limit_returns_429_and_retry_after(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tokenizer_path, checkpoint_path, _ = make_tiny_service_artifacts(tmp_path)
    configure_service(monkeypatch, tokenizer_path, checkpoint_path)
    monkeypatch.setenv("XRFM_API_RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setenv("XRFM_API_RATE_WINDOW_SECONDS", "60")
    from api.main import app

    with TestClient(app) as client:
        first = client.post("/v1/tokenize", json={"text": "hello"})
        second = client.post("/v1/tokenize", json={"text": "hello"})
    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) >= 1


def test_concurrency_limit_rejects_second_inflight_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tokenizer_path, checkpoint_path, _ = make_tiny_service_artifacts(tmp_path)
    configure_service(monkeypatch, tokenizer_path, checkpoint_path)
    monkeypatch.setenv("XRFM_API_MAX_CONCURRENT", "1")
    from api.main import app

    with TestClient(app) as client:
        original_generate = app.state.xrfm.engine.generate_with_metadata
        started = threading.Event()
        release = threading.Event()
        first_result: dict[str, Any] = {}

        def blocking_generate(*args: Any, **kwargs: Any) -> Any:
            started.set()
            release.wait(timeout=5)
            return original_generate(*args, **kwargs)

        monkeypatch.setattr(app.state.xrfm.engine, "generate_with_metadata", blocking_generate)
        timer = threading.Timer(2, release.set)
        timer.start()

        def first_request() -> None:
            first_result["response"] = client.post("/v1/completions", json={"prompt": "hello"})

        thread = threading.Thread(target=first_request)
        try:
            thread.start()
            assert started.wait(timeout=2)
            second = client.post("/v1/completions", json={"prompt": "hello"})
            assert second.status_code == 429
        finally:
            release.set()
            timer.cancel()
            thread.join(timeout=5)
        assert not thread.is_alive()
        assert first_result["response"].status_code == 200
