"""Additional adversarial checks for per-application API protection."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_phase0_api import configure_service, make_tiny_service_artifacts


def test_chunked_request_body_is_limited_without_content_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tokenizer_path, checkpoint_path, _ = make_tiny_service_artifacts(tmp_path)
    configure_service(monkeypatch, tokenizer_path, checkpoint_path)
    monkeypatch.setenv("XRFM_API_MAX_BODY_BYTES", "128")
    from api.main import app

    def chunks() -> Iterator[bytes]:
        yield b'{"text":"'
        yield b"x" * 512
        yield b'"}'

    with TestClient(app) as client:
        response = client.post("/v1/tokenize", content=chunks(), headers={"Content-Type": "application/json"})
    assert response.status_code == 413, response.text


def test_invalid_protection_limit_fails_startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tokenizer_path, checkpoint_path, _ = make_tiny_service_artifacts(tmp_path)
    configure_service(monkeypatch, tokenizer_path, checkpoint_path)
    monkeypatch.setenv("XRFM_API_MAX_CONCURRENT", "0")
    from api.main import app

    with pytest.raises(RuntimeError, match="XRFM_API_MAX_CONCURRENT"):
        with TestClient(app):
            pass


def test_emergency_kill_is_not_blocked_by_exhausted_rate_quota(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tokenizer_path, checkpoint_path, _ = make_tiny_service_artifacts(tmp_path)
    configure_service(monkeypatch, tokenizer_path, checkpoint_path)
    monkeypatch.setenv("XRFM_API_RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setenv("XRFM_AGENT_ENABLED", "true")
    monkeypatch.setenv("XRFM_AGENT_ALLOW_LOCAL_DEMO", "true")
    monkeypatch.setenv("XRFM_AGENT_ROOT", str(tmp_path))
    from api.main import app

    with TestClient(app) as client:
        assert client.post("/v1/tokenize", json={"text": "hello"}).status_code == 200
        assert client.post("/v1/agent/control/kill").status_code == 200
        assert client.post("/v1/tokenize", json={"text": "hello"}).status_code == 429
