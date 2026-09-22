from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


def test_agent_json_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("XRFM_AGENT_ROOT", str(tmp_path))
    monkeypatch.delenv("XRFM_AGENT_API_KEY", raising=False)
    import api.routes.agent as agent_route
    agent_route._runtime = None
    from api.main import app

    with TestClient(app) as client:
        response = client.post("/v1/agent/run", json={"request": "Find the largest file in this project and summarize what it is."})
    assert response.status_code == 200
    body = response.json()
    assert body["failed"] is False  # tool failure is contained and converted into an answer
    assert body["events"]
    assert body["observations"]


def test_agent_stream_endpoint(monkeypatch, tmp_path):
    (tmp_path / "example.txt").write_text("hello", encoding="utf-8")
    monkeypatch.setenv("XRFM_AGENT_ROOT", str(tmp_path))
    monkeypatch.delenv("XRFM_AGENT_API_KEY", raising=False)
    import api.routes.agent as agent_route
    agent_route._runtime = None
    from api.main import app

    with TestClient(app) as client:
        response = client.post("/v1/agent/stream", json={"request": "Find the largest file in this project and summarize what it is."})
    assert response.status_code == 200
    assert "event: execute" in response.text
    assert "event: done" in response.text


def test_agent_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("XRFM_AGENT_ROOT", str(tmp_path))
    monkeypatch.setenv("XRFM_AGENT_API_KEY", "test-secret")
    import api.routes.agent as agent_route
    agent_route._runtime = None
    from api.main import app

    with TestClient(app) as client:
        assert client.post("/v1/agent/run", json={"request": "hello"}).status_code == 401
        assert client.post("/v1/agent/run", headers={"X-XRFM-Agent-Key": "test-secret"}, json={"request": "hello"}).status_code == 200
    monkeypatch.delenv("XRFM_AGENT_API_KEY", raising=False)
