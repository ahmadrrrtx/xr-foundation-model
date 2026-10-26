"""Phase 0 regressions for application-level agent kill and revocation controls."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from xrfm.agent import (
    AgentRuntime,
    PermissionPolicy,
    RuntimeLimits,
    ScriptedBackend,
    ToolExecutor,
    ToolRegistry,
    ToolSpec,
)


def _new_control() -> Any:
    from xrfm.agent import AgentControl

    return AgentControl()


class _ControlBlockingBackend(ScriptedBackend):
    def __init__(self, path: str) -> None:
        self.path = path

    def chat(self, messages: Any, tools: Any) -> Any:
        Path(self.path).write_text("started", encoding="utf-8")
        time.sleep(10)
        return super().chat(messages, tools)


def _wait_until_started(path: Path) -> None:
    deadline = time.monotonic() + 3
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists(), "backend did not start"


def _configure_local_demo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XRFM_AGENT_ENABLED", "true")
    monkeypatch.setenv("XRFM_AGENT_ALLOW_LOCAL_DEMO", "true")
    monkeypatch.setenv("XRFM_AGENT_ROOT", str(tmp_path))
    monkeypatch.delenv("XRFM_AGENT_API_KEY", raising=False)


def test_kill_switch_cancels_active_runtime_and_is_app_scoped(tmp_path: Path) -> None:
    control = _new_control()
    independent_control = _new_control()
    cancel_event = threading.Event()
    registration = control.register(cancel_event)
    started_path = tmp_path / "backend-started"
    runtime = AgentRuntime(
        _ControlBlockingBackend(str(started_path)),
        ToolExecutor(ToolRegistry()),
        RuntimeLimits(max_steps=1, max_wall_seconds=20),
    )
    result: dict[str, Any] = {}

    def execute() -> None:
        result["state"] = runtime.run("hello", cancel_event=cancel_event)

    thread = threading.Thread(target=execute)
    thread.start()
    _wait_until_started(started_path)
    control.kill()
    thread.join(timeout=2)
    control.unregister(registration)

    assert not thread.is_alive()
    assert result["state"].failed
    assert "cancelled" in (result["state"].final_answer or "")
    with pytest.raises(RuntimeError, match="kill switch"):
        control.register(threading.Event())

    independent_registration = independent_control.register(threading.Event())
    independent_control.unregister(independent_registration)
    control.resume()
    resumed_registration = control.register(threading.Event())
    control.unregister(resumed_registration)


def test_revocation_cancels_active_runs_and_survives_resume() -> None:
    control = _new_control()
    active_cancellation = threading.Event()
    registration = control.register(active_cancellation)
    control.revoke(permission="filesystem.read", tool="read_file")
    control.unregister(registration)
    assert active_cancellation.is_set()

    policy = PermissionPolicy(allowed_permissions={"filesystem.read", "search.read"})
    control.apply_policy(policy)
    spec = ToolSpec("read_file", "test", {"type": "object"}, permissions=("filesystem.read",))
    assert not policy.authorize(spec).allowed
    assert "filesystem.read" not in policy.allowed_permissions
    assert "read_file" in policy.denied_tools

    control.kill()
    control.resume()
    resumed_policy = PermissionPolicy(allowed_permissions={"filesystem.read"})
    control.apply_policy(resumed_policy)
    assert not resumed_policy.authorize(spec).allowed
    with pytest.raises(ValueError):
        control.revoke()


def test_http_kill_switch_rejects_new_runs_and_resume_is_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_local_demo(monkeypatch, tmp_path)
    from api.main import app

    with TestClient(app) as client:
        assert client.post("/v1/agent/run", json={"request": "hello"}).status_code == 200
        killed = client.post("/v1/agent/control/kill")
        assert killed.status_code == 200
        assert killed.json()["killed"] is True
        rejected = client.post("/v1/agent/run", json={"request": "hello"})
        assert rejected.status_code == 503
        assert "kill switch" in rejected.text
        assert client.post("/v1/agent/control/resume").status_code == 200
        assert client.post("/v1/agent/run", json={"request": "hello"}).status_code == 200
        assert client.post("/v1/agent/control/revoke", json={}).status_code == 400


def test_http_kill_switch_cancels_inflight_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_local_demo(monkeypatch, tmp_path)
    import api.routes.agent as route
    from api.main import app

    started_path = tmp_path / "http-backend-started"
    runtime = AgentRuntime(
        _ControlBlockingBackend(str(started_path)),
        ToolExecutor(ToolRegistry()),
        RuntimeLimits(max_steps=1, max_wall_seconds=20),
    )
    monkeypatch.setattr(route, "_get_runtime", lambda: runtime)
    result: dict[str, Any] = {}

    with TestClient(app) as client:

        def execute() -> None:
            try:
                result["response"] = client.post("/v1/agent/run", json={"request": "hello"})
            except Exception as exc:  # retain thread failures for the main assertion
                result["error"] = exc

        thread = threading.Thread(target=execute)
        try:
            thread.start()
            deadline = time.monotonic() + 5
            while not started_path.exists() and "response" not in result and "error" not in result:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
            response_body = result["response"].json() if "response" in result else result
            assert started_path.exists(), f"backend did not start; result={response_body}"
            killed = client.post("/v1/agent/control/kill")
            thread.join(timeout=3)

            assert killed.status_code == 200
            assert not thread.is_alive()
            assert "error" not in result
            assert result["response"].status_code == 200
            assert result["response"].json()["failed"] is True
            assert client.post("/v1/agent/run", json={"request": "hello"}).status_code == 503
            assert client.get("/v1/agent/control/status").json()["active_runs"] == 0
        finally:
            if thread.is_alive() and app.state.xrfm.agent_control is not None:
                app.state.xrfm.agent_control.kill()
            thread.join(timeout=25)


def test_control_endpoints_require_configured_authentication(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_local_demo(monkeypatch, tmp_path)
    monkeypatch.setenv("XRFM_AGENT_API_KEY", "control-secret")
    from api.main import app

    with TestClient(app) as client:
        assert client.get("/v1/agent/control/status").status_code == 401
        response = client.get("/v1/agent/control/status", headers={"X-XRFM-Agent-Key": "control-secret"})
        assert response.status_code == 200
        assert response.json()["killed"] is False
