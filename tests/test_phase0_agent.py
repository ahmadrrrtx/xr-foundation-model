"""Adversarial agent containment regressions for Phase 0."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from xrfm.agent import (
    AgentPhase,
    AgentRuntime,
    AuditLog,
    Decision,
    DecisionType,
    OpenAICompatibleBackend,
    RuntimeLimits,
    SafeFilesystemTools,
    ScriptedBackend,
    ToolCall,
    ToolExecutor,
    ToolRegistry,
    ToolSpec,
    sanitize_observation,
)
from xrfm.agent.tools import JSONLimits, ToolError, _validate_schema, enforce_json_bounds


def _slow_tool(_: dict[str, Any]) -> str:
    time.sleep(0.5)
    return "finished"


def _invalid_output_tool(_: dict[str, Any]) -> dict[str, bool]:
    return {"unexpected": True}


def _pid_sleep_tool(arguments: dict[str, Any]) -> str:
    Path(arguments["path"]).write_text(str(os.getpid()), encoding="utf-8")
    time.sleep(10)
    return "finished"


class _BlockingBackend(ScriptedBackend):
    def __init__(self, seconds: float = 2.0) -> None:
        self.seconds = seconds

    def chat(self, messages: Any, tools: Any) -> Decision:
        time.sleep(self.seconds)
        return Decision(DecisionType.ANSWER, "late answer", answer="too late")


class _SlowBackend(ScriptedBackend):
    def chat(self, messages: Any, tools: Any) -> Decision:
        time.sleep(0.35)
        return super().chat(messages, tools)


class _PIDBlockingBackend(ScriptedBackend):
    def __init__(self, path: str) -> None:
        self.path = path

    def chat(self, messages: Any, tools: Any) -> Decision:
        Path(self.path).write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(10)
        return Decision(DecisionType.ANSWER, "late answer", answer="too late")


class _DecisionBackend(ScriptedBackend):
    def __init__(self, decision: Decision) -> None:
        self.decision = decision

    def chat(self, messages: Any, tools: Any) -> Decision:
        return self.decision


def test_agent_public_mode_without_auth_fails_at_startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XRFM_AGENT_ENABLED", "true")
    monkeypatch.setenv("XRFM_AGENT_ROOT", str(tmp_path))
    monkeypatch.delenv("XRFM_AGENT_API_KEY", raising=False)
    monkeypatch.delenv("XRFM_AGENT_ALLOW_LOCAL_DEMO", raising=False)
    monkeypatch.delenv("XRFM_AGENT_BASE_URL", raising=False)
    from api.main import app

    with pytest.raises(RuntimeError, match="agent|API key|demo"):
        with TestClient(app):
            pass


def test_agent_public_mode_with_auth_but_no_backend_fails_at_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XRFM_AGENT_ENABLED", "true")
    monkeypatch.setenv("XRFM_AGENT_ROOT", str(tmp_path))
    monkeypatch.setenv("XRFM_AGENT_API_KEY", "secret")
    monkeypatch.delenv("XRFM_AGENT_ALLOW_LOCAL_DEMO", raising=False)
    monkeypatch.delenv("XRFM_AGENT_BASE_URL", raising=False)
    from api.main import app

    with pytest.raises(RuntimeError, match="BASE_URL|backend|demo"):
        with TestClient(app):
            pass


def test_agent_public_mode_requires_declared_decision_capability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XRFM_AGENT_ENABLED", "true")
    monkeypatch.setenv("XRFM_AGENT_ROOT", str(tmp_path))
    monkeypatch.setenv("XRFM_AGENT_API_KEY", "secret")
    monkeypatch.setenv("XRFM_AGENT_BASE_URL", "http://provider.invalid")
    monkeypatch.delenv("XRFM_AGENT_NATIVE_TOOL_CALLING", raising=False)
    monkeypatch.delenv("XRFM_AGENT_STRUCTURED_OUTPUT", raising=False)
    from api.main import app

    with pytest.raises(RuntimeError, match="capability|NATIVE|STRUCTURED"):
        with TestClient(app):
            pass


def test_agent_runtime_factory_does_not_share_request_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XRFM_AGENT_ROOT", str(tmp_path))
    monkeypatch.setenv("XRFM_AGENT_ALLOW_LOCAL_DEMO", "true")
    import api.routes.agent as route

    route._runtime = None
    first = route._get_runtime()
    second = route._get_runtime()
    assert first is not second
    assert first.executor.audit is not second.executor.audit


def test_runtime_wall_deadline_interrupts_a_blocked_backend() -> None:
    runtime = AgentRuntime(
        _BlockingBackend(),
        ToolExecutor(ToolRegistry()),
        RuntimeLimits(max_steps=1, max_wall_seconds=0.5),
    )
    started = time.monotonic()
    state = runtime.run("hello")
    elapsed = time.monotonic() - started
    assert state.failed
    assert elapsed < 1.5


def test_backend_cancellation_interrupts_blocked_execution() -> None:
    cancel_event = threading.Event()
    timer = threading.Timer(0.5, cancel_event.set)
    timer.start()
    runtime = AgentRuntime(
        _BlockingBackend(10.0),
        ToolExecutor(ToolRegistry()),
        RuntimeLimits(max_steps=1, max_wall_seconds=20.0),
    )
    started = time.monotonic()
    try:
        state = runtime.run("hello", cancel_event=cancel_event)
    finally:
        timer.cancel()
    elapsed = time.monotonic() - started
    assert state.failed
    assert "cancelled" in (state.final_answer or "")
    assert elapsed < 1.5


def test_runtime_rejects_unregistered_and_contradictory_backend_decisions() -> None:
    decisions = [
        Decision(DecisionType.TOOL_CALL, "escape", (ToolCall("unregistered", {}),)),
        Decision(DecisionType.ANSWER, "contradiction", (ToolCall("unregistered", {}),), answer="must not pass"),
    ]
    for decision in decisions:
        state = AgentRuntime(_DecisionBackend(decision), ToolExecutor(ToolRegistry())).run("hello")
        assert state.failed
        assert not state.observations
        assert state.step == 1


@pytest.mark.skipif(os.name != "posix", reason="POSIX process liveness probe")
def test_runtime_wall_deadline_kills_a_backend_that_already_started(tmp_path: Path) -> None:
    pid_path = tmp_path / "backend.pid"
    runtime = AgentRuntime(
        _PIDBlockingBackend(str(pid_path)),
        ToolExecutor(ToolRegistry()),
        RuntimeLimits(max_steps=1, max_wall_seconds=4.0),
    )
    started = time.monotonic()
    state = runtime.run("hello")
    elapsed = time.monotonic() - started
    assert state.failed
    assert elapsed < 6.0
    assert pid_path.exists(), "the backend must have started before its deadline"
    pid = int(pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_agent_stream_emits_request_before_backend_finishes() -> None:
    runtime = AgentRuntime(_SlowBackend(), ToolExecutor(ToolRegistry()))
    events = runtime.stream("hello")
    started = time.monotonic()
    first = next(events)
    elapsed = time.monotonic() - started
    assert first.phase == AgentPhase.REQUEST
    assert elapsed < 0.2
    list(events)


def test_agent_http_stream_does_not_buffer_backend_work(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XRFM_AGENT_ENABLED", "true")
    monkeypatch.setenv("XRFM_AGENT_ALLOW_LOCAL_DEMO", "true")
    monkeypatch.setenv("XRFM_AGENT_ROOT", str(tmp_path))
    import api.routes.agent as route
    from api.main import app

    runtime = AgentRuntime(_SlowBackend(), ToolExecutor(ToolRegistry()))
    monkeypatch.setattr(route, "_get_runtime", lambda: runtime)
    with TestClient(app) as client:
        started = time.monotonic()
        with client.stream("POST", "/v1/agent/stream", json={"request": "hello"}) as response:
            lines = response.iter_lines()
            assert next(lines) == "event: request"
            assert time.monotonic() - started < 0.2
            assert "event: done" in list(lines)


def test_tool_timeout_is_a_hard_wall_clock_bound() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec("slow", "test", {"type": "object"}, timeout_seconds=0.05),
        _slow_tool,
    )
    started = time.monotonic()
    result = ToolExecutor(registry).execute(ToolCall("slow", {}))
    elapsed = time.monotonic() - started
    assert not result.ok
    assert "timed out" in (result.error or "")
    assert elapsed < 0.25
    assert result.duration_ms >= 40


@pytest.mark.skipif(os.name != "posix", reason="POSIX process liveness probe")
def test_timeout_kills_a_tool_child_that_already_started(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(ToolSpec("pid-sleep", "test", {"type": "object"}, timeout_seconds=4.0), _pid_sleep_tool)
    pid_path = tmp_path / "tool.pid"
    started = time.monotonic()
    result = ToolExecutor(registry).execute(ToolCall("pid-sleep", {"path": str(pid_path)}))
    elapsed = time.monotonic() - started
    assert not result.ok
    assert "timed out" in (result.error or "")
    assert elapsed < 6.0
    assert pid_path.exists(), "the tool must have started before its deadline"
    pid = int(pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_tool_cancellation_is_a_hard_wall_clock_bound() -> None:
    registry = ToolRegistry()
    registry.register(ToolSpec("slow", "test", {"type": "object"}, timeout_seconds=10.0), _slow_tool)
    cancel_event = threading.Event()
    timer = threading.Timer(0.05, cancel_event.set)
    timer.start()
    started = time.monotonic()
    try:
        result = ToolExecutor(registry).execute(ToolCall("slow", {}), cancel_event=cancel_event)
    finally:
        timer.cancel()
    elapsed = time.monotonic() - started
    assert not result.ok
    assert "cancelled" in (result.error or "")
    assert elapsed < 0.25


def test_schema_validation_covers_nested_constraints_and_outputs() -> None:
    schema = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
                "maxItems": 2,
            }
        },
        "required": ["values"],
        "additionalProperties": False,
    }
    with pytest.raises(ToolError):
        _validate_schema(schema, {"values": [1, "wrong"]})
    with pytest.raises(ToolError):
        _validate_schema(schema, {"values": []})

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "structured",
            "test",
            {"type": "object"},
            output_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        ),
        _invalid_output_tool,
    )
    result = ToolExecutor(registry).execute(ToolCall("structured", {}))
    assert not result.ok
    assert "output" in (result.error or "").lower()


def test_draft_2020_12_prefix_items_and_json_resource_bounds() -> None:
    schema = {
        "type": "array",
        "prefixItems": [{"type": "string"}, {"type": "integer"}],
        "minItems": 2,
        "maxItems": 2,
    }
    _validate_schema(schema, ["valid", 2])
    with pytest.raises(ToolError):
        _validate_schema(schema, ["invalid", "not-an-integer"])

    limits = JSONLimits(max_bytes=64, max_items=3, max_depth=3, max_string_bytes=8)
    with pytest.raises(ToolError, match="total JSON byte"):
        enforce_json_bounds("😀😀😀", JSONLimits(max_bytes=10, max_items=3, max_depth=3, max_string_bytes=100))
    with pytest.raises(ToolError, match="item"):
        enforce_json_bounds([1, 2, 3, 4], limits)
    deep: list[object] = []
    current = deep
    for _ in range(5):
        nested: list[object] = []
        current.append(nested)
        current = nested
    with pytest.raises(ToolError, match="depth"):
        enforce_json_bounds(deep, JSONLimits(max_bytes=1000, max_items=100, max_depth=3, max_string_bytes=100))


def test_redaction_covers_equals_bearer_multiline_and_negative_bounds() -> None:
    secret = "supersecret123"
    value = sanitize_observation(
        {
            "password": secret,
            "text": f"api_key = {secret}\nAuthorization: Bearer abcdefghijklmnop",
            "private": "-----BEGIN PRIVATE KEY-----\nabc\ndef\n-----END PRIVATE KEY-----",
        }
    )
    rendered = json.dumps(value)
    assert secret not in rendered
    assert "abcdefghijklmnop" not in rendered
    assert "abc\ndef" not in rendered
    assert "[REDACTED]" in rendered
    with pytest.raises(ValueError):
        sanitize_observation("secret", max_chars=-1)
    with pytest.raises(ValueError):
        sanitize_observation("secret", max_chars=10**12)


def test_filesystem_limits_and_secret_canaries_are_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "large.txt").write_text("x" * 10000, encoding="utf-8")
    (tmp_path / ".env").write_text("PASSWORD=do-not-exfiltrate", encoding="utf-8")
    tools = SafeFilesystemTools(tmp_path)
    with pytest.raises((ToolError, ValueError)):
        tools.read_file({"relative": "large.txt", "max_chars": -1})
    with pytest.raises((ToolError, ValueError)):
        tools.list_files({"relative": ".", "max_items": -1})
    with pytest.raises(ToolError, match="sensitive|forbidden|denied"):
        tools.read_file({"relative": ".env", "max_chars": 20})
    with pytest.raises(ToolError, match="escapes"):
        tools.read_file({"relative": "../outside.txt", "max_chars": 20})
    assert ".env" not in tools.list_files({"relative": ".", "max_items": 10})["files"]

    original_read_text = Path.read_text

    def fail_whole_file_read(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.name == "large.txt":
            raise AssertionError("whole-file read_text is not bounded")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_whole_file_read)
    result = tools.read_file({"relative": "large.txt", "max_chars": 10})
    assert len(result["text"]) == 10
    assert result["truncated"] is True
    exact = tools.read_file({"relative": "large.txt", "max_chars": 10000})
    assert exact["truncated"] is False
    monkeypatch.setattr(tools, "MAX_SCAN_FILES", 0)
    with pytest.raises(ToolError, match="file scan"):
        tools.list_files({"relative": ".", "max_items": 10})


def test_provider_rejects_non_http_urls_and_embedded_credentials() -> None:
    with pytest.raises(ValueError, match="HTTP"):
        OpenAICompatibleBackend("file:///etc/passwd", "test-model")
    with pytest.raises(ValueError, match="credentials"):
        OpenAICompatibleBackend("https://user:password@provider.invalid", "test-model")


def test_native_provider_tool_call_id_and_capabilities_are_honest(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "provider-call-42",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"text":"hello"}'},
                        }
                    ],
                }
            }
        ]
    }

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self, *args: Any) -> bytes:
            return json.dumps(payload).encode()

    requests: list[Any] = []

    def fake_urlopen(request: Any, *args: Any, **kwargs: Any) -> Response:
        requests.append(request)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    backend = OpenAICompatibleBackend("http://provider.invalid", "test-model")
    decision = backend.chat([], [ToolSpec("echo", "test", {"type": "object"})])
    assert decision.tool_calls[0].call_id == "provider-call-42"
    info = backend.get_model_info()
    assert info.native_tool_calling is False
    assert info.structured_output is False
    assert info.reasoning is False

    capable = OpenAICompatibleBackend(
        "http://provider.invalid", "test-model", native_tool_calling=True, structured_output=True
    )
    capable.chat([], [ToolSpec("echo", "test", {"type": "object"})])
    outgoing = json.loads(requests[-1].data.decode())
    assert outgoing["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert outgoing["tool_choice"] == "auto"
    assert outgoing["response_format"] == {"type": "json_object"}


def test_audit_retention_is_bounded() -> None:
    audit = AuditLog(max_records=2)
    audit.record("one", password="do-not-record")
    audit.record("two")
    audit.record("three")
    assert len(audit.records) == 2
    assert [record.event for record in audit.records] == ["two", "three"]
    assert "do-not-record" not in json.dumps(audit.export())
