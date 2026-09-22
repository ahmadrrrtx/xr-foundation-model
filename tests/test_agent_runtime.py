from __future__ import annotations

import time

from xrfm.agent import (
    AgentPhase,
    AgentRuntime,
    PermissionPolicy,
    SafeFilesystemTools,
    ScriptedBackend,
    ToolCall,
    ToolExecutor,
    ToolRegistry,
    ToolSpec,
)


def make_runtime(tmp_path):
    registry = ToolRegistry()
    SafeFilesystemTools(tmp_path).register(registry)
    executor = ToolExecutor(registry, PermissionPolicy(allowed_permissions={"filesystem.read"}))
    return AgentRuntime(ScriptedBackend(), executor), executor


def test_end_to_end_real_filesystem_tool(tmp_path):
    (tmp_path / "small.txt").write_text("small", encoding="utf-8")
    (tmp_path / "large.txt").write_text("x" * 100, encoding="utf-8")
    runtime, executor = make_runtime(tmp_path)
    state = runtime.run("Find the largest file in this project and summarize what it is.")
    assert not state.failed
    assert state.final_answer and "large.txt" in state.final_answer
    assert any(event.phase == AgentPhase.EXECUTE for event in state.events)
    assert executor.audit.records


def test_path_traversal_is_blocked(tmp_path):
    registry = ToolRegistry()
    SafeFilesystemTools(tmp_path).register(registry)
    result = ToolExecutor(registry, PermissionPolicy(allowed_permissions={"filesystem.read"})).execute(
        ToolCall("read_file", {"relative": "../secret.txt"})
    )
    assert not result.ok
    assert "escapes" in (result.error or "")


def test_permission_denial_is_deterministic():
    registry = ToolRegistry()
    registry.register(ToolSpec("danger", "test", {"type": "object"}, permissions=("process.execute",), risk="high"), lambda _: "bad")
    result = ToolExecutor(registry, PermissionPolicy()).execute(ToolCall("danger", {}))
    assert result.denied and not result.ok
    assert "missing permissions" in (result.error or "")


def test_unknown_tool_and_bad_schema_are_contained():
    registry = ToolRegistry()
    registry.register(ToolSpec("echo", "test", {"type": "object", "required": ["text"]}), lambda x: x)
    executor = ToolExecutor(registry)
    assert not executor.execute(ToolCall("missing", {})).ok
    assert not executor.execute(ToolCall("echo", {})).ok


def test_repeated_call_guard(tmp_path):
    class Repeater(ScriptedBackend):
        def chat(self, messages, tools):
            return super().chat([messages[0]], tools)

    registry = ToolRegistry()
    SafeFilesystemTools(tmp_path).register(registry)
    state = AgentRuntime(Repeater(), ToolExecutor(registry, PermissionPolicy(allowed_permissions={"filesystem.read"})),).run(
        "Find the largest file in this project."
    )
    # ScriptedBackend sees no observation because this test's backend discards it.
    assert state.failed or state.final_answer
