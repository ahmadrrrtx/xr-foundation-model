"""Provider tool-call identity and observation-correlation regressions."""

from __future__ import annotations

from typing import Any

import pytest

from xrfm.agent import (
    AgentRuntime,
    BackendExecutor,
    Decision,
    DecisionType,
    RuntimeLimits,
    ScriptedBackend,
    ToolCall,
    ToolExecutor,
    ToolRegistry,
    ToolSpec,
)


def _index_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"index": arguments["index"]}


def _duplicate_call(index: int) -> ToolCall:
    return ToolCall("echo", {"index": index}, call_id="shared-provider-call")


class _DuplicateDecisionBackend(ScriptedBackend):
    def chat(self, messages: list[Any], tools: list[ToolSpec]) -> Decision:
        return Decision(
            DecisionType.TOOL_CALL,
            "Two calls incorrectly share one provider ID.",
            (_duplicate_call(0), _duplicate_call(1)),
        )


class _DuplicateAcrossTurnsBackend(ScriptedBackend):
    def chat(self, messages: list[Any], tools: list[ToolSpec]) -> Decision:
        completed = sum(message.role == "tool" for message in messages)
        if completed >= 2:
            return Decision(DecisionType.ANSWER, "Both calls ran.", answer="completed")
        return Decision(DecisionType.TOOL_CALL, "Reuse one provider ID.", (_duplicate_call(completed),))


def _echo_spec() -> ToolSpec:
    return ToolSpec(
        "echo",
        "Return the supplied index.",
        {
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
            "additionalProperties": False,
        },
    )


def test_decision_rejects_duplicate_provider_call_ids() -> None:
    with pytest.raises(ValueError, match="duplicate tool-call id"):
        BackendExecutor().execute(_DuplicateDecisionBackend(), [], [_echo_spec()], timeout=5)


def test_runtime_rejects_cross_turn_id_reuse_before_second_execution() -> None:
    registry = ToolRegistry()
    registry.register(_echo_spec(), _index_tool)
    state = AgentRuntime(
        _DuplicateAcrossTurnsBackend(),
        ToolExecutor(registry),
        RuntimeLimits(max_steps=3, max_wall_seconds=20, max_tool_calls=2),
    ).run("exercise exact observation correlation")

    assert state.failed
    assert "duplicate tool-call id" in (state.final_answer or "")
    assert len(state.observations) == 1
    assert state.observations[0].call_id == "shared-provider-call"
