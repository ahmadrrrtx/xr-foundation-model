"""Large-tree and aggregate-I/O boundary regressions for Phase 0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from xrfm.agent import (
    AgentRuntime,
    BackendExecutor,
    BackendTimeoutError,
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
)
from xrfm.agent.tools import JSONLimits, ToolError


def _payload_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"index": arguments["index"], "payload": ["x" * 1_000 for _ in range(5)]}


class _PairThenAnswerBackend(ScriptedBackend):
    def chat(self, messages: list[Any], tools: list[ToolSpec]) -> Decision:
        completed = sum(message.role == "tool" for message in messages)
        if completed >= 4:
            return Decision(DecisionType.ANSWER, "Aggregate observation is complete.", answer="done")
        calls = tuple(
            ToolCall("payload", {"index": completed + offset}, call_id=f"call-{completed + offset}")
            for offset in range(2)
        )
        return Decision(DecisionType.TOOL_CALL, "Two bounded observations requested.", calls)


def test_directory_bound_is_enforced_during_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tools = SafeFilesystemTools(tmp_path)
    monkeypatch.setattr(tools, "MAX_SCAN_DIRECTORIES", 3)
    consumed = 0

    class DirectoryEntry:
        def __init__(self, index: int) -> None:
            self.path = str(tmp_path / f"directory-{index}")

        def is_dir(self, follow_symlinks: bool = False) -> bool:
            return True

        def is_file(self, follow_symlinks: bool = False) -> bool:
            return False

    def entries() -> Any:
        nonlocal consumed
        for index in range(100):
            consumed += 1
            yield DirectoryEntry(index)

    class ScanContext:
        def __init__(self) -> None:
            self.iterator = entries()

        def __iter__(self) -> Any:
            return self

        def __next__(self) -> Any:
            return next(self.iterator)

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            self.iterator.close()

    monkeypatch.setattr("xrfm.agent.tools.os.scandir", lambda path: ScanContext())
    with pytest.raises(ToolError, match="directory scan"):
        tools.list_files({"relative": ".", "max_items": 10})

    # The root consumes one slot; two children fill the remaining slots. The
    # third child must fail immediately instead of enqueueing the whole fanout.
    assert consumed == 3


def test_execution_resource_limits_reject_non_finite_and_wrong_types() -> None:
    invalid_limits = (
        lambda: JSONLimits(max_bytes=True),
        lambda: JSONLimits(max_bytes=0),
        lambda: JSONLimits(max_items=1.5),
        lambda: JSONLimits(max_depth=-1),
        lambda: JSONLimits(max_string_bytes=float("nan")),
        lambda: RuntimeLimits(max_steps=True),
        lambda: RuntimeLimits(max_tool_calls=1.5),
        lambda: RuntimeLimits(max_wall_seconds=float("nan")),
        lambda: RuntimeLimits(max_wall_seconds=float("inf")),
    )
    for factory in invalid_limits:
        with pytest.raises(ValueError):
            factory()

    registry = ToolRegistry()
    for timeout in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="finite"):
            registry.register(
                ToolSpec(f"bad-{str(timeout)}", "invalid timeout", {"type": "object"}, timeout_seconds=timeout),
                _payload_tool,
            )
    with pytest.raises(ValueError, match="finite"):
        OpenAICompatibleBackend("https://provider.invalid", "test", timeout=float("nan"))

    for timeout in (float("nan"), float("inf")):
        with pytest.raises(BackendTimeoutError, match="finite"):
            BackendExecutor().execute(ScriptedBackend(), [], [], timeout=timeout)


def test_aggregate_observations_stay_within_composed_bounds() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "payload",
            "Return a bounded test payload.",
            {
                "type": "object",
                "properties": {"index": {"type": "integer"}},
                "required": ["index"],
                "additionalProperties": False,
            },
        ),
        _payload_tool,
    )
    json_limits = JSONLimits(max_bytes=16_000, max_items=100, max_depth=8, max_string_bytes=2_000)
    runtime_limits = RuntimeLimits(max_steps=3, max_wall_seconds=30, max_tool_calls=4)
    state = AgentRuntime(
        _PairThenAnswerBackend(),
        ToolExecutor(registry, limits=json_limits),
        runtime_limits,
    ).run("aggregate bounded observations")

    assert not state.failed
    assert len(state.observations) == runtime_limits.max_tool_calls
    observation_bytes = sum(len(result.as_observation().encode("utf-8")) for result in state.observations)
    message_bytes = sum(len(message.content.encode("utf-8")) for message in state.messages)
    envelope_allowance = 1_024
    assert 0 < observation_bytes <= runtime_limits.max_tool_calls * (json_limits.max_bytes + envelope_allowance)
    assert message_bytes <= observation_bytes + len(state.request.encode("utf-8"))
    assert len(state.events) <= 2 + 4 * runtime_limits.max_steps + 3 * runtime_limits.max_tool_calls
    tool_messages = [message for message in state.messages if message.role == "tool"]
    assert json.loads(tool_messages[-2].content)["output"]["index"] == "2"
