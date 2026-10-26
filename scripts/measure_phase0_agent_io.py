#!/usr/bin/env python3
"""Measure Phase 0 large-tree and aggregate agent I/O bounds without network access."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xrfm.agent import (
    AgentRuntime,
    Decision,
    DecisionType,
    RuntimeLimits,
    SafeFilesystemTools,
    ScriptedBackend,
    ToolCall,
    ToolExecutor,
    ToolRegistry,
    ToolSpec,
)
from xrfm.agent.tools import ToolError


def _large_payload(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"index": arguments["index"], "payload": ["x" * 12_000 for _ in range(5)]}


class _TwelveCallBackend(ScriptedBackend):
    def chat(self, messages: list[Any], tools: list[ToolSpec]) -> Decision:
        completed = sum(message.role == "tool" for message in messages)
        if completed >= 12:
            return Decision(DecisionType.ANSWER, "Aggregate measurement complete.", answer="done")
        calls = tuple(
            ToolCall("payload", {"index": completed + offset}, call_id=f"measure-{completed + offset}")
            for offset in range(2)
        )
        return Decision(DecisionType.TOOL_CALL, "Two bounded observations requested.", calls)


def _trace_call(function: Any) -> tuple[Any, dict[str, float | int]]:
    tracemalloc.start()
    tracemalloc.reset_peak()
    started = time.perf_counter()
    try:
        result = function()
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, {
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "current_traced_bytes": current,
        "peak_traced_bytes": peak,
    }


def measure_tree(scratch_root: Path | None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="xrfm-agt003-tree-", dir=scratch_root) as temporary:
        root = Path(temporary)
        for directory_index in range(999):
            directory = root / f"directory-{directory_index:04d}"
            directory.mkdir()
            for file_index in range(10):
                (directory / f"file-{file_index:02d}.txt").touch()
        for file_index in range(10):
            (root / f"root-{file_index:02d}.txt").touch()
        (root / "root-00.txt").write_bytes(b"x")

        tools = SafeFilesystemTools(root)
        largest, largest_metrics = _trace_call(lambda: tools.largest_file({"relative": "."}))
        assert largest["relative"] == "root-00.txt"
        listing, listing_metrics = _trace_call(lambda: tools.list_files({"relative": ".", "max_items": 500}))
        assert listing["count"] == 500 and listing["truncated"] is True
        listing_metrics["serialized_output_bytes"] = len(
            json.dumps(listing, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )

        (root / "overflow.txt").touch()
        try:
            tools.largest_file({"relative": "."})
        except ToolError as exc:
            file_bound_error = str(exc)
        else:
            raise AssertionError("10,001 files did not fail closed")
        assert "file scan" in file_bound_error

        (root / "overflow.txt").unlink()
        (root / "overflow-directory").mkdir()
        try:
            tools.largest_file({"relative": "."})
        except ToolError as exc:
            directory_bound_error = str(exc)
        else:
            raise AssertionError("1,001 directories did not fail closed")
        assert "directory scan" in directory_bound_error

        return {
            "regular_files_at_limit": 10_000,
            "directories_including_root_at_limit": 1_000,
            "largest_file": largest,
            "largest_file_metrics": largest_metrics,
            "bounded_listing_metrics": listing_metrics,
            "file_10001_error": file_bound_error,
            "directory_1001_error": directory_bound_error,
        }


def measure_aggregate() -> dict[str, Any]:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "payload",
            "Return a bounded measurement payload.",
            {
                "type": "object",
                "properties": {"index": {"type": "integer"}},
                "required": ["index"],
                "additionalProperties": False,
            },
        ),
        _large_payload,
    )
    executor = ToolExecutor(registry)
    limits = RuntimeLimits(max_steps=7, max_wall_seconds=120, max_tool_calls=12)
    runtime = AgentRuntime(_TwelveCallBackend(), executor, limits)
    state, metrics = _trace_call(lambda: runtime.run("measure aggregate bounded observations"))
    assert not state.failed and state.final_answer == "done"
    assert len(state.observations) == limits.max_tool_calls

    observation_bytes = sum(len(result.as_observation().encode("utf-8")) for result in state.observations)
    message_bytes = sum(len(message.content.encode("utf-8")) for message in state.messages)
    envelope_allowance = 1_024
    aggregate_bound = limits.max_tool_calls * (executor.limits.max_bytes + envelope_allowance)
    event_bound = 2 + 4 * limits.max_steps + 3 * limits.max_tool_calls
    assert 0 < observation_bytes <= aggregate_bound
    assert message_bytes <= observation_bytes + len(state.request.encode("utf-8"))
    assert len(state.events) <= event_bound
    return {
        **metrics,
        "tool_observations": len(state.observations),
        "decisions": len(state.decisions),
        "events": len(state.events),
        "event_bound": event_bound,
        "serialized_observation_bytes": observation_bytes,
        "serialized_message_content_bytes": message_bytes,
        "aggregate_observation_bound_bytes": aggregate_bound,
        "per_tool_output_bound_bytes": executor.limits.max_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    if args.scratch_root is not None:
        args.scratch_root.mkdir(parents=True, exist_ok=True)

    report = {
        "status": "pass",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scope": "CPU parent-process tracemalloc and serialized state; not RSS, GPU, or provider evidence",
        "large_tree": measure_tree(args.scratch_root),
        "aggregate_runtime": measure_aggregate(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
