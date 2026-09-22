"""Small deterministic benchmark harness for the agent runtime."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .backends import ModelBackend
from .execution import ToolExecutor
from .runtime import AgentRuntime


@dataclass(frozen=True)
class AgentBenchmarkCase:
    name: str
    request: str
    expected_tool: str | None
    success: Callable[[str], bool]


@dataclass(frozen=True)
class AgentBenchmarkResult:
    name: str
    passed: bool
    selected_tool: str | None
    steps: int
    failed: bool


def run_benchmark(backend: ModelBackend, executor: ToolExecutor, cases: list[AgentBenchmarkCase]) -> list[AgentBenchmarkResult]:
    results = []
    for case in cases:
        state = AgentRuntime(backend, executor).run(case.request)
        selected = None
        for decision in state.decisions:
            if decision.tool_calls:
                selected = decision.tool_calls[0].name
                break
        answer = state.final_answer or ""
        results.append(AgentBenchmarkResult(case.name, not state.failed and selected == case.expected_tool and case.success(answer), selected, state.step, state.failed))
    return results


def reference_cases() -> list[AgentBenchmarkCase]:
    return [
        AgentBenchmarkCase("largest_file", "Find the largest file in this project and summarize what it is.", "largest_file", lambda text: "largest_file" in text),
        AgentBenchmarkCase("direct_answer", "What can this deterministic demo do?", None, lambda text: bool(text)),
    ]
