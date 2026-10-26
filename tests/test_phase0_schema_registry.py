"""Versioned, bounded Draft 2020-12 tool-schema registry regressions."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from xrfm.agent import AgentPhase, AgentRuntime, ScriptedBackend, ToolExecutor, ToolRegistry, ToolSpec
from xrfm.agent.tools import ToolError


def _echo(arguments: dict[str, Any]) -> dict[str, Any]:
    return arguments


def _spec(name: str = "read") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Read one bounded value.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        permissions=("filesystem.read",),
    )


def _registry(spec: ToolSpec) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(spec, _echo)
    return registry


def test_schema_registry_identity_is_deterministic_and_order_independent() -> None:
    alpha = _spec("alpha")
    beta = _spec("beta")
    first = ToolRegistry()
    second = ToolRegistry()
    first.register(alpha, _echo)
    first.register(beta, _echo)
    second.register(beta, _echo)
    second.register(alpha, _echo)

    assert first.schema_version == second.schema_version
    assert first.schema_version.startswith("sha256:")
    assert len(first.schema_version) == len("sha256:") + 64
    manifest = first.schema_manifest()
    assert manifest["schema_draft"] == "2020-12"
    assert manifest["schema_uri"] == "https://json-schema.org/draft/2020-12/schema"
    assert [tool["name"] for tool in manifest["tools"]] == ["alpha", "beta"]


def test_schema_registry_identity_tracks_schema_and_execution_metadata() -> None:
    base = _spec()
    baseline = _registry(base).schema_version
    changed_schema = replace(
        base,
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string", "maxLength": 100}},
            "additionalProperties": False,
        },
    )
    changed_timeout = replace(base, timeout_seconds=base.timeout_seconds + 1)
    changed_description = replace(base, description="A different provider-facing description.")

    assert _registry(changed_schema).schema_version != baseline
    assert _registry(changed_timeout).schema_version != baseline
    assert _registry(changed_description).schema_version != baseline


def test_schema_registry_manifest_is_bounded() -> None:
    registry = _registry(replace(_spec(), environment={"note": "x" * 100_001}))
    with pytest.raises(ToolError, match="tool registry.*bound"):
        _ = registry.schema_version


def test_runtime_context_emits_schema_registry_identity() -> None:
    registry = ToolRegistry()
    runtime = AgentRuntime(ScriptedBackend(), ToolExecutor(registry))
    state = runtime.run("hello")
    context = next(event for event in state.events if event.phase is AgentPhase.CONTEXT)
    assert context.data["schema_version"] == registry.schema_version
