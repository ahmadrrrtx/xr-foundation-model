"""Run the real XR agent loop against this repository's filesystem."""
from __future__ import annotations

import json
from pathlib import Path

from xrfm.agent import AgentRuntime, PermissionPolicy, SafeFilesystemTools, ScriptedBackend, ToolExecutor, ToolRegistry


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = ToolRegistry()
    SafeFilesystemTools(root).register(registry)
    policy = PermissionPolicy(allowed_permissions={"filesystem.read"})
    executor = ToolExecutor(registry, policy=policy)
    runtime = AgentRuntime(ScriptedBackend(), executor)
    state = runtime.run("Find the largest file in this project and summarize what it is.")
    print("EVENTS")
    for event in state.events:
        print(json.dumps(event.to_dict(), ensure_ascii=False))
    print("FINAL")
    print(state.final_answer)
    print("AUDIT")
    print(json.dumps(executor.audit.export(), indent=2))


if __name__ == "__main__":
    main()
