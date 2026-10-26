"""Run the explicitly local-only deterministic XR agent demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xrfm.agent import AgentRuntime, PermissionPolicy, SafeFilesystemTools, ScriptedBackend, ToolExecutor, ToolRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-demo", action="store_true", help="acknowledge deterministic local-only demo mode")
    args = parser.parse_args()
    if not args.local_demo:
        parser.error("this scripted backend requires explicit --local-demo; it is not a model-quality demonstration")

    root = Path(__file__).resolve().parents[1]
    registry = ToolRegistry()
    SafeFilesystemTools(root).register(registry)
    policy = PermissionPolicy(allowed_permissions={"filesystem.read"})
    executor = ToolExecutor(registry, policy=policy)
    runtime = AgentRuntime(ScriptedBackend(), executor)
    print("LOCAL-ONLY DETERMINISTIC DEMO: no trained agent or Urdu-capability claim")
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
