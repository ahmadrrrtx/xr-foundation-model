# XRFM agent-runtime migration guide

## Existing raw model users

No change is required. Continue importing `xrfm.XRFM`, `xrfm.generate`, `xrfm.Trainer`, and the existing tokenizer/model APIs. The new runtime is additive and does not rewrite Transformer, inference, training, or data modules.

## Add the runtime

```python
from pathlib import Path
from xrfm.agent import (
    AgentRuntime, PermissionPolicy, SafeFilesystemTools,
    ScriptedBackend, ToolExecutor, ToolRegistry,
)

registry = ToolRegistry()
SafeFilesystemTools(Path.cwd()).register(registry)
executor = ToolExecutor(
    registry,
    PermissionPolicy(allowed_permissions={"filesystem.read"}),
)
state = AgentRuntime(ScriptedBackend(), executor).run("Find the largest file in this project.")
```

## Use a local/remote model

Replace `ScriptedBackend()` with:

```python
from xrfm.agent import OpenAICompatibleBackend
backend = OpenAICompatibleBackend(
    "http://127.0.0.1:8081",  # llama.cpp / compatible server
    "your-model-id",
)
state = AgentRuntime(backend, executor).run("...")
```

The adapter is runtime-enforced and still validates every call. It does not make a base model natively tool-trained.

## Security changes to account for

* Register only tools that are required.
* Grant exact permission strings; an absent permission is denied.
* Keep project roots explicit; filesystem tools reject `..` escapes.
* Never put API keys in prompts, tool descriptions, or tool outputs.
* Keep `auto_confirm=False` for mutating/high-risk tools.
* Do not add a shell/Python execution tool without a separate sandbox and policy review.
* Treat model output and tool output as untrusted.

## Compatibility

The package remains Python 3.10+ and uses only existing core dependencies for the deterministic runtime. HTTP adapter networking uses the Python standard library. The optional FastAPI app now exposes `/v1/agent/run` and `/v1/agent/stream`; set `XRFM_AGENT_API_KEY` before exposing it beyond a trusted local environment. Full JSON Schema libraries, MCP, and native XRFM backend remain planned extensions.
