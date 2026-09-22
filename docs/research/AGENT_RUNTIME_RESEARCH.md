# Agent runtime research

**Date:** 2026-09-22

## Findings

Modern agent systems converge on a bounded perceive → decide → act → observe loop. ReAct is adaptive and useful when the next step depends on tool output; plan-and-execute is useful for stable multi-stage workflows. XR should combine them: produce a short structured plan/decision, execute at most a bounded batch of calls, then re-plan after each observation.

### Message and protocol conventions

Hugging Face tool-use integrations commonly represent an assistant tool call as an `assistant` message with `tool_calls`, and the result as a `tool` message whose content is a string. Chat templates differ across models (ChatML, Hermes, Mistral, Qwen), so model-specific formatting belongs in the backend, not the runtime. MCP standardizes interoperability around JSON-RPC and the primitives tools, resources, and prompts; it intentionally does not define the agent's planning policy.

XR therefore defines a small internal protocol with stable typed records: `Message`, `ToolSpec`, `ToolCall`, `ToolResult`, `Decision`, and `AgentEvent`. An MCP bridge can map MCP tools into `ToolSpec` later without making MCP a runtime dependency.

### Structured output

Reliability order: constrained decoding/schema-guided provider mode, grammar-constrained local decoding, then prompt-only JSON with validation and bounded repair. XR's initial implementation supports parsing plus deterministic JSON-schema checks and explicitly labels that path runtime-enforced, not native tool training.

### Runtime state

The loop is an explicit state machine: REQUEST → CONTEXT → DECIDE → AUTHORIZE → EXECUTE → OBSERVE → VERIFY → COMPLETE/FAILED. Every transition is an event and every tool invocation has a correlation id. Hidden chain-of-thought is not required or persisted. The runtime records concise decision rationale and evidence references, not private scratchpad text.

### Safety

Tool descriptions and tool outputs are untrusted data. Authorization is deterministic and outside the model. Each tool declares permissions, risk, timeout, confirmation, and environment restrictions. Filesystem access is rooted to an approved directory; process execution is not enabled by default. High-risk or mutating calls require explicit confirmation. Tool output is size-limited, redacted for common secret patterns, and wrapped as observation data rather than instructions.

### Failure controls

Hard limits are mandatory: maximum steps, wall-clock budget, output size, retries, repeated-call detection, cancellation, schema validation, unavailable-tool errors, and policy denial. Tool errors become typed observations; they do not grant extra authority. A failed verification ends or triggers a bounded re-plan.

## Ecosystem references

* Hugging Face tool use: https://huggingface.co/docs/transformers/chat_extras
* MCP architecture: https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture
* ReAct / plan-execute patterns: research synthesis used for design; XR does not copy a framework implementation.
* Prompt-injection threat model: https://arxiv.org/html/2601.17548v1

## XR-native decisions

1. Runtime is model-agnostic and does not import `torch` at its boundary.
2. Protocol objects are JSON serializable and versioned.
3. One tool call may be executed per decision by default; parallel execution is an explicit future optimization.
4. Verification is a runtime phase, not a request for hidden reasoning.
5. Tool output never changes permissions or system policy.
