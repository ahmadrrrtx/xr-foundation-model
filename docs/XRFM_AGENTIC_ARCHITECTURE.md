# XRFM Agentic Architecture

**Status:** v0.1 implemented on 2026-09-22.

## Truthful product boundary

```text
XRFM neural model (small from-scratch Transformer, research track)
                         │ ModelBackend interface
                         ▼
XR Intelligence Runtime
  context → structured decision → authorization → tool execution
  → untrusted observation → verification → next decision/final answer
```

The runtime is not the neural model. It can run the native XRFM model, an external local model, or a remote/BYOK OpenAI-compatible endpoint. No external weights are bundled.

## Components

* **Protocol:** JSON-serializable `Message`, `ToolSpec`, `ToolCall`, `ToolResult`, `Decision`, and `AgentEvent` records.
* **Backend:** `ModelBackend.chat(messages, tools) -> Decision`. Backends translate model-specific templates and outputs into the XR protocol.
* **Decision engine:** the backend produces one of ANSWER, TOOL_CALL, ASK_USER, REFUSE, or COMPLETE. Rationale is concise structured metadata; private chain-of-thought is neither required nor stored.
* **Tool registry:** names, descriptions, JSON-schema subset, permissions, risk, timeout, confirmation, and environment policy.
* **Executor:** validates schema → checks deterministic policy → optional confirmation → bounded execution → sanitizes output → typed result.
* **Runtime:** explicit bounded state machine with max steps, wall time, tool-call budget, duplicate-call detection, cancellation-compatible event stream, and typed failures.
* **Safety:** least-privilege permission sets, root-confined filesystem tools, no arbitrary code execution, no process tool by default, audit records, secret redaction, output bounds. Tool output is always untrusted data.
* **Memory:** current v0.1 keeps session context in `AgentState`; persistent memory is deliberately not silently enabled. A future memory adapter must preserve provenance and trust labels.

## State machine

`REQUEST → CONTEXT → DECIDE → AUTHORIZE → EXECUTE → OBSERVE → VERIFY → COMPLETE | FAILED`

Every transition emits an `AgentEvent`. The verifier currently performs protocol-level verification (successful typed observation, bounded output, terminal decision). Domain-specific verifiers are an extension point.

## Deployment modes

1. **No-model deterministic demo:** ScriptedBackend; useful for CI and security testing.
2. **Local model:** OpenAICompatibleBackend pointed at llama.cpp/Ollama/vLLM.
3. **Native XRFM:** future `NativeXRFMBackend` wraps `GenerationEngine`; current neural model is not tool-trained and should only be used for research.
4. **Remote/BYOK:** OpenAI-compatible endpoint with credentials outside prompts and tool output.

## Explicit non-goals

The runtime does not execute model-generated Python or shell commands, infer permissions from natural language, expose private chain-of-thought, or claim native tool-calling ability for a model that was not trained for it.
