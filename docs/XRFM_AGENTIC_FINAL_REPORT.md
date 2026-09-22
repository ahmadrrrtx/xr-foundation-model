# XRFM Agentic Final Report

**Date:** 2026-09-22 · **Release:** additive agent runtime v0.1

## 1. What exactly is XRFM now?

XRFM is two deliberately separated layers:

1. **XRFM neural model:** a from-scratch, decoder-only PyTorch Transformer with RoPE, RMSNorm, SwiGLU, pre-norm residuals, weight tying, byte-level BPE, training/checkpointing, KV-cached inference, and evaluation. It is a small research LM, not a frontier foundation model.
2. **XR Intelligence Runtime:** a model-neutral execution system that turns a request into structured decisions, validates and authorizes tool calls, executes approved tools, treats outputs as untrusted observations, verifies bounded progress, and produces a final answer.

## 2. What is trained and what runs on an existing model?

The existing XRFM checkpoint/model track is trained as ordinary language modeling on a small public-domain corpus. It has no demonstrated instruction-following or native tool-use training. The first capable runtime deployments should use an existing instruct model through `OpenAICompatibleBackend` (llama.cpp, Ollama, vLLM, or a BYOK endpoint). That is explicitly runtime integration, not XR claiming to have trained the external model.

The deterministic `ScriptedBackend` is included for reproducible integration/security testing and is not an LLM.

## 3. How reasoning and decision making work

The runtime does not require or expose private chain-of-thought. A backend returns a structured `Decision`: ANSWER, TOOL_CALL, ASK_USER, REFUSE, or COMPLETE, with concise rationale and typed tool calls. The runtime controls state transitions and records events, decisions, observations, and audit metadata.

## 4. How tools are selected and executed

The model sees `ToolSpec` records with name, description, JSON-schema input, output metadata, permissions, risk, timeout, confirmation, and environment restrictions. The model proposes a `ToolCall`; XR then performs deterministic schema validation, permission authorization, optional user confirmation, bounded execution, sanitization, and typed observation. Model-generated Python or shell is never executed.

The shipped `SafeFilesystemTools` provides root-confined, read-only `list_files`, `largest_file`, and `read_file` tools. Path traversal is rejected. There is no arbitrary process tool.

## 5. How verification and safety work

The state machine is:

`REQUEST → CONTEXT → DECIDE → AUTHORIZE → EXECUTE → OBSERVE → VERIFY → COMPLETE/FAILED`.

The runtime enforces max steps, wall time, tool-call budget, duplicate-call detection, schema checks, timeouts, policy denial, output bounds, and audit logging. Tool outputs are untrusted data, are size-bounded/redacted, and cannot modify policy. High-risk or mutating tools require confirmation by default. Persistent memory is not silently enabled.

## 6. Can it run locally and with different models?

Yes. The deterministic demo runs with only the XRFM package and local filesystem. The external adapter speaks a Chat Completions-compatible HTTP protocol suitable for llama.cpp, Ollama, vLLM, or compatible hosted endpoints. A future native adapter can wrap the existing `GenerationEngine`; the runtime does not need to change.

A real demo was executed in this audit: the runtime selected `largest_file`, authorized `filesystem.read`, inspected the repository, observed `checkpoints/checkpoint_step_500.pt` as the largest regular file, and returned that observation. The demo used real filesystem state, not a fake output.

## 7. What was implemented and tested?

Implemented:

* `xrfm.agent.protocols`: typed XR protocol/state/event records.
* `xrfm.agent.backends`: backend abstraction, deterministic demo backend, OpenAI-compatible adapter.
* `xrfm.agent.tools`: registry, dependency-free JSON-schema subset validator, safe root-confined filesystem tools.
* `xrfm.agent.safety`: permissions, confirmation policy, audit log, bounded observation sanitization.
* `xrfm.agent.execution`: validation → authorization → timeout-bounded execution → typed result.
* `xrfm.agent.runtime`: bounded event-emitting state machine.
* `xrfm.agent.evaluation`: initial benchmark case/result harness.
* `scripts/run_agent_demo.py` and `tests/test_agent_runtime.py`.
* research, architecture, implementation-plan, training-strategy, audit, and README documents.

Measured in this session:

* Existing repository suite: **492 passed, 1 skipped**.
* New agent tests: **5 passed**.
* End-to-end real filesystem demo: **completed successfully**.
* Local Qwen2.5-0.5B GGUF loaded by llama.cpp server and answered a plain chat request through the adapter. A tool-use request from this very small model produced contradictory JSON; the adapter contained it and did not execute an unauthorized/unknown call. This is evidence of adapter containment, not a claim of tool-call quality.

## 8. Current limitations

* The native XRFM model is not an agent-capable model; no native XRFM backend is yet shipped.
* OpenAI-compatible model output still uses parse-and-validate rather than grammar-constrained decoding. Provider/llama.cpp guided decoding should be added before production use.
* The dependency-free schema validator intentionally supports a subset of JSON Schema; complex schemas need a full validator dependency.
* The runtime's verification is protocol-level, not domain-complete. File creation/report workflows need domain-specific verifiers.
* Persistent memory, MCP transport/bridge, SSE agent API, cancellation tokens, parallel independent tool calls, and sandboxed process execution are planned, not represented as complete capabilities.
* No fabricated agent benchmark score is claimed. The new benchmark harness needs a larger frozen corpus and measured runs across backends/models.

## 9. Migration and roadmap

Existing raw generation users do not need to change anything. Import `xrfm.agent` for the runtime. Register tools explicitly, grant minimal permissions, and choose a backend. Do not pass secrets in prompts or tool descriptions.

Next milestones:

1. agent API endpoint with typed SSE events and authentication;
2. model-specific Qwen/Hermes/Mistral templates and guided decoding;
3. MCP tool discovery bridge with provenance and per-server policy;
4. larger benchmark suite, adversarial injection cases, latency/cost reporting;
5. domain verifiers and persistent provenance-aware memory;
6. tool-use SFT for an XRFM native model only after evaluation demonstrates it is necessary.

## Bottom line

XRFM now has an actual executable, bounded agent loop while remaining scientifically honest: the runtime architecture and safety controls are XR's implementation; capable language behavior comes from an external model until an XRFM tool-use model is trained and measured. The existing Transformer is preserved as a first-class research path rather than mislabeled as a capable agent brain.
