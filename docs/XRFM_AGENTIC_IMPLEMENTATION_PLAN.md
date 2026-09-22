# XRFM agentic implementation plan

## Phase A — contracts and audit (complete)

* Objective: understand current XRFM and freeze neural model as control path.
* Files: `docs/research/XRFM_CURRENT_STATE_AUDIT.md`.
* Acceptance: source-verified execution trace; existing tests remain green. **Met: 492 passed, 1 skipped.**

## Phase B — minimal safe runtime (complete)

* Objective: implement model-neutral protocol, tool registry, validation, policy, executor, bounded loop.
* Files: `src/xrfm/agent/{protocols,tools,safety,execution,backends,runtime}.py`.
* Tests: protocol serialization, schema errors, path traversal, denial, timeout, repeated-call guard, end-to-end real filesystem inspection.
* Rollback: remove `xrfm.agent`; neural packages are untouched.
* Acceptance: no arbitrary code execution; actual tool runs; typed events; all legacy tests green.

## Phase C — external model adapter (complete in initial form)

* Objective: connect llama.cpp/Ollama/vLLM/BYOK through one HTTP adapter.
* Files: `backends.py`, model ecosystem docs.
* Acceptance: Chat Completions response can be parsed into XR decisions; all calls still validate and authorize.
* Next: model-specific chat templates and strict guided decoding.

## Phase D — server and observability (complete in initial form)

* Objective: add `/v1/agent/run` and SSE typed events without changing raw completions.
* Dependencies: FastAPI optional extra.
* Implemented: `api/routes/agent.py`, optional `XRFM_AGENT_API_KEY`, `XRFM_AGENT_ROOT`, OpenAI-compatible environment configuration.
* Tests: request validation, API-key denial/acceptance, event ordering, contained tool failure. **Met: 8 API tests passed.**
* Remaining: production authentication provider, cancellation propagation into model calls, metrics export.
* Rollback: endpoint is additive.

## Phase E — evaluation and training data

* Objective: add benchmark JSONL and measured metrics for tool selection, argument correctness, execution, recovery, termination, latency, token/cost budgets, and injection resistance.
* Acceptance: no fabricated results; reports include environment/model/backend/version.

## Phase F — native XRFM tool-use research

* Objective: tool-use SFT and structured decoding only after Phase E demonstrates a gap.
* Acceptance: native-only tool metrics separated from runtime-enforced metrics; trained checkpoints and provenance released separately.
