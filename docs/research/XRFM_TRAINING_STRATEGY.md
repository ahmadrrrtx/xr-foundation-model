# XRFM training strategy for agentic capability

**Date:** 2026-09-22

## What should be trained vs enforced

| Capability | Strategy | Reason |
|---|---|---|
| Language modeling | Pretraining / continued pretraining | Neural model capability |
| General instruction following | Instruction SFT + preference optimization | Model behavior |
| Tool selection and argument generation | Tool-use SFT on diverse schemas and trajectories | Learned competence, measured by tool benchmarks |
| JSON/schema validity | Constrained decoding + validator at runtime; optionally SFT | Syntax is a safety boundary, not a hope |
| Permission and authorization | Deterministic runtime policy | Never delegate authority to model weights |
| Timeouts, budgets, retries, cancellation | Deterministic runtime | Control flow and safety |
| Verification | Tool-specific validators + structured verification state; SFT can improve proposals | Evidence must be machine-checkable |
| Memory retrieval | Runtime memory interface; train retrieval/reranking only if needed | Keep facts outside weights |

## Staged plan

1. **Baseline:** run the existing XRFM model only as a native research backend; do not claim agent competence.
2. **Continued pretraining:** scale tokenizer/context/data only after data manifests and eval are stable.
3. **Instruction SFT:** high-quality conversations with explicit refusal and uncertainty examples.
4. **Tool-use SFT:** single-call and multi-call examples with correct schemas, invalid arguments, unavailable tools, tool errors, and malicious observations. Include model-specific chat templates.
5. **Trajectory SFT:** bounded ReAct/plan-execute traces with concise plans, decisions, observations, and verification outcomes. Exclude private chain-of-thought as a product requirement.
6. **Preference optimization:** prefer correct tool, least privilege, no unnecessary calls, concise final answer, honest uncertainty, and early termination.
7. **Distillation:** distill capable external tool models into smaller XRFM candidates only after runtime benchmarks show the target behavior.

## Data format

Each example should carry `messages`, `tools`, expected `decision`, expected `tool_calls`, tool observations, final answer, safety labels, provenance, and license. Tool outputs must include adversarial cases and should never be treated as trusted instructions in training.

## Evaluation gates before expensive training

* Tool selection ≥ 90% on a fixed deterministic suite.
* Schema-valid arguments ≥ 98% under runtime validation; native-only rate reported separately.
* No unauthorized high-risk action in adversarial tests.
* Multi-step completion and verification measured on held-out tools.
* Regression on ordinary instruction following and perplexity tracked separately.

These are proposed acceptance gates, not current measurements.
