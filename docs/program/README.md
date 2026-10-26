# XRFM Native Urdu–English Agent Program

**Audit date:** 2026-10-22  
**Repository audited:** `ahmadrrrtx/xr-foundation-model`  
**Audited revision:** `b8bea87683ef3ae1c839bf1891b7cc882273de83` (`main`, reconfirmed on the audit date)  
**Program state:** **NO-GO for GPU training, public agent exposure, or a capability release**

This directory is the implementation-control package for a sustainable, open-source, native Urdu–English agentic LLM supporting Urdu script, English, Roman Urdu, and code-switching. It is based on source inspection, executable probes, the full test suite, and primary-source research—not on repository claims alone.

**Canonical location:** the repository copy at `docs/program/` is now canonical for implementation status and evidence. The earlier `/home/user/xrfm-urdu-agent-plan` directory and Google Drive files are dated planning snapshots, not live status sources.

## Read in this order

1. [`01_DEEP_AUDIT.md`](01_DEEP_AUDIT.md) — verified findings, reproductions, research implications, and release blockers.
2. [`02_IMPLEMENTATION_PLAN.md`](02_IMPLEMENTATION_PLAN.md) — target architecture, phases, dependencies, gates, compute strategy, and release criteria.
3. [`03_EXECUTION_LEDGER.md`](03_EXECUTION_LEDGER.md) — the **only canonical source for task and gate status**.
4. [`04_DECISION_RISK_REGISTER.md`](04_DECISION_RISK_REGISTER.md) — binding plan decisions, change control, owners, and risks.
5. [`05_TOOL_DATA_EVAL_MATRIX.md`](05_TOOL_DATA_EVAL_MATRIX.md) — researched tools, data/evaluation candidates, licenses, and source links.
6. [`templates/EXPERIMENT_RECORD.md`](templates/EXPERIMENT_RECORD.md) and [`templates/DATA_SOURCE_REVIEW.md`](templates/DATA_SOURCE_REVIEW.md) — mandatory records for experiments and data admission.
7. [`evidence/audit_summary.json`](evidence/audit_summary.json) — compact machine-readable audit evidence.

## Current position

- **What exists:** a promising small decoder-only LM skeleton, tokenizer/data/agent/training modules, 505 passing tests, and a deterministic scripted agent demo.
- **What does not exist:** a usable current pretrained checkpoint, demonstrated native Urdu/Roman-Urdu capability, execution-verified agentic behavior from XRFM weights, a leakage-safe corpus, or a safe public agent runtime.
- **Readiness for the stated ambition:** approximately **2/10**. This is a research/engineering starting point, not a nearly finished model.
- **Immediate next work:** assign owners, then execute `TRN-001`, `AGT-001`–`AGT-004`, and `API-001` from the ledger. Do not start a large training run.

## Differentiation that is still defensible

UrduLM, Alif, Qalb, UrduLLaMA, UrduMMLU, UrduBench, UrBLiMP, and Roman-Urdu resources already exist. The project should therefore not claim novelty merely for training or evaluating an Urdu model. Its defensible contribution is the combination of:

1. **Native bilingual weights trained from random initialization**, with continued-pretrained systems used only as clearly labeled baselines/teachers.
2. **Four first-class language surfaces:** Urdu in Perso-Arabic script, English, natural Roman Urdu, and Urdu–English code-switching.
3. **Verifiable agency:** schema-valid calls are necessary but insufficient; success is graded by final environment state, policy compliance, recovery, and unauthorized-action prevention.
4. **Provenance and reproducibility:** source-level manifests, data/license decisions, contamination records, exact-resume evidence, pinned evaluations, and native-speaker review.
5. **Small, deployable systems:** measured quality per parameter/token/watt/dollar rather than an unbounded parameter-count race.

## No-drift protocol

- The **plan owns scope**, the **ledger owns status**, and the **decision register owns consequential choices**. Do not copy changing status into other documents.
- Only one work package may be `IN_PROGRESS` per owner. A task is `DONE` only when its evidence link, tests, and reviewer are recorded.
- A phase cannot start because time has passed; it starts only when its dependency gate is `PASS` in the ledger.
- Any change to model scale, data mixture, native-from-scratch commitment, tool permissions, benchmark set, external teacher, or release claim requires a new decision record **before implementation**.
- Never mark a model “native” if its weights were initialized from another pretrained model. Never call the scripted backend an LLM result.
- No training beyond a smoke test without exact-resume, dtype, OOM, data-manifest, and budget evidence.
- No public agent endpoint without fail-closed authentication, per-request isolation, strict schemas, bounded I/O, cancellation, injection tests, and audit review.
- No “SOTA,” “production-ready,” “safe,” “best,” or “open data” claim without the referenced evaluation and license evidence.
- Benchmark test sets and their near-duplicates are never training data. Tokenizer fitting also excludes validation/test documents.
- Keep secrets out of chat, Git, Drive, notebooks, manifests, and experiment logs. Kaggle credentials are not required for the current phase.

## Storage and collaboration boundaries

- **Git:** source, tests, configs, schemas, small manifests, this control package.
- **Google Drive:** private copies of plans, review reports, signed gate records, and archives; not active training I/O and not a secret store.
- **Kaggle:** disposable GPU smoke tests and small gated experiments after hardware discovery; not assumed to sustain a large from-scratch run.
- **Artifact hub (recommended Hugging Face):** versioned tokenizers, safetensors weights, model/data cards, and evaluation bundles after their gates.

## Audit integrity

The audit clone had no tracked source modifications. Tests and probes created only local ignored artifacts. The plan does not represent fixes as already implemented; every remediation remains explicitly pending in the ledger.
