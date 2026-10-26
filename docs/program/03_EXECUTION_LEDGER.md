# Living Execution Ledger

**Canonical status document. The repository copy at `docs/program/` is authoritative.** Last initialized: 2026-10-22; Phase 0 implementation started on local branch `phase-0-stabilization` on 2026-10-22.  
**Overall program:** **NO-GO**.  
**Next action:** publish the validated `AGT-002` schema-registry increment and observe its hosted CI run, then complete `AGT-003` large-tree/aggregate measurements; pinned real-provider disconnect/soak probes, independent review, and CUDA/second-GPU evidence on real target hardware remain mandatory. Do not infer `G1` from CPU-only or hosted-CPU passes.

## Status rules

- `VERIFIED`: audit/research work completed with linked evidence; not an implementation claim.
- `READY`: dependencies are satisfied and work may be claimed.
- `IN_PROGRESS`: one active task per owner; record assignee and start date.
- `BLOCKED`: name the unresolved dependency or external decision.
- `HOLD`: intentionally not scheduled; cannot begin without the stated gate/decision.
- `DONE`: implementation, tests, independent review, and permanent evidence are complete.
- `REJECTED`: evaluated and consciously not adopted; retain the reason.

A task is not `DONE` because code was written. Its evidence must include a commit/artifact hash, test command/result, reviewer, and date. Update this ledger in the same commit as the work. Do not duplicate changing statuses in the plan or README.

## Owners (all human assignees currently UNASSIGNED)

| Code | Accountable role |
|---|---|
| M | Project maintainer / final scope decision |
| CE | Core/API engineer |
| TE | Training/inference engineer |
| DE | Data engineer |
| DS | Data steward / license and consent review |
| EL | Evaluation lead |
| ULL | Urdu language lead |
| RUR | Roman-Urdu/code-switch reviewers |
| AE | Agent/runtime engineer |
| SE | Security reviewer |
| RE | Release/operations engineer |

One person may hold multiple roles, but `REL-001` and `REL-002` require reviewers independent of the corresponding implementation.

## Gate board

| Gate | State | Evidence / blocker | Decision owner |
|---|---|---|---|
| `G0` Audit and program charter | **PASS** | Deep audit, research matrix, plan, and this ledger created at the audited revision | M |
| `G1` Trustworthy small training/local runtime | **FAIL** | Repository-wide local quality and the complete hosted Python 3.10–3.12 CI matrix pass on `f659701`; real-provider soak, independent review, same-hardware CUDA, and second-GPU/DDP evidence remain open | M + TE + SE |
| `G2` Evaluation and admission policy operational | **PENDING** | Requires `G1`; owners, baseline harness, gold protocols, and source review not yet operational | M + EL + DS + ULL |
| `G3` Frozen pilot corpus and tokenizer | **PENDING** | Requires `G2`; no approved corpus or downstream-selected tokenizer | DS + DE + ULL + RUR |
| `G4` Native pilot validates/rejects scaling | **PENDING** | Requires `G3`; no native pilot | TE + EL + M |
| `G5` Useful and bounded bilingual agent | **PENDING** | Requires `G4`; no XRFM-weight tool-use result | AE + EL + SE + ULL + RUR |
| `G6` Reproducible release candidate | **PENDING** | Requires `G5`; no release candidate or independent sign-off | M + RE + SE |

A `FAIL` is a measured state, not a punishment. It prevents dependent work from being misrepresented as ready.

## Work packages

### Governance and audit

| ID | Work package | Owner | Dependencies | Status | Exit evidence / next action |
|---|---|---|---|---|---|
| `GOV-001` | Independent source, artifact, test, API, tokenizer, data, and agent audit | M | None | **VERIFIED** | [`01_DEEP_AUDIT.md`](01_DEEP_AUDIT.md); revision `b8bea876…`; 505 tests; executable probes |
| `GOV-002` | Primary-source Urdu/Roman/agent/toolchain research | EL + DS | `GOV-001` | **VERIFIED** | [`05_TOOL_DATA_EVAL_MATRIX.md`](05_TOOL_DATA_EVAL_MATRIX.md); source/license caveats recorded |
| `GOV-003` | Gated implementation plan and no-drift controls | M | `GOV-001`, `GOV-002` | **VERIFIED** | [`02_IMPLEMENTATION_PLAN.md`](02_IMPLEMENTATION_PLAN.md), this ledger, decision register |
| `GOV-004` | Assign named owners, reviewers, communication path, and escalation SLA | M | `G0` | **READY** | Signed RACI in decision register; name at least one accountable owner per role and native-review capacity |
| `GOV-005` | Branch protection, required checks, issue templates, security disclosure, release ownership | M + RE | `GOV-004` | **BLOCKED** | Repository settings evidence; CI checks required; private vulnerability channel tested |

### Training, inference, API, and packaging — `G1`

| ID | Work package | Owner | Dependencies | Status | Exit evidence / next action |
|---|---|---|---|---|---|
| `TRN-001` | Correct AMP optimizer-step/skip and scheduler/global-step semantics | TE | `G0` | **BLOCKED** | CPU regressions and full suite pass; physical CUDA overflow/successful-step smoke and independent review still required |
| `TRN-002` | Capability-tested FP32/FP16/BF16 policy and hardware report | TE | `TRN-001` | **BLOCKED** | T4/P100 or actual target report: dtype, stability, throughput, memory, PyTorch/driver; no unsupported silent BF16 |
| `TRN-003` | Coordinated OOM abort/retry and DDP fault handling | TE | `TRN-001` | **BLOCKED** | Single-/two-GPU fault injection; no partial gradient reuse, rank divergence, or deadlock |
| `TRN-004` | Atomic exact-resume checkpoints and complete identity/state | TE | `TRN-001`, `TRN-005` | **BLOCKED** | CPU batch/metrics/optimizer/scheduler/RNG/final-weight equivalence and atomic rollback pass; same-hardware CUDA and independent review pending |
| `TRN-005` | Epoch-correct sampler and explicit pad/label/attention contracts | TE | `G0` | **IN_PROGRESS** | `set_epoch`, nonzero pad-ID, ignore-index, variable-length, and full-suite tests pass; independent review pending |
| `TRN-006` | Compose causal/padding masks; fix joint sampling and usage accounting | TE | `G0` | **BLOCKED** | Causal/padding and top-k+top-p regressions remain green. Strict future-token invariance, numerical multi-layer KV-cache parity, cached/uncached greedy equivalence, and deterministic stop/truncation/usage accounting pass in [`TRN-006_contract_after.txt`](evidence/phase0/TRN-006_contract_after.txt), [`TRN-006_focused_after.txt`](evidence/phase0/TRN-006_focused_after.txt), and [`TRN-006_full_after.txt`](evidence/phase0/TRN-006_full_after.txt); independent review remains |
| `TRN-007` | Reproducible quality gates and target-hardware smoke workflow | TE + RE | `TRN-001…006` | **BLOCKED** | Repository-wide Ruff/Ruff-format/Black, mypy (93 files), build, CI smoke, Bandit/pip-audit, and 564 tests pass on local Python 3.10 and 3.13 in [`TRN-007_py310_full_after.txt`](evidence/phase0/TRN-007_py310_full_after.txt) and [`TRN-007_stream_ci_full_after.txt`](evidence/phase0/TRN-007_stream_ci_full_after.txt); earlier exact-commit clean-checkout evidence is retained in [`TRN-007_clean_checkout_ci.txt`](evidence/phase0/TRN-007_clean_checkout_ci.txt). Hosted run 26 on `f659701` passes all eight jobs, including Python 3.10–3.12 ([`TRN-007_hosted_ci_f659701.json`](evidence/phase0/TRN-007_hosted_ci_f659701.json)); GPU workflow, dependency tasks, and independent review remain |
| `API-001` | Single API lifecycle owner, correct liveness/readiness, no stale globals | CE | `G0` | **BLOCKED** | Per-app lifespan/tokenizer/checkpoint/engine state and liveness/readiness/models integration pass real tiny-checkpoint tests, isolated legacy fixtures, clean-checkout validation, and the hosted 3.10–3.12 matrix on `f659701`; independent review remains before `DONE` |
| `API-002` | Completion/model/tokenize error and accounting contracts | CE | `API-001`, `TRN-006` | **BLOCKED** | Accounting/truncation/stop/streaming metadata remain covered in [`API-AGENT_after.txt`](evidence/phase0/API-AGENT_after.txt). Unary-stream 409, tokenizer 400, declared/chunked body 413, per-app rate/concurrency 429 with slot cleanup, sanitized 500, and readiness 503 contracts now pass in [`API-002_contract_after.txt`](evidence/phase0/API-002_contract_after.txt), [`API-002_focused_after.txt`](evidence/phase0/API-002_focused_after.txt), and [`API-002_full_after.txt`](evidence/phase0/API-002_full_after.txt); the hosted 3.10–3.12 matrix passes on `f659701`, and independent review remains |
| `API-003` | Valid-checkpoint end-to-end service integration test | CE | `API-001`, `API-002` | **BLOCKED** | Real startup→readiness→models→completion fixture passes locally with weights-only tiny checkpoint, in the clean-checkout suite, and throughout the hosted 3.10–3.12 matrix on `f659701`; independent review remains |
| `PKG-001` | Preserve XRFM Foundational Model name; prevent unrelated PyPI installation/publication and modernize metadata | M + CE | `G0` | **DONE** | Maintainer selected no rename (`D-018`); source/Docker install instructions retained, unsafe PyPI instruction removed, PEP 639 metadata build passed: [`PKG-001_build.txt`](evidence/phase0/PKG-001_build.txt); external PyPI publication remains blocked |

### Agent containment — `G1`

| ID | Work package | Owner | Dependencies | Status | Exit evidence / next action |
|---|---|---|---|---|---|
| `AGT-001` | Fail-closed deployment auth and per-request runtime/audit isolation | AE + SE | `G0` | **IN_PROGRESS** | Public startup now requires auth, explicit root/provider/capability; scripted mode requires explicit local-demo opt-in and every request builds fresh runtime/audit state. Before/after evidence: [`API-AGENT_regression_before.txt`](evidence/phase0/API-AGENT_regression_before.txt), [`API-AGENT_after.txt`](evidence/phase0/API-AGENT_after.txt); independent security review pending |
| `AGT-002` | Full JSON Schema 2020-12 input/output/decision validation | AE | `G0` | **BLOCKED** | Draft 2020-12 `prefixItems`, nested/input/output validation, and UTF-8 bounds remain green. Canonical bounded schema manifests now have deterministic order-independent SHA-256 identities that track schema and execution metadata; runtime context events emit the identity. Before/after and full-suite evidence is in [`AGT-002_registry_before.txt`](evidence/phase0/AGT-002_registry_before.txt), [`AGT-002_registry_after.txt`](evidence/phase0/AGT-002_registry_after.txt), and [`AGT-002_full_after.txt`](evidence/phase0/AGT-002_full_after.txt); increment-specific hosted CI and independent review remain |
| `AGT-003` | Repair redaction and enforce internal bounded I/O | AE + SE | `AGT-002` | **BLOCKED** | Equals/Bearer/multiline redaction, negative/huge-limit rejection, traversal/sensitive-canary denial, streamed scan caps, and bounded reads pass; large-tree memory/aggregate measurements and review remain |
| `AGT-004` | Hard deadlines, cancellation, disconnects, and genuine streaming | AE | `AGT-001` | **BLOCKED** | Parent-enforced spawn isolation now covers custom/external backend capability inspection and chat as well as tools; blocked-backend wall deadline, external cancellation, started-child PID kill, strict decision validation, disconnect propagation, and genuine streaming pass in [`AGT-004_backend_deadline_after.txt`](evidence/phase0/AGT-004_backend_deadline_after.txt). Baseline [`AGT-004_backend_deadline_before.txt`](evidence/phase0/AGT-004_backend_deadline_before.txt) took 2.0003s under a 0.5s budget; pinned real-provider disconnect/soak and independent review remain |
| `AGT-005` | Honest backend capability/schema/tool-call protocol | AE | `AGT-002` | **BLOCKED** | False-by-default capabilities, explicit startup declaration, native function schemas/history, JSON response format, provider ID preservation, and HTTP(S)-only URLs have local protocol tests; pinned real-provider capability probes/review remain |
| `AGT-006` | Sandboxed tools, audit retention/redaction, accurate timing, kill switch | AE + SE | `AGT-001…005` | **BLOCKED** | Process isolation/PID kill, bounded redacted per-request audit, retention caps, and measured `duration_ms` are covered. Authenticated per-application kill/resume/status and permission/tool revocation now reject new work, cancel registered in-flight backend execution, and constrain fresh policies; before/after evidence is in [`AGT-006_control_before.txt`](evidence/phase0/AGT-006_control_before.txt) and [`AGT-006_control_after.txt`](evidence/phase0/AGT-006_control_after.txt). Independent escape review remains |

### Evaluation and agent benchmarks — `G2` and release evaluation

| ID | Work package | Owner | Dependencies | Status | Exit evidence / next action |
|---|---|---|---|---|---|
| `EVAL-001` | Versioned benchmark registry and sealed-set access policy | EL + DS | `G1` | **BLOCKED** | Revision/license/split/prompt/metric/chance records; all core splits excluded from training |
| `EVAL-002` | Pinned XRFM LM-evaluation adapter and reproducible bundles | EL + TE | `EVAL-001`, `API-003` | **BLOCKED** | Per-sample outputs, config/hashes, CIs; dummy model fixtures validate scoring |
| `EVAL-003` | Rerun size-matched and Urdu-specific baselines under one protocol | EL | `EVAL-002` | **BLOCKED** | Qwen/OLMo plus available Urdu baseline bundles; no copied vendor scores |
| `EVAL-004` | Native four-surface gold suite and blind human-review protocol | ULL + RUR + EL | `EVAL-001`, `GOV-004` | **BLOCKED** | Double annotation/adjudication, agreement report, guidelines, sealed dev/test split |
| `EVAL-005` | Exact/near/semantic contamination detection and exclusion manifests | EL + DE | `EVAL-001` | **BLOCKED** | Known-positive/negative fixtures; benchmark and sealed-task exclusions reproducible |
| `AGTE-001` | Final-state tool simulator and native `XRFM-UrduAgentBench` | AE + EL + ULL + RUR | `EVAL-001`, `EVAL-004` | **BLOCKED** | Matched tasks across four surfaces; final-state/policy/no-tool/clarification/error-recovery grades |
| `AGTE-002` | BFCL-style structured-call evaluation adapter | EL + AE | `AGTE-001`, `AGT-005` | **BLOCKED** | AST/schema/execution metrics separated; pinned benchmark revision and contamination check |
| `AGTE-003` | AgentDojo-style injection, canary, and RUAB-equivalent safety suite | SE + EL + RUR | `AGTE-001`, `AGT-006` | **BLOCKED** | Utility/attack-success/exfiltration by surface; known attacks caught and false positives measured |
| `AGTE-004` | Full agent utility: pass@1, pass^4, policy, recovery, language gaps | EL | `SFT-003`, `AGTE-001…003` | **BLOCKED** | Frozen suite with ≥200 tasks/surface or justified CI; base/no-tool/scripted-oracle comparisons |
| `AGTE-005` | Blind native-speaker agent conversation review | ULL + RUR | `AGTE-004` | **BLOCKED** | Adequacy/naturalness/cultural/task rubric, agreement, severity-1 issue disposition |
| `AGTE-006` | Release security and safety-transfer evaluation | SE + EL | `AGTE-004`, `SERV-001` | **BLOCKED** | Zero unauthorized mutations/canary exfiltration in release suite; residual ASR and RUAB results published |

### Data and tokenizer — `G2`/`G3`

| ID | Work package | Owner | Dependencies | Status | Exit evidence / next action |
|---|---|---|---|---|---|
| `DATA-001` | Ratify Tier A/B/C source, attribution, share-alike, teacher, and withdrawal policy | DS + M | `G1` | **BLOCKED** | Signed policy and completed template; legal specialist consulted where needed |
| `DATA-002` | Source inventory, original-license resolution, consent-based native collection | DS + ULL + RUR | `DATA-001` | **BLOCKED** | Every candidate has provenance/terms decision; conflicting mirrors resolved or excluded; consent records secured |
| `DATA-003` | Streaming DataTrove-style ingestion/normalization/quality/PII pipeline | DE | `DATA-002` | **BLOCKED** | Immutable raw refs, raw/normalized views, source hashes, processing reports, restartability |
| `DATA-004` | Human-validated Urdu/English/Roman/mixed language-script classifier | DE + ULL + RUR | `EVAL-004`, `DATA-003` | **BLOCKED** | Per-class precision/recall and confidence routing; GlotLID/script baseline comparison |
| `DATA-005` | Scalable exact+LSH dedup and group/source-aware splits before tokenizer fitting | DE | `DATA-003`, `EVAL-005` | **BLOCKED** | Known-cluster fixtures, scale benchmark, zero prohibited cross-split clusters/tokenizer exposure |
| `DATA-006` | Corpus v0 stratified native review, statistics, and signed freeze | DS + ULL + RUR | `DATA-001…005` | **BLOCKED** | No unknown admitted licenses; review sample and issue log; frozen manifest/card/hashes |
| `TOK-001` | Training-only BPE/unigram 24K/32K/48K tournament | TE + ULL + RUR | `DATA-005` | **BLOCKED** | Conformance plus fertility/fallback/throughput by surface/domain; current tokenizer control |
| `TOK-002` | Downstream finalist selection and HF export/conformance | TE + EL | `TOK-001`, `BASE-001` | **BLOCKED** | Identical small-LM comparison, chosen-tokenizer decision, round-trip and serving parity |

### Native base model — `G4`

| ID | Work package | Owner | Dependencies | Status | Exit evidence / next action |
|---|---|---|---|---|---|
| `BASE-001` | S0 conformance: overfit, NaN/gradient, resume, HF export, deterministic generation | TE | `G1` | **BLOCKED** | Repeatable tiny-run bundle and reference logits/generation |
| `BASE-002` | S1 equal-compute tokenizer/mixture proxy arms | TE + EL | `G3`, `BASE-001` | **BLOCKED** | A/B/C equal-compute results, per-surface bpb and downstream diagnostics, selected recipe |
| `BASE-003` | S2 50–100M random-init native pilot, two seeds and learning-curve checkpoints | TE + EL | `BASE-002`, `OPS-001` | **BLOCKED** | 25/50/100% checkpoints, full eval/human error analysis, compute/data scaling projection |
| `BASE-004` | OLMo-core versus TorchTitan scaling bake-off at fixed small budget | TE | `BASE-003` | **BLOCKED** | Same model/data/tokens; correctness/resume/throughput/memory/export/maintenance comparison |
| `BASE-005` | Conditional S3 300M, then separately approved S4 0.5–1B | TE + M | `G4`, `BASE-004`, new budget decision | **HOLD** | Cannot start without measured scaling slope, clean unique-data budget, and compute approval |

### Agentic post-training — `G5`

| ID | Work package | Owner | Dependencies | Status | Exit evidence / next action |
|---|---|---|---|---|---|
| `SFT-001` | Canonical versioned chat/decision template and loss masks | AE + TE | `G4`, `AGT-005` | **BLOCKED** | Train/serve/eval token-for-token parity; tool/system content not assistant-target loss |
| `SFT-002` | Reviewed four-surface SFT and verifiable tool trajectory data | ULL + RUR + AE + DS | `SFT-001`, `AGTE-001…003` | **BLOCKED** | Provenance/teacher labels, human sample review, hard negatives, no benchmark leakage |
| `SFT-003` | TRL SFT baseline and constrained-decoding integration | TE + AE | `SFT-002`, `SERV-001` | **BLOCKED** | Base-vs-SFT utility/schema/language results; invalid calls always blocked |
| `SFT-004` | Optional reviewed DPO or RLVR experiment | TE + EL + M | `SFT-003`, new decision | **HOLD** | Only with preference/verifier validation and reward-hacking/false-accept analysis |

### Serving, operations, and release — `G5`/`G6`

| ID | Work package | Owner | Dependencies | Status | Exit evidence / next action |
|---|---|---|---|---|---|
| `SERV-001` | HF reference and vLLM/XGrammar compatibility/performance | TE + AE | `G4`, `TRN-007` | **BLOCKED** | Reference-vs-server logits/generation, schema constraints, throughput/latency/memory report |
| `SERV-002` | Production API/network/tool hardening and privacy-preserving observability | AE + SE | `G5` | **BLOCKED** | AuthZ, SSRF allowlists, sandboxes, request/rate limits, redacted metrics, kill-switch tests |
| `SERV-003` | Load, soak, cancellation, upgrade, and rollback verification | RE + SE | `SERV-002` | **BLOCKED** | Load/latency/error budget report; rollback to last verified model/runtime rehearsed |
| `SERV-004` | Optional GGUF/llama.cpp edge export and parity | TE + RE | `SERV-001` | **HOLD** | Separate quantization quality and output-parity report; not required for v1 server release |
| `OPS-001` | Immutable experiment/artifact records and durable checkpoint archive | RE + TE | `G3` | **BLOCKED** | Experiment template, JSONL, hashes, configs, environment, budget, recovery links for every run |
| `OPS-002` | Model/data cards, evaluation bundle, licenses/attribution, negative results | RE + DS + EL | `G5` | **BLOCKED** | Complete public review draft with per-surface results and limitations |
| `OPS-003` | Monitoring, vulnerability response, drift reports, and incident runbook | RE + SE | `SERV-002` | **BLOCKED** | Alert/escalation/disable/rollback drill; no sensitive-content logging |
| `OPS-004` | Versioned maintenance and continual-data update protocol | RE + DS + EL | `G6` | **HOLD** | Scheduled security/eval review and replay/regression procedure for corpus updates |
| `REL-001` | Independent native-language and cultural release review | ULL + RUR | `OPS-002`, `AGTE-005` | **BLOCKED** | Signed findings; all severity-1 issues closed or release scope narrowed |
| `REL-002` | Independent security/privacy/license release review | SE + DS | `OPS-002`, `AGTE-006`, `SERV-003` | **BLOCKED** | Signed residual-risk decision; no open critical/high unauthorized path |
| `REL-003` | Release-candidate tag, reproducibility check, and promote/rollback decision | M + RE | `REL-001`, `REL-002` | **BLOCKED** | Clean-checkout reproduction, hashes, RC soak, signed `G6` decision and rollback target |

## Blocked external inputs (not reasons to bypass gates)

| Input | Needed for | Current handling |
|---|---|---|
| Named accountable maintainers/reviewers | `GOV-004` and all later gates | Must be assigned locally; no names invented in this ledger |
| Native Urdu and Roman-Urdu review capacity | `EVAL-004`, `DATA-006`, `AGTE-005`, `REL-001` | Required; automated judging is not a substitute |
| Original publisher terms for conflicting data mirrors | `DATA-002` | Exclude until resolved; mirror metadata alone is insufficient |
| Actual CUDA and second-GPU hardware, including Kaggle quota/session measurements when used | `TRN-001…004`, `BASE-002/003` | Local probe is CPU-only (`torch 2.14.1+cpu`, zero CUDA devices) in [`API-AGENT_hardware_probe.txt`](evidence/phase0/API-AGENT_hardware_probe.txt); run same-hardware CUDA and two-GPU tests only on real target resources, with no CPU substitution |
| Additional compute for S3/S4 | `BASE-005` | Separate budget decision after S2; Kaggle is not assumed sufficient |

## Change log

| Date | Change | Evidence |
|---|---|---|
| 2026-10-22 | Initialized control package; `G0=PASS`, `G1=FAIL`, all later gates pending; no implementation task marked done | Audited revision `b8bea87683ef3ae1c839bf1891b7cc882273de83` |
| 2026-10-22 | Started Phase 0 on local `phase-0-stabilization`; repository `docs/program/` became canonical; maintainer directed no project rename | `D-018`; `PKG-001` in progress; no training or release gate claimed |
| 2026-10-22 | Added regression-first training/attention/sampling tests before semantic edits; audited revision fails all 10 focused tests, including exact CPU resume | [`TRAIN-001_regression_before.txt`](evidence/phase0/TRAIN-001_regression_before.txt); `TRN-001` started; no `G1` closure claimed |
| 2026-10-22 | Fixed AMP step/skip accounting, capability-aware precision policy, fail-stop OOM cleanup, causal+explicit mask composition, and joint top-k/top-p sampling | [`TRAIN-001_amp_masks_after.txt`](evidence/phase0/TRAIN-001_amp_masks_after.txt): 82 focused/legacy tests passed; checkpoint/data-position and hardware evidence remain open |
| 2026-10-22 | Added atomic v2 checkpoints, scaler/RNG/state/identity restoration, deterministic epoch/data positioning, pad-aware masks, and exact script resume | [`TRAIN-001_full_after_checkpoint.txt`](evidence/phase0/TRAIN-001_full_after_checkpoint.txt): 518 passed; [`TRAIN-001_ci_smoke_after.txt`](evidence/phase0/TRAIN-001_ci_smoke_after.txt) and [`TRAIN-001_quality.txt`](evidence/phase0/TRAIN-001_quality.txt) pass; CPU exact-resume equivalence includes batches, metrics, optimizer/scheduler, RNG, and final weights; CUDA/DDP and independent review remain open |
| 2026-10-22 | Added real-lifespan API and adversarial agent regressions before API/agent semantic edits; all 12 fail on audited code | [`API-AGENT_regression_before.txt`](evidence/phase0/API-AGENT_regression_before.txt); includes real tiny checkpoint, prompt accounting, auth/isolation, hard timeout, schema, redaction, bounds, streaming, provider ID/capability, and audit retention probes |
| 2026-10-22 | Implemented per-app checkpoint-backed API lifecycle, shared truncation-safe generation metadata/accounting, fail-closed per-request agent containment, Draft 2020-12/JSON bounds, bounded filesystem/redacted audit, spawn-killed tool deadlines/cancellation, live streaming, and honest provider payloads; added CI dependency/security checks and explicit local-demo flag | [`API-AGENT_after.txt`](evidence/phase0/API-AGENT_after.txt): 91 focused passed; [`API-AGENT_full_after.txt`](evidence/phase0/API-AGENT_full_after.txt): 540 passed, 5 warnings; [`API-AGENT_quality.txt`](evidence/phase0/API-AGENT_quality.txt), [`API-AGENT_security.txt`](evidence/phase0/API-AGENT_security.txt), [`API-AGENT_ci_smoke.txt`](evidence/phase0/API-AGENT_ci_smoke.txt), [`API-AGENT_build.txt`](evidence/phase0/API-AGENT_build.txt), and [`API-AGENT_local_demo_after.txt`](evidence/phase0/API-AGENT_local_demo_after.txt) pass. `G1` remains `FAIL`: provider-call cancellation, repository-wide quality, independent review, and CUDA/second-GPU evidence are unresolved |
| 2026-10-22 | Closed the local blocked-backend deadline gap with spawn-isolated capability inspection/chat, bounded JSON decision validation, parent kill/cancellation, and picklable custom-backend contract; preserved explicit deterministic inline demo path | Before [`AGT-004_backend_deadline_before.txt`](evidence/phase0/AGT-004_backend_deadline_before.txt): 2.0003s despite 0.5s budget; after [`AGT-004_backend_deadline_after.txt`](evidence/phase0/AGT-004_backend_deadline_after.txt): 28 passed, including started-backend PID kill and cancellation; [`AGT-004_full_after.txt`](evidence/phase0/AGT-004_full_after.txt): 544 passed; [`AGT-004_quality.txt`](evidence/phase0/AGT-004_quality.txt), [`AGT-004_security.txt`](evidence/phase0/AGT-004_security.txt), [`AGT-004_build.txt`](evidence/phase0/AGT-004_build.txt), and [`AGT-004_local_demo_after.txt`](evidence/phase0/AGT-004_local_demo_after.txt) pass. Real-provider soak, repository-wide quality, review, and hardware gates remain open |
| 2026-10-22 | Closed the repository-wide local quality baseline with safe mechanical cleanup, unambiguous variables, and localized type annotations; preserved behavior and package identity | Baseline [`AGT-004_repo_quality_remaining.txt`](evidence/phase0/AGT-004_repo_quality_remaining.txt): 447 Ruff findings/four Black files. After [`TRN-007_quality_after.txt`](evidence/phase0/TRN-007_quality_after.txt): Ruff, Ruff format, Black, and mypy (91 files) pass; [`TRN-007_full_after.txt`](evidence/phase0/TRN-007_full_after.txt): 544 passed, 5 warnings; [`TRN-007_security.txt`](evidence/phase0/TRN-007_security.txt), [`TRN-007_build.txt`](evidence/phase0/TRN-007_build.txt), [`TRN-007_ci_smoke.txt`](evidence/phase0/TRN-007_ci_smoke.txt), and [`TRN-007_clean_install.txt`](evidence/phase0/TRN-007_clean_install.txt) pass. `G1` remains `FAIL`: clean-checkout CI, real-provider soak, independent review, and hardware evidence remain open |
| 2026-10-22 | Clean-checkout testing exposed order-dependent legacy API readiness tests; replaced ambient `checkpoints/` discovery in that fixture with isolated tiny tokenizer/checkpoint artifacts and explicit CPU service configuration | Before [`API-001_clean_checkout_before.txt`](evidence/phase0/API-001_clean_checkout_before.txt): 2 failed, 542 passed because readiness/models returned 503 before later tests created checkpoints. After [`API-001_isolated_fixture_after.txt`](evidence/phase0/API-001_isolated_fixture_after.txt): 5 passed from an empty working directory; [`API-001_full_after.txt`](evidence/phase0/API-001_full_after.txt): 544 passed. Exact fresh-clone full CI-equivalent rerun remains pending |
| 2026-10-22 | Re-ran the complete local CI-equivalent workflow from a fresh clone of pre-consolidation local commit `24db736f0ee8f1d6af16c5e7bd74c2632444889b` after removing legacy fixture ambient-state dependence | [`TRN-007_clean_checkout_ci.txt`](evidence/phase0/TRN-007_clean_checkout_ci.txt): clean tree and editable source confirmed; Ruff, Ruff format, Black, mypy (91 files), 544 tests, Bandit, pip-audit, build, 60-step CI training smoke, wheel contents, clean install, packaged config, agent import, and CLI all pass on Python 3.13.16. Hosted 3.10–3.12 matrix, real-provider soak, review, and hardware evidence remain open; `G1=FAIL` |
| 2026-10-26 | Published the first remote Phase 0 branch after the restored workspace omitted Git metadata; reconstructed the complete validated tree directly on the real `origin/main` audited baseline instead of creating unrelated history | `origin/main=b8bea87683ef3ae1c839bf1891b7cc882273de83` was verified without credentials; `phase-0-stabilization` was published as consolidated commit `1f77206bbe0941cdd96b654815068bcb94d3ed74`. Earlier commit IDs remain historical pre-consolidation local evidence, not claimed remote ancestors; the consolidated branch commit and subsequent increments own the remote history. Reconstructed-branch Ruff/Ruff-format/Black/mypy and 544 tests pass in [`PUSH-001_reconstructed_validation.txt`](evidence/phase0/PUSH-001_reconstructed_validation.txt). No gate status changed |
| 2026-10-26 | Implemented authenticated application-scoped agent kill/resume/status and permission/tool revocation with atomic run registration, in-flight cancellation, fresh-policy enforcement, and shutdown cleanup; resume deliberately preserves revocations | Before [`AGT-006_control_before.txt`](evidence/phase0/AGT-006_control_before.txt): 5 failed. After [`AGT-006_control_after.txt`](evidence/phase0/AGT-006_control_after.txt): 5 passed; [`AGT-006_focused_after.txt`](evidence/phase0/AGT-006_focused_after.txt): 37 passed; [`AGT-006_full_after.txt`](evidence/phase0/AGT-006_full_after.txt): 549 passed, 5 warnings. [`AGT-006_quality.txt`](evidence/phase0/AGT-006_quality.txt), [`AGT-006_security.txt`](evidence/phase0/AGT-006_security.txt), [`AGT-006_build.txt`](evidence/phase0/AGT-006_build.txt), and [`AGT-006_local_demo_after.txt`](evidence/phase0/AGT-006_local_demo_after.txt) pass. Independent escape review and other `G1` blockers remain; `G1=FAIL` |
| 2026-10-26 | Added per-application actual-byte body limits (declared and chunked), bounded fixed-window client rate tracking, immediate in-flight rejection with cleanup, and unary/tokenizer 400/409 mappings; emergency agent controls remain authenticated and outside the request-rate quota. Enabled hosted CI pushes on the active stabilization branch | Before [`API-002_contract_before.txt`](evidence/phase0/API-002_contract_before.txt): 5 failed, 1 passed. After [`API-002_contract_after.txt`](evidence/phase0/API-002_contract_after.txt): 6 passed; three additional chunked/startup/emergency-control checks pass within [`API-002_focused_after.txt`](evidence/phase0/API-002_focused_after.txt): 51 passed and [`API-002_full_after.txt`](evidence/phase0/API-002_full_after.txt): 558 passed, 5 warnings. [`API-002_quality.txt`](evidence/phase0/API-002_quality.txt), [`API-002_security.txt`](evidence/phase0/API-002_security.txt), [`API-002_build.txt`](evidence/phase0/API-002_build.txt), and [`API-002_ci_smoke.txt`](evidence/phase0/API-002_ci_smoke.txt) pass. Hosted run observation and independent review remain; `G1=FAIL` |
| 2026-10-26 | Added strict causality, numerical KV-cache, cached/uncached generation, and deterministic stop/truncation/usage evidence; no production semantic change was required for `TRN-006` | Archived audited revision `b8bea876…`: [`TRN-006_contract_before.txt`](evidence/phase0/TRN-006_contract_before.txt) has 3 failed, 3 passed (the three legacy accounting APIs are absent). Fixed branch: [`TRN-006_contract_after.txt`](evidence/phase0/TRN-006_contract_after.txt) 6 passed, [`TRN-006_focused_after.txt`](evidence/phase0/TRN-006_focused_after.txt) 80 passed, and [`TRN-006_full_after.txt`](evidence/phase0/TRN-006_full_after.txt) 564 passed. Independent review remains |
| 2026-10-26 | Published API-002 as `55ea3a8a42b36c6ef3d00978bfff5db748c887cb`; hosted run 25 then exposed a Python 3.10 streaming-test failure and cancelled 3.12 under fail-fast. Reproduced locally, identified Starlette 1.7 TestClient whole-response buffering, replaced that transport with a direct-ASGI live-event assertion, and disabled matrix fail-fast | [`TRN-007_hosted_ci_55ea3a8.json`](evidence/phase0/TRN-007_hosted_ci_55ea3a8.json) records the hosted failure. [`TRN-007_py310_reproduction_before.txt`](evidence/phase0/TRN-007_py310_reproduction_before.txt): 1 failed, 557 passed; [`TRN-007_stream_transport_probe.txt`](evidence/phase0/TRN-007_stream_transport_probe.txt) shows the request event generated at 0.033s but the old TestClient response opened at 1.954s after backend completion. [`TRN-007_py310_stream_after.txt`](evidence/phase0/TRN-007_py310_stream_after.txt) passes; full suites [`TRN-007_py310_full_after.txt`](evidence/phase0/TRN-007_py310_full_after.txt) and [`TRN-007_stream_ci_full_after.txt`](evidence/phase0/TRN-007_stream_ci_full_after.txt) each pass 564 tests. [`TRN-007_stream_ci_quality_after.txt`](evidence/phase0/TRN-007_stream_ci_quality_after.txt), [`TRN-007_stream_ci_security_after.txt`](evidence/phase0/TRN-007_stream_ci_security_after.txt), [`TRN-007_stream_ci_build_after.txt`](evidence/phase0/TRN-007_stream_ci_build_after.txt), and [`TRN-007_stream_ci_training_smoke_after.txt`](evidence/phase0/TRN-007_stream_ci_training_smoke_after.txt) pass. Corrected hosted matrix and `G1` blockers remain open |
| 2026-10-26 | Completed the corrected hosted CI matrix for `f6597012b33dc8c3604e5f9145c9880bb0087fa8`; fail-fast is disabled so every Python version reports independently | [`TRN-007_hosted_ci_f659701.json`](evidence/phase0/TRN-007_hosted_ci_f659701.json): run 26 completed successfully; lint, mypy, Python 3.10/3.11/3.12 tests, package/clean install, dependency/agent security, and training smoke all pass. This closes only the hosted-CPU evidence item; real-provider, review, CUDA, and second-GPU blockers remain |
| 2026-10-26 | Added bounded canonical Draft 2020-12 registry manifests with deterministic order-independent SHA-256 schema/spec identities and runtime context provenance | Before [`AGT-002_registry_before.txt`](evidence/phase0/AGT-002_registry_before.txt): 4 failed. After [`AGT-002_registry_after.txt`](evidence/phase0/AGT-002_registry_after.txt): 4 passed; [`AGT-002_focused_after.txt`](evidence/phase0/AGT-002_focused_after.txt): 37 passed; [`AGT-002_full_after.txt`](evidence/phase0/AGT-002_full_after.txt): 568 passed, 5 warnings. [`AGT-002_py310_registry_after.txt`](evidence/phase0/AGT-002_py310_registry_after.txt), [`AGT-002_quality_after.txt`](evidence/phase0/AGT-002_quality_after.txt), [`AGT-002_security_after.txt`](evidence/phase0/AGT-002_security_after.txt), [`AGT-002_build_after.txt`](evidence/phase0/AGT-002_build_after.txt), and [`AGT-002_ci_smoke_after.txt`](evidence/phase0/AGT-002_ci_smoke_after.txt) pass. Increment-specific hosted CI and independent review remain; `G1=FAIL` |
