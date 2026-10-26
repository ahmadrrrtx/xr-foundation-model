# Decision and Risk Register

This document owns consequential technical/product decisions and residual risks. It does not own task status; use the execution ledger for that.

## Decision states

- **ACTIVE-PLAN:** binding default for implementation. Deviating requires a new superseding decision before the work.
- **PROPOSED:** requires the named maintainer decision; do not implement as settled.
- **SUPERSEDED / REJECTED:** retained for history with rationale and evidence.

These are plan decisions, not claims that the user has personally signed each one. `GOV-004` must assign the human approvers.

## Architecture, scope, and operating decisions

| ID | State | Decision | Why / consequence | Supersession trigger |
|---|---|---|---|---|
| `D-001` | **ACTIVE-PLAN** | “Native” released weights are trained from random initialization. | Prevents a continued-pretrained checkpoint from being relabeled as the native deliverable. Teachers/baselines remain allowed with labels. | Only an explicit program-charter change; any derivative becomes separately named |
| `D-002` | **ACTIVE-PLAN** | v1 supports four measured surfaces: Urdu script, English, natural Roman Urdu, and code-switching. | Avoids an Urdu-only claim while making Roman/mixed performance visible rather than averaged away. | New native data/eval evidence and maintainer decision |
| `D-003` | **ACTIVE-PLAN** | v1 agency is bounded to read-only/search/retrieval/calculator/structured extraction and simulators; no arbitrary shell or mutating real-world actions. | Matches small-model capability and creates a testable security envelope. | Separate threat model, tool policy, and release gate |
| `D-004` | **ACTIVE-PLAN** | Do not attempt 3B/8B+ from scratch. Pilot at 50–100M; scale to 300M and then 0.5–1B only through measured gates. | Protects compute/data sustainability and forces a stop/pivot option. | S2 learning curve, clean unique-data budget, and compute approval |
| `D-005` | **ACTIVE-PLAN** | Prefer a conventional HF-compatible decoder-only architecture; keep neurotopo/MoE/multimodal out of the production path. | Enables TRL/evaluation/vLLM interoperability and reduces maintenance. | A controlled ablation shows a material quality/efficiency advantage |
| `D-006` | **ACTIVE-PLAN** | Repair XRFM for S0–S2; do not rewrite the training stack immediately. Bake off OLMo-core and TorchTitan before larger scale. | Preserves useful existing work while avoiding a bespoke distributed framework at scale. | Fixed-budget correctness/throughput/resume/export comparison |
| `D-007` | **ACTIVE-PLAN** | DataTrove-style streaming is the default corpus architecture; NeMo Curator is conditional. | Solves scale without adopting a GPU-heavy stack prematurely. | Measured GPU-dedup/filter requirement |
| `D-008` | **ACTIVE-PLAN** | Evaluation and data-admission controls precede corpus/tokenizer/model scale. | Prevents contaminated or unmeasurable “progress.” | Not expected to change; any exception is a release-risk acceptance |
| `D-009` | **ACTIVE-PLAN** | Data uses Tier A/B/C review; unknown/conflicting/NC/ND terms are excluded from default training until specifically resolved. | A compilation license is not proof of rights in every underlying document. | Original-publisher terms or qualified legal/data-steward decision |
| `D-010` | **ACTIVE-PLAN** | Natural Roman/code-switched text, transliteration, translation, and synthetic generation remain separate strata; synthetic Roman starts capped at 25% of that partition. | Prevents synthetic text from masquerading as natural capability or overwhelming spelling variation. | Natural-data ablation and native-review evidence |
| `D-011` | **ACTIVE-PLAN** | Tokenizer winner requires conformance, per-surface efficiency, and identical downstream LM evidence. | Compression alone can harm morphology/generalization and embedding cost matters. | Reproducible tournament and downstream comparison |
| `D-012` | **ACTIVE-PLAN** | TRL SFT is the first post-training method; DPO/RLVR are optional gated experiments. | Establishes a stable baseline before preference or reward optimization. | Reviewed preference pairs or validated deterministic verifier |
| `D-013` | **ACTIVE-PLAN** | Constrained decoding is an accuracy aid, never the security boundary; JSON Schema validation, authorization, and sandboxing always run. | Grammar-valid calls can still be malicious or unauthorized. | No exception for vLLM/XGrammar/provider-native tools |
| `D-014` | **ACTIVE-PLAN** | MCP is an optional interoperability adapter, not a trust guarantee. | MCP tool descriptions/annotations and cross-server outputs can be untrusted. | Only with equivalent policy enforcement and AgentDojo-style tests |
| `D-015` | **ACTIVE-PLAN** | Kaggle is for discovered, interruption-safe small experiments; Drive is for reports/archives, not training I/O or secrets. | Prevents quota/session/storage assumptions from corrupting runs. | Actual account/hardware measurement and separate larger-compute decision |
| `D-016` | **ACTIVE-PLAN** | External teachers/judges require cost, license, retention, privacy, and native-quality approval. | Prevents private data leakage and unreviewed synthetic bias. | Written source/teacher record; local alternative preferred for sensitive data |
| `D-017` | **ACTIVE-PLAN** | Existing MIT code license is retained for project code unless maintainers decide otherwise; weight/data licenses remain pending source review. | Open-source code does not settle model/data obligations. | Maintainer/legal decision with attribution/share-alike analysis |
| `D-018` | **ACTIVE-PLAN** | **Maintainer decision (2026-10-22): retain the name XRFM Foundational Model and the existing `xrfm` import/package identity. Do not rename it. Do not publish or instruct installation of the unrelated public PyPI `xrfm` distribution; external PyPI publication is blocked until the maintainer separately resolves distribution ownership without changing the project name.** | Resolves the collision through a publication prohibition rather than a prohibited rename. Local/source installs and Docker builds remain valid. | Only the maintainer may revise the distribution-publication policy; the project name itself is not open for agent-initiated change |
| `D-019` | **ACTIVE-PLAN** | If native-from-scratch scaling fails, stopping or shipping a smaller tool-augmented model is valid; a continued-pretraining pivot is a separate derivative decision. | Preserves honesty and prevents scope drift under compute pressure. | S2/S3 evidence and explicit maintainer approval |

## Required decision-record template

Copy this block for every new decision:

```markdown
### D-NNN — Short title
- Date:
- State: PROPOSED | ACTIVE-PLAN | SUPERSEDED | REJECTED
- Decision owner(s) and independent reviewer(s):
- Context and evidence (permanent links):
- Decision:
- Alternatives considered and why rejected:
- Consequences (quality, security, data, cost, schedule):
- Linked ledger tasks and gates:
- Migration / rollback:
- Supersedes / superseded by:
```

A scope-changing decision is incomplete until linked plan/ledger rows are updated in the same change.

## Open maintainer decisions

These do not justify starting blocked work:

1. Named human owner/reviewer assignments and time commitment.
2. **Resolved by maintainer:** the project remains XRFM Foundational Model (`D-018`); only external distribution ownership/publication remains blocked, not local development.
3. Which native reviewers will cover formal Urdu, regional/informal Urdu, Roman spelling variation, and code-switching.
4. Whether any external teacher/judge is permitted, and under what data-retention/cost constraints.
5. Final model/data license after source obligations are known.
6. Whether S3/S4 compute is available after the S2 evidence exists.

## Risk register

Severity reflects impact if the risk occurs; likelihood is the current qualitative estimate before mitigation. Every risk has an observable tripwire.

| ID | Risk | Severity / likelihood | Early signal / tripwire | Mitigation and contingency | Owner |
|---|---|---|---|---|---|
| `R-001` | Underlying text rights differ from compilation/mirror license | Critical / High | `unknown`, NC/ND, conflicting labels, missing original URL | Tier review, original-publisher resolution, exclusion by default; remove affected source and descendants if discovered later | DS |
| `R-002` | Private messages/social data contain PII or lack consent | Critical / Medium-High | Personal identifiers, group names, missing consent, unexpected secrets | Opt-in collection, minimization, restricted raw access, sampled audits; quarantine/delete affected source and rotate any exposed secret | DS + ULL + SE |
| `R-003` | Benchmark or near-duplicate leakage inflates results | High / High | Suspicious score jumps, overlap/cluster hits, tokenizer trained on holdout | All-split exclusion, group splits before tokenizer, known-positive contamination tests; invalidate results and retrain affected run | EL + DE |
| `R-004` | Synthetic/transliterated Roman Urdu dominates and looks unlike real users | High / High | Natural Roman bpb/task gap, reviewer “unnatural” labels, synthetic share >25% | Separate strata/cap, natural-data collection, blind human eval; narrow Roman support if unresolved | RUR + DE |
| `R-005` | English or Urdu regresses as mixture changes | High / Medium | Per-surface bpb/benchmark regression hidden by aggregate score | A/B/C controls, replay, per-surface gates; revert mixture checkpoint | TE + EL |
| `R-006` | 50–100M pilot lacks capacity for useful agency | Medium / High | Flat learning curve, poor final-state success despite schema validity | Treat pilot as diagnostic; use retrieval/tools; scale only if evidence supports; stop rather than overclaim | M + TE |
| `R-007` | GPU overflow/OOM/resume defects waste or corrupt long runs | Critical / High | Scheduler/step mismatch, NaNs, rank divergence, batch mismatch after resume | `TRN-001…007`, exact-resume and fault-injection gates; restore last verified checkpoint with changed config | TE |
| `R-008` | Prompt injection or malicious tool output causes unauthorized action/exfiltration | Critical / Medium-High | Canary appears in output, policy bypass, unexpected cross-tool flow | Least privilege, no mutating v1 tools, schema/auth/sandbox, injection suite; disable tools and roll back | AE + SE |
| `R-009` | Public default exposes local files or cross-request state | Critical / Medium | Missing-key startup succeeds, shared audit/state grows, root too broad | Fail closed, loopback default, per-request state, approved root; take endpoint offline on detection | AE + SE |
| `R-010` | Timeout/streaming illusion causes resource exhaustion | High / High | Blocked threads accumulate, first event only at completion, deadline exceeded | Process/cooperative cancellation, queue streaming, concurrency limits; shed load and disable offending tool | AE + RE |
| `R-011` | Kaggle session/quota ends mid-run or hardware dtype differs | High / High | Checkpoint/upload exceeds session, BF16 instability, quota exhaustion | Hardware discovery, short resumable increments, durable archives; move to approved compute or reduce scale | TE + RE |
| `R-012` | LLM judge or translated benchmark rewards the wrong behavior | High / Medium-High | Judge disagrees with native reviewers, prompt sensitivity, inconsistent rankings | Blind native review, execution metrics, multiple surfaces, judge calibration; suspend disputed metric | EL + ULL |
| `R-013` | Poisoned/malicious sources or tool servers manipulate model/agent | Critical / Medium | Source distribution shifts, instruction-like documents, tool-description anomalies | Provenance/reputation, content-is-data labeling, source review, trusted MCP allowlist; quarantine source/server | DS + SE |
| `R-014` | Small maintainer team causes drift, unreviewed releases, or abandoned artifacts | High / Medium-High | Tasks without assignees, stale ledger, missing independent sign-off | RACI, one-active-task rule, automated checks, documented release/rollback; reduce scope rather than skip review | M + RE |
| `R-015` | Package/name collision causes dependency confusion | High / Medium | `pip install xrfm` resolves unrelated code | `D-018`: retain project name, remove public-PyPI install instructions, permit source/Docker installs, and block external PyPI publication until ownership is resolved | M + CE |
| `R-016` | Marketing/research claims outrun evidence | High / Medium | “SOTA,” “safe,” “production,” “native” without revisioned artifacts | Claim checklist tied to model card/eval bundle; retract/correct and block release | M + EL |
| `R-017` | Data/model update causes silent temporal or behavioral drift | High / Medium | Changed outputs at same version, stale benchmark revision, regression after corpus refresh | Immutable manifests, replay/regression suite, new semantic versions, rollback target | RE + EL |

## Risk-acceptance rule

Critical/high residual risks cannot be accepted silently by a developer. They require a written decision by the maintainer and security/data owner, affected-scope disclosure, and a rollback/containment plan. If the risk involves unauthorized actions, secret exfiltration, unknown training-data rights, or private data without consent, the default is **STOP**, not acceptance.
