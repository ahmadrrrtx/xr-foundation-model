# Sustainable Implementation Plan

## 0. Charter

### Objective

Build an open-source **native Urdu–English decoder-only language model and bounded agent runtime** that handles four surfaces deliberately:

1. Urdu in Perso-Arabic script (`ur_Arab`);
2. English (`en_Latn`);
3. natural Roman Urdu (`ur_Latn`, with spelling variation);
4. natural Urdu–English code-switching (`mixed`).

“Native” means the released native model's weights are trained **from random initialization** under XRFM's data/tokenizer/model recipe. A model initialized from Qwen, Llama, OLMo, Alif, Qalb, or another pretrained checkpoint may be a baseline, teacher, or separately named derivative, but must never be substituted silently for the native deliverable.

### Initial product boundary

The first useful agent is a **bounded bilingual research/document assistant**, not an open-ended computer controller. Its initial tools are read-only document retrieval/read, bounded search, calculator/structured extraction, and deterministic mock services for evaluation. Arbitrary shell, code execution, purchases, account changes, and other mutating tools are outside v1.

This narrow boundary is intentional: a 50–100M pilot can test language and protocol learning, but it is not expected to be a broad reasoning model. A useful agent release requires a later scale gate and execution-verified task success.

### Non-goals until explicitly re-decided

- Training a 3B/8B+ model from scratch.
- Claiming general SOTA or human-level Urdu understanding.
- MoE, neurotopo, multimodal, voice, autonomous coding, or open-world tool use.
- Fine-tuning a large pretrained model and calling it the native model.
- Using benchmark test data, private chats, unknown-license text, or unreviewed teacher output to enlarge the corpus.
- Deploying an unauthenticated public agent or treating MCP/tool annotations as a security boundary.

## 1. Program scorecard

Every released result must be reported by language/script and linked to code, data, tokenizer, model, benchmark revisions, seeds, prompts, decoding settings, and hardware. Do not compare raw perplexity across different tokenizers; use held-out **bits per byte (bpb)** for cross-tokenizer language modeling comparisons.

| Dimension | Required measures | Release interpretation |
|---|---|---|
| Data governance | Source/license coverage, consent/provenance, PII actions, hashes, dedup clusters, contamination exclusions | 100% of admitted records have a source decision; zero `unknown` licenses in the frozen training set |
| Tokenization | Round-trip accuracy, bytes/token and tokens/word by surface, byte-fallback rate, special-token behavior, encode/decode speed | Lossless on conformance corpus; no ordinary Urdu/Roman text reduced systematically to per-byte encoding; final choice also requires downstream evidence |
| Base language | Held-out bpb by surface/domain; UrBLiMP; UrduMMLU; Belebele `urd_Arab` and `urd_Latn`; English retention suite | Above-chance is only a pilot sanity check; scale/release decisions require learning curves and baseline-relative confidence intervals |
| Roman/code-switch | Natural held-out bpb, transliteration/translation tests, custom native generation/task suite, RUAB or equivalent safety-transfer protocol | Mechanical transliteration and natural Roman Urdu are reported separately; no synthetic-only success claim |
| Agent utility | Final-state task success, pass@1, pass^k consistency, policy violations, clarification/no-tool accuracy, error recovery, citations where required | JSON validity alone never counts as agent success |
| Agent security | Unauthorized side effects, secret-canary exfiltration, prompt-injection attack success, cross-request leakage, authorization/schema bypasses | Zero unauthorized mutating actions and zero canary exfiltration in the release suite; all residual attack rates disclosed |
| Human quality | Blind native-speaker adequacy, fluency, naturalness, cultural appropriateness, task success; agreement and disagreement analysis | LLM-as-judge may assist analysis but cannot be the sole release gate |
| Systems | Exact resume, finite-gradient behavior, throughput, memory, tokens/budget, latency, cancellation, streaming, artifact hashes | No capability claim if the run cannot be reproduced and resumed within stated tolerances |

### Baseline set

Pin revisions before use and rerun them locally under the same prompts/protocol; do not copy vendor leaderboard numbers into XRFM result tables.

- **Primary size/system baselines:** Qwen3-0.6B and OLMo-2-0425-1B; Llama-3.2-1B if its access terms are accepted.
- **Urdu-specific references:** UrduLM-100M artifacts if actually available and terms permit; Alif and Qalb as resource-permitting continued-pretraining references, clearly labeled by parameter count and provenance.
- **Controls:** random/majority answer baselines, current XRFM tokenizer/model, no-tool agent, scripted-tool oracle, and an English-only or no-Roman data ablation.
- **External judges/teachers:** optional only after cost, license, retention, and privacy approval. Never send private/consented-only local data to an external API by default.

## 2. Reference architecture and toolchain

### Model and interoperability

Use a conventional decoder-only Transformer with RMSNorm, SwiGLU, RoPE, tied/input-output embedding only if validated, and a **Hugging Face-compatible configuration/export contract**. Keep the architecture Llama-compatible where practical so that Transformers, TRL, vLLM, and GGUF conversion do not each require a bespoke runtime. GQA, longer context, or other changes require an ablation and decision record.

Keep the research `neurotopo` package out of the production dependency path. Preserve the repaired XRFM implementation as the small reference model; do not build a new parallelism framework.

### Selected defaults versus gated options

| Layer | Default for the next increment | Conditional alternative / rule |
|---|---|---|
| Data processing | DataTrove-style streaming jobs plus XRFM source manifests | NeMo Curator only after a measured GPU-dedup/filter need |
| Language labels | GlotLID + deterministic script features + custom human-validated Roman/code-switch classifier | UrduHack for Urdu normalization diagnostics, not as a gold language judge |
| Tokenizer | Hugging Face Tokenizers/SentencePiece BPE and unigram bake-off | No winner chosen by vocabulary size or compression alone |
| Small pilot | Repaired XRFM training loop | Any framework migration must reproduce the same tiny run first |
| Scaling | Bake off OLMo-core and TorchTitan at the same small model/token budget | Select on correctness, exact resume, throughput, memory, export, and maintenance—not stars |
| Post-training | TRL SFT after HF contract | DPO/preferences only with reviewed pairs; RLVR only after SFT and verifier baselines |
| Evaluation | Pinned XRFM adapter to LM Evaluation Harness; local versioned Urdu/agent tasks | Lighteval may replace the adapter only after output-equivalence tests |
| Agent protocol | Existing XRFM state machine, rebuilt for streaming/cancellation; JSON Schema 2020-12 at every boundary | MCP SDK as an interoperability adapter, never as proof of trust or safety |
| Constrained output | XGrammar through supported vLLM deployment; strict runtime validation always | If unsupported, fail closed or use a non-agent text response; never regex-parse and execute |
| Serving | Transformers reference first; vLLM after compatibility/performance tests | llama.cpp/GGUF optional for edge after numerical/output parity tests |
| Experiment record | Append-only JSONL plus the experiment template and immutable artifact hashes | A hosted tracker is optional and cannot be the only record |

## 3. Phase and gate sequence

Gate status is maintained only in the execution ledger.

---

### Phase 0 — Stabilize the execution and security skeleton

**Gate:** `G1 — trustworthy small training and local agent runtime`  
**Dependencies:** `G0` audit/charter complete.  
**Ledger work:** `TRN-*`, `API-*`, `AGT-*`, `PKG-001`.

#### Work

1. Convert every finding in the deep audit into a failing regression test before changing semantics.
2. Repair AMP optimizer-step detection; advance scheduler/global step only after a real update. Implement device-specific FP32/FP16/BF16 policy.
3. Make OOM a coordinated failed attempt; clear partial gradients and resume only from known state. Add deterministic `set_epoch()` and correct pad/mask contracts.
4. Implement atomic, complete checkpoints and exact-resume tests including RNG, scaler, data position, and data/tokenizer/config identity.
5. Compose causal/padding/document masks and define top-k/top-p/truncation accounting.
6. Repair API lifespan/readiness and require a real model load/model-list/completion integration test.
7. Make agent authentication fail closed outside explicit local development; use per-request runtime/audit state.
8. Replace the partial schema validator with Draft 2020-12 validation; enforce input/output, min/max, total bytes/items/depth, and internal tool bounds.
9. Fix redaction, bounded streaming file I/O, audit retention, provider capability declarations, tool-call IDs, hard cancellation, and genuine event streaming.
10. Keep the maintainer-approved **XRFM Foundational Model** name unchanged. Prevent accidental installation/publication of the unrelated PyPI `xrfm` distribution: use source/Docker installation instructions and block external PyPI publication until distribution ownership is separately resolved.

#### Required evidence

- New tests fail on the audited revision and pass on the fix.
- CPU exact-resume equivalence and same-hardware CUDA equivalence within declared numerical tolerance.
- T4 or P100 smoke if Kaggle will be used; a second GPU is required for distributed OOM/DDP tests.
- Adversarial agent tests for secrets, negative/huge limits, large files/directories, malformed decisions, unauthorized tools, blocked calls, disconnects, and cross-request isolation.
- API lifespan test performs a real completion from a valid tiny test checkpoint.
- Ruff, Black, mypy, tests, build, and security checks pass in CI.

#### `G1` stop/go

**GO** only when all blocker/critical findings are closed with evidence and no known unauthorized execution/exfiltration path remains in the local v1 configuration.  
**STOP** if exact resume, deadline enforcement, or strict execution validation cannot be made reliable; do not compensate with documentation.

**Rollback:** retain the last known local-only demo behind explicit development flags; never ship the audited agent configuration as a public default.

---

### Phase 1 — Establish evaluation and data governance before corpus scale

**Gate:** `G2 — evaluation and admission policy operational`  
**Dependencies:** `G1`.  
**Ledger work:** `GOV-004+`, `EVAL-001…005`, `DATA-001…002`, `AGTE-001…003`.

#### Work

1. Assign accountable roles: maintainer, core/training, data steward, evaluation lead, Urdu language lead, Roman-Urdu reviewers, agent/security reviewer, and release manager. One person may hold several roles, but separation of review and implementation is required at release.
2. Create a benchmark registry with exact revisions, licenses, prompts, few-shot policy, metrics, and expected chance levels. Exclude **all splits** of core benchmarks from training by default.
3. Implement the XRFM adapter and run baselines end to end. Store sample-level outputs and confidence intervals, not only aggregate scores.
4. Build a sealed, native-speaker-authored development/test suite across the four surfaces. Include spelling variants, mixed technical English, dialect/register notes, no-tool/clarification cases, and safety-relevant matched intents.
5. Create a final-state tool simulator and custom `XRFM-UrduAgentBench`: parallel Urdu/English/Roman/code-switched versions of the same underlying task, with separate natural authoring/review rather than automatic translation alone.
6. Integrate BFCL-style schema/call metrics and AgentDojo-style injection/canary tests. Use τ-bench as a protocol reference for interactive final-state grading; do not claim its supported languages include Urdu.
7. Adopt the data-source review template and Tier A/B/C policy. Inventory candidates and resolve conflicting mirror licenses with original publishers where possible.
8. Establish opt-in native data collection: contributor consent, allowed uses, withdrawal process where feasible, PII minimization, compensation/attribution policy, and no private-group scraping.

#### Minimum human-review protocol

- Gold evaluation items: double annotation and adjudication; report an appropriate agreement statistic and per-category disagreements. Agreement below 0.67 triggers guideline revision; target at least 0.80 before freezing high-stakes/safety subsets.
- Corpus audit: a stratified sample from every source/domain/surface, with at least 20% double review. Any confirmed critical privacy/license issue blocks the affected source, not merely the sampled document.
- Reviewers are blind to model identity for comparative generation evaluation.

#### `G2` stop/go

**GO** when baselines and XRFM dummy/tiny models produce reproducible evaluation bundles, contamination tests work, gold guidelines are stable, and source-admission owners are assigned.  
**STOP** if native-speaker evaluation or provenance review is unavailable. Automated translation/LLM judging cannot substitute for this gate.

**Rollback:** keep evaluation data private/sealed while publishing only schemas, guidelines, and synthetic examples that do not reveal answers.

---

### Phase 2 — Build corpus v0 and select the tokenizer empirically

**Gate:** `G3 — frozen pilot corpus and tokenizer`  
**Dependencies:** `G2`.  
**Ledger work:** `DATA-003…006`, `TOK-001…002`.

#### Corpus pipeline

Each document/segment record must carry:

```text
source_id, source_revision, original_URI_or_contributor_ID, acquisition_time,
content_hash, language_script, natural_or_synthetic, domain, license_tier,
license_evidence, consent_basis, PII_action, quality_scores,
dedup_cluster, split, benchmark_exclusions, processing_version
```

Pipeline order:

1. ingest immutable raw references and hashes;
2. extract/normalize into parallel **raw-view** and **normalized-view** representations;
3. identify language/script with confidence and route ambiguous/mixed text for review;
4. apply source-specific quality and license filters;
5. minimize/redact PII under a documented policy without claiming perfect anonymization;
6. exact-dedup streamingly, then MinHash/LSH near-dedup within and across sources;
7. split by document/source/near-duplicate cluster before tokenizer fitting;
8. decontaminate against all registered benchmarks and sealed tasks;
9. compute per-source/domain/surface statistics and native-review samples;
10. freeze a signed manifest and processing report.

Do not over-normalize Roman Urdu into one spelling or erase code-switching. Preserve original text, and train/evaluate augmentation as a separate labeled arm. Cap synthetic/transliterated Roman data within the Roman partition until natural-data ablations justify more; the initial cap is **25%**, changeable only by decision record.

#### Pilot mixture experiments

Final ratios must come from held-out bpb and downstream ablations. For the small proxy, compare at least:

- **A — Urdu-forward:** 50% Urdu script / 40% English / 10% Roman+mixed;
- **B — code-switch-forward:** 40% Urdu script / 35% English / 25% Roman+mixed;
- **C — control:** 50% Urdu script / 50% English / 0% Roman+mixed.

These are sampling targets for controlled experiments, not claims about the final corpus. Keep domain and quality fixed across arms as far as possible. Limit repeated epochs and report unique versus repeated tokens.

#### Tokenizer tournament

Train on the training split only. Compare BPE and unigram at 24K/32K/48K candidates, with byte fallback for true out-of-vocabulary robustness and explicitly defined chat/tool special tokens. Evaluate:

- exact round trips, NFC/raw-view behavior, digits, URLs, English code, Urdu diacritics, ZWJ/ZWNJ, punctuation, and spelling variants;
- bytes/token, tokens/word, sequence-length distribution, and fallback rate for each surface/domain;
- embedding-parameter/memory cost and training/serving throughput;
- a small identical downstream LM for the finalists.

A smaller token count is not automatically better if it damages morphological/generalization performance. The current tokenizer remains the control.

#### `G3` stop/go

**GO** when the corpus has no unknown admitted licenses, split/decontamination manifests pass, native review finds no unresolved critical issue, tokenizer conformance passes 100%, and the downstream/token efficiency record selects a winner.  
**STOP** if natural Roman/code-switch coverage is too small to evaluate separately; collect more data rather than hiding the gap with transliteration.

**Rollback:** freeze all prior manifests; a failed candidate never overwrites the last approved corpus/tokenizer.

---

### Phase 3 — Train a small native base and make a scale/no-scale decision

**Gate:** `G4 — native pilot validates or rejects scaling`  
**Dependencies:** `G3`.  
**Ledger work:** `BASE-001…005`, `OPS-001`.

#### Progressive runs

1. **S0 conformance:** 10–30M or smaller, tiny data, overfit batch, gradient/NaN tests, checkpoint/resume, HF export, deterministic generation.
2. **S1 proxy:** 30–50M, identical compute across mixture/tokenizer arms; used for selection, not released as capable.
3. **S2 native pilot:** 50–100M from random initialization, 2,048 context initially, two seeds for the selected recipe; checkpoint at 25%, 50%, and 100% of the token budget.
4. **S3 scale candidate:** 300M only if S2 shows a useful learning curve and governance remains clean.
5. **S4 conditional scale:** roughly 0.5–1B only after a new compute/data/utility decision. No jump to 3B/8B in this plan.

#### Budget rule

Do not promise a token budget before measuring data and hardware. For each candidate record:

```text
approx_training_FLOPs ≈ 6 × non_embedding_parameters × training_tokens
measured_tokens_per_second, measured_memory, utilization, energy/time,
unique_tokens_available, allowed_epoch_cap, session interruption overhead
```

Choose the budget from measured throughput, unique-data policy, pilot scaling slope, and an explicit interruption-safe session budget. Twenty tokens/parameter may be a comparison point, not a universal law. A Kaggle session/quota is a constraint to measure, not an assumption to hide.

#### Scaling criteria

`G4` requires:

- finite, reproducible optimization and exact-resume evidence;
- improved held-out bpb over the current tokenizer/model control on Urdu and Roman/mixed strata without an unexplained English collapse;
- above-chance core diagnostics with confidence intervals, while acknowledging that 100M MMLU-like scores may remain weak;
- improving or clearly interpretable learning curves across 25/50/100% checkpoints on at least two of the three non-English strata;
- a documented advantage from the selected tokenizer/mixture against the control;
- native-speaker error analysis and a compute-quality projection for S3/S4.

**GO** to S3 only if additional scale is supported by the measured curve and available clean unique data.  
**STOP/PIVOT** if the curve is flat, data is exhausted/contaminated, or a smaller model plus retrieval/tools is more efficient. A pivot to continued pretraining becomes a separate derivative project and requires a new decision; it is not a silent native-model fallback.

---

### Phase 4 — Agentic post-training with verifiable tasks

**Gate:** `G5 — useful and bounded bilingual agent`  
**Dependencies:** a `G4`-approved base checkpoint; tool simulators and security suite from `G2`.  
**Ledger work:** `SFT-001…004`, `AGTE-004…006`, `SERV-001`.

#### Chat/tool contract

- Version one canonical chat template and decision schema across SFT, evaluation, Transformers, and serving.
- Mask loss so system prompts, tool outputs, and untrusted documents are not treated as assistant target behavior.
- Support `answer`, `tool_call`, `ask_user`, and `refuse`; `verify` is an explicit runtime operation, not private chain-of-thought.
- Preserve provider/tool call IDs and correlate observations exactly.
- Validate every decision and tool input/output at runtime even when constrained decoding is enabled.

#### Data recipe

Start with high-quality, reviewed examples rather than a volume target:

- native-authored Urdu, English, Roman Urdu, and code-switched instructions;
- no-tool answers, necessary clarification, single/multiple tool use, irrelevant-tool rejection, tool errors, timeouts, denial, and recovery;
- grounded document answers with required citations and unanswerable cases;
- matched-intent safety examples and indirect prompt injection inside tool/document content;
- teacher-generated examples labeled by teacher/version/prompt/license and sampled for human review;
- hard negatives with malformed JSON, wrong argument types, extra fields, and unauthorized requests.

SFT comes first. DPO requires genuinely reviewed preference pairs and a documented annotation protocol. RLVR is optional only after deterministic verifiers show that reward hacking and false accepts are controlled.

#### Agent release thresholds

Before public release, on a frozen suite with at least 200 final-state tasks per language surface (or a wider confidence-interval justification):

- constrained structured output validity at least 99.5%; **100% of invalid/unauthorized calls blocked** regardless of decoder;
- statistically significant utility improvement over base and no-tool controls;
- report pass@1 and pass^4 consistency separately; do not relabel one as the other;
- English↔Urdu and English↔Roman/mixed task-success gaps no greater than 10 percentage points, or release language support is narrowed explicitly;
- zero unauthorized mutating actions and zero secret-canary exfiltration in the release security suite;
- AgentDojo/RUAB-style residual attack/safety failures disclosed by category and surface;
- blind native-speaker review finds no unresolved severity-1 language/cultural/safety failure.

These thresholds do not prove general safety. They define the limited v1 release envelope.

#### `G5` stop/go

**GO** only for the evaluated tools and language surfaces.  
**STOP** if utility depends on unconstrained malformed output, if injection can bypass policy, or if one supported surface is substantially weaker; narrow the release rather than averaging away the failure.

---

### Phase 5 — Serving, release, and sustainable maintenance

**Gate:** `G6 — reproducible release candidate`  
**Dependencies:** `G5`.  
**Ledger work:** `SERV-002…004`, `OPS-002…004`, `REL-001…003`.

#### Release engineering

1. Export safetensors and pinned HF config/tokenizer/chat template; compare logits/generation against the XRFM reference.
2. Test vLLM structured output and streaming under load; GGUF/llama.cpp is optional and requires its own parity tests.
3. Add authentication/authorization, rate and concurrency limits, request limits, SSRF/network allowlists, per-tool sandboxes, cancellation, observability without sensitive content, and a global tool kill switch.
4. Publish model/data cards, intended/out-of-scope uses, per-surface results, known bias/privacy/copyright limits, benchmark revisions, artifact hashes, and exact reproduction commands.
5. Package an evaluation bundle with sample-level outputs and confidence intervals. Include negative results.
6. Obtain independent language and security sign-off. Tag a release candidate first; promote only after soak testing and rollback verification.

#### Storage layout

- Git controls source and small versioned manifests.
- Drive stores private plans, review sign-offs, reports, and archives; it is not a high-throughput data/checkpoint filesystem.
- Kaggle stores disposable session state and small approved datasets within discovered account limits.
- The artifact hub stores final versioned weights/tokenizers/cards/evaluation bundles.

#### Maintenance triggers

- monthly dependency/security triage;
- quarterly or data/model-change reevaluation, whichever comes first;
- immediate review on a benchmark revision, tool-schema change, new source, critical vulnerability, or material drift report;
- continual-data updates use frozen incremental manifests plus English/Urdu/Roman replay and regression evaluation, never silent corpus replacement;
- retain the last verified model/runtime for immediate rollback and keep tool permissions independently revocable.

## 4. Kaggle and compute policy

Kaggle is appropriate for hardware discovery, T4/P100 dtype smoke tests, tokenizer/data experiments, S0/S1, and possibly interruption-safe S2 work. It is not a commitment that free compute can train a broadly capable 1B+ native model.

Before every Kaggle run, record the actual allocated GPU, memory, session limit, remaining quota, local disk, dataset size, throughput, checkpoint duration, and upload/archive time. Use `/kaggle/working` only for active state and checkpoint to durable storage at verified intervals. Do not place API keys in the plan, Git, notebook outputs, or Drive documents.

A larger run requires a separate budget decision after S2. If no suitable compute is available, the valid outcome is a smaller evaluated model or a stop—not an untested multi-week run.

## 5. Immediate execution order

The next three increments are deliberately small:

1. **Increment 1 — owner assignment + regression harness:** `GOV-004`, `TRN-001…007`, `API-001…003`.
2. **Increment 2 — agent containment:** `AGT-001…006`, with local-only deployment until security review.
3. **Increment 3 — evaluation/data control plane:** `EVAL-001…005`, `DATA-001…002`, `AGTE-001…003`.

Only after those gates should the team invest in corpus scale, tokenizer tournaments, Kaggle pilot runs, or post-training. This order minimizes wasted compute and prevents a polished demo from hiding unsafe or unverifiable foundations.
