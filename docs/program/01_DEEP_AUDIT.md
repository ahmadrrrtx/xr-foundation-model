# Deep Audit and Research Synthesis

**Revision:** `b8bea87683ef3ae1c839bf1891b7cc882273de83`  
**Audit date:** 2026-10-22  
**Verdict:** useful research skeleton; **not ready** for a native Urdu–English agentic model program without a stabilization phase.

## 1. What was actually verified

### Repository and environment

- Reconfirmed that upstream `main` still resolves to the audited commit.
- Installed the package and development dependencies in the audit environment.
- Full suite: **505 passed, 5 warnings**. Under coverage, the same suite took 205.03 seconds.
- Coverage command: `coverage run --source=src/xrfm,api -m pytest -q --disable-warnings`.
- Result: **79% aggregate line coverage** (6,711 statements, 1,437 missed). This is not 79% behavioral assurance: completion, data CLI, distributed, tracking, and failure paths have material gaps.
- Ruff and Black failures, 12 mypy errors in nine files, and setuptools license deprecation warnings were reproduced. The wheel/sdist build itself succeeded.
- The only committed checkpoint is a 76,805,341-byte legacy artifact incompatible with the current model. No current usable weights or meaningful native evaluation artifacts were found.
- The tiny corpus checksum matches its manifest. Its problems are provenance/semantics and split construction, not a checksum mismatch.
- No tracked source file was changed during the audit.

### Previously reproduced behavior retained as baseline evidence

| Area | Verified result | Interpretation |
|---|---|---|
| Model scale | Packaged-vocabulary configurations span roughly 264K to 19.7M parameters | Current configurations are small-LM experiments, not evidence of a broadly capable agent |
| Tokenizer | Urdu/Arabic/Chinese samples round-trip, but unfamiliar text falls back to individual bytes | Losslessness is not efficiency or learned language coverage |
| Tiny corpus | 5,524,848 bytes, 108,319 lines; 1,777,645 tokens; 3.0607 chars/token; 236,158 tokens/s in the local measurement | Useful pipeline smoke data only |
| Split construction | Tokenizer fitting intersected about 89.7% of validation/test lines in the reproduction | Evaluation/tokenizer leakage must be fixed before trusting fertility or downstream comparisons |
| API lifespan | Real startup exposed stale model globals, loading/not-ready health, empty model list, and completion 503 | Passing route tests do not demonstrate a working model service |
| Agent | Scripted demo works without an LLM; native XRFM tool-use training is not demonstrated | Protocol execution is not model agency |
| Timeout | A 0.05-second tool timeout returned after about 0.502 seconds | Thread timeout does not bound wall time or terminate work |
| CI smoke | CPU smoke reached 60 steps, loss 6.24→2.207, and resumed | Useful basic optimization evidence; not exact-resume or GPU evidence |
| Data build | Retained 16/33 documents, emitted 998 tokens and splits 14/2/0; manifest licenses became `unknown` | Current pipeline is not yet a governed corpus builder |
| Package identity | PyPI name `xrfm` belongs to an unrelated project | A distribution name decision is required before publication |

## 2. New execution-breaking findings

Severity meanings: **Blocker** prevents a responsible training/release gate; **Critical** can expose data or violate a security boundary; **High** silently corrupts semantics, accounting, or evaluation; **Medium** blocks scale or maintainability.

### TRN-01 — Successful AMP optimizer steps are reported as skipped

**Severity: Blocker for GPU training.**  
**Code:** `src/xrfm/training/mixed_precision.py`, `MixedPrecisionLoader.step`; `src/xrfm/training/loop.py`, `train_step`/`training_loop`.

The wrapper treats the return value of `GradScaler.step()` as a boolean success signal. Standard optimizers such as AdamW normally return `None` after a successful step. The local successful-scaler probe called `optimizer.step()`, observed its `None` return, and then observed `MixedPrecisionLoader.step()` report `False`.

Consequences in the loop include incorrect `step_performed` accounting and skipped normal logging/checkpoint/validation behavior while schedule/current-step state can still advance. This is not evidence that physical GPU training was run; it is a direct API-semantics defect that must be fixed and then tested on CUDA.

**Required correction:** detect overflow/skip from scaler state/scale transition or an explicit optimizer-step hook, advance scheduler and global optimizer step only on an actual update, and add tests for successful step, overflow skip, accumulation boundary, and scheduler alignment. See [PyTorch AMP guidance](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html).

### TRN-02 — OOM recovery can corrupt gradients and diverge across ranks

**Severity: Blocker for distributed training.**  
**Code:** `src/xrfm/training/loop.py`.

CUDA OOM handling can alter accumulation state without completing the corresponding backward pass. Partial gradients are not reliably invalidated, and ranks can make different local decisions. In DDP, that can produce divergence or deadlock rather than a recoverable retry.

**Required correction:** fail the whole coordinated attempt cleanly, clear partial gradients, synchronize an abort decision across ranks, and retry only from a known checkpoint with a new memory configuration. Add single-GPU and two-GPU fault-injection tests. Silent local continuation is prohibited.

### TRN-03 — Checkpoints do not implement exact resume

**Severity: Blocker for long runs.**  
**Code:** `src/xrfm/training/checkpoint.py`, `src/xrfm/training/loop.py`, `src/xrfm/training/distributed.py`.

Checkpoint state omits required combinations of Python/NumPy/Torch CPU/CUDA RNG, scaler state, epoch and sampler/dataloader position, and tokenizer/data-manifest identity. `DistributedSampler.set_epoch()` is not called by the loop, so shuffle order is not epoch-correct even before resume is considered.

**Required correction:** atomic checkpoints with complete state and identity hashes; a resume test that compares successive batches, RNG draws, optimizer/scaler/scheduler state, metrics, and final parameters against an uninterrupted run. “A checkpoint loads” is not acceptance evidence.

### TRN-04 — Dtype selection does not implement its claimed fallback

**Severity: High; Kaggle-relevant.**  
**Code:** `src/xrfm/training/mixed_precision.py`.

Any CUDA device selects bfloat16. There is no real capability-based fallback to float16 despite comments suggesting one. This matters for T4/P100-class hardware and must not be assumed away.

**Required correction:** explicit dtype policy plus actual numerical smoke tests on the target device: FP32, FP16+scaling where appropriate, and BF16 only when supported and stable. Record device, driver, PyTorch, throughput, memory, and gradient-overflow behavior.

### TRN-05 — Variable-length collation hardcodes padding ID 0

**Severity: High.**  
**Code:** `xrfm_collate_fn` in `src/xrfm/training/distributed.py`.

Probe output for a two-row batch was `[[2044, 2045], [7, 0]]`; the second row was padded with input ID `0`, not an injected tokenizer/model pad ID. If ID 0 is a real token or the model does not mask it consistently, semantics and loss can be corrupted.

**Required correction:** explicit special-token contract, correct padding and attention masks, labels padded with the ignore index, and tests using a vocabulary whose ID 0 is not the pad token.

### TRN-06 — An explicit attention mask disables causality

**Severity: High.**  
**Code:** `src/xrfm/models/attention/multi_head.py`.

The causal mask is created only when no explicit mask is supplied. A padding mask therefore replaces, rather than composes with, the causal constraint. In an executable probe, editing only token 1 changed token 0's output by **4.525887489318848** under an all-ones explicit mask; a causal decoder should leave token 0 unchanged.

The current standard collator does not return a mask, so this is not claimed to affect every existing training run. It becomes execution-critical as soon as padding/packed masks are used.

**Required correction:** combine causal, padding, and document-boundary masks with tested broadcast semantics in both optimized and fallback paths. Add future-token invariance and KV-cache equivalence tests.

### INF-01 — Top-p is ignored when top-k is present

**Severity: High for inference reproducibility.**  
**Code:** `src/xrfm/inference/sampling.py`.

The function returns through the top-k branch before top-p is applied. A direct probe with both supplied showed that even an invalid `top_p` was silently ignored; the API's Pydantic layer would reject that particular numeric value, but valid top-p values are still bypassed by the same branch.

Prompt truncation can also desynchronize generated text from API token accounting.

**Required correction:** define and test one ordering for temperature, repetition penalty, top-k, and top-p; test the joint distribution/filters and prompt/completion usage after truncation.

### AGT-01 — Credential redaction fails for common forms

**Severity: Critical.**  
**Code:** `sanitize_observation` in `src/xrfm/agent/safety.py`.

Synthetic probes produced:

```text
api_key=supersecret123  -> api_key=supersecret123: [REDACTED]
Bearer abcdefghijklmnop -> Bearer abcdefghijklmnop: [REDACTED]
password=hunter2        -> password=hunter2: [REDACTED]
api_key: supersecret123 -> api_key: [REDACTED]
```

The replacement logic preserves the original match when there is no colon. These were synthetic test strings, not user credentials.

**Required correction:** replace the secret capture group rather than the label based on string splitting; add multiline, Unicode, nested-object, Bearer, equals/colon, and log-export tests. Treat redaction as defense in depth, not permission to send secrets to the model.

### AGT-02 — Filesystem result bounds are bypassable and reads are unbounded in memory

**Severity: Critical for public agent exposure.**  
**Code:** `SafeFilesystemTools` and the hand-written schema validator in `src/xrfm/agent/tools.py`.

Schemas specify maxima but no minima. `read_file(max_chars=-1)` passed validation and returned **10,010 characters from a 10,011-character file**, bypassing the intended bound. The implementation also calls `read_text()` on the entire file before slicing, so even a positive character limit does not bound memory. Directory listing sorts/materializes traversal results before applying its limit.

**Required correction:** full JSON Schema validation including minima, internal enforcement independent of caller schema, bounded streaming reads, aggregate output byte/depth/item limits, and traversal deadlines. Add adversarial tests for negative/zero/huge limits, deep nesting, large files, symlink/race cases, and enormous directories.

### AGT-03 — Public exposure fails open when no API key is configured

**Severity: Critical deployment default.**  
**Code:** `api/routes/agent.py`.

`_check_api_key` permits requests when `XRFM_AGENT_API_KEY` is unset. The same API exposes filesystem-read tools rooted at the current working directory by default. This is not claimed to have been remotely exploited; it is an unsafe configuration path.

**Required correction:** fail startup for non-loopback/public deployments without explicit authentication and an approved root; bind locally by default; add authorization, rate/concurrency limits, and deployment-mode tests. Read-only does not mean non-sensitive.

### AGT-04 — Timeouts and streaming do not provide their advertised boundaries

**Severity: High/Critical depending on tool.**  
**Code:** `src/xrfm/agent/execution.py`, `src/xrfm/agent/runtime.py`, `api/routes/agent.py`.

A worker thread cannot be killed by `Future.result(timeout=...)`; executor shutdown can wait for it. The previously measured 0.05-second timeout returned after about 0.502 seconds. The runtime deadline is checked only around loop iterations, not inside backend calls or every tool call. `AgentRuntime.stream()` collects all events from `run()` before yielding them, so the API does not truly stream agent progress.

**Required correction:** deadline propagation, cooperative cancellation for safe in-process tools, process/container isolation with hard resource limits for untrusted or blocking tools, and a queue-backed streaming state machine. Test disconnect, cancellation, blocked backend, blocked tool, and total wall time.

### AGT-05 — Backend capabilities and structure are asserted, not established

**Severity: High.**  
**Code:** `src/xrfm/agent/backends.py`.

Every OpenAI-compatible endpoint is reported as supporting native tools, structured output, and reasoning. The request does not actually advertise the registry's tools or request JSON-Schema output. Native provider call IDs are not preserved. Greedy `{.*}` extraction is not a strict protocol parser, and malformed decision types can escape as errors.

**Required correction:** explicit capability configuration/probing, real tool/schema payloads where supported, XGrammar/provider constraints where available, strict parsing and validation, provider-ID preservation, and compatibility tests for each supported server. Constraints never replace execution-time authorization.

### AGT-06 — Tool output schemas and request isolation are incomplete

**Severity: High.**  
**Code:** `src/xrfm/agent/execution.py`, `src/xrfm/agent/protocols.py`, `api/routes/agent.py`.

`ToolSpec.output_schema` is not enforced. Observation sanitization limits individual strings but not total bytes, item count, or nesting. `ToolResult.duration_ms` is not populated despite timing being started. A singleton runtime shares a growing audit log across requests.

**Required correction:** validate inputs and outputs using JSON Schema 2020-12, enforce aggregate limits, create per-request state/audit with retention limits, redact audit details, and record accurate timings and cancellation outcome.

### DATA-01 — Near-deduplication and splitting do not scale or guarantee isolation

**Severity: High for corpus work.**  
**Code:** `src/xrfm/data/dedup.py`, split/tokenization pipeline.

Near-deduplication is pairwise **O(n²)** and in-memory. Separately, tokenizer fitting was reproduced as intersecting about 89.7% of validation/test lines. Text splits being line-disjoint after shuffling does not repair tokenizer or near-duplicate leakage.

**Required correction:** streaming exact dedup plus MinHash/LSH or equivalent candidate generation; source/document-group-aware split before tokenizer fitting; benchmark decontamination before training; publish cluster and exclusion manifests.

### API-01 — Model service readiness and contract tests are insufficient

**Severity: High.**  
**Code:** `api/main.py`, `api/routes/*`, API tests.

The real-lifespan probe reproduced stale model globals, not-ready health, empty model listing, and completion 503. Existing tests accept loading/healthy ambiguity and do not require a real completion.

**Required correction:** one lifecycle owner for model/tokenizer/engine state, readiness distinct from liveness, non-200 readiness where appropriate, deterministic startup failure, and lifespan tests requiring `/v1/models` plus a real generation after a valid test checkpoint loads.

## 3. Why the green suite is still valuable—but insufficient

The 505 tests establish many local invariants and make the repository worth repairing. They did not catch the defects above because important boundaries are absent or mocked:

- physical CUDA dtype/overflow/OOM behavior;
- exact resume and sampler order;
- joint causal/padding masks;
- real API startup and real completion;
- adversarial schema values and secret shapes;
- wall-clock cancellation and genuine streaming;
- native model tool-use success;
- corpus-scale deduplication and contamination.

The first engineering milestone is therefore **not more features**. It is converting each verified defect into a regression test and making the existing architecture trustworthy at small scale.

## 4. Research synthesis: what changed in the Urdu landscape

Primary sources are linked here and tabulated in [`05_TOOL_DATA_EVAL_MATRIX.md`](05_TOOL_DATA_EVAL_MATRIX.md).

### Native/from-scratch work is no longer a blank field

[UrduLM](https://arxiv.org/html/2601.17664) reports a 33 GB, roughly 13-million-row Urdu corpus, an estimated 5–6B tokens, custom BPE tokenizers, and 100M-class variants trained for three epochs. Its own limitations acknowledge domain/dialect imbalance, translated/adapted evaluations, LLM-judge bias, lack of systematic bias analysis, and that 1B+ is needed for production quality. The paper says data usage is limited to research purposes; artifact terms must be checked independently rather than inferred from “open-source release.”

**Implication:** XRFM should cite and reproduce relevant baselines where licenses permit. Its contribution must be bilingual/native code-switching, verifiable agency, stronger governance, and reproducible independent evaluation—not a claim to be the first Urdu LM.

### Continued-pretrained Urdu systems are baselines, not the native deliverable

[Alif](https://github.com/traversaal-ai/alif-urdu-llm) and [Qalb](https://arxiv.org/html/2601.08141) demonstrate continued pretraining and Urdu instruction tuning on stronger pretrained foundations. They are important comparators and possible teachers subject to license/privacy/cost review. They do not satisfy this program's definition of native-from-random-initialization weights.

### Evaluation is now strong enough to prevent self-deception

- [UrduMMLU](https://arxiv.org/html/2606.07167v2): 26,389 primarily native Urdu MCQs across 26 subjects/five domains.
- [UrduBench](https://github.com/traversaal-ai/urdubench_leaderboard): human-validated translated reasoning evaluations; translated origin must remain labeled.
- [UrBLiMP](https://aclanthology.org/2026.findings-acl.29/): grammatical minimal pairs for Urdu linguistic competence.
- [Belebele](https://huggingface.co/datasets/facebook/belebele): Urdu in Arabic and Latin scripts, useful for script-sensitive reading evaluation.
- [UQuAD+](https://doi.org/10.1145/3759455): 20,000 human-annotated native Urdu MRC pairs with answerable/unanswerable questions; reported CC BY-NC-SA makes it evaluation-only for this program unless separately approved.
- [RUAB](https://aclanthology.org/2026.abjadnlp-1.37/): 400 safety-relevant intents in English, mixed Roman Urdu, and transliterated Roman Urdu (1,200 inputs), human reviewed; its paper reports safety degradation under code-switching/transliteration.

**Implication:** a single aggregate “Urdu score” is prohibited. Report per script/language/domain, confidence intervals, contamination status, and human evaluation. Do not train on these test sets.

### Roman Urdu needs natural data and provenance, not only transliteration

[Roman-Urdu-Parl](https://dl.acm.org/doi/10.1145/3464424) reports 6.37M parallel pairs. Public mirrors carry conflicting license labels, so a mirror's metadata is not sufficient approval. [Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar) is a useful transliteration candidate with a documented CC0/CC-BY scheme, but its current Hub viewer reports a schema-generation failure; direct-file validation is required. The UCI/GitHub Roman Urdu resources also show conflicting license labels across mirrors.

**Implication:** keep natural Roman Urdu, mechanical transliteration, translated text, and synthetic code-switching as separate strata. Never let synthetic Roman text silently dominate the “natural Roman Urdu” metric or training mixture. Private chats require explicit consent and provenance; scraping personal messages is not an acceptable shortcut.

### Tooling should be assembled, not reinvented

- **Data:** [DataTrove](https://github.com/huggingface/datatrove) is the default candidate for scalable processing; [NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) is optional when its GPU capabilities are measurably needed.
- **Language tooling:** [GlotLID](https://github.com/cisnlp/GlotLID) for initial language/script labels and [UrduHack](https://github.com/urduhack/urduhack) for Urdu-specific diagnostics/normalization experiments. Neither is a reliable Roman-Urdu/code-switch gold labeler by itself.
- **Training:** keep repaired XRFM for the small control/pilot; bake off [OLMo-core](https://github.com/allenai/OLMo-core) and [TorchTitan](https://github.com/pytorch/torchtitan) only before scaling. Avoid building a new distributed framework.
- **Post-training:** [TRL](https://github.com/huggingface/trl) after a Hugging Face-compatible model contract exists; [Open Instruct](https://github.com/allenai/open-instruct) is an optional reference for later verifiable/RL workflows.
- **Evaluation:** [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) or Lighteval behind a pinned XRFM adapter; custom Urdu/agent tasks remain versioned local tasks.
- **Agency:** [MCP](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) for interoperability, [jsonschema](https://github.com/python-jsonschema/jsonschema) for Draft 2020-12 validation, [XGrammar](https://github.com/mlc-ai/xgrammar)/[vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) for supported constrained decoding, and [AgentDojo](https://github.com/ethz-spylab/agentdojo) for injection evaluation.
- **Interactive agent evaluation:** [τ-bench](https://github.com/sierra-research/tau2-bench) is a reference for final-state and user-simulation evaluation; its built-in language packs do not remove the need for native Urdu/Roman-Urdu tasks.

## 5. License and openness caution

A repository license, a dataset-compilation license, and rights in underlying documents are different things. FineWeb2, for example, states ODC-By for the dataset and also points to [Common Crawl's Terms of Use](https://commoncrawl.org/terms-of-use). Common Crawl does not relicense every crawled page. Therefore this plan uses:

- **Tier A:** owned/commissioned, public-domain, or explicit commercial-compatible source terms suitable for the intended use;
- **Tier B:** potentially usable but requiring source-level/legal review, including web-compilation terms and share-alike questions;
- **Tier C / HOLD:** noncommercial, no-derivatives, unknown, conflicting, or unprovenanced material; excluded from default training.

No external corpus is marked “approved” by this audit. Open-source project code does not automatically make every training source open licensed. The data steward must approve source records before corpus freeze.

## 6. Release-blocking conclusion

The current project should remain a local research prototype. The following are all **NO-GO** today:

1. multi-hour or multi-GPU training;
2. claims of exact resume or hardware-portable mixed precision;
3. a public agent endpoint with filesystem access;
4. claims of safe credential redaction, bounded tool output, or hard timeouts;
5. claims of native Urdu/Roman-Urdu/model-tool capability;
6. corpus training using `unknown` or conflicting license metadata;
7. benchmark claims based on the current split/tokenizer pipeline.

The fastest responsible path is to repair the small system, establish evaluation/data governance, then run a modest native pilot. The detailed sequence and objective gates are in the implementation plan and ledger.
