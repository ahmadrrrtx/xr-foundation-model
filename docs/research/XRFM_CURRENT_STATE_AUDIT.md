# XRFM — Current-State Audit (Agentic Program, Phase 0)

**Date:** 2026-09-22 · **Auditor:** XR lead ML systems engineering (agent session)
**Repository:** https://github.com/ahmadrrrtx/xr-foundation-model @ `main` (HEAD `fb470fb`, "feat(phase-1): build XRFM data foundation")
**Environment of this audit:** CPU-only sandbox — 2 vCPU Xeon-class, 2 GB RAM, Python 3.13.14, PyTorch 2.14.0+cpu (freshly installed), Debian 13.

> **Method.** Every claim below marked **[EXECUTED]** was verified by running code in
> this audit session; claims marked **[READ]** were verified by reading source; claims
> marked **[DOC]** come from repository documentation and were cross-checked against
> source where they matter. Documentation was treated as untrusted until verified.

---

## 1. Executive summary

XRFM today is **one coherent, tested, honest small-scale LM system** — not a chatbot,
not an agent, and not a frontier model:

* A from-scratch decoder-only transformer (~1.3 M params in the SMALL preset) with
  RoPE, RMSNorm, SwiGLU, pre-norm residuals, weight tying, and a verified KV cache. **[READ]**
* A byte-level BPE tokenizer (vocab 2048; specials `<|pad|>=2044, <|bos|>=2045,
  <|eos|>=2046, <|unk|>=2047`) with lossless Unicode round-trip. **[EXECUTED]**
* A serious Phase-1 data pipeline (document schema → normalization → language/quality/PII
  filters → dedup → splits → mixing → tokenization → packing → sharding → manifests). **[READ]**
* A seeded, resumable training loop with real validation; historical evidence: 5,000-step
  run on ~1.77 M-token corpus → val PPL 301.55 on a 1.32 M-param model. **[DOC, consistent with code]**
* **492 passed, 1 skipped** in the full test suite on this machine (219 s, torch 2.14+cpu). **[EXECUTED]**

What XRFM is **not** today, verified directly:

* The committed/packaged model has **no instruction-following, no chat format, no
  roles, no tool calling**. An untrained instance generated `' horroundverroundroundal,
  ra light light light'` for "Hello world" — expected for random weights, but it also
  demonstrates there is no chat template or structured-output machinery anywhere in
  the codebase. **[EXECUTED]**
* The best trained checkpoint (historical, `checkpoint_step_5000.pt`, not committed —
  only the incompatible legacy `checkpoint_step_500.pt` with vocab 50304 is in git)
  produces **grammatical but degenerate repetitive prose** per the project's own
  reports. It cannot be used as an agent's language engine. **[DOC + READ]**
* There is **no agent runtime of any kind**: no tool protocol, no decision engine,
  no execution loop, no permissions, no memory beyond the `search/` RAG helper. **[READ]**

**Conclusion:** the neural-model track and the agentic track must be separated. The
existing transformer stack is a legitimate *research track and future native backend*,
but the agentic system must run on a *model adapter layer* that can use capable
open-weight models now and XRFM's own model later, without rewriting the runtime.

---

## 2. Repository map (what actually exists)

```
src/xrfm/            THE library (src-layout; only thing the wheel ships)
├── config/          typed+validated dataclasses (ModelConfig/TrainingConfig/DatasetConfig/XRFMConfig), YAML loader
├── tokenization/    Tokenizer ABC + byte-level BPE (vocab.json packaged)
├── data/            Phase-1 pipeline: 25 modules (schema→pipeline), CLI `xrfm data`
├── models/          XRFMModel(=GPTModel): embedding, MHA+RoPE, RMSNorm, SwiGLU, blocks
├── training/        Trainer facade + TrainingLoop (DDP/FSDP hooks), optimizer/scheduler/AMP/checkpoint/metrics
├── inference/       generate() functional API + GenerationEngine (KV cache) + sampling primitives
├── evaluation/      evaluate() → PPL + top-1/top-k benchmarks
├── optimization/    SDPA wrapper, INT8/INT4 quant, speculative decoding (experimental)
├── search/          local RAG (indexer/retriever/agent) — the ONLY "agent-like" code; no tools, no loop
├── research/neurotopo/  experimental non-transformer research model (isolated; 16 phase tests)
└── experiment/      run tracking

api/                 FastAPI server: /health /v1/models /v1/completions(+stream) /v1/tokenize /v1/search /metrics
scripts/             train/eval/smoke entry points
config/              tiny.yaml (264K), config.yaml (SMALL 1.3M), medium.yaml, v1.1-medium.yaml
model/, tokenizer/, training/, inference/, evaluation/, optimization/   DEPRECATED root shims (repo-only)
docs/                architecture, ADR-0001/0002, audits, research, phase reports (~40 docs)
tests/               40 test files; 492 passing [EXECUTED]
data/datasets/       corpus.txt (~1.77M tokens public-domain), golden dataset under tests/data
```

Dependency direction is enforced and acyclic: `config → tokenization → data → models →
training → inference/evaluation`; `optimization/search/research` consume core; `api/`
depends on `xrfm`, never the reverse (verified by `tests/test_package.py`). **[READ]**

## 3. Execution-flow trace (instantiation → generation)

1. `xrfm.load_config(path|None)` → `XRFMConfig` (validated in `__post_init__`; cross-field
   rules like `d_model % n_heads == 0`; no CWD dependence — packaged default YAML). **[READ]**
2. `XRFMModel(cfg)` → embedding (weight-tied, `padding_idx=cfg.model.pad_token_id`) →
   N × `TransformerBlock`(RMSNorm → MHA(explicit causal additive mask, RoPE half-split,
   optional SDPA) → RMSNorm → SwiGLU) → final RMSNorm → tied `lm_head`. Forward returns
   `(logits, present_key_values|None)`; accepts 1-D or 2-D `input_ids`. **[READ]**
3. `xrfm.generate(model, prompt, tokenizer, ...)` → `GenerationEngine.generate`:
   prompt forward with `use_cache=True` → per-token loop with KV reuse → sampling
   (greedy/temperature/top-k/top-p, repetition penalty, stop token, stop sequences,
   max_seq_len guard, batch==1 only) → `GenerationResult(text, token_ids, prompt_token_ids)`. **[READ]**
4. API server: lifespan loads packaged tokenizer, builds `GPTModel(vocab_size=tok.vocab_size())`,
   opportunistically loads the newest `checkpoints/checkpoint_step_*.pt` (strict state dict),
   exposes completions/stream/tokenize/search. **[READ]**

Verified by execution in this session: model construction (214 K-param 2-layer test
model), tokenizer special-token contract, greedy generation path end-to-end. **[EXECUTED]**

## 4. Architectural strengths (keep — do NOT rewrite)

| Component | Why it's a strength |
|---|---|
| Typed config system (`config/schema.py`) | Validation-at-construction, cross-field invariants, exact dict round-trip, hashable configs → reproducible runs. The agentic runtime should copy this pattern. |
| Tokenizer contract (`tokenization/interface.py`) | Explicit special-token ids; `decode(encode(x)) == x` for all Unicode; no hard-coded pad ids anywhere downstream. |
| `GenerationEngine` | Correct KV-cached loop with stop tokens/sequences, repetition penalty, seq-len guard; ground-truth tested (`test_audit_verification.py` checks cache equivalence ≈1e-7). |
| Explicit causal masking in MHA | Post-audit fix; manual path and SDPA path both causal, tested. |
| Training loop + checkpointing | Seeded, resumable (LR schedule state saved), `ignore_index` loss masking, config+seed embedded in checkpoint `extra`, weights_only-safe loading. |
| Phase-1 data pipeline | Document-centric with manifests/dataset_id/checksums — directly reusable for future tool-use SFT data. |
| Package discipline (ADR-0001) | One-way deps, src-layout, wheel ships only `xrfm`, tests enforce boundaries. New `xrfm.agent` work must respect this. |
| Honesty culture | README/FINAL_REPORT state limitations plainly (PPL 301, degenerate text, unverified DDP). The agentic layer inherits this standard. |

## 5. Weaknesses / missing components (for the agentic mission)

**Missing entirely (must be built):**
1. Chat/role formatting — no chat template, no system/user/assistant/tool message types.
2. Tool protocol — no tool schema, registry, validation, execution.
3. Agent execution loop — no plan/decide/act/observe/verify state machine.
4. Decision representation — nothing distinguishes ANSWER vs ACTION vs ASK vs REFUSE.
5. Permissions/safety — no capability model, no sandboxing, no audit log, no confirmation flow.
6. Memory — no conversation/session memory abstraction (`search/` is a flat TF-IDF-ish RAG index, not memory).
7. Model adapter layer — everything is hard-bound to `XRFMModel`; no way to plug external models (transformers/GGUF/OpenAI-compatible).
8. Structured output / constrained decoding — sampling has top-k/top-p but no grammar/schema constraints, no JSON mode.
9. Streaming to callers as typed events — API streams raw text deltas only; no event protocol (tool calls, decisions, observations).
10. Agent evaluation — benchmarks are PPL/top-k only; no tool-selection/argument/task-completion suite.

**Weak but present:**
* `GenerationEngine.generate` is batch==1 only; `generate_batch` ignores KV cache (O(n²) per step). Acceptable at this scale; documented.
* `inference/kv_cache.py` preallocated buffer is EXPERIMENTAL/unused (dead-or-experimental doc says so). Leave it.
* `optimization/` (quant, speculative) unexercised at scale; secondary by ADR. Leave it.
* API picks "latest checkpoint by filename sort" — brittle for multi-run workspaces; the agent server will use explicit model selection instead.
* No repetition of *stop-sequence trimming*: generation stops when a stop sequence appears but returns the full token stream; callers must trim. Minor.

## 6. Code quality assessment

* **Tests:** 492 passing [EXECUTED]; strong ground-truth tests (RoPE/RMSNorm/SwiGLU vs
  independent references; causality probes; KV-cache equivalence; tokenizer round-trips;
  config validation; pipeline determinism). This is above-average rigor for a repo this size.
* **Typing/lint:** py.typed, mypy configured (not fully green historically), ruff+black at
  line-length 120; CI runs lint/type/test.
* **Docs:** unusually good audit trail (FORENSIC_AUDIT 49 findings, GAP_ANALYSIS, FINAL_AUDIT,
  ADRs). Several docs contain *historical* claims (e.g. ROADMAP "all phases complete") that
  contradict the honest README — the README/final-report framing supersedes them.
* **Duplication:** root-level shims (`model/`, `tokenizer/`, …) duplicate pre-Phase-0 paths
  intentionally (deprecated, removal ≥ v2.0). `research/neurotopo` duplicates some top-level
  neurotopo packages via deprecation re-exports. Not to be touched.

## 7. Components that must NOT be rewritten (control surfaces)

1. `models/` transformer math + `attention/` (ground-truth tested; the CONTROL model for NeuroTopo research).
2. `tokenization/` byte-level BPE + contract.
3. `training/` loop, checkpointing, scheduler state.
4. `data/` Phase-1 pipeline (needed for future SFT corpora).
5. `config/schema.py` validation pattern (extend, don't replace).
6. Public API surface in `xrfm/__init__.py` (`load_config`, `XRFM`, `generate`, `evaluate`, `Trainer`, …).
7. All 492 existing tests must keep passing after agentic work lands.

## 8. Components that must change / be added

* **Add** `src/xrfm/agent/` (new subpackage): protocols, backends, runtime, tools, decision,
  execution, memory, safety, evaluation. Depends on core one-way (core must not import agent).
* **Extend** (not rewrite) `inference/`: the native XRFM backend wraps `GenerationEngine`;
  structured-output/constrained decoding for the native path is an agent-layer concern
  (token-level logit masking hook) — a later phase, since the native model can't tool-call yet anyway.
* **Add** an agent server app (`agent_api/` or extend `api/`): event-streaming agent endpoint.
* **Keep** `api/` completions server untouched for the raw-model product surface.

## 9. Technical debt & dependency risks

| Item | Risk | Mitigation |
|---|---|---|
| torch is a hard dependency of the core package | Agent runtime would inherit a ~2 GB dep even for API-only use | Agent layer must import torch lazily/only in the native backend; the OpenAI-compatible and scripted backends must work torch-free. **[DESIGN RULE]** |
| Deprecated root shims | Confusion only; scheduled removal ≥ v2.0 | None needed |
| Legacy committed checkpoint (vocab 50304) | Incompatible; kept as evidence | Do not load it in agent paths; native backend must verify vocab coherence before serving |
| DDP/FSDP unverified on GPU | Training-scale risk, not agent risk | Out of scope for this program |
| mypy not fully green | CI hygiene | Agent code ships fully typed; don't regress core |
| NeuroTopo research bundle | Experimental; isolated by ADR | Do not couple agent runtime to it |

## 10. Security concerns (current)

* API server has rate limiting + security headers middleware **[READ]**, but no auth; it
  only does text completion — no tool execution — so blast radius is small today.
* The `search/` agent indexes local text and grounds answers; no execution capability.
* **The moment tools are added, everything changes:** tool execution needs a capability/
  permission model, sandboxed filesystem scope, command allowlists, confirmation gates,
  audit logging, and prompt-injection hygiene (tool output is untrusted data, never
  instructions). This is designed in `docs/research/AGENT_RUNTIME_RESEARCH.md` and
  `docs/XRFM_AGENTIC_ARCHITECTURE.md`.

## 11. Inference & training limitations (facts to design around)

* Native model: 2048-token vocab, ~3.1 chars/token efficiency, 256–512 max_seq_len presets,
  batch-1 KV-cached generation at ~2,300 tok/s *training* throughput on 2 vCPU (generation
  slower per token due to Python loop). Context is far too small for agentic transcripts
  (a single tool schema list exceeds it). **[EXECUTED/DOC]**
* Trained checkpoints are prose-LMs with repetition degeneration; zero tool-use training data
  exists in the pipeline today.
* Therefore: **the first release of the XR agent runtime runs on external open-weight
  models via adapters; the native XRFM backend exists, is tested, and is honestly labeled
  "research/limited".** The training track to close this gap is specified in
  `docs/research/XRFM_TRAINING_STRATEGY.md`.

## 12. Verdict

The repository is a well-engineered *small LM research system* with an honest audit trail
and enforced package discipline. It contains **zero agentic capability**. The correct move
is additive: build `xrfm.agent` as a model-agnostic runtime with a backend adapter layer,
reuse (never rewrite) the verified core, keep all 492 tests green, and state plainly in all
docs which layer is trained by XR (small transformer, research track) and which layer runs
on existing open-weight models (the agent's language engine, v1).
