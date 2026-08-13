# XRFM — Current State Audit

**Mission:** XRFM Frontier Architecture Research — Section 1 deliverable
**Date:** 2026-08-12
**Auditor:** Agent Mode (Arena.ai)
**Repository:** https://github.com/ahmadrrrtx/xr-foundation-model.git
**Audited commit (HEAD at clone):** `main` as of 2026-08-12 (the forensic-v2 remediation has been merged; HEAD ≈ `1876a33`-class state)
**Environment of this audit:** CPU-only, Python 3.13, PyTorch 2.6.0+cpu

> **Method.** Every structural claim below was verified by reading source and, where
> feasible, executing it in this sandbox (test run, model instantiation, parameter
> count, a 10-step training smoke test, tokenizer round-trip). Findings are marked
> **OBSERVED** (read/executed) or **RESEARCH-BACKED**. This audit does **not**
> modify the model; the existing GPT-style Transformer is preserved as the
> **CONTROL MODEL**.

---

## 1. Executive Summary

XRFM is a small, from-scratch, **decoder-only GPT-style Transformer** implemented in
pure PyTorch. It is *not* a foundation model in capability terms, but it is a
genuinely trainable, reproducible, tested LM *system* at small scale. A prior
forensic audit (2026-08-08) found and fixed 49 issues (broken causality,
character-level tokenizer, fake mixed precision, API import failure, non-resumable
checkpointing, toy data, vocabulary split-brain, etc.). The current state reflects
those fixes.

**What exists today (OBSERVED):**

| Subsystem | Status | Notes |
|---|---|---|
| Model architecture | ✅ Working | Pre-norm decoder Transformer: RMSNorm + MHA(RoPE) + SwiGLU, weight tying |
| Tokenizer | ✅ Working | Byte-level BPE v2; lossless Unicode round-trip verified |
| Training loop | ✅ Working | AdamW, cosine+warmup, grad accum, grad clip, seeded, resume, validation hook |
| Checkpointing | ✅ Working | Saves model/optim/scheduler/config/seed; `map_location=cpu`, `weights_only` |
| Inference | ✅ Working | KV-cached autoregressive generation; temp/top-k/top-p/repetition penalty/stop seq |
| Evaluation | ✅ Working | Token PPL (padding-corrected), strided PPL, top-k accuracy |
| Data pipeline | ✅ Working | Line-boundary splits, exact dedup, padding with `-100` loss masking |
| Distributed | ⚠️ Scaffold only | DDP/FSDP wrappers; single-process/CPU-tested, never run multi-GPU |
| Tests | ✅ 227 pass | 222 core + 5 API (verified this audit); includes 20 ground-truth tests |
| API / serving | ✅ Imports & runs | FastAPI; loads tokenizer + latest checkpoint |
| Optimization | ⚠️ Partial | Quantization (INT8/INT4) tested; flash-attn wrapper; speculative decoding scaffold |
| GPU readiness | ⚠️ Untested | bf16 autocast path exists but no GPU was available to verify |

**Headline metrics from prior executed run (SMALL config, CPU):**
1,318,528 params; 5,000 steps / 10.24 M tokens; val loss 7.24 → 5.71; **val PPL 301.55**
(random baseline ≈ 2048). This is a *system-correctness* result, not a quality result.

---

## 2. Current Architecture

### 2.1 High-level data flow

```
config.yaml ──► ConfigLoader ──► GPTModel
                                       │
input_ids ──► XRFMEmbedding (weight-tied) ──► [TransformerBlock × L] ──► RMSNorm ──► lm_head(=E^T) ──► logits
                                       ▲                                   ▲
                                       │                                   │
                            mask / past_kv (KV cache)              next-token CE loss
```

`GPTModel.forward` (`model/gpt.py`) returns `(logits, present_key_values)`. The
training loop shifts targets for next-token prediction; inference uses the KV cache.

### 2.2 Component detail

| Component | File | Specification (OBSERVED) |
|---|---|---|
| **Embedding** | `model/embedding.py` | `nn.Embedding(vocab, d_model, padding_idx=0)`, Xavier-uniform init; shares weight with `lm_head` |
| **Block** | `model/layers/transformer_block.py` | Pre-norm: `x = x + dropout(attn(norm1(x)))`; `x = x + dropout(ffn(norm2(x)))` |
| **Attention** | `model/attention/multi_head.py` | Manual MHA: W_q/k/v/o, head split, **explicit additive causal mask** (built in), SDPA/flash path as accelerator, manual fallback is causal by construction; KV cache |
| **RoPE** | `model/attention/rope.py` | Rotary embeddings, base 10000, **half-split** `(-x2,x1)` convention (not interleaved); supports cache offset; recomputes freqs each forward (perf, not correctness) |
| **RMSNorm** | `model/layers/rmsnorm.py` | `weight * x / sqrt(mean(x^2)+eps)`, eps=1e-6, learnable scale, no bias |
| **SwiGLU** | `model/layers/swiglu.py` | `W3(silu(W1(x)) * W2(x))`; 3 projections, Xavier init; bias configurable |
| **LM head** | `model/gpt.py` | `nn.Linear(d_model, vocab, bias=False)`, weight-tied to embedding by default |
| **Loss** | `training/loop.py` | `F.cross_entropy(..., ignore_index=-100)` over shifted targets; averaged over grad-accum cycle |

### 2.3 Available configurations

| Config | d_model | layers | heads | d_ff | seq | vocab (config) | Measured params | Non-emb params |
|---|---|---|---|---|---|---|---|---|
| `config/tiny.yaml` | 64 | 2 | 4 | 256 | 128 | 1024 | **198,592** | 133,056 |
| `config/config.yaml` (SMALL, default) | 128 | 4 | 4 | 512 | 256 | 2048 | **1,318,528** | 1,056,384 |
| `config/medium.yaml` | 384 | 8 | 6 | 1536 | 1024 | 8192 | **22,066,560** | 18,920,832 |

> The runtime passes the tokenizer's **actual** `vocab_size` to `GPTModel(...)`, so
> the config `vocab_size` field is a default/override target. Measured above with
> each config's declared vocab.

### 2.4 Tokenizer (OBSERVED, verified)

`tokenizer/bpe.py` is **byte-level BPE v2**: base vocabulary = 256 UTF-8 byte
values (latin-1 view, tiktoken convention); merges learned over byte sequences;
whitespace/newlines preserved; PAD/BOS/EOS/UNK reserved; `decode(encode(x)) == x`
for arbitrary Unicode. Verified this audit:

```
vocab_size: 2048
roundtrip: "Hello, world! 你好"  ->  "Hello, world! 你好"   (lossless)
```

Training the tokenizer on the first 200 KB of the corpus to target 2048 is fast
(seconds); full-corpus training to larger vocab is supported.

### 2.5 Data pipeline (OBSERVED)

`xrfm/data/loader.py`:
- Splits **by line boundaries** with exact-line dedup (no train/val leakage).
- Preserves newlines/paragraph structure.
- Chunks token IDs to `max_seq_len`; pads short chunks with a real `pad_id`.
- Targets: next token, padded positions → `-100` (ignored by CE).
- Corpus shipped: `data/datasets/corpus.txt` = **5,524,848 bytes, 953,961 words,
  108,319 lines** (≈1.77 M tokens at the tokenizer's rate; 11 public-domain books
  + a Python code slice). `sample.txt` (one paragraph ×100) remains only for smoke tests.

### 2.6 Training system (OBSERVED)

- **Optimizer:** AdamW, β=(0.9,0.999) [configurable; prior reports recommend (0.9,0.95)],
  eps 1e-8, weight decay 0.1. No parameter-group exclusions (norms/biases/embeddings
  also get wd — a minor deviation from OLMo-style recipes).
- **Scheduler:** linear warmup + cosine decay to 0; `state_dict`/`load_state_dict`
  implemented (resume restores LR correctly).
- **Loop:** gradient accumulation, DDP `no_sync`, grad-clip (norm 2, max 1.0),
  optional `error_if_nonfinite`, optional bf16 autocast on GPU (honest fp32 no-op on CPU),
  OOM-skip on CUDA, validation hook, JSONL metrics writer, seeded shuffling + worker seeds.
- **Checkpoint:** `model/optimizer/scheduler` state + `step/loss/best_loss` +
  `extra` (config, seed, ignore_index, PyTorch version); `torch.load(..., map_location="cpu",
  weights_only=True)`.

**10-step smoke test (this audit, TINY, CPU):** loss ≈ 7.6 (≈ ln 2048) → **7.01**
in 0.12 s for 10 steps — confirms the learning loop is live.

### 2.7 Inference & evaluation (OBSERVED)

- `inference/engine.py`: KV-cached `generate()` (prefill + single-token steps),
  temperature/top-k/top-p, `repetition_penalty`, `stop_token_id`, `stop_sequences`,
  optional `torch.compile`. Batch generation without cache.
- `inference/kv_cache.py` / `inference/sampling.py`: standalone KV cache class
  (unused by the engine, which uses a list-of-tuples — harmless dead code) and the
  sampling math (verified correct in prior audit).
- `evaluation/perplexity.py`: padding-corrected PPL (`ignore_index`), strided
  long-sequence PPL (overlap counted once), `evaluate_checkpoint` with timing.

### 2.8 Tests (OBSERVED, executed)

```
$ python -m pytest tests/ -q
222 passed (core), 5 passed (tests/test_api.py with fastapi+httpx installed)
```
Coverage includes ground-truth/reference tests for RoPE, RMSNorm, SwiGLU, weight
typing, KV-cache equivalence, causality, scheduler resume, tokenizer fidelity,
loss masking, determinism, and API import/startup.

---

## 3. Current Strengths

1. **Mathematically correct, readable Transformer.** Every core primitive
   (RMSNorm, RoPE, SwiGLU, MHA, weight tying, KV cache) has an independent-
   reference test. The code is small enough to read end-to-end — a strong base
   for a controlled *baseline* against any new architecture.
2. **Honest causality.** Causal masking is now built into the attention module
   and does not depend on an optimized-kernel import succeeding.
3. **Correct tokenizer.** Lossless byte-level BPE with real Unicode support — a
   prerequisite for any non-trivial experiment.
4. **Reproducibility is taken seriously.** Seeds, config-in-checkpoint, scheduler
   state, map_location, weights_only, deterministic-shuffle option.
5. **Real (if small) data and real metrics.** Line-split deduped corpus,
   padding-corrected PPL, strided evaluation, overfit test, executed baseline numbers.
6. **Fair-comparison discipline already present.** Prior work explicitly refuses
   to compare against models trained on different data/compute/tokenizer/params
   without a protocol row — exactly the discipline the Frontier mission requires
   for the Transformer-vs-NeuroTopo control experiment.
7. **Modular package layout.** `model/`, `training/`, `tokenizer/`, `inference/`,
   `evaluation/`, `xrfm/` are cleanly separated, making it feasible to add a
   sibling `xrfm/` neuro-topology stack without disturbing the control.

---

## 4. Current Weaknesses

1. **It is a tiny model trained on a tiny corpus.** 1.3 M params / 10 M tokens
   produce degenerate, repetitive text. The system works; the *model* is not capable.
2. **Distributed training is unverified.** DDP/FSDP are scaffolding; no multi-GPU
   run has ever executed. FSDP does not set `device_id`/`sync_module_states`.
3. **No external benchmarks.** Only intrinsic PPL/loss. No WikiText, MMLU,
   HellaSwag, etc. (meaningless at this scale, but absent).
4. **Tokenizer efficiency is modest.** 2048 vocab at ~3.1 chars/token is far from
   Llama-3-grade (~3.3 chars/token is actually *good* per-char; the issue is vocab
   coverage and merge quality on the small corpus, not the algorithm).
5. **Initialization is Xavier-everywhere.** No GPT-2 residual-stream scaling or
   Llama-style 0.02 small init — a known stability risk at 32+ layers (fine today).
6. **Weight decay is applied indiscriminately** (norms/biases/embeddings included).
7. **Dead/unintegrated code:** standalone `GradientAccumulator`, `KVCache`,
   speculative decoding, parts of `optimization/` are not wired into the engine.
8. **CI does not train** and mypy is not green by default (it fast-fails; cleanup needed).
9. **Corpus is English-only, 19th-century-prose-heavy** with a small code slice;
   no multilingual, no modern web text, no instruction/math/reasoning data.
10. **No architecture beyond dense self-attention.** There is no graph, memory,
    uncertainty, routing, or sparsity machinery — which is precisely the gap the
    Frontier mission targets.
11. **Generation degeneration** at small scale (repetition loops); repetition
    penalty now exists but no beam search / contrastive search.
12. **A legacy incompatible checkpoint** (`checkpoints/checkpoint_step_500.pt`,
    vocab 50304, empty optimizer state) is still on disk as evidence and cannot resume.

---

## 5. Current Bottlenecks

| Bottleneck | Nature | Impact on Frontier work |
|---|---|---|
| **Scale** | 1.3 M params / 10 M tokens | Cannot draw scientific conclusions about a new architecture at this size; need at least 15–100 M params + 1–10 B tokens |
| **Compute** | CPU-only sandbox historically; free-tier single GPU target | Limits how many architecture variants / ablations can be run; experimental design must be cheap |
| **Data** | 1.77 M-token English-only corpus | Need a documented free-license data strategy (FineWeb-Edu, Wikipedia, etc.) |
| **No distributed validation** | DDP/FSDP never run on real hardware | Scaling beyond one GPU is currently theoretical |
| **Attention cost** | Dense O(n²) manual MHA | This is the *compute* bottleneck the new architecture must address — and the control to beat |
| **No long-context support beyond max_seq_len** | No RoPE scaling, no sliding window | Long-context claims cannot be tested yet |
| **Evaluation is intrinsic-only** | No calibrated uncertainty / abstention / OOD harness | The uncertainty/abstention goals require a new evaluation stack |

---

## 6. Current Baseline Metrics

All numbers below are **executed** in-repo (not extrapolated), per `XRFM_FINAL_REPORT.md`
and this audit's verification.

### 6.1 SMALL baseline (the canonical control)

| Metric | Value |
|---|---|
| Config | `config/config.yaml` (XRFM-SMALL) |
| Parameters | 1,318,528 (weight-tied) |
| d_model / layers / heads / d_ff / seq | 128 / 4 / 4 / 512 / 256 |
| Vocab | 2048 (byte-level BPE) |
| Optimizer | AdamW (0.9,0.95), lr 3e-4, wd 0.1, clip 1.0, warmup 200, cosine |
| Data | `corpus.txt`, ~1.77 M tokens; 6,015 train / 360 val chunks |
| Tokens seen | 10.24 M (5,000 steps × batch 8 × seq 256) |
| Train loss | 7.69 → 4.78 (best 4.30) |
| **Val loss** | **7.24 → 5.7089** |
| **Val PPL** | **301.55** (random ≈ 2048) |
| Throughput | ~2,300 tok/s (~1.1 s/step) on 2-vCPU CPU |
| Wall time | ~73 min |
| Peak RSS | < 1 GB |
| Precision | fp32 (CPU) |
| Seed | 42 (reproducible) |

### 6.2 Overfit sanity test

TINY on a 6-document mini-corpus, 600 steps → train loss **0.097**; greedy
generation reproduces training text verbatim. Confirms the training loop can
actually fit data.

### 6.3 Scaling pair (equal data/compute/tokenizer)

TINY (264 K) vs SMALL (1.32 M), same ~1.64 M tokens → val PPL **777.1 vs 586.0**.
Larger wins at equal data — the expected scaling direction, and a template for the
controlled Transformer-vs-NeuroTopo comparison.

### 6.4 Test status

- **227/227 tests pass** (222 core + 5 API) in this audit.
- Prior forensic audit: 217/217; the delta reflects tests added since.

---

## 7. Current Compute Requirements

| Workload | Hardware observed | Wall time | Memory |
|---|---|---|---|
| Full test suite | 2 vCPU CPU | ~6–8 s | < 1 GB |
| SMALL 5,000-step baseline | 2 vCPU Xeon @ 2.6 GHz, no GPU | ~73 min | < 1 GB RSS |
| TINY overfit (600 steps) | same | ~8 s | < 512 MB |
| 10-step smoke (TINY) | same | 0.12 s | < 512 MB |
| MEDIUM (22 M params) | **not executed**; target free T4 / 1× GPU, bf16 | estimated hours-days depending on tokens | ~2–6 GB VRAM at seq 1024 bf16 (estimate) |
| LARGE (70–90 M) | **not executed**; requires 24 GB-class GPU | days | 16–24 GB VRAM (estimate) |

**Checkpoint storage:** ~16 MB/ckpt at SMALL (incl. optimizer state). A 5,000-step
run with `checkpoint_every=500` produced ~498 MB before pruning.

**Dependency footprint:** `torch`, `pyyaml`, `numpy` are the only runtime
requirements (`fastapi`/`uvicorn` for serving; `pytest` for dev). MIT/BSD-style
licenses throughout.

---

## 8. Implications for the Frontier Architecture Mission

1. **The CONTROL MODEL is ready.** The existing GPT-Transformer is small, correct,
   tested, and reproducible. It is an *ideal* ablation baseline: parameter-matched,
   same tokenizer, same data, same optimizer/scheduler, same compute budget. It
   must **not** be deleted or rewritten.
2. **The experimental harness (data/tokenizer/training/checkpoint/eval) can be
   reused.** New architecture code should live in a sibling package
   (`xrfm/core`, `xrfm/topology`, … per the mission's layout) and plug into the
   same `TrainingLoop`-style loop and evaluation, so comparisons are apples-to-apples.
3. **The biggest gap is scale and data, not code.** Any scientific conclusion about
   a neuro-topological architecture requires (a) ≥10–100 M params and (b) ≥1–10 B
   free-license tokens on at least one GPU. CPU-only results can only validate
   correctness/overfitting, not architectural superiority.
4. **No existing graph/memory/uncertainty/sparsity code to build on.** The Frontier
   design starts from a clean slate for the novel components; only the dense
   Transformer primitives and the training/eval scaffolding are reused.
5. **Honesty discipline is already cultural.** The repo's prior audits
   ruthlessly documented overclaiming. The Frontier work must continue that:
   "hypothesis / candidate / observed / not yet demonstrated," never "Transformer
   killer."

---

## 9. What Was Verified in This Audit (evidence)

- Read all core source: `model/gpt.py`, `model/attention/multi_head.py`,
  `model/attention/rope.py`, `model/layers/{transformer_block,rmsnorm,swiglu}.py`,
  `model/embedding.py`, `training/{loop,optimizer,scheduler,checkpoint}.py`,
  `tokenizer/bpe.py`, `xrfm/data/loader.py`, `inference/engine.py`,
  `evaluation/perplexity.py`, all configs, `pyproject.toml`, `requirements.txt`.
- Read prior audits/reports: `XRFM_FINAL_REPORT.md`, `AUDIT_CLOSURE.md`,
  `ARCHITECTURE_REVIEW.md`, `UPGRADE_PLAN_A_PLUS.md`,
  `docs/audit/{FORENSIC_AUDIT,BASELINE}.md`,
  `docs/architecture/XRFM_TARGET_SPEC.md`, `docs/research/MODEL_COMPARISON.md`,
  `DECISIONS.md`.
- Executed: full test suite (227 pass); instantiated all 3 configs and counted
  params; trained 10 live steps (loss decreased); verified byte-level BPE Unicode
  round-trip; measured corpus size.

---

## 10. Open Items Carried Forward

- Distributed (DDP/FSDP) validation on real GPUs — **unverified**.
- GPU bf16 path — **code exists, not executed** in this audit.
- MEDIUM/LARGE training runs — **not executed** (compute).
- External benchmark harness — **to be built** for Frontier evaluation.
- Uncertainty/abstention/OOD evaluation — **to be designed** (mission Section 12).
- Legacy `checkpoint_step_500.pt` — kept as evidence; should remain gitignored/
  documented, not used.

---

*This audit intentionally does not implement or modify the model. It establishes
the baseline against which every Frontier architecture candidate will be measured.*
