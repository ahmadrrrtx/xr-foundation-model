# XRFM Architecture (Phase 0)

**Status:** current as of Phase 0 (2026-09-18). Companion documents:
`docs/adr/0001-xrfm-core-architecture.md` (why) and
`docs/architecture/PHASE_0_AUDIT.md` (what the architecture looked like
before and why it changed).

---

## 1. Package boundary — one sentence

**`xrfm` is the library; everything else in the repository is an
application, a script, a preset, or data.**

The library lives at `src/xrfm/` and is the only thing `pip install`
ships. The API server (`api/`), training/eval scripts (`scripts/`),
microbenchmarks (`benchmark/`), YAML presets (`config/`), corpora
(`data/`), and UI assets (`webui/`, `vercel_app/`) are *consumers* that
depend on `xrfm` — never the reverse.

## 2. Directory tree and module responsibilities

```text
src/xrfm/
├── __init__.py          # THE public API (see §3). Everything else is internal.
├── py.typed             # PEP 561: package is type-annotated
├── cli.py               # minimal CLI: xrfm info / xrfm validate-config
├── config/              # typed, validated configuration
│   ├── schema.py        #   ModelConfig / TrainingConfig / DatasetConfig / XRFMConfig
│   ├── loader.py        #   YAML → XRFMConfig (explicit path | packaged default; never CWD)
│   └── config.default.yaml  # packaged default preset (XRFM-SMALL)
├── tokenization/        # tokenizer contract + byte-level BPE
│   ├── interface.py     #   Tokenizer ABC + special-token contract (pad/bos/eos/unk ids)
│   ├── bpe.py           #   BPETokenizer (historical name: BytePairEncoder) + vocab.json resource
│   └── encode.py/decode.py  # functional helpers
├── data/                # dataset system (representation ≠ splitting ≠ packing ≠ manifests)
│   ├── splits.py        #   deterministic line/document splits (seed is consumed; shuffle honored)
│   ├── packing.py       #   validated fixed-length chunking (overlap < max_seq_len enforced)
│   ├── dataset.py       #   TextDataset: padded (input, target) pairs; position-based loss masking
│   ├── manifest.py      #   dataset provenance (hashes, token counts)
│   └── loader.py        #   compat re-exports (pre-Phase-0 path)
├── models/              # the transformer
│   ├── gpt.py           #   XRFMModel (alias GPTModel): forward contract in the docstring
│   ├── embedding.py     #   XRFMEmbedding (weight tying, optional padding_idx)
│   ├── attention/       #   multi-head attention, RoPE
│   └── layers/          #   RMSNorm, SwiGLU, TransformerBlock
├── training/            # optimization loop + Trainer facade
│   ├── trainer.py       #   Trainer: stable public boundary (model + config → TrainResult)
│   ├── loop.py          #   TrainingLoop: steps, grad accumulation, DDP/FSDP hooks, resume
│   ├── optimizer.py / scheduler.py / mixed_precision.py
│   ├── checkpoint.py    #   save/load + reproducibility metadata (config hash, seed)
│   ├── distributed.py   #   DDP/FSDP wrappers, collate, distributed dataloader
│   └── metrics.py       #   JSONL metrics sink
├── inference/           # generation
│   ├── generate.py      #   generate(): the functional text-in/text-out API
│   ├── engine.py        #   GenerationEngine: KV-cached token-level loop
│   ├── sampling.py      #   greedy / temperature / top-k / top-p primitives
│   └── kv_cache.py      #   experimental preallocated cache
├── evaluation/          # model + batches → metrics (never needs the trainer)
│   ├── runner.py        #   evaluate(model, dataloader) → perplexity + accuracy suite
│   ├── perplexity.py    #   strided token-level perplexity
│   └── benchmarks.py    #   Benchmark ABC + TextCompletionAccuracy / TopKAccuracy
├── optimization/        # SECONDARY: SDPA attention, INT8/INT4 quantization, speculative decoding
├── search/              # SECONDARY: local RAG (indexer, retriever, agent) — depends on core
├── research/neurotopo/  # EXPERIMENTAL: the NeuroTopo research model, isolated from core
└── experiment/          # run tracking helpers (ExperimentRecord)
```

Repository-root directories that are **not part of the wheel**:

```text
api/            FastAPI server app (depends on xrfm; optional [api] extra)
scripts/        thin entry points (train, eval, smoke tests)
benchmark/      microbenchmarks
config/         per-size YAML presets (config.yaml / tiny / medium / v1.1-medium)
data/           training corpora + manifests
checkpoints/    run artifacts
webui/, vercel_app/, deployment/, notebooks/
model/, tokenizer/, training/, inference/, evaluation/, optimization/
                DEPRECATED import shims → xrfm.* (repo-checkout only; removed ≥ v2.0)
```

## 3. Public API

Users import from `xrfm` and do not need to know the internal layout:

```python
import xrfm

xrfm.__version__                      # single version source (pyproject reads it)

cfg    = xrfm.load_config("config/tiny.yaml")   # or load_config() → packaged default
model  = xrfm.XRFM(cfg)                         # = XRFMModel = GPTModel (aliases)
tok    = xrfm.BPETokenizer.pretrained()         # packaged vocab.json (works anywhere)
train  = xrfm.TextDataset(cfg.data, tok, split="train")
trainer = xrfm.Trainer(model, config=cfg)
result = trainer.train(train)                   # TrainResult(final_loss, checkpoint_path, ...)

out     = xrfm.generate(model, "Once upon a time", tokenizer=tok, max_new_tokens=64,
                        temperature=0.8, top_k=50)
metrics = xrfm.evaluate(model, val_loader)      # perplexity + top-1/top-5
```

Everything not reachable from `xrfm.__all__` is internal and may change
between minor versions. Two deliberate exceptions carry their own
compatibility promise:

* `xrfm.research.neurotopo.*` — experimental, but its *own* package; may
  change without notice (documented risk, not a core promise).
* The deprecated shims (`model.*`, `tokenizer.*`, ..., `xrfm.core`,
  `xrfm.neurotopo`, ...) — kept working until at least v2.0, then removed.

## 4. Dependency direction (acyclic)

```text
                 config ─────────────┐
                   ↓                 │
   tokenization → data ──────────────┤
                   ↓                 ↓
                 models ←──────── (config schema objects)
                   ↓
                training
                   ↓
        inference / evaluation
                   ↑
      optimization · search · research/*      (secondary, consumers only)
                   ↑
                api/ · scripts/               (repository applications)
```

Rules enforced by review and by tests (`tests/test_package.py`):

1. Nothing in core imports from `optimization`, `search`, `research`,
   `api`, or scripts.
2. `models` / `training` / `inference` / `evaluation` never parse YAML —
   they consume typed config objects from `xrfm.config.schema`.
3. The API server imports XRFM; XRFM never imports the API (verified:
   importing `xrfm` pulls in neither `fastapi` nor `uvicorn`).

## 5. Configuration system

* **Schema is executable truth** (`config/schema.py`): every field is
  validated at construction — including cross-field rules such as
  `d_model % n_heads == 0` and `warmup_steps < max_steps` — with errors
  naming the field and the received value. Invalid configs cannot reach
  model construction.
* **No CWD dependence**: `load_config(path)` for explicit files;
  `load_config(None)` uses the packaged default resource. `cd / && python
  -c "import xrfm; xrfm.load_config()"` works (tested).
* **No dead fields**: every `DatasetConfig` field has an effect; the seed
  is consumed by the split; YAML presets carry no version string.
* **Serialization round-trips**: `XRFMConfig.to_dict()` / `from_dict()`;
  checkpoints embed the serialized config and its hash.

## 6. Data system contracts

* Splits are line/document-boundary, deterministic, `seed`-aware
  (`shuffle=False` → sequential holdout; `shuffle=True` → seeded shuffle).
* Packing rejects `overlap >= max_seq_len` and `max_seq_len <= 0` with
  `PackingError` (the pre-Phase-0 code infinite-looped).
* Padding ids come from the **tokenizer contract**
  (`tokenizer.pad_token_id`), never hard-coded; loss masking is
  position-based, so content tokens that equal the pad id keep targets.
* The last real position of each chunk is masked (`-100`) — the model is
  no longer trained to emit the padding token at chunk ends.

## 7. Model / training / inference / evaluation systems

* `XRFMModel(input_ids) → (logits, cache)`; full forward contract in
  `src/xrfm/models/gpt.py`. Architecture unchanged in Phase 0 (RoPE,
  RMSNorm, SwiGLU, pre-norm, weight tying, KV cache).
* `Trainer` = stable boundary; `TrainingLoop` remains importable for
  research control. Device policy: config `training.device`, else auto.
* `generate(model, prompt, tokenizer, ...)` for text-level generation;
  `GenerationEngine` for token-level control; sampling primitives stay
  internal to `xrfm.inference`.
* `evaluate(model, dataloader)` requires only a model and batches.

## 8. Extension points

| Want to add... | Do this |
|---|---|
| a new tokenizer algorithm | subclass `xrfm.tokenization.Tokenizer`; implement the special-token contract |
| a new dataset format | produce line lists → `xrfm.data.splits` + `xrfm.data.packing`; or subclass `TextDataset` |
| a new benchmark | subclass `xrfm.evaluation.Benchmark`; plug into `evaluate` |
| a new optimizer/scheduler | extend `training/optimizer.py` / `scheduler.py`; expose via `TrainingConfig` |
| new model architectures | add to `xrfm/models/`; must accept `ModelConfig` (or its own typed config in `config/schema.py`) |
| research experiments | put them under `xrfm/research/` — never import them from core |

## 9. Explicitly out of scope for core (Phase 0)

RAG/search quality, the API server feature set, the web UIs, NeuroTopo
research, quantization/speculative decoding performance, distributed
scaling experiments, benchmark expansion, and any new model capabilities.
Secondary features must consume the core through the public API — PRs that
make core depend on a secondary layer violate this architecture (see the
ADR).
