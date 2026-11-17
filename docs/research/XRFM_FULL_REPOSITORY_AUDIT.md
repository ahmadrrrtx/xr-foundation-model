# XR Foundation Model (XRFM) — Full Repository Forensic Audit

**Audit Date:** 2026-11-17 (Asia/Karachi)  
**Repository:** `https://github.com/ahmadrrrtx/xr-foundation-model`  
**Audited Branch:** `main`  
**Exact Commit SHA:** `90674742d8e6a5caf65336e4b64c949696e33bc0` (`Merge pull request #3 from ahmadrrrtx/phase-0-stabilization`)  
**Package Version:** `xrfm == 1.0.1` (`src/xrfm/__init__.py:32`)  
**Auditor Role:** Principal AI Research Scientist, Senior ML Engineer, Multilingual/Tokenizer Specialist, Cybersecurity AI Architect & MLOps Auditor  
**Authorization Scope:** Audit, Research, Safe Lightweight Testing, Documentation & Planning ONLY (no code rewrites, no model training, no paid compute, no GitHub push).

---

## 1. Phase Zero — Execution Environment & Repository Verification

### 1.1 Local Sandbox Hardware & Runtime Profile

All local verification commands were executed inside a strictly resource-constrained container:

| Property | Measured Value | Verification Command | Operational Implication |
|---|---|---|---|
| **OS / Kernel** | Linux `e2b.local` `6.1.189` x86_64 | `uname -a` | Standard Linux userland; no host GPU drivers |
| **Python** | `Python 3.13.18` | `python3 --version` | Newer than CI matrix (`3.10`–`3.12`); surfaces Python 3.13 compatibility |
| **System RAM** | `1.9 GiB` total (`~1.4 GiB` available, `0 B` swap) | `free -h` | **Strict OOM ceiling**: running full pytest in one process or loading >500M weights will OOM-kill |
| **Disk Space** | `25 GB` root, `20–21 GB` available (`~5.0 GB` used after CPU torch) | `df -h /home/user` | Sufficient for code, metadata, and lightweight CPU wheels; strictly forbids multi-GB corpus/weight downloads |
| **GPU / CUDA** | None (`nvidia-smi` not found; `torch.cuda.is_available() == False`) | `nvidia-smi`, `torch.cuda` | Zero physical CUDA/BF16/FP16/DDP/FSDP verification possible locally |
| **Git** | `git version 2.47.3` | `git --version` | Full shallow-clone & history inspection available |
| **Network** | Active HTTPS (`HTTP 200` to GitHub, Hugging Face, Kaggle docs) | `curl -sI` | Metadata API queries and primary-source documentation verification enabled |

**Packages Installed Specifically for Safe Audit Execution (`--no-cache-dir`):**
- `torch==2.15.0+cpu` (from `https://download.pytorch.org/whl/cpu`, 209.5 MB wheel, `filelock-4.2.4`, `fsspec-2026.9.0`)
- `fastapi==0.163.1`, `starlette==1.10.0`, `uvicorn==0.54.0`, `opentelemetry-api==1.45.2` (for API test suite)
- `xrfm==1.0.1` editable install (`pip install --no-cache-dir --no-deps -e /home/user/xr-foundation-model`)
- *Pre-existing in sandbox:* `numpy==2.3.5`, `PyYAML==6.0.3`, `jsonschema==4.26.0`, `pytest==9.0.3`, `httpx==0.28.1`.
- *Intentionally NOT installed:* CUDA toolkits, `transformers`, `accelerate`, `peft`, `trl`, `bitsandbytes`, `vllm` (none required to audit XRFM Native; installing CUDA wheels would waste ~3–5 GB disk).

### 1.2 Repository Verification & Git Inventory

- **Repository Root:** `/home/user/xr-foundation-model` (shallow clone `--depth 100`, no LFS weight pulls beyond tracked repo files)
- **Remote Origin:** `https://github.com/ahmadrrrtx/xr-foundation-model.git`
- **HEAD Commit:** `90674742d8e6a5caf65336e4b64c949696e33bc0`
- **Recent Commit Lineage:**
  - `9067474` — Merge PR #3 (`phase-0-stabilization`)
  - `1f77206` — `feat: stabilize phase 0 execution and security skeleton`
  - `b8bea87` — `feat: expose agent runtime API and event streaming`
  - `54e44c8` — `feat: add XRFM agentic runtime`
  - `fb470fb` — `feat(phase-1): build XRFM data foundation`
  - `9c4b087` — `feat(phase-0): freeze architecture — single src-layout xrfm package`
  - `421fac0` — `feat(neurotopo): XRFM-NeuroTopo v1 implementation (Phases 0-16)`
  - `4a90b7e` — `feat(v1.1): forensic audit remediation + GPU readiness + training preparation`
- **Git Tags / Releases:** **None** (`git tag` returns empty, despite `.github/workflows/release.yml` and `CHANGELOG.md` referencing historical `v0.1`–`v1.0.1` milestones).
- **Total Tracked Files (`git ls-files`):** **448 files**.

### 1.3 Complete Directory Map & Excluded Categories

```text
xr-foundation-model/ (448 tracked files)
├── src/xrfm/                  [CANONICAL LIBRARY — 108 files]
│   ├── __init__.py            Public API surface, __version__ = "1.0.1"
│   ├── cli.py                 `xrfm info | validate-config | data ...`
│   ├── config/                Typed dataclasses (schema.py, loader.py, config.default.yaml)
│   ├── tokenization/          Tokenizer ABC, BytePairEncoder (v2), encode/decode, vocab.json (2,048 tokens)
│   ├── models/                XRFMModel/GPTModel, XRFMEmbedding, TransformerBlock, MultiHeadAttention, RoPE, RMSNorm, SwiGLU
│   ├── data/                  Phase 0 TextDataset + Phase 1 20-module DataPipeline (schema..sharding)
│   ├── training/              Trainer, TrainingLoop, CheckpointLoader (v2), MixedPrecisionLoader, DDP/FSDP wrappers
│   ├── inference/             generate(), GenerationEngine, sampling.py, KVCache (experimental)
│   ├── evaluation/            evaluate(), compute_perplexity(_strided), TextCompletionAccuracy, TopKAccuracy
│   ├── optimization/          SDPA wrapper (flash_attention.py), INT8/INT4 quantization, SpeculativeDecoder (experimental)
│   ├── agent/                 Model-agnostic bounded tool runtime (protocols, runtime, safety, spawn isolation)
│   ├── search/                Local TF/BM25-style RAG indexer/retriever/agent
│   ├── research/neurotopo/    Experimental post-transformer graph/GRU/associative-memory research bundle (24 files)
│   └── {core,dynamics,knowledge,memory,neurons,neurotopo,nt_evaluation,nt_training,routing,topology,uncertainty}/
│                              11 deprecated in-package compatibility shims emitting DeprecationWarning -> xrfm.research.neurotopo
├── {model,tokenizer,training,inference,evaluation,optimization}/
│                              6 deprecated repo-root compatibility shims (31 files) forwarding to src/xrfm/*
├── api/                       FastAPI server (/health, /ready, /v1/models, /v1/completions, /v1/tokenize, /v1/agent/*, /v1/search)
├── config/                    Model/training YAML presets (tiny.yaml, config.yaml, medium.yaml, v1.1-medium.yaml, config.legacy-v1.yaml)
├── configs/data/              Phase 1 data pipeline configs (pretrain.yaml, sources.yaml)
├── scripts/                   14 CLI/training/eval/smoke scripts (train_custom_model.py, ci_smoke_test.py, gpu_smoke_test.py, etc.)
├── tests/                     47 pytest files + tests/data/golden/ (544 total test cases)
├── checkpoints/               checkpoint_step_500.pt (74 MB legacy pre-audit checkpoint, vocab=50304, d_model=256)
├── data/                      datasets/{corpus.txt (5.3MB), code_slice.py (386KB), sample.txt (30KB)}, manifests/
├── deployment/                Dockerfiles (CPU/GPU), docker-compose.yml, gunicorn_conf.py, huggingface_space/
├── webui/ & vercel_app/       Duplicate static HTML/JS frontends + serverless proxy stubs (8 files each)
├── notebooks/                 XRFM_Training_Colab.ipynb
└── docs/ & research/          102 historical audit, ADR, program ledger (`docs/program/`), and NeuroTopo reports
```

**Explicitly Excluded Categories from Deep Line-by-Line Code Audit (with Justification):**
1. **Generated Caches (`__pycache__/`, `.pytest_cache/`, `src/xrfm.egg-info/`):** Untracked runtime bytecode/build metadata created during test execution.
2. **Binary Weight Tensors inside `checkpoints/checkpoint_step_500.pt` (74 MB):** Inspected safely via `torch.load(..., map_location="cpu", weights_only=True)` for keys/shapes/metadata only; raw float arrays not dumped.
3. **Static Frontend Assets (`webui/logo.png`, `vercel_app/logo.png`):** Binary PNG logos irrelevant to ML architecture.
4. **Raw Text Training Samples (`data/datasets/corpus.txt` 5.3 MB, `code_slice.py` 386 KB):** Sampled programmatically for byte/character length, encoding cleanliness, and BPE scaling benchmarks rather than read manually line-by-line.

---

## 2. Phase One (A) — Repository & Transformer Architecture Audit

### 2.1 Subsystem Dependency Map

```text
xrfm.config (schema.py, loader.py)          [Zero internal deps; pure stdlib + PyYAML]
      │
      ▼
xrfm.tokenization (interface.py, bpe.py)    [Zero torch deps; pure Python + vocab.json]
      │
      ├──────────────────────────────┐
      ▼                              ▼
xrfm.data (dataset.py, pipeline.py)  xrfm.models (gpt.py, embedding.py, layers/*, attention/*)
      │                              │      └── optional import: xrfm.optimization.flash_attention
      └──────────────┬───────────────┘
                     ▼
        xrfm.training (trainer.py, loop.py, checkpoint.py, mixed_precision.py, distributed.py)
                     │
      ┌──────────────┴───────────────┐
      ▼                              ▼
xrfm.inference (engine.py, generate) xrfm.evaluation (perplexity.py, benchmarks.py, runner.py)
                     │
                     ▼
      Secondary / Outer Consumers: api/*, xrfm.agent/*, xrfm.search/*, scripts/*
```

The one-way dependency boundary documented in `docs/adr/0001-xrfm-core-architecture.md` is **genuinely respected** in `src/xrfm/`: core modules do not import `api/`, `agent/`, or `research/neurotopo/`.

### 2.2 Detailed Transformer Inspection (`src/xrfm/models/`)

| Component | File & Lines | Implementation Reality | Critical Technical Findings |
|---|---|---|---|
| **Top-Level Model** | `src/xrfm/models/gpt.py:36-210` | `GPTModel` (aliased as `XRFMModel` and `XRFM`). Decoder-only pre-norm transformer. Accepts `XRFMConfig`, `ModelConfig`, YAML path, or `None`. Supports `vocab_size` override and weight-tied `lm_head` (`lm_head.weight = embedding.embedding.weight`, `bias=False`). | **DEFECT-ARCH-01 (`use_swiglu` ignored):** `ModelConfig.use_swiglu` (`schema.py:84`) is validated and stored in every YAML preset, but `GPTModel.__init__` (`gpt.py:114-126`) never passes `use_swiglu` to `TransformerBlock`, and `TransformerBlock` (`transformer_block.py:55`) unconditionally constructs `SwiGLU`. Setting `use_swiglu: false` silently builds a SwiGLU model. |
| **Attention** | `src/xrfm/models/attention/multi_head.py:25-219` | Standard Multi-Head Attention (**MHA**, `n_kv_heads == n_heads`; **no GQA/MQA**). Builds explicit causal upper-triangular `-inf` mask (`diagonal=cache_len+1`), composes 2D/3D/4D `0/1` padding/custom masks additively, and delegates to `flash_attention_forward` (`F.scaled_dot_product_attention`). | **DEFECT-ARCH-02 (SDPA Fast-Path Defeated):** `MultiHeadAttention.forward` (`multi_head.py:143-199`) *always* materializes an explicit float `(q_len, kv_len)` `additive_mask` tensor (`causal_mask`, plus `explicit_mask` when `mask` is passed) and passes `mask=additive_mask` to `flash_attention_forward()`. Inside `flash_attention.py:162-171`, `if mask is not None:` calls `F.scaled_dot_product_attention(..., attn_mask=mask, is_causal=False)`. On CUDA, passing an explicit dense float `attn_mask` with `is_causal=False` **disables the FlashAttention-2 kernel** (which requires `attn_mask=None, is_causal=True`), falling back to slower MemoryEfficient or O(N²) math kernels and allocating O(L²) mask memory! |
| **Rotary Embeddings** | `src/xrfm/models/attention/rope.py:19-118` | Pairwise half-rotation (`rotate_half`), `base=10000.0`, `offset=cache_len` for KV-cached decoding. | **DEFECT-ARCH-03 (RoPE Recomputed Every Forward + Dtype Cast):** `RoPE.__init__` registers `inv_freq` buffer (`rope.py:52`), **never uses `self.inv_freq`**, and recomputes `torch.arange`, `1.0 / (self.base ** ...)`, `torch.outer`, `cos`, and `sin` in `float32` on **every single attention layer of every forward step** (`rope.py:73-93`). Furthermore, `x * cos + rotate_half(x) * sin` promotes `bf16`/`fp16` `Q` and `K` tensors to `float32`! Under `torch.autocast`, linear layers cast inputs, so `F.scaled_dot_product_attention` receives `float32` `Q, K` alongside `bf16/fp16` `V` — either causing dtype mismatch errors in Flash/MemEfficient kernels or forcing slow FP32 fallback! Also, `rope_theta` (`base=10000.0`) is hard-coded and not configurable via `ModelConfig`. |
| **Normalization** | `src/xrfm/models/layers/rmsnorm.py:32-96` | `self.weight * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)` (`eps=1e-6`). | **DEFECT-ARCH-04 (RMSNorm Precision):** Computes `x.pow(2)` in input dtype rather than upcasting `x.float()` before squaring and casting back (`(x.float() * torch.rsqrt(...)).type_as(x)`). In `fp16` (required on Kaggle P100/T4 when BF16 is unsupported), `x.pow(2)` overflows FP16 (`max 65504`) whenever activations exceed `255.9`, causing NaN activations. |
| **Feed-Forward** | `src/xrfm/models/layers/swiglu.py:34-125` | `W_3(SiLU(W_1(x)) * W_2(x))` with configurable `bias`. | Clean, readable, shape-validated. Uses `d_ff = 4 * d_model` in presets (`config.yaml`: `128 -> 512`, `medium.yaml`: `384 -> 1536`) rather than `~8/3 * d_model` rounded to 256 multiple, increasing parameter count per block by ~12.5% vs parameter-matched Llama/Qwen blocks. |
| **Embeddings & Init** | `src/xrfm/models/embedding.py:33-155`, `gpt.py:134` | `nn.Embedding` initialized with `xavier_uniform_`; `lm_head` initialized with `normal_(0.0, 0.02)` **and then immediately overwritten** when `weight_tied=True` (`self.lm_head.weight = self.embedding.embedding.weight`). | **DEFECT-ARCH-05 (Initialization Scaling & Embedding Lookup Overhead):** (1) Because `lm_head.weight` is tied *after* `normal_(0, 0.02)`, the tied weight actually gets `xavier_uniform_` (`limit = sqrt(6 / (vocab_size + d_model))`). For `vocab_size=50304, d_model=384`, `std ≈ 0.006` (3.3× smaller than GPT-2/Llama `0.02`), and residual output projections (`W_o`, `W_3`) lack `1 / sqrt(2 * n_layers)` depth scaling. (2) `XRFMEmbedding.forward` (`embedding.py:146`) runs `if (input_ids >= self.vocab_size).any(): max_id = int(input_ids.max().item())` **and** `TrainingLoop.train_step` (`loop.py:390`) runs `(batch_input_ids < 0).any() or (batch_input_ids >= vocab_size).any()` on every step — triggering synchronous GPU→CPU `.any()` host-device syncs on CUDA! |

### 2.3 Measured Parameter Counts of Shipped Presets

Verified programmatically via `sum(p.numel() for p in XRFMModel(load_config(p)).parameters())`:

| Config File | `vocab_size` | `d_model` | `n_layers` | `n_heads` | `d_ff` | `max_seq_len` | `use_bias` | Exact Parameters |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `config/tiny.yaml` | 1,024 | 64 | 2 | 4 | 256 | 128 | `True` | **198,592** (0.20M) |
| `config/config.yaml` (packaged default) | 2,048 | 128 | 4 | 4 | 512 | 256 | `True` | **1,318,528** (1.32M) |
| `config/v1.1-medium.yaml` | 8,192 | 384 | 7 | 6 | 1,536 | 1,024 | `False` | **19,666,560** (19.67M) |
| `config/medium.yaml` | 8,192 | 384 | 8 | 6 | 1,536 | 1,024 | `True` | **22,066,560** (22.07M) |
| `config/config.legacy-v1.yaml` | 50,304 | 256 | 6 | 8 | 1,024 | 512 | `True` | **19,613,952** (19.61M, historical) |

### 2.4 Duplicated, Experimental, or Disconnected Modules

1. **17 Compatibility Shim Packages (42 files):**
   - Repo-root `model/`, `tokenizer/`, `training/`, `inference/`, `evaluation/`, `optimization/` (31 files) re-export `src/xrfm/*` with `DeprecationWarning`.
   - In-package `src/xrfm/{core,dynamics,knowledge,memory,neurons,neurotopo,nt_evaluation,nt_training,routing,topology,uncertainty}/` (11 packages) re-export `src/xrfm/research/neurotopo/*`.
   - *Assessment:* Documented and tested (`tests/test_compat_shims.py`), excluded from wheel build (`pyproject.toml:66`), safe to preserve until v2.0.
2. **Duplicate Web Frontends (`webui/` vs `vercel_app/`):** Both contain nearly identical `index.html`, `app.js`, `style.css`, `vercel.json`, `api/generate.js`, `api/search.js` (16 files total).
3. **Experimental Unused Classes:** `KVCache` (`src/xrfm/inference/kv_cache.py`), `GradientAccumulator` (`src/xrfm/training/distributed.py`), and `SpeculativeDecoder` (`src/xrfm/optimization/speculative_decoding.py`) are marked `EXPERIMENTAL (v1.1)` and unused by `TrainingLoop` or `GenerationEngine`. Notably, `SpeculativeDecoder._draft_logits_at_position` (`speculative_decoding.py:265-275`) is a stub returning `torch.ones(1, vocab_size)` (uniform distribution), making rejection resampling mathematically inaccurate!
4. **Isolated NeuroTopo Bundle (`src/xrfm/research/neurotopo/`):** 24 files + 16 test files (`test_neurotopo_phase1..16.py`) implementing an experimental dynamic-graph GRU + associative memory model. Completely separate from Track A (`XRFMModel`) and Track B (`CoderSec`), accounting for **74.7% of total test suite wall-clock runtime** (169.8s of 226.5s).

---

## 3. Phase One (B) — Complete Tokenizer Audit

### 3.1 Implementation & Correctness Analysis (`src/xrfm/tokenization/bpe.py`)

- **Algorithm:** Pure-Python byte-level BPE (`BytePairEncoder` / `BPETokenizer`, version `xrfm-bpe-v2`). Maps UTF-8 bytes `0..255` to single-character `latin-1` strings (`bytes([b]).decode("latin-1")`), splits input via `_WS_RE = re.compile(r"\s+|\S+")` attaching leading whitespace runs to the following non-whitespace token (`_split_words`, `bpe.py:130-148`), and learns pairwise merges.
- **Unicode & Round-Trip Losslessness:** Because every string is encoded to UTF-8 bytes and base tokens `0..255` always exist in `self.vocab`, `decode(encode(text)) == text` holds **100% losslessly** across English, Urdu (Arabic script), Roman Urdu, Arabic, Hindi (Devanagari), Spanish, French, emojis, and source code indentation/newlines.

### 3.2 Critical Tokenizer Defects & Bottlenecks

1. **DEFECT-TOK-01 — Catastrophic Byte Fallback (1.00 bytes/token) on Non-Latin Scripts in Packaged `vocab.json`:**
   - Packaged `src/xrfm/tokenization/vocab.json` (`vocab_size=2048`) was trained exclusively on English Tiny Shakespeare (` gentle`+`man`, ` s`+`erv`).
   - Empirical measurement conducted during this audit across the 7 target languages + Python:

   | Language / Script | Sample Words | UTF-8 Bytes | Token Count | Tokens / Word | Bytes / Token | Raw Byte Fallback (`id < 256`) |
   |---|---:|---:|---:|---:|---:|---:|
   | **English (`en`)** | 13 | 116 | 44 | 3.38 | 2.64 | 13.6% |
   | **Roman Urdu (`ur-Latn`)** | 20 | 115 | 58 | 2.90 | 1.98 | 32.8% |
   | **Spanish (`es`)** | 13 | 115 | 63 | 4.85 | 1.83 | 34.9% |
   | **French (`fr`)** | 15 | 116 | 63 | 4.20 | 1.84 | 47.6% |
   | **Python (`code`)** | 17 | 180 | 102 | 6.00 | 1.76 | 51.0% |
   | **Urdu (`ur`, Arabic script)** | 15 | 158 | **158** | **10.53** | **1.00** | **100.0%** |
   | **Arabic (`ar`, Arabic script)** | 13 | 171 | **171** | **13.15** | **1.00** | **100.0%** |
   | **Hindi (`hi`, Devanagari)** | 16 | 258 | **258** | **16.12** | **1.00** | **100.0%** |

   - *Consequence:* Urdu, Arabic, and Hindi receive **zero** subword merges in the packaged tokenizer — every UTF-8 byte becomes a separate token (2 bytes/char for Arabic/Urdu, 3 bytes/char for Devanagari), inflating sequence length by **3.5×–5×** and wasting 75%–80% of attention context and KV cache!

2. **DEFECT-TOK-02 — Unscalable Pure-Python Training Complexity $O(M \cdot N)$:**
   - `train_on_text` (`bpe.py:224-262`) stores the entire corpus as `word_splits: list[list[str]]` (every word occurrence duplicated in memory, **not** deduplicated by word frequency!) and rescans every token of every word occurrence on **every single merge step**.
   - Measured benchmark on `data/datasets/corpus.txt`:
     - `50,000` chars, 252 merges: `3.74 s` (`14.8 ms/merge`)
     - `100,000` chars, 252 merges: `7.67 s` (`30.4 ms/merge`)
     - `200,000` chars, 252 merges: `15.39 s` (`61.1 ms/merge`)
   - Extrapolating linearly in both corpus size and merge count: training a **32,000-vocab** tokenizer on **50 MB** of multilingual text would take `~0.061s * 250 * (31740/252) ≈ 1,920 seconds * 250` → **>50 hours** and exhaust 2 GB RAM! Even `scripts/train_custom_model.py:69` had to hack around this by truncating training text to `400,000` characters.

3. **DEFECT-TOK-03 — Pre-tokenization Regex Merges Punctuation Into Words (`\s+|\S+`):**
   - `_WS_RE = re.compile(r"\s+|\S+")` (`bpe.py:29`) does **not** separate punctuation, digits, or operators from words (unlike GPT-2/GPT-4/Qwen regexes). Thus `"function(x):"` is treated as a single tokenizable unit `" function(x):"`, fragmenting vocabulary capacity across `word`, `word,`, `word.`, `word)`, `word:` variants and harming both code and multilingual morphology.

4. **DEFECT-TOK-04 — Special Tokens Are Not Parsed by `encode()` and `encode_text(..., add_special_tokens=True)` Is a No-Op:**
   - `BytePairEncoder.encode("Hello <|eos|> world")` encodes `<|eos|>` into 7 ordinary byte/subword tokens `[60, 124, 101, 470, 124, 62]` rather than `2046`, while `decode([2046])` outputs `'<|eos|>'` — breaking round-trip symmetry for special tokens (`decode(encode(decode([2046]))) == '<|eos|>'`, `encode('<|eos|>') != [2046]`).
   - Worse, `src/xrfm/tokenization/encode.py:48-51` literally has `if add_special_tokens ...: pass`! Callers passing `add_special_tokens=True` silently get zero BOS/EOS tokens.

5. **DEFECT-TOK-05 — Sparse Dead Vocabulary IDs When Early-Stopping on Small Corpora:**
   - `train_on_text` reserves `<|pad|>..unk` at `vocab_size_target - 4 .. vocab_size_target - 1` (`bpe.py:194-199`). When a small corpus runs out of pairs with frequency `>= 2`, merging stops early (e.g., at ID 266), leaving IDs `267..2043` completely unassigned while `vocab_size()` (`max(id) + 1`, `bpe.py:378`) still returns `2048` (`len(vocab) == 271 != vocab_size() == 2048`).
   - Additionally, `src/xrfm/data/tokenization.py:20-48` (`_hash_tokenizer_config`) hashes **only** `vocab_size`, special token IDs, and class name — **not** the vocabulary or merges! Two completely different 32K tokenizers trained on English vs Urdu produce the **exact same `TokenizerInfo.hash`** in Phase 1 dataset manifests! (By contrast, `tokenizer_fingerprint()` in `interface.py:34` hashes `vocab` and `merges`, creating two inconsistent hash functions across `xrfm.data`!)

---

## 4. Phase One (C & D) — Training System & Distributed/Hardware Execution Audit

### 4.1 Training Engine Audit (`src/xrfm/training/`)

 Phase 0 stabilization (`1f77206`) significantly improved `TrainingLoop`, `CheckpointLoader`, and `MixedPrecisionLoader`, fixing several classic bugs. Specifically:
- **Genuinely Implemented & Verified on CPU:**
  - Next-token prediction with `ignore_index=-100` (`loop.py:402-406`).
  - Position-based target masking in `TextDataset` (`dataset.py:196-215`) and `xrfm_collate_fn` (`distributed.py:345-372`), avoiding hard-coded `pad_id=0` bugs.
  - Gradient accumulation (`grad_accum_steps`) with loss scaling (`loss / grad_accum_steps`), DDP `no_sync()` on intermediate micro-steps, and overflow-skip handling that zeroes partial gradients without advancing `scheduler` or `current_step` (`loop.py:408-457`).
  - Format-v2 atomic checkpoints (`tempfile.mkstemp` + `fsync` + `os.replace`, `weights_only=True` load, RNG state serialization for Python/NumPy/Torch/CUDA, config/manifest/tokenizer hash verification, exact epoch + `batch_in_epoch` resume positioning in `checkpoint.py` and `loop.py:252-298`).
  - Cosine LR schedule with linear warmup (`scheduler.py`).

- **Confirmed Training Defects & Operational Gaps:**
  1. **DEFECT-TRN-01 — No Weight Decay Parameter Grouping:** `OptimizerLoader` (`loop.py:166-170`, `optimizer.py:79-85`) passes `raw_model.parameters()` as a single flat group with `weight_decay=0.1` (`TrainingConfig` default). Consequently, **1D `RMSNorm.weight` parameters, linear biases, and token embeddings all receive `0.1` decoupled weight decay**, contradicting standard Transformer practice (GPT-3, Llama, Qwen, Chinchilla apply `weight_decay=0.0` to norms and biases).
  2. **DEFECT-TRN-02 — Cosine Scheduler Exceeds `base_lr` If `step > max_steps`:** In `SchedulerLoader.get_lr` (`scheduler.py:109-110`), `progress = (step - warmup_steps) / (max_steps - warmup_steps)` is **not clamped to `1.0`** and has no `min_lr` floor (typically `0.1 * base_lr`). If a resumed run continues past `max_steps`, `cos(pi * progress)` increases back toward `base_lr`!
  3. **DEFECT-TRN-03 — No Gradient Checkpointing (Activation Checkpointing) Anywhere:** Neither `TransformerBlock` nor `GPTModel` nor `TrainingConfig` implements `torch.utils.checkpoint.checkpoint`. On a 16 GB T4/P100 GPU, scaling `XRFM Native` to 125M–350M parameters at `seq_len=2048` will hit activation memory limits rapidly without activation checkpointing.
  4. **DEFECT-TRN-04 — Resume Iterator Fast-Forward Materializes All Skipped Batches (`loop.py:567-572`):** On resume at `_batch_in_epoch = B`, the loop executes `for _ in range(self._batch_in_epoch): next(iterator)`, reading, collating, and discarding `B` full batches from disk/workers before training resumes, rather than slicing the sampler indices.
  5. **DEFECT-TRN-05 — Validation Does Not Move Batches to `self.device` Automatically in `perplexity.py` / `benchmarks.py`:** `compute_perplexity` (`perplexity.py:92`) calls `model(batch_input_ids)` directly on whatever device the DataLoader yields (CPU!), without moving `batch_input_ids` and `batch_target_ids` to `next(model.parameters()).device`! On a CUDA GPU, calling `xrfm.evaluate(cuda_model, val_loader)` **crashes immediately with `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu`!** (Verified in `perplexity.py:75-108` and `benchmarks.py:114-126`!).

### 4.2 Hardware & Distributed Execution Matrix (Truth Table)

Per audit rules, every hardware/distributed feature is classified strictly into **(1) Implemented**, **(2) Unit-tested**, **(3) Exercised on real hardware**, or **(4) Merely planned**:

| Feature | Code Location | Status Classification | Evidence & Critical Caveats |
|---|---|---|---|
| **CPU Execution (FP32)** | `models/*`, `training/*` | **3. Exercised on hardware** | Verified in this audit (`544 passed`) and CI workflows. |
| **Single-GPU CUDA** | `loop.py:214-217`, `scripts/gpu_smoke_test.py` | **2. Unit-tested (mocked/CPU); NOT hardware-verified** | `docs/program/03_EXECUTION_LEDGER.md:42,67-68` explicitly records `G1 = FAIL` because physical CUDA runs have not been executed. **Additionally, `perplexity.py` & `benchmarks.py` lack `.to(device)` and will crash on CUDA!** |
| **Mixed Precision (`fp16` / `bf16`)** | `training/mixed_precision.py` | **2. Unit-tested on CPU mocks; broken edge cases for CUDA** | `MixedPrecisionLoader` logic is clean, **but** `RoPE.apply_rotary_emb` (`rope.py:99`) upcasts `Q, K` to `float32` while `V` stays `fp16/bf16`, and `RMSNorm` (`rmsnorm.py:96`) squares `fp16` without `float32` upcast (overflow risk on P100/T4). |
| **DDP (Multi-GPU)** | `training/distributed.py:137-165`, `loop.py:135-142` | **1. Implemented (wrappers) / 2. Lightly unit-tested in single-process** | `wrap_model_ddp` and `no_sync()` exist, **but** `TrainingLoop.__init__` wraps `model` in DDP *before* moving `model` to `self.device` (`loop.py:138-142` runs before `self.device` is even resolved at `loop.py:214`, and `TrainingLoop` **never calls `self.model.to(self.device)`**!). Unless the caller manually moves `model.to(cuda:local_rank)` before `Trainer(model)`, DDP wraps a CPU model! Never exercised on 2 GPUs (`03_EXECUTION_LEDGER.md:69`). |
| **FSDP** | `training/distributed.py:168-217` | **1. Implemented (minimal wrapper only) — Broken with `CheckpointLoader`** | `wrap_model_fsdp` wraps the root model without transformer-layer auto-wrap policy, and `TrainingLoop` saves checkpoints via `CheckpointLoader.save_checkpoint(get_raw_model(self.model), ...)` instead of `save_fsdp_checkpoint()`, which fails or saves corrupted sharded flat tensors under `FULL_SHARD`. |
| **Gradient Checkpointing** | None | **4. Merely planned / Absent** | Zero implementation in `models/` or `training/`. |
| **FlashAttention-2** | `optimization/flash_attention.py` | **1. Implemented (SDPA wrapper) — Defeated by dense mask** | Uses `torch.backends.cuda.sdp_kernel` (deprecated in PyTorch 2.15, emitting `FutureWarning`) and receives an explicit dense float mask from `MultiHeadAttention`, disabling the FlashAttention kernel on GPU. |

---

## 5. Phase One (E) — Dataset Pipeline Audit (`src/xrfm/data/`)

The repository contains **two coexisting data pipelines**:
1. **Phase 0 `TextDataset` (`src/xrfm/data/dataset.py`):** Reads an entire plain-text file into RAM (`f.read()`), splits lines, encodes the entire split string at once (`tokenizer.encode(joined)`), and chunks into Python lists (`list[list[int]]`). Fine for `<10 MB` smoke tests; **unusable at scale**.
2. **Phase 1 `DataPipeline` (`src/xrfm/data/pipeline.py` + 19 modules):** Document-centric pipeline (`Document` schema, normalization, language ID, quality filter, PII filter, exact SHA-256 dedup, MinHash near-dedup, contamination check, hash-based document split, data mixing, tokenization, sequence packing, `.npy` sharding with `mmap_mode="r"`, and `DatasetManifestV2`).

### Critical Dataset Pipeline Defects

1. **DEFECT-DATA-01 — Pipeline Is Entirely In-Memory (`list[Document]`), NOT Streaming:**
   - Despite comments claiming *"streaming where appropriate"* (`pipeline.py:25`), every single stage (`ingest`, `normalize`, `filter_language`, `filter_quality`, `filter_pii`, `dedup`, `split`, `tokenize`, `pack`, `write_shards`) materializes full Python `list[Document]`, `list[TokenizedDocument]`, and `stream: list[int]` in RAM! Additionally, `ingest()` writes a duplicate `raw.jsonl` copy of all input documents (`pipeline.py:148-151`). On a 2 GB RAM machine (or 29 GB Kaggle RAM for multi-GB corpora), `DataPipeline.run()` will OOM.
   - Nor is there any Hugging Face streaming downloader (`datasets.load_dataset(..., streaming=True)`) or DataTrove reader implemented — `SourceRegistry` (`sources.py`) only stores YAML metadata!

2. **DEFECT-DATA-02 — `HeuristicLanguageDetector` Fails Completely on 5 of the 7 Target Languages:**
   - Verified empirically during this audit (`src/xrfm/data/language.py:133-200`):
     - **Urdu (`ur`):** Misclassified as **Arabic (`ar`)** (`ar=0.90, ur=0.81`) because lines 163–165 literally hardcode `scores["ur"] = scores.get("ar", 0.0) * 0.9` — making it **mathematically impossible** for Urdu (`0.9 * ar`) to ever beat Arabic (`ar`)! Urdu-specific codepoints (`ٹ ڈ ڑ ں ہ ے گ چ پ ژ`, `U+0679`, `U+0688`, `U+0691`, `U+06BA`, `U+06C1`, `U+06D2`, `U+06AF`, `U+0686`, `U+067E`, `U+0698`) are never checked!
     - **Roman Urdu (`ur-Latn`):** Misclassified as `en` with `confidence=0.00` (rejected by default threshold `0.6`). Zero Roman Urdu lexicon, character n-gram, or code-switch detection exists.
     - **Spanish (`es`) & French (`fr`):** Both misclassified as `en` with `confidence=0.00` (rejected by threshold `0.6`!). Neither `es` nor `fr` appears anywhere in `language.py`.
     - Only **English (`en`)**, **Arabic (`ar`)**, and **Hindi (`hi`, Devanagari)** can ever be detected.

3. **DEFECT-DATA-03 — Near-Deduplication Is $O(N^2)$ Pairwise (`src/xrfm/data/dedup.py`):**
   - `deduplicate_near` compares MinHash signatures pairwise across documents without LSH banding (`O(N^2)`), quadratic above ~10,000 documents.

4. **DEFECT-DATA-04 — `xrfm data build --config` Silently Ignores 90% of `configs/data/pretrain.yaml`:**
   - In `src/xrfm/data/cli.py:92-97`, `cmd_build` parses `cfg_dict = yaml.safe_load(f)` and constructs `PipelineConfig` passing **only** `dataset_name`, `dataset_version`, `output_dir`, and `sequence_length` — silently discarding `language`, `quality`, `pii`, `exact_dedup`, `near_dedup`, `split`, `tokenization`, `packing`, `sharding`, and `contamination` settings from `pretrain.yaml`!

5. **DEFECT-DATA-05 — `ShardedTokenDatasetMap` Drops Every `L`-th Token Target Across Sequence Boundaries (`tokenized_dataset.py:113-116`):**
   - `write_shards` saves chunks of length `L` (`sequence_length`), and `ShardedTokenDatasetMap.__getitem__` constructs `input_tensor = input_ids` (`0..L-1`) and `target_list = list(input_ids[1:]) + [self.ignore_index]`. Instead of packing `L + 1` tokens per row (`x = row[:-1], y = row[1:]`), every packed sequence masks out its final token (`1/L` loss of training signal and broken cross-chunk continuity) and allocates a Python `list` on every `__getitem__` call.

6. **DEFECT-DATA-06 — Invalid / Placeholder Source Registry Entries (`configs/data/sources.yaml`):**
   - Line 157 lists `uri: https://huggingface.co/datasets/urdu` (`license: varies`) — a non-existent/placeholder URI. `openwebtext` has `license: unknown`. Only `golden`, `tiny_shakespeare`, and `python_stdlib_slice` (~466 KB total) are enabled.

---

## 6. Phase One (F) — Evaluation & Testing Audit

### 6.1 Executed Tests & Exact Results in 2 GB Sandbox

To respect the 1.9 GiB RAM limit (running all 544 tests in a single `pytest` invocation hit timeout/memory pressure inside the slow NeuroTopo training tests), the suite was executed in two clean, isolated batches covering **100% of the 47 test files (544 tests total)**:

1. **Core, Tokenizer, Data, Model, Training, Inference, Evaluation, Optimization, Search, Agent & API Suite:**
   - **Command:** `python3 -m pytest tests/ --ignore-glob="tests/test_neurotopo_*.py" -q --tb=short --durations=10`
   - **Result:** **`448 passed, 2 warnings in 56.75s`** (Exit code `0`).
   - **Warnings observed:**
     - `StarletteDeprecationWarning` on `httpx` in `starlette.testclient` (Python 3.13 / Starlette 1.10).
     - `FutureWarning: torch.backends.cuda.sdp_kernel() is deprecated` (`src/xrfm/optimization/flash_attention.py:101`).
2. **Experimental NeuroTopo Suite (Phases 1–16):**
   - **Command:** `python3 -m pytest tests/test_neurotopo_*.py -q --tb=short --durations=5`
   - **Result:** **`96 passed, 12 warnings in 169.78s`** (Exit code `0`; `test_nt_trains_on_real_corpus` alone took `86.99s`).
   - **Combined Total:** **`544 passed, 0 failed, 14 warnings`**, working tree verified clean (`git status --short` empty).

### 6.2 Critical Evaluation Defects & Missing Test Coverage

1. **DEFECT-EVAL-01 — `TextCompletionAccuracy` and `TopKAccuracy` Score Padded `-100` Targets (`benchmarks.py:121-126, 198-203`):**
   - Verified empirically during this audit: on a batch with 6 real tokens and 10 padded (`-100`) positions, `compute_perplexity` properly evaluated `6` tokens, whereas `TextCompletionAccuracy` and `TopKAccuracy` evaluated all **`16` positions**! Because model `argmax`/`topk` indices are in `[0, vocab_size)` and never equal `-100`, every padded position is counted as a **wrong prediction**, artificially deflating accuracy proportional to batch padding!
2. **DEFECT-EVAL-02 — Missing `.to(device)` in `perplexity.py` and `benchmarks.py`:**
   - All evaluation tests pass only because both model and DataLoader tensors are on CPU. Evaluating a GPU-resident `XRFMModel` fails with a device mismatch `RuntimeError`.
3. **DEFECT-EVAL-03 — No Bits-Per-Byte (BPB) Metric:**
   - `perplexity.py` computes only token-level `exp(NLL / N_tokens)`. Token-level perplexity is **mathematically incomparable across different tokenizers or vocabulary sizes** (e.g., 2K vs 16K vs 32K vs 50K, or Urdu byte-fallback vs subword BPE). `BPB = total_NLL_nats / (total_UTF8_bytes * ln(2))` is completely absent.
4. **Missing Test Categories:**
   - **CUDA / Mixed-Precision Numerical Tests:** Zero tests catching `RoPE` FP32 upcasting of `bf16/fp16` Q/K or `RMSNorm` FP16 overflow or GPU evaluation device placement.
   - **Multilingual & Roman Urdu Tests:** Zero tests for `ur`, `ur-Latn`, `ar`, `hi`, `es`, `fr` language detection, tokenizer fertility ceilings, script normalization (Arabic Yeh/Kaf unification hazards vs Urdu `ی`/`ے`/`ک`), or code-switching.
   - **Config-to-Architecture Wiring Tests:** Zero tests checking that `ModelConfig.use_swiglu=False` actually changes the FFN, or that `xrfm data build --config` actually applies `pretrain.yaml` sub-configs.
   - **Coding & Cybersecurity Evaluation Tests:** Zero execution-based coding tests (HumanEval+/MBPP+) or defensive security/CWE/prompt-injection/over-refusal benchmarks for model outputs.

---

## 7. Phase One (G) — Documentation vs. Implementation Audit

The repository contains unusually candid post-audit documents (`README.md`, `docs/program/01_DEEP_AUDIT.md`, `docs/program/03_EXECUTION_LEDGER.md`), which is a major credit to the maintainer's engineering integrity. However, significant contradictions and outdated artifacts remain across the 102 Markdown files:

| Issue ID | File & Line Reference | Documented Claim | Actual Source / Repo Reality |
|---|---|---|---|
| **DOC-01** | `README.md:30-33` | Quickstart runs `cfg = xrfm.load_config("config/tiny.yaml"); model = xrfm.XRFM(cfg); tok = xrfm.BPETokenizer.pretrained(); train = xrfm.TextDataset(cfg.data, tok)` | **Crashes immediately with `ValueError`!** `tiny.yaml` sets `model.vocab_size = 1024`, whereas `BPETokenizer.pretrained()` has `vocab_size = 2048` (`pad_id = 2044`). `TextDataset` succeeds (`2044 < 2048`), `XRFM(cfg)` builds an embedding of size `1024`, and `trainer.train(train)` crashes on step 1 with `IndexError: Token ID outside [0, 1024)` unless `vocab_size=tok.vocab_size()` is passed! |
| **DOC-02** | `README.md:164` | `xrfm data build --config configs/data/pretrain.yaml ...` | `src/xrfm/data/cli.py:92-97` ignores almost every section of `pretrain.yaml` (`language`, `quality`, `pii`, `dedup`, `packing`, `sharding`). |
| **DOC-03** | `pyproject.toml:26-28`, `.github/workflows/ci.yml:93` | Supports Python `3.10`, `3.11`, `3.12` (3.13 omitted from classifiers/CI). | Tested on Python `3.13.18` in this audit; works, but `flash_attention.py:101` uses deprecated `torch.backends.cuda.sdp_kernel()`. |
| **DOC-04** | `CHANGELOG.md` & `.github/workflows/release.yml` | Extensive release history (`v0.1.0` through `v1.0.1`). | `git tag` is completely empty; zero Git tags exist in the repository. |
| **DOC-05** | `docs/scaling/MULTI_GPU_GUIDE.md`, `src/xrfm/training/loop.py:4` | Claims *"Supports DDP/FSDP, gradient accumulation, mixed precision"*. | `TrainingLoop` never moves `model` to `self.device` before DDP wrapping (`loop.py:138`), FSDP checkpointing is not wired into `TrainingLoop` (`loop.py:617` calls `CheckpointLoader` instead of `save_fsdp_checkpoint`), and `G1` hardware gate is explicitly `FAIL` in `03_EXECUTION_LEDGER.md:42`. |
| **DOC-06** | `docs/model/ARCHITECTURE.md`, `XRFM_FINAL_REPORT.md`, `docs/audit/*` | Multiple overlapping historical reports from Phases 0–39, v1.1, NeuroTopo, and Agentic sprints with conflicting parameter figures and status claims. | Only `docs/program/03_EXECUTION_LEDGER.md` accurately reflects the post-October-2026 `NO-GO` gate state, and **none** of the docs yet cover the 7-language (`en, ur, ur-Latn, ar, hi, es, fr`) + two-track (`XRFM Native` + `XRFM CoderSec`) architecture. |
| **DOC-07** | Model & Dataset Cards | Missing standardized Hugging Face `MODEL_CARD.md` and `DATASET_CARD.md` templates for Track A and Track B releases. | `docs/program/templates/` has `DATA_SOURCE_REVIEW.md` and `EXPERIMENT_RECORD.md`, but no publishable HF model/dataset cards with explicit safety/misuse boundaries for defensive cybersecurity. |

---

## 8. Synthesis: Existing Strengths vs. Top Technical Gaps

### 8.1 Genuine Engineering Strengths Worth Preserving
1. **Clean Single-Package `src/xrfm/` Layout & Typed Configs (`src/xrfm/config/schema.py`):** Zero Pydantic bloat, explicit invariant checks, CWD-independent `importlib.resources` loading.
2. **Explicit Tokenizer Special-Token Contract (`src/xrfm/tokenization/interface.py`):** Eliminates hard-coded `pad_id=0` assumptions across `TextDataset` and `xrfm_collate_fn`.
3. **Hardened Checkpoint Format v2 (`src/xrfm/training/checkpoint.py`):** Atomic `mkstemp` + `fsync` + `os.replace`, `weights_only=True` deserialization (preventing arbitrary pickle code execution), and full Python/NumPy/Torch/CUDA RNG + dataloader offset persistence.
4. **Fail-Closed Bounded Agent Runtime (`src/xrfm/agent/`):** Process-isolated (`spawn`) backend and tool deadlines, root-confined read-only filesystem tools, JSON Schema Draft 2020-12 validation, and credential-redacted audit logs — an unusually strong foundation for integrating **XRFM CoderSec** later as a safe local coding/security assistant.
5. **High Test Discipline (`544` passing unit/integration tests)** and honest governance in `docs/program/03_EXECUTION_LEDGER.md`.

### 8.2 Summary Table of Confirmed Defects (Severity-Ranked)

| Defect ID | Severity | Location | Summary |
|---|---|---|---|
| **DEFECT-TOK-01** | **CRITICAL (P0)** | `src/xrfm/tokenization/vocab.json` | 2,048-token English Shakespeare vocab causes **100% raw-byte fallback (1.00 bytes/tok)** on Urdu, Arabic, and Hindi (3.5×–5× token inflation). |
| **DEFECT-TOK-02** | **CRITICAL (P0)** | `src/xrfm/tokenization/bpe.py:224-262` | $O(M \cdot N)$ un-deduplicated Python BPE training cannot scale to 16K–50K multilingual corpora (>50 hrs, RAM OOM). |
| **DEFECT-DATA-02** | **CRITICAL (P0)** | `src/xrfm/data/language.py:133-200` | `HeuristicLanguageDetector` hardcodes `ur = 0.9 * ar` (Urdu can *never* win!) and fails (`conf=0.0`) on `ur-Latn`, `es`, and `fr`. |
| **DEFECT-EVAL-02** | **CRITICAL (P0)** | `src/xrfm/evaluation/perplexity.py:92`, `benchmarks.py:116` | Evaluation never moves batch tensors to `model.device` — **guaranteed crash on CUDA GPU**. |
| **DEFECT-ARCH-02** | **HIGH (P0)** | `src/xrfm/models/attention/multi_head.py:143-199` | Always passes dense float `additive_mask` with `is_causal=False` to SDPA, **disabling FlashAttention-2** and allocating $O(L^2)$ memory. |
| **DEFECT-ARCH-03** | **HIGH (P0)** | `src/xrfm/models/attention/rope.py:73-99` | Recomputes `inv_freq`/`cos`/`sin` every layer every step in `fp32` and **upcasts `bf16/fp16` Q/K to `fp32`** while V stays `bf16/fp16`. |
| **DEFECT-ARCH-04** | **HIGH (P0)** | `src/xrfm/models/layers/rmsnorm.py:96` | Computes `x.pow(2)` in `fp16` without `fp32` upcast — high NaN overflow risk on Kaggle T4/P100 GPUs. |
| **DEFECT-DATA-01** | **HIGH (P1)** | `src/xrfm/data/pipeline.py:144-295` | `DataPipeline` loads all documents/tokens into Python lists in RAM; no streaming HF/Parquet ingestion. |
| **DEFECT-EVAL-01** | **HIGH (P0)** | `src/xrfm/evaluation/benchmarks.py:121,199` | `TextCompletionAccuracy` & `TopKAccuracy` do not mask `ignore_index=-100`, corrupting accuracy on padded batches. |
| **DEFECT-TRN-01** | **MEDIUM (P1)** | `src/xrfm/training/loop.py:166`, `optimizer.py:79` | Applies `weight_decay=0.1` to 1D RMSNorm weights, biases, and embeddings; `TrainingLoop` also omits `model.to(self.device)` before DDP wrap. |
| **DEFECT-DATA-04** | **MEDIUM (P1)** | `src/xrfm/data/cli.py:92-97` | `xrfm data build --config` silently ignores `language`, `quality`, `pii`, `dedup`, `packing`, and `sharding` sections of YAML. |
| **DOC-01** | **MEDIUM (P0)** | `README.md:28-33` | README quickstart pairs `tiny.yaml` (`vocab=1024`) with `BPETokenizer.pretrained()` (`vocab=2048`), crashing on step 1. |
| **DEFECT-ARCH-01** | **LOW (P2)** | `src/xrfm/models/gpt.py:114-126` | `ModelConfig.use_swiglu` is validated in config but silently ignored by `GPTModel`/`TransformerBlock`. |
