# XR Foundation Model (XRFM) — Independent Technical Research Review

**Research & Verification Date:** 2026-11-17 (Asia/Karachi)  
**Target Tracks:** Track A (`XRFM Native` from-scratch multilingual research LM) & Track B (`XRFM CoderSec` pretrained specialist via LoRA/QLoRA)  
**Target Languages & Scripts:** English (`en`, Latin), Urdu (`ur`, Arabic-derived Nastaliq/Naskh), Roman Urdu (`ur-Latn`, Latin — distinct stratum), Arabic (`ar`, Arabic), Hindi (`hi`, Devanagari), Spanish (`es`, Latin), French (`fr`, Latin)  
**Evidence Standard:** Official Hugging Face Hub APIs/Model Cards/Dataset Cards, official GitHub repositories/LICENSE files, official Kaggle & Hugging Face billing documentation, and primary peer-reviewed papers. Clearly distinguishes **Verified Facts**, **Analytical Estimates**, and **Engineering Recommendations**.

---

## 1. Current Small Open-Weight Model Landscape (Track B: XRFM CoderSec)

### 1.1 Evaluation Criteria for Track B Base Models
Track B requires a **1B–3.6B parameter** openly licensed pretrained model that:
1. Carries an **unrestricted open-source license** (Apache-2.0 preferred; no custom research-only or non-commercial clauses, and ideally ungated for automated Kaggle/CI reproducibility).
2. Fits comfortably inside **Kaggle free GPUs** (16 GB VRAM Tesla P100 or 2×16 GB Tesla T4) under QLoRA (4-bit NF4) or LoRA (FP16) with `seq_len = 2048–4096`.
3. Uses a **mature, stable `AutoModelForCausalLM` architecture** fully supported by `transformers`, `peft`, `trl` (`SFTTrainer`), `bitsandbytes`, `vLLM`, and `llama.cpp`/GGUF.
4. Has a **high-fertility multilingual + code tokenizer** covering Latin, Arabic/Urdu, Devanagari, and source-code tokens without byte-fallback explosion.
5. Demonstrates strong **coding, reasoning, tool-calling, and multilingual instruction-following**.

### 1.2 Primary-Source Comparison of Verified Candidate Models (Checked 2026-11-17)

All repository SHAs, parameter counts, architecture classes, context windows, and licenses below were queried directly against `https://huggingface.co/api/models/<repo>` and `<repo>/raw/main/config.json` on **2026-11-17**:

| Candidate Model (`repo_id`) | Pinned SHA (2026-11-17) | Exact Params (`safetensors`) | Architecture (`config.json`) | Vocab / Max Context | License & Gating | Coding & Reasoning Evidence (Model Card) | Multilingual Coverage (7 Target Langs) | Fine-Tuning & Kaggle T4/P100 Compatibility | Verdict |
|---|---|---:|---|---|---|---|---|---|---|
| **`Qwen/Qwen3-1.7B`** ([card](https://huggingface.co/Qwen/Qwen3-1.7B)) | `70d244cc86cc` | **2.03B** (`1.7B` nominal, `1.4B` non-emb) | `Qwen3ForCausalLM` (28L, d=2048, GQA 16Q/8KV, tied emb) | `151,936` / `40,960` (`32K` active) | **Apache-2.0**, Ungated | Dual `<think>` / `/no_think` reasoning + tool-use; surpasses Qwen2.5-1.5B-Instruct on code/math/commonsense (`transformers>=4.51.0`). | **100+ languages & dialects** explicitly including `en, ur, ar, hi, es, fr` + strong `ur-Latn` transfer via 151K BPE. | **Excellent.** Pure dense CausalLM; native `peft` + `trl` + `bitsandbytes` 4-bit (~3.6 GB VRAM at seq 2048). | **PRIMARY RECOMMENDED TRACK B BASE (#1A)** |
| **`Qwen/Qwen2.5-Coder-1.5B-Instruct`** ([card](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)) | `2e1fd397ee46` | **1.54B** (`1.31B` non-emb) | `Qwen2ForCausalLM` (28L, d=1536, GQA 12Q/2KV, tied emb) | `151,936` / `32,768` | **Apache-2.0**, Ungated | Pretrained on **5.5T code + text-code tokens** ([arXiv:2409.12186](https://arxiv.org/abs/2409.12186)); strongest pure-code 1.5B dense specialist; 237+ public HF adapters. | Shares Qwen 151,936 vocab (encodes `ur, ar, hi, es, fr` cleanly), though post-training is heavily English/Chinese code-centric. | **Excellent.** Smallest KV cache (only 2 KV heads!), fastest T4 QLoRA/LoRA training (~3.1 GB VRAM). | **CO-PRIMARY / ABLATION BASELINE (#1B — Pure Code Specialist)** |
| **`ibm-granite/granite-4.2-3b`** ([card](https://huggingface.co/ibm-granite/granite-4.2-3b)) | `4e9c275e232f` | **3.66B** (`3B` non-emb, untied emb) | `GraniteForCausalLM` (40L, d=2560, GQA 40Q/8KV, RoPE θ=10M) | `100,352` / `131,072` (512K ext.) | **Apache-2.0**, Ungated (Released Aug 25, 2026) | Very high reasoning/coding/agentic scores reported on card: LiveCodeBench v6 `69.71`, BFCL v4 `52.41`, `<think>` / low-effort modes. | Card lists **12 tested languages** (`en, es, fr, ar`, etc.); **Urdu (`ur`) & Hindi (`hi`) NOT in tested list** (`MMLU-ProX lite` drops to `27.78` at 3B). | **Moderate.** 3.66B untied weights = ~2.4 GB NF4 base + larger activations across 40 layers; tighter on single 16 GB GPU at seq 4096. | **STRONG SECONDARY CANDIDATE (#2 — High Code/Agentic Ceiling)** |
| **`Qwen/Qwen3.5-2B`** ([card](https://huggingface.co/Qwen/Qwen3.5-2B)) | `15852e8c1636` | **2.27B** | `Qwen3_5ForConditionalGeneration` (Early-fusion VLM + **Hybrid Gated DeltaNet + Gated Attention**, MTP) | `248,320` / `262,144` | **Apache-2.0**, Ungated | Outperforms Qwen3-1.7B on MMLU-Pro (`66.5` vs `56.5`) & BFCL-V4 (`43.6`). | **201 languages**, `MMMLU 63.1`, `MMLU-ProX 52.3`, massive 248K vocab. | **HIGH ENGINEERING RISK FOR QLoRA:** Uses `Qwen3_5ForConditionalGeneration` (vision encoder + linear-attention Gated DeltaNet kernels + 248K padded tied LM head = ~1.0 GB BF16 embedding alone). Custom linear-attention ops frequently break standard `bitsandbytes` 4-bit / `trl` packing on older Tesla T4/P100 GPUs! | **DEFER TO PHASE 6 ABLATION (Do NOT use as first pipeline test)** |
| **`HuggingFaceTB/SmolLM3-3B`** ([blog](https://huggingface.co/blog/smollm3)) | `a07cc9a04f16` | **3.08B** | `SmolLM3ForCausalLM` (36L, d=2048, GQA 16Q/4KV, NoPE+YaRN) | `128,256` / `65,536` (128K YaRN) | **Apache-2.0**, Ungated | Trained on 11T tokens with fully open recipe; dual `/think` mode. | Explicitly trained/supported on **6 European languages (`en, fr, es, de, it, pt`) ONLY** — weak on `ur`, `ar`, `hi`. | Good dense 3B compatibility, though newer `smollm3` model_type requires recent `transformers`. | **REJECT as primary (misses `ur, ar, hi`)** |
| **`Qwen/Qwen2.5-Coder-3B-Instruct`** | `488639f1ff80` | **3.09B** | `Qwen2ForCausalLM` (36L, d=2048, GQA 16Q/2KV) | `151,936` / `32,768` | **`other` (Qwen Research License — NON-COMMERCIAL!)** | Strong 3B coder, but **3B checkpoint is NOT Apache-2.0** (unlike 1.5B and 7B!). | Same 151K Qwen2 vocab. | Fits T4 in 4-bit, **but fails commercial/open-license criterion!** | **REJECT (Non-commercial license trap!)** |
| **`HuggingFaceTB/SmolLM2-1.7B-Instruct`** | `31b70e2e869a` | **1.71B** | `LlamaForCausalLM` (24L, d=2048, MHA 32Q/32KV) | `49,152` / `8,192` | **Apache-2.0**, Ungated | Good English compact baseline, superseded by Qwen3-1.7B & SmolLM3. | Primarily English (`49,152` vocab has poor Urdu/Hindi fertility). | Standard `LlamaForCausalLM`. | **REJECT (English-only bias, 8K ctx)** |
| **`ibm-granite/granite-3.3-2b-instruct`** | `707f574c6205` | **2.53B** | `GraniteForCausalLM` (40L, d=2048, GQA 32Q/8KV) | `49,159` / `131,072` | **Apache-2.0**, Ungated | Superseded by `granite-4.2-3b`. | 12 enterprise languages; `49K` vocab weak on `ur`/`hi`. | Standard `GraniteForCausalLM`. | **SUPERSEDED** |
| **`google/gemma-3-1b-it`** | `dcc83ea841ab` | **1.00B** | `Gemma3ForCausalLM` | `262,144` / `32,768` | **Gemma Terms of Use** (**Manual Gated**) | Lightweight 1B, weaker coding ceiling than 1.5B–2B Qwen coders. | Strong 262K SentencePiece multilingual vocab (`ur, hi, ar, es, fr`). | Requires manual HF gate acceptance + `HF_TOKEN` on every pull; custom Gemma prohibited-use terms. | **BACKUP ONLY (Gated + custom license)** |

### 1.3 Critical Base-Model Selection Recommendation

> **Recommendation (Evidence-Backed):**  
> 1. **Test `Qwen/Qwen3-1.7B` (`sha: 70d244cc86cc`, Apache-2.0) FIRST as the primary Track B (`XRFM CoderSec`) base model**, paired with **`Qwen/Qwen2.5-Coder-1.5B-Instruct` (`sha: 2e1fd397ee46`, Apache-2.0) as the head-to-head Phase 4 baseline comparator**.  
> 2. **Why this exact pair wins:**
>    - Both are **100% Apache-2.0 licensed and ungated** (note how our API audit caught a crucial trap: `Qwen2.5-Coder-3B-Instruct` uses the restrictive Qwen Research License `license: other`, whereas `Qwen3-1.7B` and `Qwen2.5-Coder-1.5B-Instruct` are `apache-2.0`!).
>    - Both use pure dense standard Transformer blocks (`Qwen3ForCausalLM` / `Qwen2ForCausalLM`) with GQA, SwiGLU, RoPE, RMSNorm, and the **151,936-token byte-level BPE tokenizer**, which already has dedicated subwords for Arabic/Urdu, Devanagari, Spanish, French, and code — unlike `Qwen3.5-2B` (whose VLM + Gated DeltaNet hybrid architecture risks QLoRA/kernel incompatibilities on Kaggle T4/P100) and unlike `SmolLM3-3B` / `Granite-4.2-3B` (which were not validated on Urdu/Hindi).
>    - Running a zero-shot Phase 4 bake-off between `Qwen3-1.7B` (stronger 100+ language instruction following + `/think` reasoning + native tool calling) and `Qwen2.5-Coder-1.5B-Instruct` (5.5T code-specialized pretraining) on our 7-language + coding + CyberSec baseline takes **<45 minutes on a single Kaggle T4** and lets empirical data decide whether general multilingual reasoning (`Qwen3-1.7B`) or code-heavy pretraining (`Qwen2.5-Coder-1.5B-Instruct`) adapts better under QLoRA!

---

## 2. Fine-Tuning & GPU Memory Engineering (Track B on Kaggle)

### 2.1 Critical Hardware Compatibility Matrix: Kaggle P100 vs. T4 vs. L4

From official Kaggle Notebook Specifications ([`kaggle.com/docs/notebooks`](https://www.kaggle.com/docs/notebooks), verified 2026-11-17) and NVIDIA compute capability rules:

| GPU Platform | Where Available | VRAM | Compute Capability | FP16 + GradScaler | Native BF16 | FlashAttention-2 | `bitsandbytes` NF4 QLoRA | Practical Role in XRFM |
|---|---|---:|---|---|---|---|---|---|
| **NVIDIA Tesla P100** | Kaggle Free (`1× P100`, 29 GB RAM, 4 vCPU) | `16 GB` HBM2 | **SM 6.0 (Pascal)** | Yes (fast FP16) | **NO** (`is_bf16_supported() == False`) | **NO** (requires SM 8.0+) | Slow / legacy kernels (no Turing/Ampere INT8/TensorCores) | Good for **Track A (`XRFM Native` FP16 pretraining)**; **AVOID for QLoRA** (use T4 instead). |
| **NVIDIA Tesla T4 (`×2`)** | Kaggle Free (`2× T4`, 29 GB RAM, 4 vCPU) | `2 × 16 GB` (`32 GB` aggregate) | **SM 7.5 (Turing)** | **Yes (Tensor Cores)** | **NO** (emulated/slow; **must use `fp16=True, bf16=False`**!) | **NO** (uses PyTorch SDPA `mem_efficient` xformers-style kernel) | **YES (Mature & Fast)** | **PRIMARY GPU FOR TRACK B (`XRFM CoderSec` QLoRA/LoRA)** & 2-GPU DDP verification for Track A! |
| **NVIDIA L4** | Hugging Face Jobs (`$0.80/hr`, 24 GB VRAM) / Colab Pro | `24 GB` GDDR6 | **SM 8.9 (Ada)** | Yes | **YES (Native Tensor Cores)** | **YES** | **YES** | Optional paid burst (`$0.80/hr`); **NOT required** if Kaggle `T4 ×2` is used properly. |

> **Critical Precision Rule for Kaggle T4 (`SM 7.5`):** Many Qwen3/Granite checkpoints ship with `"torch_dtype": "bfloat16"` in `config.json`. Loading them with `torch_dtype="auto"` on a Kaggle T4 either fails or triggers catastrophic slow software emulation / GradScaler crashes in PyTorch AMP. On T4, **always explicitly pass `torch_dtype=torch.float16`, `bnb_4bit_compute_dtype=torch.float16`, `fp16=True`, `bf16=False`**, and cast trainable LoRA adapter weights to `float32` (PEFT default)!

### 2.2 Explicit VRAM Budget Calculation (`Qwen3-1.7B` QLoRA on 1× T4 16 GB)

Let $P = 2.03 \times 10^9$ total parameters (`Qwen3-1.7B`), LoRA rank $r = 16$, $\alpha = 32$, applied to all 7 linear projections (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) across 28 layers ($d = 2048, d_{kv} = 1024, d_{ff} = 6144$):
- **Trainable LoRA Parameters:** Per layer $\approx 2 r (4 d + 2 d + 3 d + d_{ff}) \dots$ exact calculation:
  - `q_proj` ($2048 \to 2048$): $2 \times 16 \times 2048 = 65,536$
  - `k_proj`, `v_proj` ($2048 \to 1024$ each): $2 \times 16 \times (2048 + 1024) = 49,152 \times 2 = 98,304$
  - `o_proj` ($2048 \to 2048$): $65,536$
  - `gate_proj`, `up_proj`, `down_proj` ($2048 \leftrightarrow 6144$ each): $3 \times 16 \times (2048 + 6144) = 393,216$
  - Total per layer $= 622,592 \implies \times 28\text{ layers} = \mathbf{17,432,576}$ trainable parameters (**0.86%** of model; embeddings/LM head kept frozen in 4-bit/FP16 since Qwen's 151,936 vocab already covers all 7 languages!).

| Memory Component (`seq_len = 2048`, `micro_batch = 2`, `grad_accum = 8`, Gradient Checkpointing ON) | Exact / Analytical Estimate | Notes |
|---|---:|---|
| **Base Model Weights (4-bit NF4 + double quant)** | `~1.35 GB` | Non-embedding `1.4B * 0.55 B` + quantized/FP16 tied `151,936 × 2048` embedding (`~0.62 GB`) |
| **LoRA Adapter Weights (FP32 master) + Gradients (FP32)** | `~0.14 GB` | `17.4M × (4 + 4) bytes` |
| **Optimizer States (`paged_adamw_8bit` on 17.4M params)** | `~0.04 GB` | Even standard FP32 AdamW is only `17.4M × 8B = 0.14 GB`! |
| **Activations (with Gradient Checkpointing + SDPA `mem_efficient`)** | `~2.10 GB` | `micro_batch=2, seq=2048`, dominated by logits cross-entropy (`2 × 2048 × 151,936 × 2B ≈ 1.24 GB` — reduced to `<0.3 GB` if chunked/fused CE or `micro_batch=1` at `seq=4096`) |
| **CUDA Context + PyTorch Allocator Reserve** | `~1.20 GB` | cuBLAS workspaces + fragmentation |
| **TOTAL PEAK VRAM (1× T4 16 GB)** | **`~4.83 GB / 16.0 GB`** | **Leaves >11 GB headroom!** Even `seq_len = 4096, micro_batch = 2` fits inside **~7.8 GB** on a single T4! |

**Session Runtime Feasibility on Kaggle (`12h` hard session cap, `~30h/week` GPU quota):**
- With **Sequence Packing (`packing=True` in `trl.SFTTrainer` / `SFTConfig`)** and `max_seq_length=2048`, a curated, high-signal dataset of **40,000 instruction/code/security/multilingual examples** (~25M–30M packed tokens $\approx 13,500$ packed sequences of length 2048) with `effective_batch_size = 16` (`per_device_bs=2, grad_accum=8` on 1× T4, or `per_device_bs=2, grad_accum=4` on 2× T4 DDP) equals **~845 optimizer steps per epoch** (~2,535 steps for 3 epochs).
- At ~3.5–4.5 sec/step on T4 (`Qwen3-1.7B` QLoRA with GC), **1 epoch takes ~55–65 minutes**, and **3 full epochs complete in ~3.0 hours** — comfortably inside a single 12-hour Kaggle session with frequent checkpointing every 200 steps (`<75 MB` per LoRA adapter checkpoint in `/kaggle/working`, well inside the 20 GB limit)!

---

## 3. Tokenizer Research & Controlled Tournament Design (Track A)

### 3.1 Candidate Algorithm Comparison for XRFM Native
1. **Candidate A — Current `xrfm.tokenization.BPETokenizer` (Pure Python Byte-BPE):**
   - *Pros:* Zero C++/Rust dependencies, 100% lossless UTF-8 round-trip, explicit special-token contract.
   - *Cons:* Training is $O(M \cdot N)$ over un-deduplicated words; regex `\s+|\S+` glues punctuation to words; `encode()` is $O(k^2)$ per word in pure Python (~605 KB/s) and ignores special-token literals.
2. **Candidate B — Optimized Native `BPETokenizer` (Word-Frequency Pre-Aggregation + GPT-4/Qwen-style Unicode Regex + Pair-Index Heap):**
   - *Pros:* By counting unique pre-tokenized chunks `Counter(words)` *once* and maintaining an inverted index from `(tok_a, tok_b) -> set[word_id]`, pure-Python training speeds up by **50×–200×** while keeping 100% native XRFM Python ownership! Splitting with a Unicode category regex (separating letters `\p{L}\p{M}*` — crucial so Devanagari matras `\p{M}` and Arabic harakat stay attached to base letters! — individual digits `\p{N}{1,3}`, punctuation, and whitespace/indentation runs) dramatically improves code and Indic/Arabic morphology.
3. **Candidate C — Hugging Face `tokenizers` (Rust `ByteLevelBPETokenizer`, Apache-2.0, [`github.com/huggingface/tokenizers`](https://github.com/huggingface/tokenizers)):**
   - *Pros:* Trains a 32K–50K BPE vocabulary on **1 GB of multilingual + code text in <60 seconds on 2–4 CPU cores** in streaming iterator mode (`train_from_iterator`, `<500 MB` RAM!), exports a standard `tokenizer.json` that can also be loaded into a thin `HFTokenizerAdapter(Tokenizer)` implementing XRFM's exact `Tokenizer` ABC *or* exported directly into XRFM's `vocab.json` `(vocab, merges, special_tokens)` format!
4. **Candidate D — Google `SentencePiece` Unigram (`Apache-2.0`, [`github.com/google/sentencepiece`](https://github.com/google/sentencepiece)):**
   - *Pros:* Unigram LM tokenization (`--model_type=unigram --byte_fallback=true --split_digits=true --allow_whitespace_only_pieces=true`) often achieves slightly better morphological segmentation on agglutinative/rich-morphology scripts (`ur`, `ar`, `hi`) at small vocabularies (16K–32K).
   - *Cons:* Requires `normalization_rule_name="identity"` (default NFKC corrupts certain code/math symbols and Arabic presentation forms!) and careful whitespace `▁` preservation so Python indentation isn't mangled.

### 3.2 Vocabulary Size Trade-Off Analysis (2K vs. 16K vs. 32K vs. 50K)
In `XRFM Native` (`weight_tied=True`), embedding parameters equal $V \times d_{\text{model}}$:
- At $d_{\text{model}} = 512, L = 12$ (~50M non-embedding params):
  - `V = 2,048`: Embedding = `1.05M` params (2%), **but** 7 languages + 8 programming languages cannot fit (~200 merges per script $\implies$ severe fragmentation, long sequences, slow attention).
  - `V = 16,384`: Embedding = `8.39M` params (14%). Viable for a 20M–50M model, ~1,500–2,500 subwords per script family.
  - **`V = 32,768` (Sweet Spot for 50M–150M Native Multilingual + Code):** Embedding = `16.78M` params at $d=512$ (or `25.17M` at $d=768$). Provides dedicated subwords for Latin (`en, es, fr, ur-Latn`), Arabic/Urdu (`ar, ur`), Devanagari (`hi`), and code keywords/indentation without starving transformer depth on small compute budgets.
  - `V = 50,304`: Embedding = `25.76M` params at $d=512$ (`38.6M` at $d=768$). Under-trains rare tail tokens unless the pretraining run exceeds `~2B–3B` tokens.

### 3.3 Controlled Tokenizer Benchmark Protocol
Every candidate (`A, B, C, D` × `{2048, 16384, 32768, 50304}`) must be trained on the **exact same temperature-sampled multilingual + code training split** (`T = 0.5` sampling across `en, ur, ur-Latn, ar, hi, es, fr, code`) and evaluated on a **locked held-out benchmark** measuring:
1. **Exact UTF-8 Round-Trip Accuracy (`%`):** Must be `100.00%` on all 7 languages + Python/C++/Rust/Bash indentation + edge-case Unicode (zero-width non-joiner `U+200C` in Urdu/Persian, Devanagari conjuncts `क्ष/त्र/ज्ञ`, Arabic diacritics).
2. **Fertility (Tokens / Word) & Compression (UTF-8 Bytes / Token)** reported **independently per language** (never averaged into a single hiding number) and **Parity Ratio** $\text{Fertility}(\text{lang}) / \text{Fertility}(\text{en})$ (target $\le 1.6\times$ for `ur, ar, hi` at `32K`).
3. **Code Whitespace Efficiency:** Tokens per 4-space / 8-space indent line in Python/YAML.
4. **Throughput:** `MB/s` encode and decode on CPU.
5. **Vocabulary Occupancy (`%` active IDs on held-out 10 MB dev set)** to detect dead vocabulary slots.
6. **Downstream Model Comparison Metric — Bits-Per-Byte (`BPB`):**
   $$\text{BPB} = \frac{\sum_{i=1}^{N_{\text{tok}}} -\ln P(t_i \mid t_{<i})}{N_{\text{UTF8\_bytes}} \cdot \ln 2}$$
   Because `BPB` normalizes total negative log-likelihood by the *raw UTF-8 byte length of the held-out text* rather than token count, a 16K BPE model, a 32K Unigram model, and a 50K BPE model can be compared fairly!

---

## 4. Multilingual, Roman Urdu, Coding & Defensive Cybersecurity Dataset Research

### 4.1 Multilingual Pretraining & Instruction Datasets (Verified 2026-11-17)

| Dataset (`repo_id` / URL) | Pinned SHA (2026-11-17) | License & Governance Tier | Verified Target Subsets & Sizes | Quality, Deduplication & PII Risks | Streaming & Sampling Strategy (Zero Full Download) |
|---|---|---|---|---|---|
| **`HuggingFaceFW/fineweb-2`** ([HF](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)) | `af9c13333eb9` | **ODC-By-1.0** + Common Crawl ToU (**Tier B — Review/Attribution**) | **Verified directly via HF Tree API:**<br>• `urd_Arab`: **2 shards (`6.40 GB`)**, test `68 MB`<br>• `urd_Latn`: **1 shard (`122.9 MB`)**, test `1.0 MB`<br>• `hin_Deva`: **8 shards (`31.92 GB`)**, test `84.6 MB`<br>• `arb_Arab`: **25 shards (`98.69 GB`)**, test `71.9 MB`<br>• `spa_Latn`: **146 shards (`593.82 GB`)**, test `56.9 MB`<br>• `fra_Latn`: **135 shards (`502.82 GB`)**, test `63.9 MB` | Language-specific quality thresholds + MinHash dedup applied per subset by HuggingFaceFW. Still web-crawled text: requires XRFM secondary PII/secret pass & contamination check. | **Stream individual Parquet shards via HTTP range / `datasets` `streaming=True` or `hf_hub_download` of 1 shard at a time on Kaggle**; sample balanced quotas (e.g., 50M–100M tokens/lang) into compact `.npy` shards. |
| **`HuggingFaceFW/fineweb-edu`** ([HF](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)) | `87f09149ef47` | **ODC-By-1.0** + CC ToU (**Tier B**) | English (`en`) educational web text (`sample-10BT` subset available). | Classifier-filtered (`>=3` educational score); very high prose/technical explanation density for small LMs (proven in SmolLM2/3). | Stream `sample-10BT` (`streaming=True`), take deterministic hash-sampled slice for English Track A pretraining. |
| **`ai4bharat/sangraha`** ([HF](https://huggingface.co/datasets/ai4bharat/sangraha)) | `8b813c3f62d3` | **CC-BY-4.0** (**Tier A/B — Permissive with Attribution**) | `verified` subset covers **Urdu (`urd`)** and **Hindi (`hin`)** (human-verified scrapes, OCR-verified PDFs, Indic transcriptions). | Much cleaner editorial/educational Indic & Urdu quality than raw mC4/OSCAR. | Stream `verified/urd` and `verified/hin` splits via `streaming=True`. |
| **`CohereLabs/aya_dataset`** & **`aya_collection_language_split`** ([HF](https://huggingface.co/datasets/CohereLabs/aya_dataset)) | `f9ea04583f02` / `a3af2fde4b4c` | **Apache-2.0** (`aya_dataset` human-annotated!) (**Tier A**) | **Verified in README:** Explicitly includes **`eng, urd, arb, hin, spa, fra`** human-written & expert-curated instruction/response pairs (~204K human annotations in `aya_dataset`). | Gold-standard human-written multilingual instruction data; note `aya_collection` includes translated subsets (filter to human/permissive sub-datasets only!). | Lightweight (~200 MB total; language splits are a few MB each), ideal for Track B multilingual SFT! |
| **`allenai/MADLAD-400`** ([HF](https://huggingface.co/datasets/allenai/MADLAD-400)) | `9d886a76bd8f` | Card metadata `odc-by`, paper mentions CC-BY-4.0 (**HOLD until revision license reconciled**) | 419 languages (`ur, hi, ar, es, fr, ur_Latn` noisy). | Document-level audited Common Crawl, noisier than FineWeb-2. | **HOLD** — FineWeb-2 + Sangraha provide cleaner, better-documented subsets. |
| **`uonlp/CulturaX`** ([HF](https://huggingface.co/datasets/uonlp/CulturaX)) | `6a8734bc69fe` | `license: None` on Hub, auto-gated, mC4+OSCAR (**HOLD**) | 167 languages. | Missing unified license declaration on HF Hub metadata. | **HOLD / DO NOT INGEST BY DEFAULT.** |

### 4.2 Dedicated Roman Urdu (`ur-Latn`) & Code-Switching Strategy
Roman Urdu is **non-standardized** (e.g., *"mujhe" / "mujy" / "mjhy"*, *"kya" / "kia"*), frequently mixed with English technical identifiers (*"Yeh Python script JWT token verify karti hai lekin expired signature par exception throw nahi kar rahi"*), and cosmetically overlaps with Roman Hindi (`hin_Latn`).
1. **Why Mechanical Transliteration Alone Fails:** Transliterating formal Arabic-script Urdu news/literature (`یہ ایک انتہائی پیچیدہ مسئلہ ہے`) via rule-based/Aksharantar models produces unnatural, hyper-Persianized/formal Latin strings that real Pakistani/South Asian developers never type, and misses authentic Urdu-English code-switching around programming keywords!
2. **Verified Public Sources & Their Caveats:**
   - `HuggingFaceFW/fineweb-2` (`urd_Latn` subset, `122.9 MB` train, `1.0 MB` test, ODC-By): Authentic web-mined Roman Urdu — **must be audited** because LID frequently confuses `urd_Latn`, `hin_Latn` (`189.3 MB`), and Indonesian/Malay noise.
   - `ai4bharat/Aksharantar` (`sha: e418c1fc928d`, CC-BY/CC0 word-level pairs): Useful **only** for lexicon/transliteration-invariance testing and orthographic normalization diagnostics, **not** document generation.
   - `Roman-Urdu-Parl` & `UCI/GitHub Roman Urdu`: **HOLD** due to conflicting Apache-2.0 / CC-BY / GPL-3.0 mirror licenses recorded in `05_TOOL_DATA_EVAL_MATRIX.md`.
3. **Actionable 4-Stratum Roman Urdu Pipeline (Strict Separation):**
   - **Stratum RU-1 (Gold Human-Reviewed Evaluation & Seed SFT — Tier A):** Author and human-review **500 evaluation items** (sealed, never trained on!) + **1,500–3,000 natural developer/security Q&A prompts** in authentic Pakistani Roman Urdu and Roman-Urdu/English code-switching (covering Python/JS/Linux/SQL/OWASP).
   - **Stratum RU-2 (Filtered Web `fineweb-2/urd_Latn` — Tier B):** Stream the `122.9 MB` `urd_Latn` shard, apply a dedicated **Roman-Urdu Lexicon + Character-Ngram Scorer** (requiring high-specificity function words/postpositions like *hai, hain, nahi, mein, ke, ki, ko, se, aap, kya, kyunki, lekin, agar, chahiye* while rejecting Devanagari/Indonesian/spam), and tag provenance as `script=Latn, lang=ur-Latn, origin=web_natural`.
   - **Stratum RU-3 (Controlled Bilingual Code-Switched Technical Synthesis — Clearly Tagged `origin=synthetic_reviewed`):** Generate technical explanations and secure-coding dialogues grounded in permissive code/OWASP passages, enforce automated lexical/code-execution verification (every code block must parse/pass AST or unit tests!), and sample **10%–15% for native Urdu/Roman-Urdu human acceptance review** (reject batch if naturalness `< 4.0/5`).
   - **Stratum RU-4 (Transliterated Parallel Alignment — Max 15% of `ur-Latn` mix):** Paired `ur` $\leftrightarrow$ `ur-Latn` technical glossary sentences so the model links `سائبر سیکیورٹی / انکرپشن` with `cybersecurity / encryption`, explicitly labeled `origin=transliterated` and **never** counted toward natural Roman Urdu evaluation metrics.

### 4.3 Coding Datasets Research (Track A & Track B)

| Dataset | Pinned SHA / Source | License | Size & Languages | Critical Security/License Controls & Usage Plan |
|---|---|---|---|---|
| **`bigcode/the-stack-v2-dedup`** ([HF](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup)) | `611ef38fbfab` | Gated (`auto`), **per-file open-source licenses** (Software Heritage IDs) | Metadata + SWH pointers across 600+ langs (`Python, JS, TS, C, C++, Rust, Bash, SQL`). | **Never download bulk! Filter strictly to permissive SPDX licenses (`MIT`, `Apache-2.0`, `BSD-3-Clause`, `BSD-2-Clause`, `ISC`, `PSF-2.0`)** — exclude GPL/AGPL/LGPL/unlicensed files to preserve Track A/B permissive licensing! Run secret/PII regex + malware keyword filter. |
| **`OpenCoder-LLM/opc-sft-stage2`** ([HF](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2)) | `7d28f40d579e` | **MIT** (**Tier A**, Ungated) | High-quality educational code instruction, debugging, unit-test generation, and algorithmic reasoning pairs ([OpenCoder paper](https://arxiv.org/abs/2411.04905)). | **Top choice for Track B coding SFT:** MIT-licensed, ungated, clean instruction+test pairs; stream and sample ~15K–25K examples across Python, JS/TS, C/C++, Rust, Bash, SQL. |
| **`ise-uiuc/Magicoder-OSS-Instruct-75K`** ([HF](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K)) | `5f839b1f368a` | **MIT** repo tag, *synthetic via OpenAI API* (**Tier B — Review upstream ToS**) | `75K` diverse open-source-seeded coding problems. | Note OpenAI ToS synthetic provenance caveat; prefer **`opc-sft-stage2`** and **`nvidia/OpenCodeInstruct`** where clean commercial-friendly provenance is desired. |
| **`nvidia/OpenCodeInstruct`** ([HF](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)) | `8f3ba5bafe4d` | **CC-BY-4.0** (**Tier A**, Ungated) | `5M` examples (`6.86 GB` download, streamable!) with **unit tests and execution test scores** (`tests_execution_status`, `average_test_score`)! | Stream (`streaming=True`) and filter exclusively to rows where `tests_execution_status` passed and `llm_judgement` is top-tier! Zero bulk download needed. |
| **`nickrosh/Evol-Instruct-Code-80k-v1`** | `3ae930c20d54` | **`cc-by-nc-sa-4.0` (NON-COMMERCIAL!)** | 80K code instructions. | **EXCLUDE FROM TRAINING** due to `NC` (Non-Commercial) license! |

### 4.4 Defensive Cybersecurity Dataset & Knowledge Sources Research

Cybersecurity AI requires strict **capability taxonomy** and **dual-use safety filtering**. We partition defensive security tasks into five legitimate engineering competencies:
1. **Explanatory Knowledge:** Threat modeling, CIA triad, cryptography primitives, Linux permissions, TLS/DNS/TCP-IP, NIST CSF 2.0 / SSDF (`SP 800-218`), MITRE ATT&CK tactics/detections.
2. **Defensive Instruction-Following:** Hardening `sshd_config`, `systemd` sandboxing, `ufw`/`nftables` rules, `auditd`/`journalctl` log triage, incident containment playbooks.
3. **Secure Code Generation:** Writing parameterized SQL queries, constant-time HMAC verification, CSRF/XSS-safe web handlers, memory-safe Rust/C++ bounds checking.
4. **Vulnerability Identification & Secure Code Review:** Identifying CWE classes (e.g., CWE-89 SQLi, CWE-79 XSS, CWE-78 OS Command Injection, CWE-22 Path Traversal, CWE-798 Hardcoded Credentials, CWE-502 Insecure Deserialization) in source diffs with line-level root-cause explanations.
5. **Remediation & Patch Generation:** Producing minimal, functionality-preserving security patches + regression unit tests, while **refusing offensive weaponization** (exploit payload generation, reverse shells, ransomware, credential harvesting, unauthorized scanning against named targets) without **over-refusing** benign administrative/defensive questions (e.g., *"How do I check listening ports on Linux with `ss -tulnp`?"*).

**Verified Authoritative & Openly Licensed Defensive Sources (Checked 2026-11-17):**

| Source | Official URL / Repo | Verified License | Defensive Role (SFT vs. RAG) | Safety & Licensing Controls |
|---|---|---|---|---|
| **OWASP Cheat Sheet Series** | [`github.com/OWASP/CheatSheetSeries`](https://github.com/OWASP/CheatSheetSeries) | **CC-BY-SA-4.0** (verified via GitHub API) | 137+ builder-focused secure coding & hardening guides (SQLi, Auth, Docker, Linux, Logging, AI Security). Ideal for **RAG index + grounded SFT QA**. | Retain attribution; note ShareAlike (`CC-BY-SA-4.0`) if redistributing derived text corpora — keep OWASP text in a modular **RAG knowledge pack** and/or clearly attributed separate dataset subset! |
| **OWASP Top 10 (2021 & 2025)** & **OWASP ASVS 5.0** | [`github.com/OWASP/Top10`](https://github.com/OWASP/Top10), `OWASP/ASVS` | **CC-BY-SA-4.0 / CC-BY-3.0** | Canonical web/API vulnerability taxonomy & verification requirements. | Strip any raw PoC attack strings that resemble copy-paste weaponized payloads; emphasize detection & prevention. |
| **MITRE CWE Catalogue** | [`cwe.mitre.org`](https://cwe.mitre.org/about/termsofuse.html) | **MITRE CWE Terms of Use** (free research/commercial use with attribution notice) | 900+ software/hardware weakness definitions, demonstrative vulnerable/fixed C/Python/Java/JS code snippets, mitigations. | Reproduce MITRE copyright/attribution notice; never imply MITRE endorsement. |
| **MITRE ATT&CK STIX (`mitre/cti`)** | [`github.com/mitre/cti`](https://github.com/mitre/cti) | **MITRE ATT&CK License** (royalty-free research/dev/commercial use with notice; verified in `LICENSE.txt`) | Structured JSON/STIX enterprise/cloud/Linux techniques, **mitigations (`course-of-action`)**, and **detection data sources**. | Extract **descriptions + mitigations + detection analytics**; exclude red-team procedural command one-liners that lack defensive framing. |
| **NIST Publications (`SP 800-53r5`, `SP 800-218 SSDF`, `SP 800-61r3` Incident Response, `CSF 2.0`)** | [`csrc.nist.gov`](https://csrc.nist.gov/publications) | **U.S. Government Public Domain** (`17 U.S.C. § 105`) | **Tier A (Unrestricted Public Domain!)** Authoritative incident response, secure SDLC, access control, cryptographic & log management guidance. | Highest license cleanliness on Earth — zero ShareAlike restrictions; ideal backbone for defensive policy/IR SFT & RAG! |
| **`CyberNative/Code_Vulnerability_Security_DPO`** | [`huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO`](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO) (`sha: 1788f0a6ab36`) | **Apache-2.0** (verified via HF API) | Paired `chosen` (secure code) vs `rejected` (vulnerable code) across Python, C++, JS, Java, Go, Rust, PHP, Ruby. | **Audit & AST-validate every code pair** before SFT/DPO admission (synthetic vulnerability datasets can contain toy/non-compiling snippets). |

---

## 5. Benchmark & Evaluation Landscape (Coding, CyberSec, Multilingual)

All candidate benchmarks were verified against their official repositories and licenses on **2026-11-17**:

| Benchmark | Official Repo | Verified License | What It Measures | What It Does NOT Measure | Contamination & Usage Rules |
|---|---|---|---|---|---|
| **EvalPlus (`HumanEval+` & `MBPP+`)** | [`github.com/evalplus/evalplus`](https://github.com/evalplus/evalplus) | **Apache-2.0** | Function-level Python code synthesis under 80×/35× augmented edge-case unit tests. | Multi-file repo understanding, non-Python languages, security of generated code. | **EVAL-ONLY.** Register all prompts/canonical solutions in `ContaminationRegistry` (`13-gram` + SHA-256). Run inside sandboxed subprocess with CPU/memory/timeout limits! |
| **LiveCodeBench** | [`github.com/LiveCodeBench/LiveCodeBench`](https://github.com/LiveCodeBench/LiveCodeBench) | **MIT** | Time-stamped competitive programming, code execution prediction, test output prediction, self-repair (filtrable by release date after base-model cutoff!). | Everyday systems engineering, Linux/Bash, secure coding. | **EVAL-ONLY.** Filter to problem dates after base-model training cutoff to guarantee zero pretraining leakage. |
| **MultiPL-E** | [`github.com/nuprl/MultiPL-E`](https://github.com/nuprl/MultiPL-E) | **BSD-3-Clause with ML Restriction** (*Clause 4 explicitly forbids training!*) | Translates HumanEval/MBPP into **JavaScript, TypeScript, C++, Rust, Bash, etc.** | Real idiomatic repo architecture. | **STRICT EVAL-ONLY** (legally enforced by License Clause 4 — verified directly in `nuprl/MultiPL-E/LICENSE`!). |
| **PurpleLlama `CyberSecEval 2` (and v3/v4 in `CybersecurityBenchmarks`)** | [`github.com/meta-llama/PurpleLlama`](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks) | **MIT** (verified in `CybersecurityBenchmarks/LICENSE`; note Safeguard *models* use Llama License, *benchmarks* are MIT!) | (1) **Insecure Coding Practice** (CWE static analysis via `insecure_code_detector`), (2) **Prompt Injection**, (3) **Cyberattack Helpfulness / Safe Refusal**, (4) **False-Positive Over-Refusal (`FRR`)** on benign security prompts! | Deep multi-turn patch generation or Urdu/Hindi security explanations. | **EVAL-ONLY.** Note that full LLM-judge sub-tasks require a judge model or deterministic `insecure_code_detector` semgrep/regex rules; prioritize deterministic ICD + prompt-injection + FRR subsets first! |
| **Multilingual Suite (`Belebele`, `UrduMMLU`, `UrBLiMP`, `RUAB` + Custom **`XRFM-MultiCoderSec-Bench`**)** | [`facebook/belebele`](https://huggingface.co/datasets/facebook/belebele) (`CC-BY-SA-4.0`) + XRFM internal gold suite | `CC-BY-SA-4.0` / `MIT` | `Belebele` tests parallel reading comprehension across `eng_Latn, urd_Arab, urd_Latn, arb_Arab, hin_Deva, spa_Latn, fra_Latn`! Custom **`XRFM-MultiCoderSec-Bench`** (350 human-verified items = 50 per language across the 7 languages) tests coding instructions, code explanations, debugging, defensive security Q&A, terminology consistency, and `ur-Latn`/English code-switching. | Open-ended long-horizon repo engineering. | **SEALED EVAL-ONLY.** Never include in pretraining or SFT. |

---

## 6. Compute & Platform Strategy: Kaggle + Hugging Face (Verified 2026-11-17)

### 6.1 Kaggle Free GPU Environment (Primary Compute Engine — `$0.00` Cost)
Verified from [`https://www.kaggle.com/docs/notebooks`](https://www.kaggle.com/docs/notebooks) (accessed 2026-11-17):
- **Available GPU Flavors:**
  - `GPU T4 x2`: **2 × NVIDIA Tesla T4** (`2 × 16 GB = 32 GB` VRAM total), 4 CPU cores, **29 GB system RAM**.
  - `GPU P100`: **1 × NVIDIA Tesla P100** (`16 GB` VRAM), 4 CPU cores, **29 GB system RAM**.
- **Session & Quota Limits:**
  - **Max continuous session runtime:** **12 hours** (CPU/GPU) for both interactive and `"Save & Run All"` background batch commits; **20-minute idle timeout** in interactive editor.
  - **Weekly GPU quota:** Account-specific rolling 7-day quota (standard verified Kaggle tier is typically **~30 GPU hours/week**; must be inspected on the developer's Kaggle Settings/Quota page before each run).
- **Storage & Persistence Workflow:**
  - `/kaggle/working`: **20 GB** auto-saved persistent output per notebook run.
  - `/kaggle/temp` (or `/tmp`): Ephemeral session scratchpad (larger, wiped when session ends — ideal for streaming shard caches!).
  - **Kaggle Datasets Input (`/kaggle/input/`):** Read-only mounted versioned datasets (up to **200 GB per dataset**), zero copy time at session boot!
  - **Kaggle Secrets:** Built-in `kaggle_secrets.UserSecretsClient()` securely injects `HF_TOKEN` into runtime memory without ever writing secrets into notebook cells or logs.

### 6.2 Hugging Face Hub, PRO Subscription & `$4` Compute Credits Strategy
Verified from [`https://huggingface.co/docs/hub/pro`](https://huggingface.co/docs/hub/pro), [`billing`](https://huggingface.co/docs/hub/billing), [`jobs-pricing`](https://huggingface.co/docs/hub/jobs-pricing), and [`spaces-zerogpu`](https://huggingface.co/docs/hub/spaces-zerogpu) (accessed 2026-11-17):
- **Free / Included with HF PRO (Zero Credit Burn):**
  - Hosting public and private **Model repositories** (`safetensors` checkpoints, LoRA adapters, tokenizer artifacts, model cards).
  - Hosting versioned **Dataset repositories** (curated Parquet/`.npy` shards, manifests, evaluation gold sets).
  - Hosting up to **10 ZeroGPU Gradio Spaces** with **8× daily ZeroGPU usage quota** and highest queue priority (ideal for the Phase 8 `XRFM CoderSec` + `XRFM Native` live demo without spending a cent of credits!).
  - Higher Hub API rate limits and bandwidth for streaming FineWeb-2 / OpenCodeInstruct shards into Kaggle.
- **Official Hugging Face Jobs Pay-As-You-Go Pricing (`jobs-pricing`, verified 2026-11-17):**
  - Billed **by the minute** only while `Starting` or `Running` (default `--timeout 30m`, must always set explicit `--timeout` cap!):
    - `CPU Basic` (2 vCPU, 16 GB RAM, 50 GB disk): **`$0.01 / hr`**
    - `CPU Upgrade` (8 vCPU, 32 GB RAM, 50 GB disk): **`$0.03 / hr`**
    - `Nvidia T4 - small` (4 vCPU, 15 GB RAM, **16 GB VRAM**, 50 GB disk): **`$0.40 / hr`**
    - `Nvidia T4 - medium` (8 vCPU, 30 GB RAM, **16 GB VRAM**, 100 GB disk): **`$0.60 / hr`**
    - `1x Nvidia L4` (8 vCPU, 30 GB RAM, **24 GB VRAM**, 400 GB disk): **`$0.80 / hr`**
    - `Nvidia A10G - small` (4 vCPU, 15 GB RAM, **24 GB VRAM**, 110 GB disk): **`$1.00 / hr`**
    - `Nvidia A100 - large` (12 vCPU, 142 GB RAM, **80 GB VRAM**, 1000 GB disk): **`$2.50 / hr`**
  - ZeroGPU overage beyond daily PRO quota costs **`$1.00 per 10 minutes` (`$6.00/hr`)** — **WARNING: Disable pay-as-you-go ZeroGPU overage or monitor billing so public demo traffic cannot drain the `$4` credit balance in 40 minutes!**

### 6.3 Optimal Allocation of the Developer's `~$4.00` Credit Balance
With `~$4.00` total compute credits:
- `$4.00` buys **only 5.0 hours on L4 (`$0.80/hr`)** or **1.6 hours on A100 (`$2.50/hr`)**, whereas **Kaggle provides ~30 hours/week of `2× T4 (32 GB)` or `P100 (16 GB)` for `$0.00`!**
- **Strict Fiscal Recommendation:**
  1. **Spend `$0.00` of HF credits during Phases 0–5:** Run all data shard preparation (CPU Kaggle notebooks: 30 GB RAM, 4 cores, free!), tokenizer training, baseline evaluations, Track B QLoRA training, and Track A small-scale pretraining on **Kaggle Free Notebooks**, pushing checkpoints/manifests to **HF Hub repositories** via `HF_TOKEN`.
  2. **Reserve the `~$4.00` HF Credit Balance strictly for optional, explicitly approved high-leverage bursts** (if ever needed after developer verification of `https://huggingface.co/settings/billing`):
     - e.g., a **30-minute `1x Nvidia L4` (`$0.40` total)** native BF16 + FlashAttention-2 verification/merge job (`hf jobs run --flavor l4x1 --timeout 30m`), or `CPU Upgrade` (`$0.03/hr` = 10 hours for `$0.30`) for streaming Parquet dataset filtering if local/Kaggle sessions are busy.
  3. **Never trigger any HF Job, Inference Endpoint, or paid Space hardware without explicit per-job written spending approval from M. Ahmad.**

---

## 7. Primary Sources & Citations Index (All Verified 2026-11-17)

1. **Qwen3-1.7B Official Model Card & Config:** `https://huggingface.co/Qwen/Qwen3-1.7B` (`sha: 70d244cc86cc`, Apache-2.0, accessed 2026-11-17).
2. **Qwen2.5-Coder-1.5B-Instruct Official Model Card & Technical Report:** `https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct` (`sha: 2e1fd397ee46`, Apache-2.0); Hui et al. (2024), *Qwen2.5-Coder Technical Report*, `https://arxiv.org/abs/2409.12186` (accessed 2026-11-17).
3. **IBM Granite-4.2-3B Official Model Card:** `https://huggingface.co/ibm-granite/granite-4.2-3b` (`sha: 4e9c275e232f`, Apache-2.0, released 2026-08-25, accessed 2026-11-17).
4. **Qwen3.5-2B Official Model Card:** `https://huggingface.co/Qwen/Qwen3.5-2B` (`sha: 15852e8c1636`, Apache-2.0, accessed 2026-11-17).
5. **SmolLM3 Technical Blog & Model Card:** Bakouch et al. (2025), `https://huggingface.co/blog/smollm3` & `https://huggingface.co/HuggingFaceTB/SmolLM3-3B` (`sha: a07cc9a04f16`, Apache-2.0, accessed 2026-11-17).
6. **FineWeb-2 Multilingual Dataset & Subsets (`urd_Arab`, `urd_Latn`, `hin_Deva`, `arb_Arab`, `spa_Latn`, `fra_Latn`):** `https://huggingface.co/datasets/HuggingFaceFW/fineweb-2` (`sha: af9c13333eb9`, ODC-By-1.0, tree verified 2026-11-17).
7. **AI4Bharat Sangraha & Aksharantar:** `https://huggingface.co/datasets/ai4bharat/sangraha` (`sha: 8b813c3f62d3`, CC-BY-4.0) & `https://huggingface.co/datasets/ai4bharat/Aksharantar` (`sha: e418c1fc928d`, accessed 2026-11-17).
8. **Cohere Labs Aya Dataset:** `https://huggingface.co/datasets/CohereLabs/aya_dataset` (`sha: f9ea04583f02`, Apache-2.0, accessed 2026-11-17).
9. **OpenCoder SFT Stage 2 & NVIDIA OpenCodeInstruct:** `https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2` (`sha: 7d28f40d579e`, MIT) & `https://huggingface.co/datasets/nvidia/OpenCodeInstruct` (`sha: 8f3ba5bafe4d`, CC-BY-4.0, accessed 2026-11-17).
10. **The Stack v2 Dedup:** `https://huggingface.co/datasets/bigcode/the-stack-v2-dedup` (`sha: 611ef38fbfab`, Software Heritage per-file OSS licenses, accessed 2026-11-17).
11. **OWASP Cheat Sheet Series & MITRE ATT&CK / CWE:** `https://github.com/OWASP/CheatSheetSeries` (CC-BY-SA-4.0), `https://github.com/mitre/cti/blob/master/LICENSE.txt`, `https://cwe.mitre.org/about/termsofuse.html`, `https://csrc.nist.gov/publications` (US Public Domain, accessed 2026-11-17).
12. **Benchmarks — EvalPlus, LiveCodeBench, MultiPL-E, PurpleLlama CyberSecEval 2, Belebele:**
    - `https://github.com/evalplus/evalplus` (Apache-2.0)
    - `https://github.com/LiveCodeBench/LiveCodeBench` (MIT)
    - `https://github.com/nuprl/MultiPL-E/blob/main/LICENSE` (BSD-3-Clause with ML Training Restriction)
    - `https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks` (MIT; Bhatt et al., 2024, `https://arxiv.org/abs/2404.13161`)
    - `https://huggingface.co/datasets/facebook/belebele` (`sha: 7899cdfa4e1e`, CC-BY-SA-4.0)
13. **Platform Docs — Kaggle Notebooks & Hugging Face Jobs/PRO/ZeroGPU:**
    - `https://www.kaggle.com/docs/notebooks` (accessed 2026-11-17)
    - `https://huggingface.co/docs/hub/jobs-pricing` (accessed 2026-11-17)
    - `https://huggingface.co/docs/hub/pro` & `https://huggingface.co/docs/hub/billing` & `https://huggingface.co/docs/hub/spaces-zerogpu` (accessed 2026-11-17).
