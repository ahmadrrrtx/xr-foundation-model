# XR Foundation Model (XRFM) — Phased Implementation Roadmap (Phases 0–8)

**Document Date:** 2026-11-17  
**Baseline Audited Commit:** `90674742d8e6a5caf65336e4b64c949696e33bc0` (`G0=PASS, G1=FAIL`)  
**Guiding Principle:** Every phase ends with an explicit, falsifiable **Go / No-Go Gate**. Zero GPU quota or Hugging Face credits may be spent on training until foundation correctness, contamination gates, and base-model baselines are verified.

---

## Master Phase & Compute Allocation Overview

| Phase | Name | Primary Compute | Est. GPU Hours | Est. HF Credits Spent | Blocking Gate |
|---|---|---|---:|---:|---|
| **Phase 0** | Repo Truth Alignment, Cleanup & Defect Fixes | Local CPU / GitHub CI | `0.0 h` | `$0.00` | `GATE-0`: Zero doc inflation; all P0/P1 core bugs fixed; 560+ unit tests green |
| **Phase 1** | Shared Multilingual/Code/Sec Data Governance & Eval Suite | Local CPU / Free HF Hub | `0.0 h` | `$0.00` | `GATE-1`: License registry enforced; 7-lang LID >95% F1; held-out eval suite frozen & hashed |
| **Phase 2** | Track A Tokenizer Tournament & Streaming Shard Pipeline | Local CPU / Kaggle CPU | `0.0 h` | `$0.00` | `GATE-2`: 32K tokenizer `<2.2x` fertility ratio vs English across `ur/ar/hi/es/fr/ur-Latn`; zero byte-fallback pathology |
| **Phase 3** | Track A (`XRFM Native`) Architecture Upgrade & Pretraining | Kaggle `2× T4` (`$0`) | `~22.0 h` (1 wk quota) | `$0.00` | `GATE-3`: Monotonic `BPB` decrease across all 7 langs + code; resume bit-exactness verified |
| **Phase 4** | Track B (`XRFM CoderSec`) Base Model Bake-Off & SFT Dataset | Kaggle `1× T4` (`$0`) | `~2.5 h` | `$0.00` | `GATE-4`: Unadapted base scores logged; 40K SFT dataset passes AST, secret, safety & 0%-contam audit |
| **Phase 5** | Track B (`XRFM CoderSec`) QLoRA Fine-Tuning & Ablations | Kaggle `1× T4` (`$0`) | `~8.0 h` | `$0.00` | `GATE-5`: Beats unadapted base on Coding + Defensive Sec + Multilingual without catastrophic forgetting |
| **Phase 6** | Comprehensive Cross-Track Evaluation, Red-Teaming & Stress Tests | Kaggle `1× T4` (`$0`) | `~4.0 h` | `$0.00` *(opt `$0.40` L4)* | `GATE-6`: Full EvalPlus, CyberSecEval 2, Belebele, Over-Refusal & Roman-Urdu robustness matrix complete |
| **Phase 7** | Bounded Security RAG, Agent Runtime & HF ZeroGPU Demo | HF Hub + ZeroGPU (`$0`) | `0.0 h` | `$0.00` | `GATE-7`: Citation accuracy >90%; sandbox escape = 0%; UI clearly labels Track A vs Track B |
| **Phase 8** | Reproducible Release Packaging, Model Cards & Technical Report | Local CPU + HF Hub | `0.0 h` | `$0.00` | `GATE-8`: Clean clone reproduces evals; pinned SHAs, licenses & honest claims audited |
| **TOTAL** | **Full Dual-Track Program** | **Kaggle Free + HF PRO** | **~36.5 GPU-h (~1.3 wks)** | **`$0.00`–`$0.40`** | ** Reserve `~$3.60–$4.00` HF credits intact!** |

---

## Detailed Phase-by-Phase Execution Plan

### Phase 0: Repository Stabilization, Truth Alignment & Core Defect Remediation
- **Objectives:** Eliminate all documentation inflation (`G1` blocker in `docs/program/03_EXECUTION_LEDGER.md`), unify duplicate frontends, quarantine legacy checkpoints, and fix all confirmed P0/P1 core bugs identified in `XRFM_FULL_REPOSITORY_AUDIT.md`.
- **Exact Engineering Tasks:**
  1. **Truth Alignment:** Rewrite `README.md`, `MODEL_CARD.md`, and `docs/specs/v1.1.0_spec.md` to accurately distinguish current repository state, **Track A (`XRFM Native`)**, and **Track B (`XRFM CoderSec`)**. Fix `README.md` quickstart (`tiny.yaml` `vocab_size: 2048` matching `vocab.json`). Move `checkpoints/checkpoint_step_500.pt` out of default runtime path into quarantined legacy status (`git rm` or `artifacts/legacy_unverified/`). Consolidate `vercel_app/` and `webui/` so static assets have one source of truth.
  2. **Model Precision & FlashAttention Fixes (`src/xrfm/models/`):**
     - `attention/rope.py`: Cache `cos/sin` buffers, compute angles in `float32`, cast output back to `x.dtype` (`fp16`/`bf16` safe!).
     - `layers/rmsnorm.py`: Upcast variance reduction to `float32` (`fp16` P100/T4 overflow fix).
     - `attention/multi_head.py`: Pass `mask=None` when `mask is None and cache_len == 0` so `flash_attention_forward` executes native `is_causal=True` FlashAttention/MemEfficient SDPA ($O(L)$ memory). Add optional `n_kv_heads` (GQA) and honor `use_swiglu=False`. Remove unconditional `.any()` CUDA sync per step.
  3. **Training & Evaluation Fixes (`src/xrfm/training/`, `src/xrfm/evaluation/`):**
     - `loop.py`: Move `self.model.to(self.device)` *before* `setup_ddp`/`setup_fsdp`.
     - `optimizer.py`: Exclude 1D tensors (`RMSNorm`, biases) from `weight_decay`.
     - `scheduler.py`: Clamp `progress <= 1.0` with `min_lr_ratio`.
     - `perplexity.py` & `benchmarks.py`: Move inputs/targets `.to(device)` and mask `ignore_index = -100` in `TextCompletionAccuracy` and `TopKAccuracy`; add `compute_bits_per_byte()`.
  4. **CI Fixes (`.github/workflows/`):** Add missing `fastapi`, `httpx`, `uvicorn` test deps in `tests.yml` and remove `|| true` silent lint suppression.
- **Expected Deliverables:** Clean PR/commit series on `src/xrfm/*`, `README.md`, `.github/workflows/*`, plus regression tests covering FP16/BF16 forward-backward, `-100` accuracy masking, GQA, and GPU device placement.
- **Go / No-Go Gate (`GATE-0`):**
  - `pytest` passes 100% including new FP16/BF16, GQA, and `-100` tests (`0` failures).
  - `python scripts/train.py --config config/tiny.yaml` runs cleanly from scratch without pre-existing checkpoint dependencies.
  - Zero unsupported benchmark numbers or ambiguous authorship claims remain in `README.md` or `docs/`.

---

### Phase 1: Shared Governance, Multilingual LID & Frozen Evaluation Suite
- **Objectives:** Upgrade the shared data pipeline (`src/xrfm/data/`) to stream bounded chunks, accurately classify all 7 language varieties (`en, ur, ur-Latn, ar, hi, es, fr`) + code, enforce SPDX license tiers, and freeze the held-out evaluation benchmark hashes *before* any training data is built.
- **Exact Engineering Tasks:**
  1. **Source & License Registry (`configs/data/sources.yaml`, `src/xrfm/data/sources.py`):** Encode all approved sources from `XRFM_RESEARCH_REVIEW.md` with pinned HF dataset commit SHAs, SPDX licenses, and `allowed_for_training: bool` (`MultiPL-E` and `Belebele` locked to `eval_only`).
  2. **7-Language + Roman-Urdu LID (`src/xrfm/data/language.py`):** Fix the `ur = 0.9 * ar` bug using Urdu-specific Perso-Arabic letters (`ٹ ڈ ڑ ں ہ ے گ چ پ ژ`), add Spanish/French lexical & diacritic detectors, implement the `RomanUrduCodeSwitchDetector` (`ur-Latn`), and wire optional `GlotLID`/`fastText` verification.
  3. **Streaming Pipeline & CLI (`src/xrfm/data/pipeline.py`, `dedup.py`, `cli.py`):** Fix `cmd_build` to load full nested YAML; add LSH banded MinHash (`O(N)`); fix `ShardedTokenDatasetMap` to emit `L + 1` tokens per row; fix `TokenizerInfo` to use `tokenizer_fingerprint()`.
  4. **Freeze `XRFM-MultiCoderSec-Bench` & Contamination Index:** Assemble and SHA-256-register held-out evaluation sets:
     - General Multilingual: `Belebele` (`eng_Latn, urd_Arab, urd_Latn, arb_Arab, hin_Deva, spa_Latn, fra_Latn`) + held-out `FineWeb-2` test shards (`BPB`).
     - Coding: `HumanEval+`, `MBPP+`, `MultiPL-E` (`js, cpp, rs, sh`), `LiveCodeBench` subset.
     - Security & Calibration: `CyberSecEval 2` subset + **350-item `XRFM-MultiCoderSec-Bench`** (50 items × 7 language varieties covering secure code review, defensive hardening, Roman-Urdu debugging, benign sysadmin anti-overrefusal, and malicious exploit refusal).
- **Expected Deliverables:** Frozen eval manifest `configs/eval/frozen_benchmarks_v1.json` + contamination bloom/13-gram index + unit tests on 7-language LID.
- **Go / No-Go Gate (`GATE-1`):**
  - LID achieves $\ge 95\%$ macro-F1 on held-out 7-language + code validation samples (specifically `ur` vs `ar` $\ge 98\%$ and `ur-Latn` vs `en`/`hi-Latn` $\ge 90\%$).
  - Contamination filter demonstrably catches 100% of injected synthetic benchmark leaks in unit tests.

---

### Phase 2: Track A (`XRFM Native`) Tokenizer Tournament & Pretraining Shards
- **Objectives:** Eliminate the 100% raw-byte fallback pathology (`10.5–16.1 tok/word` on `ur/ar/hi`) by running the controlled 4-way tokenizer tournament (`XRFM_RESEARCH_REVIEW.md` §3) and building tokenized `.npy` shards for Track A.
- **Exact Engineering Tasks:**
  1. Refactor `BytePairEncoder` (`src/xrfm/tokenization/bpe.py`) with word-frequency pre-aggregation, Indic/Arabic grapheme-preserving regex (`\p{L}\p{M}*`), and special-token literal parsing; add `HFByteBPETokenizer` bridge exporting to `vocab.json`.
  2. Stream balanced `250 MB` tokenizer training sample across `en (30%), code (25%), ur (10%), ur-Latn (5%), ar (10%), hi (10%), es (5%), fr (5%)`.
  3. Train Candidates `T1` (`32,768` Byte-BPE), `T2` (`32,768` Unigram), `T3` (`49,152` Byte-BPE), and `T4` (script-protected `32,768` Byte-BPE) in `<5 minutes` on CPU.
  4. Evaluate fertility (`tok/word`), bytes/token, parity ratio vs English, Python indentation preservation, and roundtrip losslessness across all 7 languages + 6 programming languages.
  5. Stream, deduplicate, decontaminate, and pack **500M tokens** (Tier-1) and **1.7B tokens** (Tier-2, ~3.4 GB `uint16` shards) onto a versioned HF Dataset repo (`RRRTX/xrfm-native-shards-v1`) with `DatasetManifestV2`.
- **Expected Deliverables:** Winning `vocab_32k_multilingual.json`, `docs/experiments/tokenizer_tournament_v1.json`, and versioned pretraining shards on HF Hub.
- **Go / No-Go Gate (`GATE-2`):**
  - Exact lossless UTF-8 roundtrip (`100%`) across all scripts + emojis + code.
  - Raw-byte fallback `< 0.5%` on all 7 languages; fertility ratio relative to English $\le 2.0\times$ on `ur, ar, hi` and $\le 1.4\times$ on `ur-Latn, es, fr`.

---

### Phase 3: Track A (`XRFM Native`) Pretraining & Scaling Verification
- **Objectives:** Prove end-to-end training stability, Kaggle session checkpoint/resume fidelity, and genuine multilingual + code representation learning in the original `XRFMModel` architecture.
- **Exact Engineering Tasks:**
  1. **Run A1 (`xrfm-native-25m`, 500M tokens, ~4h on Kaggle `2× T4` DDP FP16):** Validate cosine LR schedule, gradient norms, throughput (`tok/sec`), and per-language validation `BPB` every 1,000 steps. Verify mid-run checkpoint save at step 2,000 and resume in a second process produces identical loss trajectory.
  2. **Run A2 (`xrfm-native-85m`, 1.7B tokens, 2× ~10h Kaggle `2× T4` sessions):** Train flagship native checkpoint (`d_model=640, L=12, GQA 10/2, seq=2048`) across 2 checkpointed sessions, exporting format-v2 `.pt` + `model.safetensors` to HF Hub.
- **Expected Deliverables:** `RRRTX/xrfm-native-25m` & `RRRTX/xrfm-native-85m` checkpoints, training telemetry JSONL, and per-language `BPB` learning curves.
- **Go / No-Go Gate (`GATE-3`):**
  - Zero NaN/Inf loss spikes; monotonic validation `BPB` reduction on **every** target language (`en, ur, ur-Latn, ar, hi, es, fr`) and Python/JS code held-out splits.
  - Coherent syntactic text & basic function completion demonstrated within honest small-model scope.

---

### Phase 4: Track B (`XRFM CoderSec`) Base Model Bake-Off & SFT Corpus Construction
- **Objectives:** Establish zero-shot/few-shot baselines on unadapted open base models *before* spending fine-tuning GPU hours, and curate the verified 40,000-example `XRFM-CoderSec-SFT-40K` dataset.
- **Exact Engineering Tasks:**
  1. **Base Model Head-to-Head (`~2.5h` on Kaggle `1× T4`):** Run the frozen Phase 1 evaluation suite (`HumanEval+`, `MBPP+`, `CyberSecEval 2` subset, `Belebele` 7-lang subset, and `XRFM-MultiCoderSec-Bench`) on:
     - **Candidate B1:** `Qwen/Qwen3-1.7B` (`sha=70d244cc86cc`, Apache-2.0, NF4 + FP16)
     - **Candidate B2:** `Qwen/Qwen2.5-Coder-1.5B-Instruct` (`sha=2e1fd397ee46`, Apache-2.0, NF4 + FP16)
  2. Lock the primary Track B base model (`Qwen3-1.7B` expected winner whenever `ur/ar/hi/ur-Latn` gap outweighs raw English code delta; otherwise document empirical decision).
  3. **Build `XRFM-CoderSec-SFT-40K` (`~28M` tokens):** Curate the 4-pillar mixture defined in `XRFM_TARGET_ARCHITECTURE.md` §3.3 (`35%` execution/AST-verified coding from `opc-sft-stage2` & `OpenCodeInstruct`, `30%` NIST/CWE/DPO defensive security & patching, `25%` `aya_dataset` + Roman-Urdu technical dialogues, `10%` RAG citation & anti-overrefusal calibration).
  4. **Strict Quality & Safety Gate on SFT Data:** Parse code blocks (`ast.parse` / `tree-sitter` / shellcheck), scan for leaked API keys (`pii.py` + secret regexes), enforce 0% benchmark 13-gram overlap (`contamination.py`), and filter out offensive weaponization instructions.
- **Expected Deliverables:** Base model comparison table (`docs/experiments/track_b_base_bakeoff.json`) + private/versioned HF dataset `RRRTX/xrfm-codersec-sft-40k` with `DatasetManifestV2`.
- **Go / No-Go Gate (`GATE-4`):**
  - 100% of SFT examples validate against ChatML schema with non-empty assistant target tokens; 0% contamination hits; 100% Tier-A/Tier-B permissive provenance.

---

### Phase 5: Track B (`XRFM CoderSec`) QLoRA Fine-Tuning & Controlled Ablations
- **Objectives:** Train `XRFM CoderSec` inside `src/xrfm/codersec/` on Kaggle T4 hardware using rigorously controlled QLoRA ablations.
- **Exact Engineering Tasks:**
  1. Implement `src/xrfm/codersec/{config.py, dataset.py, train_qlora.py, merge.py}` using `transformers` + `peft` + `trl` (`SFTTrainer`, assistant-only completion loss masking `DataCollatorForCompletionOnlyLM` / `completion_only_loss=True`, `packing=True`, `max_seq_length=2048`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.float16`, `torch_dtype=torch.float16` — avoiding T4 slow software BF16!).
  2. **Smoke Run B0 (`500 steps`, `~15 min` on T4):** Verify peak VRAM (`< 8 GB`), throughput, assistant-only loss mask, checkpoint save to `/kaggle/working`, and Hub push.
  3. **Controlled Ablation Matrix (`1 epoch` fast runs, `~1.0h` each on T4):**
     - `Ablation-B1`: Code + Defensive Security English-only (`r=16, alpha=32`).
     - `Ablation-B2`: Code + Defensive Security + 7-Language / Roman-Urdu Mix (`r=16, alpha=32`).
     - `Ablation-B3`: Rank sensitivity check (`r=32, alpha=64` vs `r=16, alpha=32`).
  4. **Full Flagship Run B-Final (`3 epochs` on winning mix, `~3.0–3.5h` single Kaggle T4 session):** Cosine LR `2e-4`, warmup `3%`, `neftune_noise_alpha=5.0`, saving checkpoints every 250 steps with evaluation on held-out dev split.
- **Expected Deliverables:** Published LoRA adapter `RRRTX/xrfm-codersec-1.7b-lora` (~80 MB) + merged FP16 `safetensors` checkpoint `RRRTX/xrfm-codersec-1.7b` + training logs.
- **Go / No-Go Gate (`GATE-5`):**
  - Statistically meaningful improvement over unadapted base model on **Defensive Security Remediation**, **Multilingual / Roman-Urdu Technical QA**, and **Citation Grounding**, with **no regression $> 1.5\%$** on `HumanEval+` / `MBPP+` English coding!

---

### Phase 6: Comprehensive Cross-Track Evaluation, Red-Teaming & Failure Analysis
- **Objectives:** Subject both Track A and Track B checkpoints (plus unadapted base models) to full benchmark evaluation, adversarial prompt injection, dual-use red-teaming, and Roman-Urdu spelling perturbation tests.
- **Exact Engineering Tasks:**
  1. Execute full `EvalPlus` (`HumanEval+`, `MBPP+`), `MultiPL-E` (`js, cpp, rs, sh`), `CyberSecEval 2` (insecure coding rate + prompt injection resilience + false-refusal rate), `Belebele` (all 7 varieties including `urd_Arab` & `urd_Latn`), and `XRFM-MultiCoderSec-Bench`.
  2. Run **Roman-Urdu Orthographic Perturbation Stress Test** (`kya/kia`, `mujhe/mujy`, `hain/hen`, mixed Urdu-English stack traces) measuring answer stability.
  3. *(Optional Approved Credit Spend — Max `$0.40`):* If a clean single-run 24 GB VRAM validation of merged FP16 weights is desired, run a 30-minute Hugging Face Job on `L4` (`$0.80/hr` $\times 0.5\text{h} = \$0.40`, `--timeout 35m`).
  4. Document concrete failure cases (where the 1.7B model still hallucinates obscure CWEs without RAG, or where 85M native model hits capacity limits).
- **Expected Deliverables:** Complete JSON + Markdown evaluation matrices in `docs/experiments/FINAL_BENCHMARK_MATRIX.md`.
- **Go / No-Go Gate (`GATE-6`):**
  - Malicious weaponization compliance rate `< 2%` across all 7 languages (including Roman-Urdu code-switched adversarial prompts); benign sysadmin/defensive false-refusal rate `< 10%`; insecure code generation rate lower than unadapted base model.

---

### Phase 7: Citation-Grounded Security RAG, Bounded Agent & ZeroGPU Space
- **Objectives:** Connect `XRFM CoderSec` to the bounded local RAG & agent runtime (`src/xrfm/search/`, `src/xrfm/agent/`) and deploy a zero-cost interactive showcase on Hugging Face ZeroGPU (included in HF PRO).
- **Exact Engineering Tasks:**
  1. Build the attributed **Defensive Security Knowledge Pack** (`OWASP Cheat Sheets`, `OWASP ASVS 5.0`, `MITRE CWE`, `MITRE ATT&CK Mitigations`, `NIST SP 800-218 / 800-61r3`) with explicit SPDX license headers (`CC-BY-SA-4.0` attribution preserved for OWASP).
  2. Add `SecurityAdvisorySearchTool` and `StaticCodeAuditTool` (bounded AST/pattern scanner, zero shell execution) to `src/xrfm/agent/tools.py`.
  3. Deploy a clean Gradio / FastAPI Space on Hugging Face **ZeroGPU** (`$0/hr` under HF PRO — **verify paid ZeroGPU overage billing is toggled OFF** in HF Billing settings!) featuring explicit model selector tabs (`Track B: XRFM CoderSec 1.7B [Qwen3-1.7B QLoRA + Security RAG]` vs `Track A: XRFM Native 85M [From-Scratch Research Preview]`).
- **Expected Deliverables:** Live reproducible demo + local CLI `xrfm agent --track codersec` + citation verification tests.
- **Go / No-Go Gate (`GATE-7`):**
  - Grounded RAG mode reduces CWE/OWASP control hallucination by $\ge 70\%$ compared to weights-only generation; agent process sandbox blocks 100% of path-traversal/network/timeout escape tests.

---

### Phase 8: Reproducible Release Packaging, Model Cards & Publication
- **Objectives:** Package code, configs, manifests, model cards, dataset cards, and a publication-grade technical report so any external researcher can audit and reproduce every number from a fresh clone.
- **Exact Engineering Tasks:**
  1. Auto-generate HF Model Cards (`MODEL_CARD_NATIVE.md`, `MODEL_CARD_CODERSEC.md`) and Dataset Cards directly from `src/xrfm/experiment/cards.py` with pinned base model SHAs, dataset SHAs, license attributions, training hardware/hours, full per-language eval tables, and dual-use safety boundaries.
  2. Tag release `v2.0.0` only after clean-environment reproduction script `scripts/verify_reproducibility.py` passes end-to-end.
  3. Update `docs/program/03_EXECUTION_LEDGER.md` from `NO-GO` to `GO` with signed artifact hashes.
- **Expected Deliverables:** Tagged GitHub release, HF Hub collection (`XRFM Native` + `XRFM CoderSec` + `Datasets` + `Space`), and final technical report.
- **Go / No-Go Gate (`GATE-8`):**
  - Fresh `git clone` + `pytest` + `verify_reproducibility.py` passes 100%; every public claim traces 1-to-1 to a committed JSON artifact in `docs/experiments/`.
