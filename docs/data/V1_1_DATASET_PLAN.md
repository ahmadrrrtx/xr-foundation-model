# XRFM v1.1 — First Real Dataset Plan (Phase 34)

**Date:** 2026-08-08 · **Goal:** a real, licensed, provenance-able pretraining
corpus for the first serious 15–20 M parameter experiment.
**Rule:** no automated billion-token downloads; size/quality/license are
established first, then a bounded first slice is recommended.

---

## 1. Candidate Sources (researched, primary sources)

| Source | Size | License | Language | Quality/filtering | Notes |
|---|---|---|---|---|---|
| **FineWeb-Edu** (HF `HuggingFaceFW/fineweb-edu`) | 1.3 T tokens (threshold-3); also score-2 = 5.4 T | **ODC-By 1.0** + CommonCrawl ToU | en | Llama-3-70B-Instruct educational-quality classifier (F1 82%); beats FineWeb on MMLU/ARC/OpenBookQA | Official sample subsets: **sample-10BT**, sample-100BT, sample-350BT (GPT-2-tokenizer counts). arXiv:2406.17557. Dedup: per-dump in FineWeb; deduped variant in SmolLM-Corpus (dedup shown to have no perf impact at 1.8B/350B scale) |
| **The Pile** (EleutherAI) | 825 GiB, 22 components | mixed per-component (permissive majority; **Books3 has copyright concerns — avoid**) | en | manually curated components (arXiv, PubMed, StackExchange, books, code…) | 825 GiB, ~300 B GPT-2 tokens. Use only permissive components (e.g., PubMed, StackExchange, Wikipedia) |
| **Wikipedia (en) dump** | ~20 GiB compressed (modern dump); ~6.4 GiB in Pile | **CC BY-SA 4.0 / CC BY-SA 3.0** | en | high quality, encyclopedic | Attribution requirements apply; use wikitext preprocessing |
| Current XRFM corpus (`corpus.txt`) | 1.77 M tokens | public-domain + PSF | en | — | **Keep only as a smoke/CI corpus**; NOT the v1.1 training corpus |

## 2. Recommendation for the First Serious Run

**Primary: FineWeb-Edu `sample-10BT`** (10 B tokens; ODC-By 1.0; English; classifier-filtered).

Rationale:
1. **License-clean** (ODC-By 1.0) — compatible with XRFM's MIT distribution.
2. **Quality-controlled** — the educational-quality filter is exactly the
   "data quality > quantity" lesson from Llama 3 / OLMo that the audit recorded.
3. **Bounded download** — `sample-10BT` is the smallest official sample;
   a 100–500 M-token slice is a few hundred MB of Parquet.
4. **Provenance-ready** — HuggingFace provides versioned configs
   (CC-MAIN-2024-38/42/46, etc.) that map directly into the Phase-33 manifest
   (`source`, `version`, `sha256`).

Fallback if FineWeb is unreachable: **Pile-permissive subset** (PubMed +
StackExchange + Wikipedia components) or a **Wikipedia (en) slice**.

## 3. Recommended First Slice (for the 15–20 M experiment)

| Budget | Tokens | Est. disk (Parquet, ~4 B/token text) | Purpose |
|---|---|---|---|
| Experiment A | **50 M** | ~200 MB | first serious pretraining; full protocol run |
| Experiment B | 100 M | ~400 MB | compare A/B learning curves |
| Experiment C | 500 M | ~2 GB | upper bound for a single T4-class GPU run |
| Experiment D | 1 B | ~4 GB | stretch; only if throughput measured ≥ 30k tok/s |

**Deterministic slicing protocol (reproducibility):**
- Use `datasets` streaming or download the sample-10BT shards; take a
  **seeded deterministic subsample** (`seed=42`) of exactly N tokens.
- Split at **document boundaries** (each FineWeb-Edu record is one document)
  into train (95%) / val (2.5%) / test (2.5%) — never cut documents.
- Hold the **val/test slices out permanently** (never trained on; the Phase-39
  protocol evaluates only on these).
- Record in the Phase-33 manifest: dataset name `fineweb-edu-sample-10BT`,
  version (CC-MAIN snapshot used), source URL, license ODC-By 1.0, preprocessing
  version (`xrfm-fineweb-v1`), sha256 of the local slice, dedup
  (`none-needed` — fineweb is already filtered; or `exact-doc-dedup` if applied).

## 4. Storage & Tooling Requirements

- Disk needed: 200 MB (A) to 4 GB (D) for the slice itself, plus the tokenized
  cache if we add one (~1 B/token in uint16 → 2 GB for 1 B tokens). A 25 GB
  sandbox fits A–B; C–D require ~30–50 GB.
- Tooling: `datasets` (HF) for download+streaming; existing `xrfm.data.loader`
  is file-based — a small `load_fineweb_slice.py` adapter converts Parquet →
  a plain-text (or Memmap uint16) local corpus consumed by the existing loader.
- Tokenizer: reuse the v1.1 byte-level BPE. **A 2048-token vocab is too small
  for web-scale text** — the v1.1 plan should retrain the tokenizer on the
  actual slice with a target of **8,192 tokens** (matches `config/v1.1-medium.yaml`
  vocab) — see Phase 35.

## 5. Explicit Non-Goals

- No raw FineWeb (15 T tokens, 44 TB) — far beyond this project's storage.
- No The-Pile Books3 (copyright risk).
- No multilingual mixture for v1.1 (English-only slice; multilingual is a
  later-version goal).
- No synthetic data (no evidence at this scale it helps; adds provenance cost).

## 6. Acceptance for the First Run

The dataset is acceptable for the first serious run when:
1. manifest (Phase 33) is generated and stored with the run;
2. val/test slices are disjoint from train by construction (document-boundary split);
3. tokenizer (vocab 8192) is trained on the TRAIN slice only;
4. slice sha256 is recorded in the run manifest.
