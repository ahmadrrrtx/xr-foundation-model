# XRFM Data Pipeline — Phase 1 Architecture

**Date:** 2026-09-18 · **Version:** xrfm-pipeline-v1 · **Status:** Implemented

This document describes the complete data foundation built in Phase 1, designed to support future training of 125M, 350M, 1B and larger models without another major rewrite.

## 1. Overview — Data Flow

```
raw sources
    ↓
source registry (configs/data/sources.yaml)
    ↓
download / ingestion
    ↓
document normalization
    ↓
metadata + provenance
    ↓
language filtering
    ↓
quality filtering
    ↓
PII / safety filtering
    ↓
exact deduplication (SHA-256)
    ↓
near-duplicate detection (MinHash / n-gram)
    ↓
document-level split / contamination controls
    ↓
dataset mixing (explicit weights)
    ↓
tokenization (tokenizer identity tracked)
    ↓
sequence packing (fixed-length, EOS handling)
    ↓
sharded training dataset (numpy .npy, deterministic)
    ↓
dataset manifest (machine-readable)
    ↓
reproducible training input
```

Every stage is deterministic, traceable, and resumable via manifests/checksums.

## 2. Source Registry

**Location:** `configs/data/sources.yaml`

Machine-readable registry, authoritative description of where XRFM training data comes from.

Each source:

```yaml
name: wikipedia_en
uri: https://dumps.wikimedia.org/enwiki/latest/
type: wiki_dump
license: CC-BY-SA-4.0
license_url: https://creativecommons.org/licenses/by-sa/4.0/
language: en
description: English Wikipedia dump
expected_size: 20GB raw, ~5B tokens
enabled: false
version: 2026-09-01
checksum: ...
snapshot_date: 2026-09-01
citation: Wikipedia contributors
redistribution: allowed
```

**Rule:** No hard-coded URLs in Python files. All sources via registry.

**Official vs Experimental:**
- Official: `golden`, `tiny_shakespeare`, `python_stdlib_slice` — license-clean, tiny, redistributable
- Candidate large sources: `wikipedia_en`, `fineweb_edu_sample`, `dolma_cc`, etc. — documented but `enabled: false`, not auto-ingested

## 3. Document Schema

**File:** `src/xrfm/data/schema.py`

Canonical internal representation:

```python
Document(
    document_id="web:abc123...",
    source="wikipedia",
    source_uri="https://en.wikipedia.org/wiki/...",
    license="CC-BY-SA-4.0",
    license_url="https://...",
    collection="encyclopedic",
    language="en",
    text="...",
    content_hash="sha256...",
    metadata={...},
)
```

Derived attributes stored separately (Dolma-inspired):

```python
DocumentAttributes(
    document_id=...,
    language_score=LanguageScore(...),
    quality_score=QualityScore(...),
    pii_score=PIIScore(...),
    dedup_info=DeduplicationInfo(...),
)
```

Final decision:

```python
ProcessingDecision(
    document_id=...,
    keep=True/False,
    reason="too_short / pii_high_risk / exact_duplicate",
    stage="quality / pii / dedup",
)
```

**Fields tracked:**
- `document_id` — stable, deterministic from source+hash
- `source`, `source_uri`, `license`, `license_url`, `collection`
- `language`, `text`, `content_hash` (SHA-256)
- `metadata` — arbitrary provenance

## 4. Normalization

**File:** `src/xrfm/data/normalization.py`

Conservative, explicit, testable:

- Unicode NFC normalization
- Newline normalization (`\r\n`, `\r` → `\n`)
- Control character removal (except `\n`, `\t`)
- Whitespace normalization (preserve code indentation by default)
- Blank line collapsing (3+ → 2)
- Empty/malformed detection

**Preserved:**
- Code indentation
- Markdown structure
- Punctuation
- Language-specific Unicode

Every transformation recorded in `metadata["normalization_modifications"]`.

## 5. Language Filtering

**File:** `src/xrfm/data/language.py`

Architecture supports:

```python
LanguageScore(
    language="en",
    confidence=0.9,
    scores={"en": 0.9, "ar": 0.1},
    detector="heuristic",
    detector_version="v1",
)
```

- Default: heuristic detector (no heavy deps, deterministic, CI-friendly)
- Optional: fastText wrapper if model path provided (documents dependency/version)
- Configurable threshold, allowed languages
- Does NOT silently delete low-confidence without recording reason

Initial corpus English-focused, but infrastructure supports `en`, `ur`, `ar`, `multilingual` without redesign.

## 6. Quality Filtering

**File:** `src/xrfm/data/quality.py`

Modular, independently testable filters:

- `minimum length` (chars, words)
- `maximum repetition` (n-gram duplicate ratio)
- `symbol/number ratio`
- `boilerplate detection`
- `malformed detection`
- `overall quality score`

Each filter produces explicit decision:

```
Document → LanguageScore → QualityScore → FilterAttributes → FinalDecision
```

Preserves metadata to answer: *Why was this document excluded?*

## 7. PII / Safety Filtering

**File:** `src/xrfm/data/pii.py`

Baseline heuristic, conservative, reports "PII heuristic filtering" (never claims PII-free).

Detects:
- Emails (regex)
- Phone numbers (US + international)
- Secrets, API keys (`sk-...`, `ghp_...`, `AKIA...`, private keys)
- SSN, credit card patterns

Reports:

```
documents scanned
documents flagged
documents removed
documents transformed
```

Configurable: remove high-risk, optionally redact medium-risk.

## 8. Exact Deduplication

**File:** `src/xrfm/data/dedup.py`

Deterministic via SHA-256 content hash:

- Algorithm: SHA-256 (documented)
- Normalization: strip before hashing (configurable)
- Scope: document-level
- Collision assumption: SHA-256 negligible
- Records: `input`, `unique`, `duplicates`, `duplicate_of`

Never silently deduplicates without statistics.

## 9. Near-Duplicate Detection

**File:** `src/xrfm/data/dedup.py`

Interface: `deduplicate_documents(...)` with pluggable algorithms.

Implemented:
- **MinHash** (lightweight deterministic version, no external deps for Phase 1, 128 permutations)
- **N-gram overlap** (Jaccard over shingles)
- SimHash placeholder (interface ready)

Pipeline:

```
exact dedup → near dedup
```

Reports: `input`, `exact duplicates`, `near duplicates`, `final unique`

Production would use datasketch LSH for trillion-token scale; Phase 1 O(n²) acceptable for golden dataset.

Reference: Dolma's Bloom-filter and document/paragraph deduplication.

## 10. Contamination Control

**File:** `src/xrfm/data/contamination.py`

Architectural boundary to prevent evaluation leakage:

- Registry of protected datasets (benchmarks, val, test)
- Exact hash check + n-gram overlap (13-gram default)
- Reports contaminated docs with protected dataset name + score

Goal: make it difficult for future training to accidentally ingest eval data.

## 11. Data Mixing

**File:** `src/xrfm/data/mixing.py`

Explicit, versioned, reproducible:

```yaml
mixture:
  web: 0.45
  books: 0.15
  code: 0.15
  educational: 0.10
  wikipedia: 0.05
  science: 0.05
  multilingual: 0.05
```

For every corpus, records:

```
source, weight, sampled token count, actual token count
```

Deterministic via seed, supports token-weighted or doc-weighted sampling.

Reference: OLMo treats mixture as explicit experiment.

## 12. Tokenization Pipeline

**File:** `src/xrfm/data/tokenization.py`

Integrates Phase 0 tokenizer contract, records:

```
tokenizer identity, version, config, hash, special token IDs
```

Reproducible from:

```
raw source + processing config + tokenizer version
```

`TokenizerInfo` includes vocab size, pad/eos/bos/unk ids, hash.

## 13. Tokenized Data Format

**Chosen format:** NumPy `.npy` shards (binary packed token files)

**Location:** `processed/<dataset>/tokens/shard-00000.npy`

**Decision rationale (ADR-0002):**
- Sequential throughput: high (contiguous binary)
- Random access: supported via memmap
- Storage efficiency: uint16 for vocab < 65535 else uint32 (auto)
- Distributed reading: trivial (each shard independent)
- Simplicity: no heavy deps (Arrow, Parquet would add deps)
- Reproducibility: checksum per shard
- Dependency footprint: minimal (numpy already required)

**Alternatives considered:**
- Arrow/Parquet: good for analytics but overhead for training
- WebDataset/tar: good for streaming but complexity
- memmap raw .bin: efficient but needs separate metadata
- JSONL: human-readable but inefficient

**Chosen:** `.npy` as pragmatic middle ground; each shard is `(num_sequences, seq_len)` array.

Future: can add `.bin` + `.meta` or Arrow without breaking manifest contract.

## 14. Sequence Packing

**File:** `src/xrfm/data/packing_extended.py`

```
documents → tokens → continuous token stream → fixed-length sequences
```

Avoids unnecessary padding, supports `sequence_length`.

Documents:
- **EOS inserted:** configurable, between docs, avoids double EOS
- **Documents may cross boundaries:** `allow_cross_document=True` (efficient) or `False` (preserves doc boundaries, wasteful)
- **Document boundaries preserved:** via `doc_boundaries` list (optional)
- **Attention masking:** not required for cross-doc packing if causal LM, but boundary info preserved for future
- **Leftovers:** `drop` (default), `pad`, or `keep` (pad to full length)

Records `discarded_tokens`, `padded_tokens`.

## 15. Sharding

**File:** `src/xrfm/data/sharding.py`

Deterministic sharding:

```python
world_size=8, rank=3 → knows exactly which shards
```

Assignment: `shard_id % world_size == rank` (round-robin, disjoint, full coverage)

**Worker partitioning (PyTorch IterableDataset issue):**

PyTorch explicitly warns IterableDataset is replicated across workers. We implement:

```python
assign_samples_to_worker(num_samples, num_workers, worker_id)
```

Contiguous split, deterministic, no duplicates, full coverage.

Tests simulate 1,2,4,8 workers and prove no duplicates.

**ShardedTokenDataset:**
- Map-style: `ShardedTokenDatasetMap` (for DataLoader with DistributedSampler)
- Iterable-style: `ShardedTokenDatasetIterable` (worker-aware)

Both support `rank/world_size` and `mmap`.

## 16. Manifest System

**File:** `src/xrfm/data/manifest_v2.py`

Machine-readable manifest, answers: *Exactly what data produced this corpus?*

```json
{
  "dataset_name": "xrfm-pretrain",
  "dataset_version": "0.1.0",
  "dataset_id": "xrfm-ds-abc123...",
  "created_at": "...",
  "git_commit": "...",
  "processing_pipeline_version": "xrfm-pipeline-v1",
  "processing_config": {...},
  "processing_config_hash": "...",
  "tokenizer": {"name": "...", "version": "...", "hash": "...", "vocab_size": ...},
  "sources": [{"name": "...", "uri": "...", "license": "...", "document_count": ..., "token_count": ...}],
  "documents_total": 123456,
  "tokens_total": 987654321,
  "shards": [{"shard_id": 0, "path": "...", "num_sequences": ..., "sha256": "..."}],
  "num_shards": 128,
  "sequence_length": 2048,
  "split": {"train_documents": ..., "method": "hash", "seed": 42},
  "checksums": {"...": "sha256..."}
}
```

## 17. Checksums and Immutability

**File:** `src/xrfm/data/checksums.py`

- SHA-256 for files, configs, dataset identity
- `dataset_id = hash(source_versions + processing_config + tokenizer_version + pipeline_version)`
- Same inputs + same pipeline + same config = same dataset identity
- Change one input → dataset_id changes (verified in tests)

Every important artifact identifiable: raw snapshots, shards, manifests, tokenizer, config.

## 18. Dataset Reporting

**File:** `src/xrfm/data/reporting.py`

Auto-generated from pipeline output:

```
documents ingested, rejected, deduplicated, retained
languages, license distribution, source distribution
avg document length, token count, token distribution
quality-score distribution, PII flags, near-duplicate counts
train/val/test sizes, tokens per source, tokens per language
storage size, processing time
```

Generates:

```
reports/dataset_report.json
reports/dataset_report.md
```

Not manually typed.

## 19. Inspection Tooling

**File:** `src/xrfm/data/cli.py`

CLI:

```bash
xrfm data build --config configs/data/pretrain.yaml --input tests/data/golden --output-dir processed/golden
xrfm data stats --manifest processed/.../manifest.json
xrfm data inspect --input tests/data/golden --num 10 --mode random
xrfm data sample --input processed/.../documents/raw.jsonl --num 10
```

Maintainer can inspect:

- 10 random documents
- 10 rejected documents
- 10 duplicate examples
- 10 low-quality examples

Without custom scripts.

## 20. CLI

**Full pipeline:**

```bash
xrfm data build --config configs/data/pretrain.yaml
```

**Stages callable individually** (via Python API, future CLI subcommands):

```python
pipeline.normalize(docs)
pipeline.filter_language(docs)
pipeline.filter_quality(docs)
pipeline.filter_pii(docs)
pipeline.dedup(docs)
pipeline.split(docs)
pipeline.tokenize(docs)
pipeline.pack(tokenized)
pipeline.shard(packed)
```

## 21. Resumability

Each stage detects already completed / partial / invalid output via manifests/checksums.

If tokenization fails at shard 47, next run does not redo 0–46 (shard files checked via existence + checksum).

Implemented via file existence checks and manifest tracking; future: more granular checkpointing.

## 22. Parallelism

Interfaces allow configurable:

```
workers, processes, batch size, chunk size
```

Deterministic behavior when seed set.

Phase 1 baseline: single-process, but architecture supports multi-process via `num_workers` in tokenization/packing.

## 23. Memory Safety

Pipeline does NOT assume corpus fits in RAM:

- Streaming JSONL reading (ingestion)
- Iterative processing (normalization, filtering)
- Sharded writing (packing → sharding streams sequences)
- Tokenization can be batched

Tested with dataset larger than available? For Phase 1 golden is tiny, but architecture avoids `all_documents = list(...)` for large datasets where possible.

## 24. Source Data Not Committed

Git contains:

```
source definitions, configs, schemas, processing code, sample data, manifests/examples, docs
```

Not huge generated corpora.

Artifact storage: `processed/` is gitignored, external hosting for large datasets.

## 25. Reproducibility Contract

Dataset reproducible from:

```
source registry + source versions + processing config + pipeline commit + tokenizer version + random seed
```

Manifest makes upstream changes visible via checksums.

## 26. Dataset Versioning

Explicit versions:

```
xrfm-pretrain-v0.1
xrfm-pretrain-v0.2
```

Version corresponds to:

```
source set + processing config + filter rules + dedup config + tokenizer + pipeline version
```

No silent overwriting of official dataset.

## 27. Official vs Experimental

- Official: `configs/data/sources.yaml` with `enabled: true` and license-clean
- Experimental: `enabled: false`, or in separate `experimental/` directory

Prevents experimental downloads becoming part of official corpus.

## 28. Final Large Corpus Not Chosen Yet

Phase 1 establishes pipeline, not gigantic corpus.

Candidate sources documented in `sources.yaml` with license, quality, language, tokens, availability, complexity, limitations.

Final billion-token selection left as explicit decision backed by evidence.

## 29. Visual Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    XRFM Data Pipeline v1                            │
├─────────────────────────────────────────────────────────────────────┤
│  Source Registry (YAML) → Ingestion (JSONL/text)                    │
│       ↓                                                             │
│  Normalization (Unicode, newlines, control chars)                   │
│       ↓                                                             │
│  Language ID (heuristic/fastText) → Quality Filter (modular)        │
│       ↓                                                             │
│  PII Filter (regex heuristics) → Dedup (exact SHA-256, near MinHash)│
│       ↓                                                             │
│  Contamination Check (protected eval sets)                          │
│       ↓                                                             │
│  Split (hash-based deterministic, no overlap)                       │
│       ↓                                                             │
│  Mixing (explicit weights, token/doc counts)                        │
│       ↓                                                             │
│  Tokenization (BPETokenizer, identity tracked)                      │
│       ↓                                                             │
│  Packing (continuous stream → fixed len, EOS handling)              │
│       ↓                                                             │
│  Sharding (.npy shards, rank/world_size aware)                      │
│       ↓                                                             │
│  Manifest (dataset_id, checksums, provenance) + Report (JSON/MD)    │
│       ↓                                                             │
│  Training Input (ShardedTokenDatasetMap/Iterable)                   │
└─────────────────────────────────────────────────────────────────────┘
```

## 30. Limitations (Honest)

- Heuristic language detector, not SOTA (fastText/GlotLID optional but not default)
- PII filtering is heuristic, not guaranteed PII-free
- Near dedup O(n²) for Phase 1, not yet LSH-optimized for trillion tokens
- No cluster-scale parallelism yet (single machine, multi-process ready but not benchmarked at scale)
- No streaming download from HF datasets yet (ingestion is file-based)
- Tokenized format .npy simple, not yet Arrow/Parquet for analytics
- Contamination registry empty by default (needs population with eval sets)

## 31. Phase 2 Readiness

**Can XRFM now safely move toward large-scale pretraining without rebuilding data architecture? Yes.**

- Document-centric storage ✅
- Source manifests + license tracking ✅
- Modular filters with explicit decisions ✅
- Exact + near dedup ✅
- Document-level deterministic splitting ✅
- Contamination boundary ✅
- Mixing system ✅
- Tokenizer versioning + manifest ✅
- Sharded format + deterministic sharding ✅
- Manifest + checksums + dataset_id ✅
- Reporting + inspection ✅
- Resumability + memory safety architecture ✅
- Training integration verified (one real training step) ✅

Next phase can focus on scaling to 1B+ tokens, populating source registry, and optimizing throughput, without rewriting core pipeline.
