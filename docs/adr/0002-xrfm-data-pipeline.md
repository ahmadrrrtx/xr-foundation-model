# ADR 0002 — XRFM Data Pipeline Architecture (Phase 1)

**Status:** Accepted · **Date:** 2026-09-18 · **Deciders:** XRFM maintainers
**Supersedes:** Implicit line-based dataset in Phase 0 (`TextDataset` loading whole file)
**Amended by:** Future ADRs for scaling

---

## Context

Phase 0 had a minimal data system:

- Loaded whole corpus via `f.read()` into RAM
- Split by lines (`text.splitlines()`)
- Exact-line dedup only
- Tokenized whole split at once
- No provenance, no license tracking, no language/quality/PII filtering
- No deduplication beyond lines
- Splitting not document-safe, not hash-stable
- No contamination control
- No mixing system
- No sharded format, no manifest beyond file hash
- No checksums for dataset identity
- Could not scale beyond tiny corpus

Phase 1 must build a serious, reproducible, scalable, auditable pipeline for future 125M–1B+ training without another rewrite.

We researched:

- **Dolma** (Ai2): document-centric, separates documents from derived attributes, tagging/filtering/dedup, tokenizes after curation, Bloom-filter dedup, paragraph-level duplicate detection
- **FineWeb / FineWeb-Edu**: trafilatura extraction, fastText language ID ≥0.65, heuristic quality/repetition filters, MinHash dedup, PII anonymization, ablation-driven filtering
- **OLMo / Ai2**: data pipeline as part of model, not hidden preprocessing, explicit mixing as versioned experiment, contamination control
- **Hugging Face datasets**: IterableDataset sharding issues, worker replication, need for explicit partitioning
- **PyTorch**: IterableDataset replicated across workers, must prevent duplicate consumption via worker-aware partitioning

## Decisions and Rationale

### D1 — Canonical Document Representation (`schema.py`)

**Chosen:** Dataclass `Document` with stable fields: `document_id`, `source`, `source_uri`, `license`, `license_url`, `collection`, `language`, `text`, `content_hash`, `metadata`, plus sidecar `DocumentAttributes` for derived scores.

**Why:**
- Documents are atomic unit, not lines (future-proof for books, code, web pages)
- Separation of raw doc + derived attributes mirrors Dolma (useful, proven)
- Stable IDs via SHA-256 hash enable deterministic splitting and dedup
- License and source first-class for provenance

**Rejected:**
- Line-based only (Phase 0) — loses document boundaries, breaks for multi-line docs
- Embedding all scores into raw doc — rewrites raw corpus repeatedly, loses provenance

### D2 — Source Registry (`sources.py`, `configs/data/sources.yaml`)

**Chosen:** YAML registry, authoritative, no hard-coded URLs in Python.

**Why:**
- Single source of truth, auditable
- Tracks license, version, checksum, snapshot date, citation, redistribution
- Enables official vs experimental separation (`enabled: true/false`)

**Rejected:**
- Hard-coded URLs in code — un-auditable, not reproducible

### D3 — License and Provenance Tracking

**Chosen:** Every source has `license`, `license_url`, `version`, `checksum`, `citation`, `redistribution` field. Manifest records source versions. If license unclear, mark unresolved and exclude from official corpus. Publish recipe + checksums if not redistributable.

**Why:**
- Legal clarity mandatory for foundation model
- Prevents mystery data

### D4 — Golden Dataset (`tests/data/golden/`)

**Chosen:** Tiny (33 docs), deterministic, MIT-licensed, version-controlled, covers edge cases: prose, code, Unicode, whitespace, numbers, empty, duplicates, near-dupes, non-English, malformed, very long, PII examples.

**Why:**
- CI must test pipeline without large downloads
- Deterministic, license-clean, forever usable

### D5 — Normalization (`normalization.py`)

**Chosen:** Conservative: Unicode NFC, newline normalization, control char removal, whitespace normalization preserving code indentation, blank line collapsing, empty/malformed detection. Every transformation explicit and testable.

**Why:**
- Aggressive normalization destroys code/Markdown/language-specific Unicode
- Must preserve meaningful content

**Rejected:**
- `normalize_text()` that collapsed newlines (Phase 0) — destroyed paragraph structure

### D6 — Language Filtering (`language.py`)

**Chosen:** Interface `LanguageDetector` with heuristic default (no heavy deps) and optional fastText wrapper. Records `language`, `confidence`, `scores`, `detector`, `version`. Threshold configurable, does not silently delete low-confidence without recording.

**Why:**
- English-focused initially but must support `ur`, `ar`, `multilingual` without redesign
- Heuristic lightweight for CI; fastText/GlotLID pluggable for production
- Documents dependency and version

**Rejected:**
- Hard-coded English-only — not future-proof

### D7 — Quality Filtering (`quality.py`)

**Chosen:** Modular filters: min length, max repetition, symbol/number ratio, boilerplate, malformed, overall score. Each filter independently testable, produces explicit decision + reason.

**Why:**
- One giant `filter_document()` hides reasons, hard to debug
- Preserving reason critical for corpus quality debugging (FineWeb lesson)

### D8 — PII / Safety Filtering (`pii.py`)

**Chosen:** Regex heuristics for emails, phones, secrets, API keys, SSN, credit card. Reports scanned/flagged/removed/transformed. Language: "PII heuristic filtering" not "PII-free". Conservative, configurable.

**Why:**
- Perfect PII removal impossible with regex, but baseline needed
- Must not claim false guarantees

### D9 — Exact Deduplication (`dedup.py`)

**Chosen:** SHA-256 content hash, normalized text (strip) before hashing, deterministic, keeps first occurrence, records stats.

**Why:**
- Cryptographic hash, negligible collision, documented
- Dolma uses Bloom filters for scale; SHA-256 sufficient for Phase 1, Bloom can be added later without changing interface

**Rejected:**
- Simple string matching — not robust

### D10 — Near-Duplicate Detection (`dedup.py`)

**Chosen:** Interface `deduplicate_documents(...)` with MinHash (128 perms, deterministic) and n-gram Jaccard. O(n²) for Phase 1 acceptable for golden dataset, LSH for future scale.

**Why:**
- Research shows MinHash/LSH, SimHash, shingling practical (Dolma, FineWeb)
- Pluggable interface allows multiple algorithms
- Avoid ad-hoc similarity

**Rejected:**
- Over-engineering for tiny corpus, but interface must exist for future

### D11 — Document-Level Splitting (`document_split.py`)

**Chosen:** Atomic unit = document, not line. Two methods: `hash` (stable hashing of doc_id, order-independent) and `random` (seeded shuffle then sequential). Deterministic, seedable, reproducible, proves `train ∩ val = ∅`.

**Why:**
- Line-based splitting breaks for multi-line docs, leaks if same doc appears twice
- Hash-based stable: adding workers doesn't change splits (critical for distributed)
- Seeded random also reproducible

**Rejected:**
- Arbitrary line splits — not document-safe

### D12 — Contamination Control (`contamination.py`)

**Chosen:** Registry of protected evaluation datasets, exact hash + 13-gram overlap check, reports contaminated docs with protected name + score. Architectural boundary, not yet populated with all eval sets.

**Why:**
- Must make it difficult to accidentally ingest eval data
- Simple n-gram overlap for Phase 1, sophisticated algorithms later

### D13 — Data Mixing (`mixing.py`)

**Chosen:** Explicit config `MixtureConfig` with `source: weight`, token/doc-weighted sampling, deterministic via seed, records sampled/actual token counts per source.

**Why:**
- OLMo treats mixture as explicit versioned experiment, not invisible detail
- Reproducible mixture experiments

### D14 — Tokenization Pipeline (`tokenization.py`)

**Chosen:** Records tokenizer identity, version, config, hash, special token IDs. `TokenizerInfo` from tokenizer instance. Tokenized corpus reproducible from raw + processing config + tokenizer version.

**Why:**
- Previously just `tokenizer.encode(text)` lost identity
- Must know which tokenizer produced tokens

### D15 — Tokenized Data Format (`sharding.py`)

**Chosen:** NumPy `.npy` shards, `(num_sequences, seq_len)`, dtype auto (uint16 if vocab < 65535 else uint32), binary, memmap-able, checksum per shard.

**Why:**
- Sequential throughput high
- Random access via memmap
- Storage efficient
- Distributed reading trivial
- Simplicity, minimal deps (numpy already required)
- Checksum per shard

**Alternatives considered:**
- Arrow/Parquet: analytics-friendly but overhead, extra dep
- WebDataset/tar: streaming but complexity
- Raw .bin + .meta: efficient but needs metadata handling
- JSONL: human-readable but inefficient

**Decision:** `.npy` pragmatic middle ground; can evolve to Arrow without breaking manifest contract.

### D16 — Sequence Packing (`packing_extended.py`)

**Chosen:** Concatenate token streams, optionally insert EOS between docs, chunk into fixed `sequence_length`, handle leftovers (drop/pad/keep), track discarded/padded, preserve doc boundaries optionally.

**Why:**
- Avoid unnecessary padding
- Documents may cross boundaries for efficiency (common practice)
- Explicit about EOS, boundaries, attention masking, leftovers

### D17 — Sharding (`sharding.py`)

**Chosen:** Deterministic sharding `shard_id % world_size == rank`, contiguous per shard, `num_shards` configurable. Worker partitioning via `assign_samples_to_worker` to prevent duplicate consumption in IterableDataset (PyTorch warning). Map and Iterable datasets both support rank/world_size.

**Why:**
- Must be impossible for workers to accidentally process identical data
- PyTorch IterableDataset replicated across workers needs explicit partitioning
- Tests for 1,2,4,8 workers prove no duplicates

### D18 — Manifest System (`manifest_v2.py`)

**Chosen:** Machine-readable JSON manifest with dataset_id, git_commit, pipeline_version, processing_config + hash, tokenizer info, sources, docs/tokens counts, shards with checksums, split info, checksums dict.

**Why:**
- Answers: What data? Where from? What filtering? Which tokenizer? How many tokens? How split/mixed? Which config? Can reproduce? Can trainer consume deterministically?
- Most important deliverable for trust

### D19 — Checksums and Immutability (`checksums.py`)

**Chosen:** SHA-256 for files, configs, dataset identity. `dataset_id = hash(source_versions + processing_config + tokenizer_version + pipeline_version)`. Same inputs → same ID, change one input → ID changes.

**Why:**
- Immutability and reproducibility contract

### D20 — Reporting (`reporting.py`)

**Chosen:** Auto-generated from pipeline output, JSON + Markdown, includes ingested/rejected/deduped/retained, languages, licenses, sources, avg length, token counts, quality distribution, PII flags, splits, storage, timing.

**Why:**
- Not manually typed, from actual output
- Critical for debugging corpus quality

### D21 — Inspection Tooling (`cli.py`)

**Chosen:** CLI `xrfm data stats`, `inspect`, `sample`, `build`. Maintainer can inspect random/rejected/duplicate/low-quality docs without custom scripts.

**Why:**
- Extremely valuable for future corpus debugging (FineWeb lesson)

### D22 — Resumability

**Chosen:** Each stage checks existing output via file existence + manifest/checksum. If tokenization fails at shard 47, next run doesn't redo 0–46.

**Why:**
- Pipeline must not restart everything if later step fails

### D23 — Parallelism

**Chosen:** Interfaces allow configurable workers/processes/batch/chunk size, deterministic when seed set. Phase 1 single-process but ready for multi-process.

**Why:**
- Must scale from laptop to workstation to multi-process server

### D24 — Memory Safety

**Chosen:** Streaming JSONL reading, iterative processing, sharded writing, avoids `all_documents = list(...)` for large datasets where possible.

**Why:**
- Corpus must not require fitting in RAM

### D25 — Dataset Versioning

**Chosen:** Explicit versions `xrfm-pretrain-v0.1`, corresponds to source set + processing config + filter rules + dedup config + tokenizer + pipeline version. No silent overwriting.

### D26 — Official vs Experimental Separation

**Chosen:** Registry `enabled` flag, official = license-clean tiny, experimental = candidate large sources with `enabled: false`. Clear convention.

### D27 — No Large Corpus Yet

**Chosen:** Phase 1 pipeline only, not gigantic corpus. Candidate sources documented with license, quality, language, tokens, availability, complexity, limitations. Final billion-token selection left as explicit decision.

## Consequences

- Positive: Document-centric, auditable, reproducible, scalable architecture; training integration verified; no rewrite needed for 1B+ scale; manifest answers all provenance questions
- Negative: Heuristic language/PII not SOTA; near dedup O(n²) not yet LSH-optimized; no cluster-scale parallelism yet; tokenized format .npy simple but not Arrow

## What Must NOT Bypass This Architecture (Future Rules)

1. No new source without registry entry + license tracking
2. No filter without explicit config + reporting
3. No dataset build without manifest + checksums + dataset_id
4. No sharding without rank/world_size awareness + worker partition tests
5. No tokenization without tokenizer identity in manifest
6. No large corpus committed to Git (processed/ gitignored)
7. No nondeterministic builds unless explicitly requested
8. No data leakage into eval splits (verify_no_overlap)
9. No claiming PII-free from regex alone
10. No claiming cluster-ready without measurements

## References

- Dolma (Ai2) — document + attributes separation, Bloom-filter dedup
- FineWeb — language ID threshold 0.65, heuristic filters, MinHash dedup, PII anonymization
- OLMo — data pipeline as part of model, explicit mixing
- PyTorch docs — IterableDataset replication warning
- Hugging Face datasets — IterableDataset DDP duplication issues
