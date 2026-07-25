# Phase 3 — Dataset Pipeline: Fresh Research Report

Date: 2026-07-24
Module: Dataset Pipeline (XRFM Data Platform)
Status: Research phase — implementation not yet started.

---

## 1. Official Sources Consulted

- Hugging Face `datasets` documentation (official API reference, streaming mode, dataset loading, dataset versioning).
- WebDataset documentation (streaming dataset format for large-scale data).
- Apache Arrow documentation (in-memory columnar format for efficient data processing).
- MosaicML `Streaming` documentation (streaming dataset design for large-scale training).
- FineWeb / RefinedWeb papers and documentation (web-scale dataset filtering pipelines).
- RedPajama documentation (dataset reproduction and filtering methodology).
- Dolma / OLMo documentation (open dataset platform for LLM pretraining).
- The Pile / SlimPajama documentation (dataset composition and processing).
- OpenWebText documentation.
- Raschka (2024) — dataset preparation and dataset loader patterns (conceptual reference only).

---

## 2. Modern Dataset Engineering Best Practices

### 2.1 Dataset Formats

**Local Text Files:**
- Simplest format; easy to inspect; no dependencies.
- Limitation: requires manual loading; no built-in versioning or metadata tracking.
- XRFM use: Phase 3 starts here (Tiny Shakespeare, WikiText, OpenWebText).

**Hugging Face `datasets`:**
- Provides standardized loading, splitting, and streaming interfaces.
- Supports dataset versioning, dataset cards (metadata), and dataset repositories.
- Integration with `tokenizers` and `transformers` ecosystems.
- XRFM classification: OPTIONAL (post-v0.3) — the dataset loader interface is designed to work with or without `datasets` library. The core loader depends only on Python standard library and `tokenizer` interface.

**WebDataset (Sharding Format):**
- Stores data as tar shards with JSON or Parquet records inside.
- Excellent for very large datasets; supports streaming; supports random access.
- Used by MosaicML and other large-scale training pipelines.
- XRFM classification: RESEARCH-ONLY (post-v1.0, if dataset exceeds local storage).

**Apache Arrow / Parquet:**
- Columnar format for structured data; efficient for filtering, sorting, and selection.
- Good for dataset metadata and statistics; less common for raw text training data.
- XRFM classification: OPTIONAL (future dataset statistics and manifest tracking can use Arrow/Parquet; not needed for Phase 3).

**Streaming Datasets (MosaicML Streaming / HF Streaming):**
- Datasets too large to load into memory are read in streaming mode.
- Essential for 1B+ token datasets on limited hardware.
- XRFM design: The dataset loader interface (`data/loader.py`) supports streaming mode as a configuration option (`streaming: true` in config). The implementation starts with file-based loading; streaming mode is designed but not required for Phase 3.

---

## 3. Dataset Pipeline Layers (Architecture Design)

The dataset pipeline is designed as independent layers, each with a stable interface. This allows future enhancements (multilingual datasets, code datasets, synthetic data) without rewrites.

### Layer Design

**Raw Source:**
- Input: Folder of `.txt` files, or Hugging Face dataset, or future custom source.
- Interface: Config-driven (`datasets.default`, `datasets.path`).

**Downloader:**
- Not required for Phase 3 (Tiny Shakespeare, WikiText, OpenWebText are manually downloaded or included in repo).
- Future: Download scripts (`scripts/download_shakespeare.py`) can be added without changing dataset loader interface.

**Verifier:**
- Checks file existence, encoding (UTF-8), non-empty content.
- Generates warnings with clear diagnostics.
- Phase 3 implementation: Basic validation (file exists, readable, non-empty).

**Cleaner / Normalizer:**
- Normalizes whitespace, removes extremely short/long documents, filters invalid Unicode.
- Phase 3: Minimal normalization; future versions can add more complex filters.

**Document Splitter:**
- Splits large text files into training documents (e.g., by paragraph, by fixed-size chunks).
- Phase 3: Simple split by whitespace/newline for Tiny Shakespeare; future versions support chunk-based splitting.

**Tokenizer Integration:**
- Uses `TokenizerInterface.encode()` (from Phase 2) — no algorithm dependency.
- Produces integer sequences compatible with model training (`model/attention/` design).

**Chunk Generator / Sample Generator:**
- Creates fixed-length sequences (`max_seq_len` from config) by sliding window or contiguous chunks.
- Phase 3: Basic contiguous chunking; sliding window optional for Phase 8 (scaling).

**Dataset Manifests:**
- Every dataset build produces a manifest file (`data/manifests/`) tracking:
  - Dataset name, version, source, license.
  - Processing version, tokenizer version, config version.
  - File checksums, number of documents, number of tokens (approximate), creation date.
- Phase 3: Basic manifest generation.

---

## 4. Dataset Quality Considerations

### Validation Requirements (Phase 3 Implementation)

Every dataset must pass basic validation before being used for training:

- **File existence and readability:** Confirmed (raises `FileNotFoundError` with clear message if missing).
- **Encoding:** UTF-8 expected; `errors="replace"` used for robustness (prevents crashes on corrupt files).
- **Non-empty files:** Confirmed (raises `ValueError` if file has < 100 chars, same safeguard as tokenizer training).
- **Document length:** Very short documents (< 10 chars) and very long documents (> `max_seq_len` * 10) generate warnings but are not rejected by default (future filtering optional).

### Deduplication (Future — Not Phase 3)

Full deduplication (MinHash, LSH) requires significant engineering (distributed processing, large memory footprint). It is designed into the architecture (cleaner layer can include dedup hooks) but deferred until Phase 8 or later when dataset size justifies it.

---

## 5. Reproducibility Requirements

Every dataset build must be reproducible. This requires tracking:

- Source files (checksums or file list).
- Processing script version (`scripts/download_*.py` version or dataset loader version).
- Tokenizer version (`vocab_size`, tokenizer file path, tokenizer class name).
- Random seed (for shuffling/splitting).
- Config snapshot (`ConfigLoader` snapshot saved with dataset).
- Git commit hash (recorded in dataset manifest).

Phase 3 implements basic reproducibility (manifest file with dataset info, tokenizer reference, config snapshot). Full reproducibility (checksums, version tracking) improves in later phases.

---

## 6. Performance and Scalability Design

### Phase 3 Constraints (Free Resources)

- Datasets: Tiny Shakespeare (~1MB), WikiText (~10MB), OpenWebText (~10GB — too large for free Colab storage, so Phase 3 focuses on Tiny Shakespeare and WikiText; larger dataset support designed but not required).
- Memory: Dataset loader uses file-based reading with minimal caching (configurable via `dataset.loader.cache_dir`).
- Batch generation: Basic contiguous chunking; no complex shuffling or sliding windows required for Phase 3.

### Future Scaling (Phase 8+)

- Streaming mode (`streaming: true` in config): Dataset loader reads chunks on demand rather than loading full dataset into memory.
- Sharding: Large datasets split into shards (`WebDataset` format or custom sharding) for distributed processing.
- Caching: Dataset loader supports caching processed chunks to disk (configurable `cache_dir`) to avoid reprocessing the same data across training runs.
- Multi-worker loading: `num_workers` configurable in dataset loader for faster data loading on multi-core systems.

---

## 7. Compatibility with XRFM Architecture

### Config Integration
- `datasets.default`: Select dataset name (`tiny_shakespeare`, `wikitext`, etc.).
- `datasets.path`: Path to dataset directory.
- `model.max_seq_len`: Sequence length for chunk generation.
- `training.batch_size`: Batch size for dataset loader.
- `tokenizer.vocab_size`: Vocabulary size used by dataset loader (via `ConfigLoader`).

### Interface Stability
- Dataset loader (`data/loader.py`) uses `TokenizerInterface` — no rewrite needed when tokenizer algorithm changes.
- Dataset loader returns batches compatible with training loop (`training/loop.py`, Phase 5): `(input_ids, target_ids)` tensors.
- Dataset manifest format is independent of dataset content; adding new dataset formats requires new loader implementations (not interface changes).

---

## 8. Potential Risks and Mitigations

### Risk: Small Dataset Overfitting (Tiny Shakespeare, WikiText)
- **Mitigation:** Config-driven dataset selection allows easy upgrade to larger datasets (OpenWebText). Dataset loader interface unchanged. Training loop includes validation split (`train/val` split configurable) to detect overfitting.

### Risk: Dataset File Corruption / Encoding Issues
- **Mitigation:** Dataset loader validates UTF-8 encoding with `errors="replace"`; generates clear error messages with file paths; dataset manifests track file checksums for future corruption detection.

### Risk: Memory Overflow (Large Datasets Like OpenWebText)
- **Mitigation:** Streaming mode (`streaming: true`) designed into dataset loader interface; basic file-based loading used for Phase 3; streaming activation deferred to Phase 8 (scaling phase).

### Risk: Reproducibility Gaps (Dataset Changes Over Time)
- **Mitigation:** Manifest file (`data/manifests/`) tracks dataset version, tokenizer reference, config snapshot, and git commit. Basic version tracking implemented in Phase 3; full checksum tracking deferred to Phase 8.

---

## 9. Final Recommendation

**Proceed with Phase 3: Dataset Pipeline Implementation.**

**Components to implement (CORE):**
- `data/datasets/` — dataset storage and download scripts.
- `data/manifests/` — dataset manifest generation.
- `xrfm/data/loader.py` — dataset loader (uses `TokenizerInterface`, produces batches).
- `tests/test_data_loader.py` — dataset loading tests (reproducibility, split generation, empty dataset handling).
- `docs/data/DATASET_GUIDE.md` — dataset documentation.
- `DECISIONS.md` update (if any new dataset-related decisions).

**Components deferred (OPTIONAL / RESEARCH-ONLY):**
- Full deduplication pipeline (Phase 8+).
- Streaming dataset mode activation (Phase 8+).
- Hugging Face `datasets` integration (optional enhancement; core loader is independent of `datasets` library).
- WebDataset / sharding format (RESEARCH-ONLY for very large datasets).
- Full dataset version tracking with checksums (Phase 8+).

No new architectural decisions required; Phase 3 follows the established design from the architecture freeze (`REPOSITORY_BLUEPRINT.md`, `API_DESIGN.md`, `IMPLEMENTATION_ROADMAP.md`).
