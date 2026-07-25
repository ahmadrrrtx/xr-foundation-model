# XRFM Dataset Guide

## Overview

The dataset pipeline loads raw text, verifies quality, splits into train/validation/test, tokenizes using the configured tokenizer (`TokenizerInterface`), and produces fixed-length chunks (`max_seq_len`) for the training loop.

## Supported Dataset Formats

### Phase 3 (Current)
- Single `.txt` file.
- Directory containing `.txt` files (first file loaded by default; future versions support multi-file merging).

### Future (Optional / RESEARCH-ONLY)
- Hugging Face `datasets` integration (optional enhancement — interface stable).
- Streaming datasets (`streaming: true` in config — designed but not activated in Phase 3).
- WebDataset / sharding format (RESEARCH-ONLY — for datasets exceeding local storage).
- Apache Arrow / Parquet for structured dataset statistics (optional — manifest uses JSON for simplicity).

## Adding New Datasets

To add a new dataset to XRFM:

1. Place the dataset file in `data/datasets/` (or reference external path in config).
2. Update `config/config.yaml` (`datasets.default`, `datasets.path`).
3. Optionally generate a manifest: call `build_manifest()` and save to `data/manifests/`.
4. Ensure the dataset file is UTF-8 encoded and non-empty (validated by `verify_text_file`).

No code rewrites are required; the dataset loader uses only the stable `TokenizerInterface`.

## Pipeline Stages

1. **Source Selection:** Config-driven (`datasets.default`).
2. **Verification:** `verify_text_file()` checks existence, encoding, and non-empty content.
3. **Normalization:** `normalize_text()` collapses whitespace.
4. **Split:** `split_dataset()` divides by line (paragraph) into train/val/test according to config ratios.
5. **Tokenization:** `tokenizer.encode()` (stable interface; algorithm-independent).
6. **Chunking:** `chunk_text()` creates fixed-length sequences (`max_seq_len`) with optional overlap.
7. **Dataset:** `XRFMTextDataset` (PyTorch `Dataset`) serves chunks with next-token targets.

## Reproducibility

Every dataset load generates a manifest (`build_manifest`) tracking:
- Dataset name and path.
- Tokenizer reference (`class_name`, `vocab_size`).
- Config snapshot (split ratios, sequence length, seed).
- File statistics (optional checksums — basic version in Phase 3; full SHA-256 deferred to Phase 8).

This ensures that experiments using the same dataset version and tokenizer version are reproducible.

## Configuration

Dataset behavior is fully configurable via `config/config.yaml`:

```yaml
datasets:
  default: tiny_shakespeare
  path: data/datasets/
  train_ratio: 0.9
  val_ratio: 0.05
  test_ratio: 0.05
  shuffle: true
  streaming: false  # Reserved for Phase 8 (scaling)
  seed: 42
```

Changing any of these values does not require code rewrites — only a new manifest for reproducibility tracking.

## Performance Notes

- Phase 3 uses file-based reading (not streaming) for simplicity on free compute.
- Memory footprint: dataset loader holds only chunk references (not full dataset) in memory if streaming is enabled; Phase 3 holds chunks in memory for speed.
- Chunk overlap (`overlap`) is configurable but set to 0 by default (Phase 3). Overlap improves context continuity but increases data size.
- Multi-worker loading (`num_workers`) is reserved for Phase 8 (scaling) when larger datasets justify parallel loading.

## Testing

Run dataset tests: `python -m pytest tests/test_data_loader.py -v`

Tests cover verification, normalization, splitting, chunking, dataset initialization, split selection, manifest generation, and tokenizer integration.

## Future Improvements (Post-v0.3)

- Streaming dataset activation (`streaming: true`).
- Full file checksum tracking in manifests.
- Deduplication pipeline (`MinHash` / `LSH` — RESEARCH-ONLY for very large datasets).
- Multi-file dataset merging.
- Hugging Face `datasets` integration layer (optional, behind interface).
- Dataset card generation (automated statistics: token count, document count, vocabulary coverage).
