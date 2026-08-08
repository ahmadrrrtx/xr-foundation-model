# XRFM — Dataset Versioning & Provenance Specification (Phase 33)

**Date:** 2026-08-08
**Implementation:** `xrfm/data/manifest.py` (`DatasetManifest`, `build_dataset_manifest`)

## 1. Requirement

Every training experiment must identify, from its artifacts alone:

| Field | Source of truth |
|---|---|
| dataset name | manifest `name` |
| dataset version | manifest `version` |
| source | manifest `source` (URL/collection) |
| license | manifest `license` |
| preprocessing version | manifest `preprocessing_version` |
| tokenizer version | manifest `tokenizer_version` (from tokenizer `_version`) |
| number of documents | manifest `num_documents` (line count after split logic) |
| token count | manifest `total_tokens` |
| train tokens | manifest `train_tokens` |
| validation tokens | manifest `val_tokens` |
| deduplication method | manifest `dedup_method` (default `exact-line-dedup`) |
| filtering method | manifest `filtering_method` |
| checksum/hash | manifest `sha256` (SHA-256 of the raw dataset file) |

## 2. Manifest Schema (v1)

```json
{
  "manifest_version": "xrfm-dataset-manifest-v1",
  "name": "tiny_corpus_v1",
  "version": "1.0.0",
  "path": "data/datasets/corpus.txt",
  "source": "Project Gutenberg (11 books) + PSF stdlib slice",
  "license": "public-domain + PSF",
  "sha256": "<64-hex>",
  "preprocessing_version": "xrfm-preprocess-v1",
  "tokenizer_version": "xrfm-bpe-v2",
  "tokenizer_vocab_size": 2048,
  "num_documents": 108319,
  "total_tokens": 1770334,
  "train_tokens": <int>,
  "val_tokens": <int>,
  "test_tokens": <int>,
  "dedup_method": "exact-line-dedup",
  "filtering_method": "gutenberg-header/footer-strip; none otherwise",
  "language": "en",
  "extra": {}
}
```

## 3. Guarantees

1. **Verifiable, not declarative.** `sha256` is computed from the actual file;
   token counts are computed with the exact split/chunk logic the loader uses,
   so manifest numbers match what training consumes.
2. **Co-located with the run.** The training entrypoint writes
   `logs/<run_id>_dataset_manifest.json` alongside `training_metrics.jsonl`
   and the checkpoint (whose `extra` records seed/config).
3. **Tokenizer version captured** so a tokenizer change invalidates comparisons
   (vocab size + `_version` string).
4. **Split consistency:** uses `split_dataset_lines` (line-boundary,
   exact-line-dedup) with the same ratios as `XRFMTextDataset`.

## 4. Current Corpus Manifest (measured)

| Field | Value |
|---|---|
| name / version | `tiny_corpus_v1` / `1.0.0` |
| path | `data/datasets/corpus.txt` |
| source | Project Gutenberg: 11 books (Swift, Carroll, Dickens, Austen, Doyle, Grimm, Shelley, Twain, Stoker, Poe, Wells) + Python 3.13 stdlib slice (PSF) |
| license | public domain (US) + PSF License |
| num_documents (lines) | 108,319 |
| total_tokens | 1,770,334 (tokenizer vocab 2048, 0.325 tok/char) |
| language | en (19th-c. prose + code) |

> **Honest classification:** this corpus is appropriate for pipeline validation
> and small-scale experiments only. It is NOT foundation-model pretraining data.
> See `docs/data/V1_1_DATASET_PLAN.md` for the v1.1 plan.

## 5. Adding a New Dataset

1. Place the file in `data/datasets/` with a `README.md` recording source,
   license, and how to re-fetch it.
2. Build its manifest:
   `python -c "from xrfm.data.manifest import build_dataset_manifest; m = build_dataset_manifest('data/datasets/<file>', tok, name=..., version=..., source=..., license=...); m.save('data/manifests/<name>_manifest.json')"`
3. Reference the manifest path in the run config / training script so every
   run records `dataset_manifest: <path>`.
