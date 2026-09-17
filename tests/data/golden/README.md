# XRFM Golden Dataset

Tiny, deterministic, redistributable, version-controlled, license-clean dataset for CI and pipeline testing.

## Purpose
Test all pipeline stages:
- normal prose
- code
- Unicode
- whitespace
- numbers
- empty/near-empty documents
- duplicate documents
- near duplicates
- non-English text
- malformed documents
- very long documents

## License
MIT (generated for XRFM, no external copyrighted text)

## Contents
- `documents.jsonl` — canonical Document JSONL format
- `manifest.json` — tiny manifest for testing

## Stats
- 30 documents
- Includes duplicates, near-duplicates, edge cases
- Covers en, ur (Arabic script), ar, code, etc.

## Usage
```bash
xrfm data build --input tests/data/golden --output-dir processed/golden-test
pytest tests/test_golden_pipeline.py
```
