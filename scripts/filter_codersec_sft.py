#!/usr/bin/env python3
"""Produce a filtered, ChatML-corrected revision of the CoderSec SFT dataset.

The published revision ``808d7505`` is intentionally left untouched because it
is the data used by the documented QLoRA experiments. This script writes a new
JSONL file plus a manifest so a corrected, versioned revision can be reviewed
and published separately.

Usage:
    python scripts/filter_codersec_sft.py \
        --input data.jsonl \
        --output codersec_sft_v1_1.jsonl \
        --manifest codersec_sft_v1_1_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE_REVISION = "808d7505152eea93e6cd4af5f2269f43712afc2d"
TARGET_VERSION = "1.1.0"

# The published file uses corrupted delimiters: "licsystem"/"licuser"/
# "licassistant" and "irit" instead of the ChatML special tokens.
_ROLE_START = re.compile(r"\blic(system|user|assistant)\b")
_ROLE_END = re.compile(r"\birit\b")
_PLACEHOLDER_CODE = re.compile(r"#\s*placeholder|result\s*=\s*x\s*\+\s*y|def\s+func_\d+\(x,\s*y\)")


def canonicalize_chatml(text: str) -> str:
    """Restore canonical ChatML special tokens without altering message text."""
    text = _ROLE_START.sub(r"<|im_start|>\1", text)
    text = _ROLE_END.sub("<|im_end|>", text)
    return text


def is_placeholder_only(record: dict[str, Any]) -> bool:
    """True when every code block is a generic non-implementation."""
    blocks = record.get("code_blocks") or []
    if not blocks:
        return False
    return all(bool(_PLACEHOLDER_CODE.search(block.get("code", ""))) for block in blocks)


def row_hash(record: dict[str, Any]) -> str:
    return hashlib.sha256(record["chatml"].encode("utf-8")).hexdigest()


def filter_records(records: list[dict[str, Any]], deduplicate: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    removed = Counter()
    for record in records:
        record = dict(record)
        original = record.get("chatml", "")
        canonical = canonicalize_chatml(original)
        if canonical != original:
            removed["chatml_canonicalized"] += 1
        record["chatml"] = canonical
        if is_placeholder_only(record):
            removed["placeholder_code_dropped"] += 1
            continue
        digest = row_hash(record)
        if deduplicate:
            if digest in seen:
                removed["duplicate_dropped"] += 1
                continue
            seen.add(digest)
        kept.append(record)
    stats = {
        "input_rows": len(records),
        "output_rows": len(kept),
        "removed": dict(removed),
        "output_sources": dict(Counter(r.get("source") for r in kept)),
        "output_domains": dict(Counter(r.get("domain") for r in kept)),
        "output_licenses": dict(Counter(r.get("license") for r in kept)),
    }
    return kept, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original JSONL dataset file")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL file")
    parser.add_argument("--manifest", type=Path, required=True, help="Destination JSON manifest")
    parser.add_argument("--keep-duplicates", action="store_true", help="Retain exact duplicate records")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept, stats = filter_records(records, deduplicate=not args.keep_duplicates)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_name": "xrfm-codersec-sft",
        "dataset_version": TARGET_VERSION,
        "source_revision": SOURCE_REVISION,
        "filtering": {
            "canonicalize_chatml": True,
            "drop_placeholder_only_code": True,
            "deduplicate_exact_chatml": not args.keep_duplicates,
        },
        "stats": stats,
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
