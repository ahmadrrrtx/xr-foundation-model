"""
CLI for XRFM data pipeline.

Design:
xrfm data ingest
xrfm data normalize
xrfm data filter
xrfm data dedupe
xrfm data split
xrfm data tokenize
xrfm data pack
xrfm data build
xrfm data stats
xrfm data inspect

All lower-level stages callable individually, high-level build orchestrates.

Implemented as subcommands for `xrfm` CLI, but also usable as `xrfm-data` entry point.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from xrfm.data.schema import Document
from xrfm.data.pipeline import PipelineConfig, DataPipeline
from xrfm.data.sources import SourceRegistry


def _load_documents_from_path(path: str) -> List[Document]:
    """Load documents from JSONL or text file."""
    docs: List[Document] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    if os.path.isdir(path):
        # Load all .jsonl files
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".jsonl"):
                    fp = os.path.join(root, f)
                    with open(fp, "r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                docs.append(Document.from_jsonl(line))
                            except Exception:
                                # Try as plain text doc
                                docs.append(Document(text=line, source="unknown"))
    else:
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        docs.append(Document.from_jsonl(line))
                    except Exception:
                        docs.append(Document(text=line, source="unknown"))
        else:
            # Plain text file: each line is a doc? Or whole file is one doc?
            # For simplicity, treat whole file as one doc, plus also split lines as docs if many lines
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            # If file has many lines, treat each non-empty line as doc
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            if len(lines) > 1 and len(content) > 1000:
                for i, line in enumerate(lines):
                    docs.append(Document(text=line, source="file", source_uri=path, document_id=f"file-{i}"))
            else:
                docs.append(Document(text=content, source="file", source_uri=path))

    return docs


def cmd_build(args: argparse.Namespace) -> int:
    from xrfm.tokenization import BPETokenizer

    # Load config if provided
    config_path = args.config
    if config_path and os.path.isfile(config_path):
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        # Build PipelineConfig from dict (simplified)
        pipeline_cfg = PipelineConfig(
            dataset_name=cfg_dict.get("dataset_name", "xrfm-pretrain"),
            dataset_version=cfg_dict.get("dataset_version", "0.1.0"),
            output_dir=cfg_dict.get("output_dir", "processed/xrfm-pretrain"),
            sequence_length=cfg_dict.get("sequence_length", 512),
        )
        # Override from args if provided
        if args.output_dir:
            pipeline_cfg.output_dir = args.output_dir
    else:
        pipeline_cfg = PipelineConfig(
            dataset_name="xrfm-pretrain",
            dataset_version="0.1.0",
            output_dir=args.output_dir or "processed/xrfm-pretrain",
            sequence_length=args.sequence_length or 512,
        )

    # Load tokenizer
    tokenizer_path = args.tokenizer or "src/xrfm/tokenization/vocab.json"
    # Try pretrained
    try:
        tokenizer = BPETokenizer.pretrained()
    except Exception:
        tokenizer = BPETokenizer()
        # If vocab exists, load
        if os.path.isfile(tokenizer_path):
            try:
                tokenizer.load(tokenizer_path)
            except Exception:
                pass
        else:
            # Train on sample data if needed
            print("No pretrained tokenizer found, using base tokenizer")

    # Load documents
    input_path = args.input
    if not input_path:
        # Try golden dataset
        golden_path = "tests/data/golden"
        if os.path.isdir(golden_path):
            input_path = golden_path
        else:
            print("No input path provided and no golden dataset found", file=sys.stderr)
            return 1

    docs = _load_documents_from_path(input_path)
    print(f"Loaded {len(docs)} documents from {input_path}")

    # Source registry
    registry_path = args.sources or "configs/data/sources.yaml"
    if os.path.isfile(registry_path):
        registry = SourceRegistry.from_yaml(registry_path)
    else:
        registry = SourceRegistry()

    pipeline = DataPipeline(config=pipeline_cfg, tokenizer=tokenizer, source_registry=registry)
    result = pipeline.run(docs)

    print(f"Build complete. Manifest: {result['manifest_path']}")
    print(f"Shards: {len(result['shard_infos'])}")
    print(f"Report: {result['report'].to_dict()}")

    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    manifest_path = args.manifest
    if not manifest_path or not os.path.isfile(manifest_path):
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    from xrfm.data.manifest_v2 import DatasetManifestV2

    manifest = DatasetManifestV2.load(manifest_path)
    print(f"Dataset: {manifest.dataset_name} v{manifest.dataset_version} id={manifest.dataset_id}")
    print(f"Documents: {manifest.documents_total} (train={manifest.documents_train} val={manifest.documents_val} test={manifest.documents_test})")
    print(f"Tokens: {manifest.tokens_total} (train={manifest.tokens_train} val={manifest.tokens_val} test={manifest.tokens_test})")
    print(f"Shards: {manifest.num_shards}, seq_len={manifest.sequence_length}")
    print(f"Sources: {len(manifest.sources)}")
    for src in manifest.sources:
        print(f"  - {src.name}: {src.document_count} docs, {src.token_count} tokens, license={src.license}")
    print(f"Tokenizer: {manifest.tokenizer.name} v{manifest.tokenizer.version} hash={manifest.tokenizer.hash} vocab={manifest.tokenizer.vocab_size}")
    print(f"Created: {manifest.created_at}, commit={manifest.git_commit}")

    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    # Inspect documents, samples
    input_path = args.input
    if not input_path:
        print("Provide --input path", file=sys.stderr)
        return 1

    docs = _load_documents_from_path(input_path)
    print(f"Total docs: {len(docs)}")

    num = args.num or 10
    mode = args.mode or "random"

    if mode == "random":
        import random

        rng = random.Random(args.seed or 42)
        sampled = rng.sample(docs, min(num, len(docs)))
    elif mode == "first":
        sampled = docs[:num]
    else:
        sampled = docs[:num]

    for i, doc in enumerate(sampled):
        print(f"\n--- Document {i} ---")
        print(f"ID: {doc.document_id}")
        print(f"Source: {doc.source} Language: {doc.language} License: {doc.license}")
        print(f"Length: {len(doc.text)} chars, {len(doc.text.split())} words, hash={doc.content_hash[:12]}")
        print(f"Text preview: {doc.text[:500]!r}")

    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    return cmd_inspect(args)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xrfm data", description="XRFM data pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    build_p = sub.add_parser("build", help="Build full dataset pipeline")
    build_p.add_argument("--config", help="Pipeline config YAML")
    build_p.add_argument("--input", help="Input documents path (jsonl or dir)")
    build_p.add_argument("--output-dir", help="Output dir")
    build_p.add_argument("--sources", help="Sources registry YAML")
    build_p.add_argument("--tokenizer", help="Tokenizer vocab path")
    build_p.add_argument("--sequence-length", type=int, help="Sequence length")

    # stats
    stats_p = sub.add_parser("stats", help="Show dataset stats from manifest")
    stats_p.add_argument("--manifest", required=True, help="Manifest JSON path")

    # inspect
    inspect_p = sub.add_parser("inspect", help="Inspect documents")
    inspect_p.add_argument("--input", required=True, help="Input path")
    inspect_p.add_argument("--num", type=int, default=10, help="Number of docs to show")
    inspect_p.add_argument("--mode", choices=["random", "first"], default="random")
    inspect_p.add_argument("--seed", type=int, default=42)

    # sample alias
    sample_p = sub.add_parser("sample", help="Sample documents")
    sample_p.add_argument("--input", required=True)
    sample_p.add_argument("--num", type=int, default=10)
    sample_p.add_argument("--mode", choices=["random", "first"], default="random")
    sample_p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    if args.command == "build":
        return cmd_build(args)
    elif args.command == "stats":
        return cmd_stats(args)
    elif args.command == "inspect":
        return cmd_inspect(args)
    elif args.command == "sample":
        return cmd_sample(args)
    else:
        parser.error(f"Unknown command {args.command}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
