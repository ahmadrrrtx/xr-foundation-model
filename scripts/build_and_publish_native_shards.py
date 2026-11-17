#!/usr/bin/env python3
"""
Phase 2 (`GATE-2`): Low-Memory PyArrow + HfFileSystem Streaming Shard Builder.

Streams directly from remote Parquet row-groups via `pyarrow.parquet.ParquetFile` +
`huggingface_hub.HfFileSystem` (peak RSS `< 300 MB`, avoiding `datasets` iterator
background hang & memory bloat):
1. Enforces SPDX & Track A provenance via `SourceRegistry.validate_for_training("track_a")`.
2. Streams all 8 domains (`en`, `code`, `ur`, `ur-Latn`, `ar`, `hi`, `es`, `fr`),
   applies quality, PII/secret, SHA-256 exact dedup, Banded MinHash-LSH near-dedup,
   and `ContaminationChecker` (`0%` frozen benchmark overlap).
3. Tokenizes with winning 32,768-vocab `HFTokenizerAdapter.pretrained_32k()`,
   packs lossless `seq_len + 1 = 2049` `uint16` causal rows, builds held-out
   per-language validation sets (`val_by_language/`), writes `DatasetManifestV2`,
   and uploads to `ahmadrrrtx/xrfm-native-shards-v1`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from xrfm.data.contamination import get_global_registry
from xrfm.data.dedup import NearDedupConfig
from xrfm.data.manifest_v2 import build_manifest_v2
from xrfm.data.pii import PIIFilterConfig
from xrfm.data.quality import QualityFilterConfig
from xrfm.data.schema import Document
from xrfm.data.sharding import ShardInfo
from xrfm.data.sources import SourceRegistry
from xrfm.data.streaming import StreamingFilterState
from xrfm.data.tokenization import TokenizerInfo
from xrfm.tokenization.hf_bridge import HFTokenizerAdapter

SHARD_STREAM_SPECS: list[dict[str, Any]] = [
    {
        "lang": "en",
        "source_name": "fineweb_edu",
        "weight": 0.28,
        "repo": "HuggingFaceFW/fineweb-edu",
        "parquet_path": "sample/10BT/000_00000.parquet",
        "text_fields": ["text"],
        "license": "ODC-By-1.0",
    },
    {
        "lang": "code",
        "source_name": "opencoder_stage2_permissive",
        "weight": 0.24,
        "repo": "OpenCoder-LLM/opc-sft-stage2",
        "parquet_path": "educational_instruct/train-00000-of-00001.parquet",
        "text_fields": ["code", "output"],
        "license": "CC-BY-4.0",
    },
    {
        "lang": "ur",
        "source_name": "fineweb2_multilingual",
        "weight": 0.11,
        "repo": "HuggingFaceFW/fineweb-2",
        "parquet_path": "data/urd_Arab/train/000_00000.parquet",
        "text_fields": ["text"],
        "license": "ODC-By-1.0",
    },
    {
        "lang": "ur-Latn",
        "source_name": "sangraha_verified_indic",
        "weight": 0.07,
        "repo": "ai4bharat/sangraha",
        "parquet_path": "synthetic/urd_Latn/wiki_urd_Latn_0000_of_0063.parquet",
        "text_fields": ["text"],
        "license": "CC-BY-4.0",
    },
    {
        "lang": "ar",
        "source_name": "fineweb2_multilingual",
        "weight": 0.10,
        "repo": "HuggingFaceFW/fineweb-2",
        "parquet_path": "data/arb_Arab/train/000_00000.parquet",
        "text_fields": ["text"],
        "license": "ODC-By-1.0",
    },
    {
        "lang": "hi",
        "source_name": "fineweb2_multilingual",
        "weight": 0.10,
        "repo": "HuggingFaceFW/fineweb-2",
        "parquet_path": "data/hin_Deva/train/000_00000.parquet",
        "text_fields": ["text"],
        "license": "ODC-By-1.0",
    },
    {
        "lang": "es",
        "source_name": "fineweb2_multilingual",
        "weight": 0.05,
        "repo": "HuggingFaceFW/fineweb-2",
        "parquet_path": "data/spa_Latn/train/000_00000.parquet",
        "text_fields": ["text"],
        "license": "ODC-By-1.0",
    },
    {
        "lang": "fr",
        "source_name": "fineweb2_multilingual",
        "weight": 0.05,
        "repo": "HuggingFaceFW/fineweb-2",
        "parquet_path": "data/fra_Latn/train/000_00000.parquet",
        "text_fields": ["text"],
        "license": "ODC-By-1.0",
    },
]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _is_val_doc(doc_id: str, val_ratio: float = 0.04) -> bool:
    digest = hashlib.sha256(doc_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "little") / float(1 << 32)
    return bucket < val_ratio


def build_and_upload_shards(
    target_tokens: int = 25_000_000,
    seq_len: int = 2048,
    num_shards: int = 8,
    hf_repo_id: str = "ahmadrrrtx/xrfm-native-shards-v1",
    upload: bool = True,
    summary_path: str = "docs/experiments/native_shards_manifest_v1.json",
) -> dict[str, Any]:
    t0 = time.perf_counter()

    registry = SourceRegistry.from_yaml("configs/data/sources.yaml")
    registry.validate_for_training(track="track_a", require_pinned_revision=True)
    for spec in SHARD_STREAM_SPECS:
        registry.assert_source_trainable(spec["source_name"], track="track_a", require_pinned_revision=True)

    tokenizer = HFTokenizerAdapter.pretrained_32k()
    tok_info = TokenizerInfo.from_tokenizer(tokenizer, name="xrfm-32k-multilingual-bpe", version="v1.0")
    eos_id = tokenizer.eos_token_id or 2
    pad_id = tokenizer.pad_token_id or 0
    row_len = seq_len + 1

    contam_checker = get_global_registry(autoload_frozen=True).checker()
    qual_cfg = QualityFilterConfig(min_chars=60, max_chars=60000, min_words=10, max_symbol_ratio=0.45)
    pii_cfg = PIIFilterConfig(remove_high_risk=True)

    hf_token = os.environ.get("HF_TOKEN")
    fs = HfFileSystem(token=hf_token)

    disk_tmp_base = "/var/tmp" if os.path.isdir("/var/tmp") else None
    scratch_dir = tempfile.mkdtemp(prefix="xrfm_native_shards_", dir=disk_tmp_base)

    try:
        tokens_dir = os.path.join(scratch_dir, "tokens")
        val_dir = os.path.join(scratch_dir, "val_by_language")
        os.makedirs(tokens_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        lang_rows: dict[str, list[np.ndarray]] = {}
        per_lang_stats: dict[str, dict[str, int]] = {}
        val_manifest_entries: dict[str, dict[str, Any]] = {}
        docs_per_source: dict[str, int] = {}
        tokens_per_source: dict[str, int] = {}
        total_train_docs = 0
        total_val_docs = 0
        total_val_tokens = 0
        contaminated_total = 0

        for spec in SHARD_STREAM_SPECS:
            lang = spec["lang"]
            src_name = spec["source_name"]
            lang_tok_budget = int(target_tokens * float(spec["weight"]))

            filter_state = StreamingFilterState(NearDedupConfig(threshold=0.85, num_perm=32, num_bands=8))
            remote_uri = f"datasets/{spec['repo']}/{spec['parquet_path']}"
            pf = pq.ParquetFile(remote_uri, filesystem=fs)

            token_carry = np.empty(0, dtype=np.uint16)
            packed_rows: list[np.ndarray] = []
            val_docs: list[Document] = []
            val_tok_buf: list[int] = []

            lang_train_tokens = 0
            lang_docs = 0
            lang_contam = 0
            doc_counter = 0

            for record_batch in pf.iter_batches(batch_size=192, columns=spec["text_fields"]):
                cols = record_batch.to_pydict()
                n_rows = record_batch.num_rows
                batch_docs: list[Document] = []
                for r_idx in range(n_rows):
                    parts = [str(cols[f][r_idx] or "") for f in spec["text_fields"] if cols.get(f)]
                    text = "\n".join(p for p in parts if p.strip()).strip()
                    if len(text) < 60:
                        continue
                    if len(text) > 10000:
                        text = text[:10000]
                    batch_docs.append(
                        Document(
                            text=text,
                            source=src_name,
                            document_id=f"{lang}-{doc_counter}",
                            language=lang,
                            license=spec["license"],
                        )
                    )
                    doc_counter += 1

                if not batch_docs:
                    continue

                kept, _ = filter_state.process_chunk(
                    batch_docs,
                    quality_config=qual_cfg,
                    pii_config=pii_cfg,
                    exact_dedup=True,
                )
                clean_kept, contam_hits = contam_checker.check_documents(kept)
                lang_contam += len(contam_hits)
                contaminated_total += len(contam_hits)

                new_train_ids: list[int] = []
                for cdoc in clean_kept:
                    ids = tokenizer.encode(cdoc.text)
                    ids.append(eos_id)
                    if _is_val_doc(cdoc.document_id) and len(val_docs) < 80:
                        val_docs.append(cdoc)
                        val_tok_buf.extend(ids)
                        total_val_docs += 1
                        total_val_tokens += len(ids)
                    else:
                        new_train_ids.extend(ids)
                        lang_train_tokens += len(ids)
                        lang_docs += 1
                        total_train_docs += 1
                        docs_per_source[src_name] = docs_per_source.get(src_name, 0) + 1
                        tokens_per_source[src_name] = tokens_per_source.get(src_name, 0) + len(ids)

                if new_train_ids:
                    arr_new = np.asarray(new_train_ids, dtype=np.uint16)
                    stream = np.concatenate([token_carry, arr_new]) if token_carry.size else arr_new
                    while stream.size >= row_len:
                        packed_rows.append(stream[:row_len].copy())
                        stream = stream[seq_len:]
                    token_carry = stream

                if lang_train_tokens >= lang_tok_budget:
                    break

            lang_rows[lang] = packed_rows

            v_jsonl = os.path.join(val_dir, f"val_{lang}.jsonl")
            with open(v_jsonl, "w", encoding="utf-8") as vf:
                for vd in val_docs:
                    vf.write(vd.to_jsonl() + "\n")

            val_seqs: list[np.ndarray] = []
            v_arr = np.asarray(val_tok_buf, dtype=np.uint16)
            while v_arr.size >= 513:
                val_seqs.append(v_arr[:513].copy())
                v_arr = v_arr[512:]
            if v_arr.size > 16 and not val_seqs:
                padded = np.full(513, pad_id, dtype=np.uint16)
                padded[: v_arr.size] = v_arr
                val_seqs.append(padded)

            if val_seqs:
                v_mat = np.stack(val_seqs, axis=0)
                v_npy = os.path.join(val_dir, f"val_{lang}_seq513.npy")
                np.save(v_npy, v_mat)
                val_manifest_entries[lang] = {
                    "docs": len(val_docs),
                    "utf8_bytes": sum(len(d.text.encode("utf-8")) for d in val_docs),
                    "tokens": len(val_tok_buf),
                    "packed_sequences_512": int(v_mat.shape[0]),
                    "jsonl": f"val_by_language/val_{lang}.jsonl",
                    "npy": f"val_by_language/val_{lang}_seq513.npy",
                }

            per_lang_stats[lang] = {
                "source": src_name,
                "train_docs": lang_docs,
                "train_tokens": lang_train_tokens,
                "packed_sequences_2048": len(packed_rows),
                "val_docs": len(val_docs),
                "contaminated_removed": lang_contam,
            }
            print(
                f"[ShardStream] {lang:8s}: {lang_docs:6d} docs, {lang_train_tokens:9,d} tokens -> "
                f"{len(packed_rows):5d} seqs (val={len(val_docs)}, contam={lang_contam})",
                flush=True,
            )

            del pf, filter_state, val_docs, val_tok_buf
            gc.collect()

        all_rows: list[np.ndarray] = []
        for rows in lang_rows.values():
            all_rows.extend(rows)
        lang_rows.clear()

        rng = np.random.default_rng(42)
        order = rng.permutation(len(all_rows))
        total_seqs = len(all_rows)
        actual_shards = min(num_shards, max(total_seqs, 1))
        base_per_shard = total_seqs // actual_shards
        rem = total_seqs % actual_shards

        shard_infos: list[ShardInfo] = []
        cursor = 0
        for sid in range(actual_shards):
            count = base_per_shard + (1 if sid < rem else 0)
            idxs = order[cursor : cursor + count]
            cursor += count
            shard_mat = np.stack([all_rows[int(k)] for k in idxs], axis=0).astype(np.uint16, copy=False)
            fname = f"train_shard-{sid:05d}-of-{actual_shards:05d}.npy"
            disk_path = os.path.join(tokens_dir, fname)
            np.save(disk_path, shard_mat)
            sha = _sha256_file(disk_path)
            shard_infos.append(
                ShardInfo(
                    shard_id=sid,
                    path=f"tokens/{fname}",
                    num_sequences=int(shard_mat.shape[0]),
                    num_tokens=int(shard_mat.shape[0] * seq_len),
                    sha256=sha,
                    dtype="uint16",
                    sequence_length=row_len,
                )
            )
            del shard_mat

        all_rows.clear()
        gc.collect()

        total_packed_train_tokens = sum(s.num_tokens for s in shard_infos)
        sources_summary = []
        for s_name in sorted({s["source_name"] for s in SHARD_STREAM_SPECS}):
            scfg = registry.get(s_name)
            if scfg:
                sources_summary.append(
                    {
                        "name": scfg.name,
                        "uri": scfg.uri,
                        "type": scfg.type,
                        "license": scfg.license,
                        "license_url": scfg.license_url,
                        "version": scfg.revision,
                        "checksum": scfg.revision,
                        "language": scfg.language,
                        "document_count": docs_per_source.get(s_name, 0),
                        "token_count": tokens_per_source.get(s_name, 0),
                    }
                )

        manifest = build_manifest_v2(
            dataset_name="xrfm-native-shards",
            dataset_version="1.0.0",
            processing_config={
                "sequence_length": seq_len,
                "row_length": row_len,
                "include_next_token": True,
                "dtype": "uint16",
                "dedup": "sha256 + minhash_lsh_8x4",
                "contamination_manifest": "configs/eval/frozen_benchmarks_v1.json",
            },
            tokenizer_info=tok_info,
            sources=sources_summary,
            splits={"train": [None] * total_train_docs, "val": [None] * total_val_docs, "test": []},  # type: ignore[list-item]
            token_counts={
                "total": total_packed_train_tokens + total_val_tokens,
                "train": total_packed_train_tokens,
                "val": total_val_tokens,
                "test": 0,
            },
            shard_infos=shard_infos,
            sequence_length=seq_len,
            split_config={"method": "sha256_hash", "val_ratio": 0.04},
            checksums={si.path: si.sha256 for si in shard_infos},
            pipeline_version="xrfm-pipeline-v2",
        )
        manifest_file = os.path.join(scratch_dir, "manifest.json")
        manifest.save(manifest_file)

        shutil.copy2(
            "src/xrfm/tokenization/tokenizer_32k_multilingual.json",
            os.path.join(scratch_dir, "tokenizer_32k_multilingual.json"),
        )

        readme_md = f"""---
license: odc-by
task_categories:
  - text-generation
language:
  - en
  - ur
  - ar
  - hi
  - es
  - fr
pretty_name: XRFM Native Multilingual + Code Pretraining Shards (v1.0)
---

# XRFM Native Pretraining Shards (`v1.0.0`)

Versioned, SPDX-governed, LSH-deduplicated, and benchmark-decontaminated `uint16` token shards (`seq_len={seq_len}`, stored as `{row_len}` lossless causal `L+1` rows) for **Track A (`XRFM Native`)**.

- **Dataset ID:** `{manifest.dataset_id}`
- **Tokenizer:** `T4` (`32,768` Script-Protected Byte-Level BPE, fingerprint `{tok_info.hash}`)
- **Total Tokens:** `{manifest.tokens_total:,}` (`train={manifest.tokens_train:,}`, `val={manifest.tokens_val:,}`)
- **Train Sequences (`{row_len}` uint16):** `{sum(s.num_sequences for s in shard_infos):,}` across `{len(shard_infos)}` `.npy` shards
- **Contamination Gate:** Checked against `configs/eval/frozen_benchmarks_v1.json` (`0%` eval leakage; `{contaminated_total}` overlapping docs removed)

## Attribution & SPDX Provenance (Tier A / Tier B Only)
- **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`, `ODC-By-1.0`, commit `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`)
- **FineWeb-2 Multilingual** (`HuggingFaceFW/fineweb-2`, `ODC-By-1.0`, commit `af9c13333eb981300149d5ca60a8e9d659b276b9`)
- **OpenCoder Stage-2** (`OpenCoder-LLM/opc-sft-stage2`, `CC-BY-4.0`, commit `7d28f40d579edd7c24402d17d0c7639f991e6f8d`)
- **AI4Bharat Sangraha** (`ai4bharat/sangraha`, `CC-BY-4.0`, commit `8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8`)
"""
        with open(os.path.join(scratch_dir, "README.md"), "w", encoding="utf-8") as rf:
            rf.write(readme_md)

        commit_sha = None
        if upload:
            if not hf_token:
                raise RuntimeError("HF_TOKEN environment variable is required to upload shards.")
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=hf_repo_id, repo_type="dataset", private=False, exist_ok=True)
            commit_info = api.upload_folder(
                repo_id=hf_repo_id,
                repo_type="dataset",
                folder_path=scratch_dir,
                commit_message=f"Publish {manifest.dataset_id} ({manifest.tokens_total:,} tokens, 8 domains, uint16 L+1 shards)",
            )
            commit_sha = getattr(commit_info, "oid", None) or str(commit_info)
            print(
                f"[HF Upload] Published to https://huggingface.co/datasets/{hf_repo_id} (commit={commit_sha})",
                flush=True,
            )

        summary = {
            "dataset_id": manifest.dataset_id,
            "hf_repo_id": hf_repo_id,
            "hf_commit": commit_sha,
            "tokenizer_fingerprint": tok_info.hash,
            "vocab_size": tok_info.vocab_size,
            "sequence_length": seq_len,
            "stored_row_length": row_len,
            "num_shards": len(shard_infos),
            "total_sequences": sum(s.num_sequences for s in shard_infos),
            "tokens_total": manifest.tokens_total,
            "tokens_train": manifest.tokens_train,
            "tokens_val": manifest.tokens_val,
            "contaminated_removed": contaminated_total,
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "per_language": per_lang_stats,
            "val_by_language": val_manifest_entries,
            "shards": [
                {
                    "shard_id": s.shard_id,
                    "path": s.path,
                    "num_sequences": s.num_sequences,
                    "num_tokens": s.num_tokens,
                    "sha256": s.sha256,
                    "dtype": s.dtype,
                }
                for s in shard_infos
            ],
        }
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2, ensure_ascii=False)
        return summary
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build & upload Track A pretraining shards")
    parser.add_argument("--target-tokens", type=int, default=25_000_000)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--hf-repo", default="ahmadrrrtx/xrfm-native-shards-v1")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    summary = build_and_upload_shards(
        target_tokens=args.target_tokens,
        seq_len=args.seq_len,
        num_shards=args.num_shards,
        hf_repo_id=args.hf_repo,
        upload=not args.no_upload,
    )
    print(f"[DONE] dataset_id={summary['dataset_id']} tokens={summary['tokens_total']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
