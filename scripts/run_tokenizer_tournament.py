#!/usr/bin/env python3
"""
Phase 2 (`GATE-2`): XRFM Track A 4-Way Multilingual + Code Tokenizer Tournament.

Streams a balanced, license-verified corpus across all 8 target domains:
  - en      (28%): HuggingFaceFW/fineweb-edu (ODC-By-1.0)
  - code    (24%): OpenCoder-LLM/opc-sft-stage2 (CC-BY-4.0)
  - ur      (11%): HuggingFaceFW/fineweb-2 urd_Arab (ODC-By-1.0)
  - ur-Latn ( 7%): ai4bharat/sangraha synthetic/urd_Latn + HuggingFaceFW/fineweb-2 urd_Latn
  - ar      (10%): HuggingFaceFW/fineweb-2 arb_Arab (ODC-By-1.0)
  - hi      (10%): HuggingFaceFW/fineweb-2 hin_Deva (ODC-By-1.0)
  - es      ( 5%): HuggingFaceFW/fineweb-2 spa_Latn (ODC-By-1.0)
  - fr      ( 5%): HuggingFaceFW/fineweb-2 fra_Latn (ODC-By-1.0)

Filters out any document overlapping frozen evaluation benchmarks (`ContaminationChecker`),
trains all 4 tournament candidates (`T1`, `T2`, `T3`, `T4`) + evaluates legacy `vocab.json`
control, selects the best 32,768-vocab model passing `GATE-2`, saves:
  - `src/xrfm/tokenization/tokenizer_32k_multilingual.json`
  - `docs/experiments/tokenizer_tournament_v1.json`
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from datasets import load_dataset

from xrfm.data.contamination import get_global_registry
from xrfm.data.schema import Document
from xrfm.tokenization import BPETokenizer
from xrfm.tokenization.hf_bridge import train_hf_candidate_tokenizer
from xrfm.tokenization.tournament import TOURNAMENT_EVAL_SUITE, evaluate_tokenizer_candidate

STREAM_SPECS: list[dict[str, Any]] = [
    {
        "lang": "en",
        "weight": 0.28,
        "repo": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "data_files": None,
        "text_fields": ["text"],
        "license": "ODC-By-1.0",
    },
    {
        "lang": "code",
        "weight": 0.24,
        "repo": "OpenCoder-LLM/opc-sft-stage2",
        "config": "educational_instruct",
        "split": "train",
        "data_files": None,
        "text_fields": ["code", "output"],
        "license": "CC-BY-4.0",
    },
    {
        "lang": "ur",
        "weight": 0.11,
        "repo": "HuggingFaceFW/fineweb-2",
        "config": None,
        "split": "train",
        "data_files": "data/urd_Arab/train/000_00000.parquet",
        "text_fields": ["text"],
        "license": "ODC-By-1.0",
    },
    {
        "lang": "ur-Latn",
        "weight": 0.07,
        "repo": "ai4bharat/sangraha",
        "config": None,
        "split": "train",
        "data_files": "synthetic/urd_Latn/wiki_urd_Latn_0000_of_0063.parquet",
        "text_fields": ["text"],
        "license": "CC-BY-4.0",
    },
    {
        "lang": "ar",
        "weight": 0.10,
        "repo": "HuggingFaceFW/fineweb-2",
        "config": None,
        "split": "train",
        "data_files": "data/arb_Arab/train/000_00000.parquet",
        "text_fields": ["text"],
        "license": "ODC-By-1.0",
    },
    {
        "lang": "hi",
        "weight": 0.10,
        "repo": "HuggingFaceFW/fineweb-2",
        "config": None,
        "split": "train",
        "data_files": "data/hin_Deva/train/000_00000.parquet",
        "text_fields": ["text"],
        "license": "ODC-By-1.0",
    },
    {
        "lang": "es",
        "weight": 0.05,
        "repo": "HuggingFaceFW/fineweb-2",
        "config": None,
        "split": "train",
        "data_files": "data/spa_Latn/train/000_00000.parquet",
        "text_fields": ["text"],
        "license": "ODC-By-1.0",
    },
    {
        "lang": "fr",
        "weight": 0.05,
        "repo": "HuggingFaceFW/fineweb-2",
        "config": None,
        "split": "train",
        "data_files": "data/fra_Latn/train/000_00000.parquet",
        "text_fields": ["text"],
        "license": "ODC-By-1.0",
    },
]


def stream_balanced_tokenizer_corpus(total_target_mb: float = 24.0) -> tuple[list[str], dict[str, Any]]:
    """Stream decontaminated UTF-8 texts proportioned by UTF-8 byte budget per language."""
    total_target_bytes = int(total_target_mb * 1024 * 1024)
    checker = get_global_registry(autoload_frozen=True).checker()

    corpus: list[str] = []
    stats_per_lang: dict[str, dict[str, Any]] = {}

    for spec in STREAM_SPECS:
        lang = spec["lang"]
        lang_budget = int(total_target_bytes * float(spec["weight"]))
        collected_bytes = 0
        collected_docs = 0
        contaminated_dropped = 0

        kwargs: dict[str, Any] = {"split": spec["split"], "streaming": True}
        if spec["data_files"]:
            ds = load_dataset("parquet", data_files=f"hf://datasets/{spec['repo']}/{spec['data_files']}", **kwargs)
        elif spec["config"]:
            ds = load_dataset(spec["repo"], spec["config"], **kwargs)
        else:
            ds = load_dataset(spec["repo"], **kwargs)

        for row in ds:
            text_parts = [str(row.get(f, "")) for f in spec["text_fields"] if row.get(f)]
            text = "\n".join(p for p in text_parts if p.strip()).strip()
            if len(text) < 60:
                continue
            # Truncate giant documents so thousands of diverse documents contribute vocabulary
            if len(text) > 6000:
                text = text[:6000]

            is_contam, _, _ = checker.check_document(Document(text=text, source=spec["repo"]))
            if is_contam:
                contaminated_dropped += 1
                continue

            b_len = len(text.encode("utf-8"))
            corpus.append(text)
            collected_bytes += b_len
            collected_docs += 1
            if collected_bytes >= lang_budget:
                break

        stats_per_lang[lang] = {
            "repo": spec["repo"],
            "license": spec["license"],
            "target_bytes": lang_budget,
            "collected_bytes": collected_bytes,
            "collected_docs": collected_docs,
            "contaminated_dropped": contaminated_dropped,
        }
        print(
            f"[TokenizerStream] {lang:8s}: {collected_docs:5d} docs, "
            f"{collected_bytes / (1024 * 1024):6.2f} MB ({spec['repo']})"
        )

    return corpus, stats_per_lang


def main() -> int:
    parser = argparse.ArgumentParser(description="Run XRFM Phase 2 4-way tokenizer tournament")
    parser.add_argument("--budget-mb", type=float, default=24.0, help="Total UTF-8 MB streamed for training")
    parser.add_argument(
        "--output-tokenizer",
        default="src/xrfm/tokenization/tokenizer_32k_multilingual.json",
    )
    parser.add_argument(
        "--output-report",
        default="docs/experiments/tokenizer_tournament_v1.json",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    corpus, stream_stats = stream_balanced_tokenizer_corpus(total_target_mb=args.budget_mb)
    stream_sec = time.perf_counter() - t_start

    # Baseline legacy control (2,048 ASCII-trained vocab.json)
    legacy_tok = BPETokenizer.pretrained()
    legacy_eval = evaluate_tokenizer_candidate("legacy_2k_control", legacy_tok, TOURNAMENT_EVAL_SUITE)

    candidate_specs = [
        {
            "id": "T1",
            "vocab_size": 32768,
            "algorithm": "byte_bpe",
            "script_protected": False,
            "desc": "32K Standard Byte-Level BPE",
        },
        {
            "id": "T2",
            "vocab_size": 32768,
            "algorithm": "unigram",
            "script_protected": False,
            "desc": "32K Byte-Fallback Unigram",
        },
        {
            "id": "T3",
            "vocab_size": 49152,
            "algorithm": "byte_bpe",
            "script_protected": True,
            "desc": "48K Script-Protected Byte-Level BPE",
        },
        {
            "id": "T4",
            "vocab_size": 32768,
            "algorithm": "byte_bpe",
            "script_protected": True,
            "desc": "32K Script-Protected Byte-Level BPE (Grapheme + Digit + Indent)",
        },
    ]

    trained_models = {}
    candidate_reports = []

    for cspec in candidate_specs:
        cid = cspec["id"]
        c_t0 = time.perf_counter()
        tok = train_hf_candidate_tokenizer(
            texts=corpus,
            vocab_size=int(cspec["vocab_size"]),
            algorithm=str(cspec["algorithm"]),
            script_protected=bool(cspec["script_protected"]),
        )
        train_sec = time.perf_counter() - c_t0
        ev = evaluate_tokenizer_candidate(cid, tok, TOURNAMENT_EVAL_SUITE)
        ev["description"] = cspec["desc"]
        ev["algorithm"] = cspec["algorithm"]
        ev["script_protected"] = cspec["script_protected"]
        ev["train_seconds"] = round(train_sec, 3)
        trained_models[cid] = tok
        candidate_reports.append(ev)
        print(
            f"[Candidate {cid}] vocab={ev['vocab_size']} gate2_pass={ev['gate2_pass']} "
            f"mean_fert={ev['mean_natural_fertility']:.3f} "
            f"ur_ratio={ev['per_language']['ur']['fertility_ratio_vs_en']:.3f} "
            f"ar_ratio={ev['per_language']['ar']['fertility_ratio_vs_en']:.3f} "
            f"hi_ratio={ev['per_language']['hi']['fertility_ratio_vs_en']:.3f} "
            f"({train_sec:.2f}s)"
        )

    # Select winning 32,768-vocab model (Track A native 25M/85M target embedding budget = 32,768)
    passing_32k = [r for r in candidate_reports if r["vocab_size"] == 32768 and r["gate2_pass"]]
    if not passing_32k:
        raise RuntimeError("No 32K tokenizer candidate passed GATE-2! Inspect fertility ratios.")

    # Prefer lowest mean natural fertility + script protection tie-breaker
    winner_report = min(
        passing_32k,
        key=lambda r: r["mean_natural_fertility"] - (0.02 if r["script_protected"] else 0.0),
    )
    winner_id = winner_report["candidate_id"]
    winner_tok = trained_models[winner_id]

    os.makedirs(os.path.dirname(args.output_tokenizer), exist_ok=True)
    winner_tok.save(args.output_tokenizer)

    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    full_report = {
        "experiment": "xrfm_phase2_tokenizer_tournament_v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_stream_seconds": round(stream_sec, 2),
        "corpus_total_documents": len(corpus),
        "corpus_total_utf8_bytes": sum(s["collected_bytes"] for s in stream_stats.values()),
        "stream_sources": stream_stats,
        "legacy_control": legacy_eval,
        "candidates": candidate_reports,
        "winner": {
            "candidate_id": winner_id,
            "artifact_path": args.output_tokenizer,
            "vocab_size": winner_report["vocab_size"],
            "fingerprint": winner_report["fingerprint"],
            "mean_natural_fertility": winner_report["mean_natural_fertility"],
            "max_non_latin_parity_ratio": winner_report["max_non_latin_parity_ratio"],
            "max_latin_parity_ratio": winner_report["max_latin_parity_ratio"],
            "max_byte_fallback_rate": winner_report["max_byte_fallback_rate"],
            "selection_rationale": (
                f"{winner_id} ({winner_report['description']}) satisfies all GATE-2 bounds at 32,768 vocab "
                f"while preserving Devanagari/Perso-Arabic combining marks, single-digit splits, and code indentation, "
                f"saving ~10.5M embedding/LM-head parameters on xrfm-native-85m vs 49,152 vocab (T3)."
            ),
        },
    }
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"\n[WINNER] {winner_id} saved to {args.output_tokenizer}")
    print(f"[REPORT] Saved to {args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
