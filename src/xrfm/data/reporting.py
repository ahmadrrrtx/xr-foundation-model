"""
Dataset reporting for XRFM.

Every corpus build should automatically generate a report:
- documents ingested, rejected, deduplicated, retained
- languages, license distribution, source distribution
- average document length, token count, token distribution
- quality-score distribution, PII flags, near-duplicate counts
- train/val/test sizes, tokens per source, tokens per language

Generates both JSON and Markdown.
From actual pipeline output, not manually typed.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from xrfm.data.schema import Document, DocumentAttributes


@dataclass
class DatasetReport:
    dataset_name: str
    dataset_version: str
    created_at: str
    # Counts
    documents_ingested: int
    documents_rejected: int
    documents_deduplicated: int
    documents_retained: int
    # Distributions
    languages: Dict[str, int]
    license_distribution: Dict[str, int]
    source_distribution: Dict[str, int]
    # Lengths
    avg_document_length: float
    min_document_length: int
    max_document_length: int
    # Tokens
    token_count: int
    token_distribution: Dict[str, int]  # per source
    tokens_per_language: Dict[str, int]
    # Quality
    quality_score_distribution: Dict[str, int]  # buckets
    pii_flags: Dict[str, int]
    near_duplicate_count: int
    exact_duplicate_count: int
    # Splits
    train_size: int
    val_size: int
    test_size: int
    train_tokens: int
    val_tokens: int
    test_tokens: int
    # Shards
    num_shards: int
    storage_size_bytes: int
    # Timing
    processing_time_seconds: float
    # Extra
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        md = []
        md.append(f"# Dataset Report: {self.dataset_name} v{self.dataset_version}")
        md.append("")
        md.append(f"Created at: {self.created_at}")
        md.append("")
        md.append("## Summary")
        md.append(f"- Documents ingested: {self.documents_ingested}")
        md.append(f"- Documents rejected: {self.documents_rejected}")
        md.append(f"- Documents deduplicated: {self.documents_deduplicated}")
        md.append(f"- Documents retained: {self.documents_retained}")
        md.append(f"- Tokens: {self.token_count}")
        md.append(f"- Shards: {self.num_shards}")
        md.append("")
        md.append("## Splits")
        md.append(f"- Train: {self.train_size} docs, {self.train_tokens} tokens")
        md.append(f"- Val: {self.val_size} docs, {self.val_tokens} tokens")
        md.append(f"- Test: {self.test_size} docs, {self.test_tokens} tokens")
        md.append("")
        md.append("## Sources")
        for src, count in self.source_distribution.items():
            md.append(f"- {src}: {count} docs, {self.token_distribution.get(src, 0)} tokens")
        md.append("")
        md.append("## Languages")
        for lang, count in self.languages.items():
            md.append(f"- {lang}: {count} docs, {self.tokens_per_language.get(lang, 0)} tokens")
        md.append("")
        md.append("## Licenses")
        for lic, count in self.license_distribution.items():
            md.append(f"- {lic}: {count} docs")
        md.append("")
        md.append("## Quality")
        md.append(f"- Avg doc length: {self.avg_document_length:.1f} chars")
        md.append(f"- Min: {self.min_document_length}, Max: {self.max_document_length}")
        md.append(f"- Exact duplicates: {self.exact_duplicate_count}")
        md.append(f"- Near duplicates: {self.near_duplicate_count}")
        md.append(f"- Quality distribution: {self.quality_score_distribution}")
        md.append(f"- PII flags: {self.pii_flags}")
        md.append("")
        md.append("## Storage")
        md.append(f"- Storage size: {self.storage_size_bytes} bytes ({self.storage_size_bytes / 1e6:.2f} MB)")
        md.append(f"- Processing time: {self.processing_time_seconds:.2f}s")
        md.append("")
        if self.extra:
            md.append("## Extra")
            md.append(f"```json\n{json.dumps(self.extra, indent=2)}\n```")
        return "\n".join(md)

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "dataset_report.json")
        md_path = os.path.join(output_dir, "dataset_report.md")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())
        return json_path, md_path


def build_report_from_pipeline(
    dataset_name: str,
    dataset_version: str,
    ingested: List[Document],
    retained: List[Document],
    rejected_reasons: Dict[str, int],
    dedup_stats: Dict[str, int],
    token_counts: Dict[str, int],
    tokens_per_source: Dict[str, int],
    tokens_per_language: Dict[str, int],
    splits: Dict[str, List[Document]],
    split_token_counts: Dict[str, int],
    quality_scores: List,
    pii_stats: Dict[str, int],
    num_shards: int,
    storage_bytes: int,
    processing_time: float,
) -> DatasetReport:
    """Build report from pipeline outputs."""

    # Languages
    lang_dist: Dict[str, int] = {}
    for doc in retained:
        lang_dist[doc.language] = lang_dist.get(doc.language, 0) + 1

    # Licenses
    lic_dist: Dict[str, int] = {}
    for doc in retained:
        lic_dist[doc.license] = lic_dist.get(doc.license, 0) + 1

    # Sources
    src_dist: Dict[str, int] = {}
    for doc in retained:
        src_dist[doc.source] = src_dist.get(doc.source, 0) + 1

    # Lengths
    lengths = [len(d.text) for d in retained] if retained else [0]
    avg_len = sum(lengths) / max(len(lengths), 1)
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

    # Quality distribution (bucket overall_score)
    qual_dist: Dict[str, int] = {"0.0-0.3": 0, "0.3-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for qs in quality_scores:
        # qs may be QualityScore
        score = getattr(qs, "overall_score", 0.0)
        if score < 0.3:
            qual_dist["0.0-0.3"] += 1
        elif score < 0.6:
            qual_dist["0.3-0.6"] += 1
        elif score < 0.8:
            qual_dist["0.6-0.8"] += 1
        else:
            qual_dist["0.8-1.0"] += 1

    report = DatasetReport(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        documents_ingested=len(ingested),
        documents_rejected=sum(rejected_reasons.values()) if rejected_reasons else len(ingested) - len(retained),
        documents_deduplicated=dedup_stats.get("exact_duplicates", 0) + dedup_stats.get("near_duplicates", 0),
        documents_retained=len(retained),
        languages=lang_dist,
        license_distribution=lic_dist,
        source_distribution=src_dist,
        avg_document_length=avg_len,
        min_document_length=min_len,
        max_document_length=max_len,
        token_count=token_counts.get("total", 0),
        token_distribution=tokens_per_source,
        tokens_per_language=tokens_per_language,
        quality_score_distribution=qual_dist,
        pii_flags=pii_stats,
        near_duplicate_count=dedup_stats.get("near_duplicates", 0),
        exact_duplicate_count=dedup_stats.get("exact_duplicates", 0),
        train_size=len(splits.get("train", [])),
        val_size=len(splits.get("val", [])),
        test_size=len(splits.get("test", [])),
        train_tokens=split_token_counts.get("train", 0),
        val_tokens=split_token_counts.get("val", 0),
        test_tokens=split_token_counts.get("test", 0),
        num_shards=num_shards,
        storage_size_bytes=storage_bytes,
        processing_time_seconds=processing_time,
        extra={"rejected_reasons": rejected_reasons},
    )

    return report
