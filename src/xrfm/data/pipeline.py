"""
End-to-end data pipeline orchestration for XRFM.

Flow:
raw sources
  -> source registry
  -> download/ingestion
  -> document normalization
  -> metadata + provenance
  -> language filtering
  -> quality filtering
  -> PII / safety filtering
  -> exact deduplication
  -> near-duplicate detection
  -> document-level split / contamination controls
  -> dataset mixing
  -> tokenization
  -> sequence packing
  -> sharded training dataset
  -> dataset manifest
  -> reproducible training input

Each stage resumable via manifests/checksums.

Memory safety: streaming where appropriate, avoid loading all docs into RAM for large corpora
(but for Phase 1 golden dataset we can hold in memory).

Deterministic and traceable.
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from xrfm.data.schema import Document
from xrfm.data.sources import SourceRegistry, SourceConfig
from xrfm.data.normalization import normalize_document
from xrfm.data.language import LanguageDetectorConfig, get_detector, filter_by_language
from xrfm.data.quality import QualityFilterConfig, filter_documents_quality
from xrfm.data.pii import PIIFilterConfig, filter_documents_pii
from xrfm.data.dedup import ExactDedupConfig, NearDedupConfig, deduplicate_documents
from xrfm.data.document_split import SplitConfig, split_documents
from xrfm.data.contamination import ContaminationRegistry, ContaminationConfig
from xrfm.data.mixing import MixtureConfig, mix_documents
from xrfm.data.tokenization import TokenizationConfig, tokenize_documents
from xrfm.data.packing_extended import PackingConfig, pack_tokenized_documents
from xrfm.data.sharding import ShardingConfig, write_shards
from xrfm.data.manifest_v2 import build_manifest_v2
from xrfm.data.reporting import build_report_from_pipeline
from xrfm.data.checksums import sha256_file


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    dataset_name: str = "xrfm-pretrain"
    dataset_version: str = "0.1.0"
    output_dir: str = "processed/xrfm-pretrain-v0.1"
    # Stage configs
    language: LanguageDetectorConfig = field(default_factory=LanguageDetectorConfig)
    quality: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    pii: PIIFilterConfig = field(default_factory=PIIFilterConfig)
    exact_dedup: ExactDedupConfig = field(default_factory=ExactDedupConfig)
    near_dedup: NearDedupConfig = field(default_factory=NearDedupConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    mixture: Optional[MixtureConfig] = None
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    packing: PackingConfig = field(default_factory=PackingConfig)
    sharding: ShardingConfig = field(default_factory=ShardingConfig)
    contamination: ContaminationConfig = field(default_factory=ContaminationConfig)
    # Pipeline options
    seed: int = 42
    pipeline_version: str = "xrfm-pipeline-v1"
    # Whether to do near dedup (can be expensive)
    do_near_dedup: bool = True
    do_exact_dedup: bool = True
    # Sequence length
    sequence_length: int = 2048

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        # Custom serialization for nested dataclasses
        def serialize(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            if isinstance(obj, list):
                return [serialize(x) for x in obj]
            return obj

        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "output_dir": self.output_dir,
            "language": asdict(self.language),
            "quality": asdict(self.quality),
            "pii": asdict(self.pii),
            "exact_dedup": asdict(self.exact_dedup),
            "near_dedup": asdict(self.near_dedup),
            "split": asdict(self.split),
            "mixture": asdict(self.mixture) if self.mixture else None,
            "tokenization": asdict(self.tokenization),
            "packing": asdict(self.packing),
            "sharding": asdict(self.sharding),
            "contamination": asdict(self.contamination),
            "seed": self.seed,
            "pipeline_version": self.pipeline_version,
            "do_near_dedup": self.do_near_dedup,
            "do_exact_dedup": self.do_exact_dedup,
            "sequence_length": self.sequence_length,
        }


class DataPipeline:
    """Orchestrates the full data pipeline."""

    def __init__(self, config: PipelineConfig, tokenizer=None, source_registry: Optional[SourceRegistry] = None):
        self.config = config
        self.tokenizer = tokenizer
        self.source_registry = source_registry or SourceRegistry()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Stage output dirs
        self.documents_dir = self.output_dir / "documents"
        self.attributes_dir = self.output_dir / "attributes"
        self.tokens_dir = self.output_dir / "tokens"
        self.manifests_dir = self.output_dir / "manifests"
        self.reports_dir = self.output_dir / "reports"

        for d in [self.documents_dir, self.attributes_dir, self.tokens_dir, self.manifests_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Timing
        self._start_time = time.time()

    def _log(self, msg: str):
        print(f"[XRFM Pipeline] {msg}")

    def ingest(self, raw_docs: List[Document]) -> List[Document]:
        """Ingestion stage: for Phase 1, raw_docs passed directly. Future: download from registry."""
        self._log(f"Ingested {len(raw_docs)} raw documents")
        # Save raw docs for provenance
        raw_path = self.documents_dir / "raw.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for doc in raw_docs:
                f.write(doc.to_jsonl() + "\n")
        return raw_docs

    def normalize(self, docs: List[Document]) -> List[Document]:
        normalized: List[Document] = []
        modifications = 0
        for doc in docs:
            result = normalize_document(doc)
            if result.is_empty or result.is_malformed:
                # Keep track but filter later? For now keep if not empty
                if not result.is_empty:
                    normalized.append(result.document)
            else:
                normalized.append(result.document)
            if result.was_modified:
                modifications += 1
        self._log(f"Normalization: {len(normalized)}/{len(docs)} kept, {modifications} modified")
        return normalized

    def filter_language(self, docs: List[Document]) -> Tuple[List[Document], Dict[str, Any]]:
        detector = get_detector(self.config.language)
        kept, rejected, scores = filter_by_language(docs, detector, self.config.language)
        self._log(f"Language filtering: {len(kept)}/{len(docs)} kept, {len(rejected)} rejected")
        return kept, {"rejected": rejected, "scores": scores}

    def filter_quality(self, docs: List[Document]) -> Tuple[List[Document], Dict[str, Any]]:
        kept, rejected, scores = filter_documents_quality(docs, self.config.quality)
        self._log(f"Quality filtering: {len(kept)}/{len(docs)} kept, {len(rejected)} rejected")
        return kept, {"rejected": rejected, "scores": scores}

    def filter_pii(self, docs: List[Document]) -> Tuple[List[Document], Dict[str, Any]]:
        kept, rejected, scores, stats = filter_documents_pii(docs, self.config.pii)
        self._log(f"PII filtering: {len(kept)}/{len(docs)} kept, {len(rejected)} removed, stats={stats}")
        return kept, {"rejected": rejected, "scores": scores, "stats": stats}

    def dedup(self, docs: List[Document]) -> Tuple[List[Document], Dict[str, Any]]:
        unique, stats, details = deduplicate_documents(
            docs,
            exact_config=self.config.exact_dedup,
            near_config=self.config.near_dedup,
            do_exact=self.config.do_exact_dedup,
            do_near=self.config.do_near_dedup,
        )
        self._log(f"Dedup: input={stats.get('input')} exact_dup={stats.get('exact_duplicates')} near_dup={stats.get('near_duplicates')} final={stats.get('final_unique')}")
        return unique, {"stats": stats, "details": details}

    def check_contamination(self, docs: List[Document]) -> Tuple[List[Document], Dict[str, Any]]:
        # For Phase 1, we use empty protected registry unless populated
        registry = ContaminationRegistry()
        # In future, load protected datasets from config
        checker = registry.checker(self.config.contamination)
        clean, contaminated = checker.check_documents(docs)
        self._log(f"Contamination: {len(clean)}/{len(docs)} clean, {len(contaminated)} contaminated")
        return clean, {"contaminated": contaminated}

    def split(self, docs: List[Document]) -> Dict[str, List[Document]]:
        splits = split_documents(docs, self.config.split)
        self._log(f"Split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
        return splits

    def mix(self, docs_by_source: Dict[str, List[Document]]) -> Tuple[List[Document], List]:
        if not self.config.mixture:
            # No mixing config, return concatenated
            all_docs = []
            for src_docs in docs_by_source.values():
                all_docs.extend(src_docs)
            return all_docs, []

        mixed, stats = mix_documents(docs_by_source, self.config.mixture, tokenizer=self.tokenizer)
        self._log(f"Mixing: {len(mixed)} docs mixed from {len(docs_by_source)} sources")
        return mixed, stats

    def tokenize(self, docs: List[Document]):
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for tokenization stage")
        tokenized, info, stats = tokenize_documents(docs, self.tokenizer, self.config.tokenization)
        self._log(f"Tokenization: {stats['total_documents']} docs -> {stats['total_tokens']} tokens")
        return tokenized, info, stats

    def pack(self, tokenized_docs):
        # Set eos token id from tokenizer if available
        if self.config.packing.eos_token_id is None and self.tokenizer:
            eos = getattr(self.tokenizer, "eos_token_id", None)
            self.config.packing.eos_token_id = eos
        if self.config.packing.pad_token_id == 0 and self.tokenizer:
            pad = getattr(self.tokenizer, "pad_token_id", None)
            if pad is not None:
                self.config.packing.pad_token_id = pad

        self.config.packing.sequence_length = self.config.sequence_length

        result = pack_tokenized_documents(tokenized_docs, self.config.packing)
        self._log(f"Packing: {result.total_tokens} tokens -> {result.total_sequences} sequences, discarded={result.discarded_tokens} padded={result.padded_tokens}")
        return result

    def shard(self, packing_result):
        # Update sharding config output dir
        self.config.sharding.output_dir = str(self.tokens_dir)
        vocab_size = self.tokenizer.vocab_size() if self.tokenizer else 50304
        shard_infos = write_shards(
            packing_result.sequences,
            self.config.sharding,
            vocab_size=vocab_size,
            sequence_length=self.config.sequence_length,
        )
        self._log(f"Sharding: {len(shard_infos)} shards written to {self.tokens_dir}")
        return shard_infos

    def build_manifest(
        self,
        sources_info: List[Dict[str, Any]],
        splits: Dict[str, List[Document]],
        token_counts: Dict[str, int],
        shard_infos,
        tokenizer_info,
        packing_result,
    ):
        # Compute token counts per split if not already
        # For simplicity, we have total tokens from packing
        # Split token counts: we need to tokenize splits separately? For Phase 1, approximate

        # Checksums
        checksums = {}
        for si in shard_infos:
            checksums[si.path] = si.sha256

        manifest = build_manifest_v2(
            dataset_name=self.config.dataset_name,
            dataset_version=self.config.dataset_version,
            processing_config=self.config.to_dict(),
            tokenizer_info=tokenizer_info,
            sources=sources_info,
            splits=splits,
            token_counts=token_counts,
            shard_infos=shard_infos,
            sequence_length=self.config.sequence_length,
            split_config=self.config.split.__dict__ if hasattr(self.config.split, "__dict__") else {},
            checksums=checksums,
            pipeline_version=self.config.pipeline_version,
        )

        manifest_path = self.manifests_dir / f"{self.config.dataset_name}-v{self.config.dataset_version}-manifest.json"
        manifest.save(str(manifest_path))
        self._log(f"Manifest saved to {manifest_path}, dataset_id={manifest.dataset_id}")
        return manifest, manifest_path

    def build_report(
        self,
        ingested: List[Document],
        retained: List[Document],
        rejected_reasons: Dict[str, int],
        dedup_stats: Dict[str, int],
        token_counts: Dict[str, int],
        tokens_per_source: Dict[str, int],
        tokens_per_language: Dict[str, int],
        splits: Dict[str, List[Document]],
        split_token_counts: Dict[str, int],
        quality_scores,
        pii_stats: Dict[str, int],
        num_shards: int,
        storage_bytes: int,
    ):
        processing_time = time.time() - self._start_time

        report = build_report_from_pipeline(
            dataset_name=self.config.dataset_name,
            dataset_version=self.config.dataset_version,
            ingested=ingested,
            retained=retained,
            rejected_reasons=rejected_reasons,
            dedup_stats=dedup_stats,
            token_counts=token_counts,
            tokens_per_source=tokens_per_source,
            tokens_per_language=tokens_per_language,
            splits=splits,
            split_token_counts=split_token_counts,
            quality_scores=quality_scores,
            pii_stats=pii_stats,
            num_shards=num_shards,
            storage_bytes=storage_bytes,
            processing_time=processing_time,
        )

        json_path, md_path = report.save(str(self.reports_dir))
        self._log(f"Report saved to {json_path} and {md_path}")
        return report

    def run(self, raw_docs: List[Document]) -> Dict[str, Any]:
        """Run full pipeline end-to-end."""

        # Track rejected reasons
        rejected_reasons: Dict[str, int] = {}

        # Stage 1: Ingest
        docs = self.ingest(raw_docs)
        ingested = list(docs)

        # Stage 2: Normalize
        docs = self.normalize(docs)

        # Stage 3: Language
        docs, lang_info = self.filter_language(docs)
        for _, score in lang_info.get("rejected", []):
            key = f"language_{score.language}"
            rejected_reasons[key] = rejected_reasons.get(key, 0) + 1

        # Stage 4: Quality
        docs, qual_info = self.filter_quality(docs)
        for _, _, reason in qual_info.get("rejected", []):
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

        # Stage 5: PII
        docs, pii_info = self.filter_pii(docs)
        for _, _, reason in pii_info.get("rejected", []):
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

        # Stage 6: Dedup
        docs, dedup_info = self.dedup(docs)
        dedup_stats = dedup_info["stats"]

        # Stage 7: Contamination
        docs, contam_info = self.check_contamination(docs)
        if contam_info["contaminated"]:
            rejected_reasons["contaminated"] = len(contam_info["contaminated"])

        # Stage 8: Split (for reporting, but we also need to keep splits)
        splits = self.split(docs)

        # For tokenization, we use train split only? Or all? For Phase 1, we tokenize train for training
        # But manifest should include all splits. We'll tokenize all retained docs for total count,
        # and separately tokenize splits for per-split token counts.

        # Mixing: if configured, mix before split? In spec, mixing after split? We do mixing before split for simplicity,
        # or mix train only. For Phase 1, if mixture configured, we apply to docs before split.
        # Here we already split, so we will tokenize each split.

        # Tokenization per split
        if self.tokenizer is None:
            raise ValueError("Tokenizer required")

        all_tokenized, tokenizer_info, all_token_stats = self.tokenize(docs)

        # Tokenize splits for per-split counts
        split_token_counts: Dict[str, int] = {}
        split_tokenized: Dict[str, List] = {}
        for split_name, split_docs in splits.items():
            if split_docs:
                tok_docs, _, stats = self.tokenize(split_docs)
                split_token_counts[split_name] = stats["total_tokens"]
                split_tokenized[split_name] = tok_docs
            else:
                split_token_counts[split_name] = 0
                split_tokenized[split_name] = []

        # For packing and sharding, we use train split by default
        train_tokenized = split_tokenized.get("train", all_tokenized)

        packing_result = self.pack(train_tokenized)

        shard_infos = self.shard(packing_result)

        # Compute storage bytes
        storage_bytes = sum(os.path.getsize(si.path) for si in shard_infos if os.path.exists(si.path))

        # Sources info for manifest
        sources_info = []
        # Group by source
        from collections import Counter

        src_counter = Counter(d.source for d in docs)
        for src_name, count in src_counter.items():
            # Lookup in registry
            src_cfg = self.source_registry.get(src_name) if self.source_registry else None
            sources_info.append(
                {
                    "name": src_name,
                    "uri": src_cfg.uri if src_cfg else "",
                    "type": src_cfg.type if src_cfg else "text",
                    "license": src_cfg.license if src_cfg else "unknown",
                    "license_url": src_cfg.license_url if src_cfg else "",
                    "version": src_cfg.version if src_cfg else "v1",
                    "checksum": src_cfg.checksum if src_cfg else "",
                    "language": src_cfg.language if src_cfg else "en",
                    "document_count": count,
                    "token_count": packing_result.tokens_per_source.get(src_name, 0),
                }
            )

        token_counts = {
            "total": all_token_stats["total_tokens"],
            "train": split_token_counts.get("train", 0),
            "val": split_token_counts.get("val", 0),
            "test": split_token_counts.get("test", 0),
        }

        # Manifest
        manifest, manifest_path = self.build_manifest(
            sources_info=sources_info,
            splits=splits,
            token_counts=token_counts,
            shard_infos=shard_infos,
            tokenizer_info=tokenizer_info,
            packing_result=packing_result,
        )

        # Report
        # Tokens per source / language
        tokens_per_source = packing_result.tokens_per_source
        tokens_per_language: Dict[str, int] = {}
        for doc in docs:
            # Approximate: use avg tokens per doc * count per language? Better: use tokenized docs
            pass
        # For simplicity, use source distribution as language proxy, but compute from tokenized
        # Let's compute tokens per language from all_tokenized + docs
        lang_token_map: Dict[str, int] = {}
        # Map doc_id -> language
        doc_lang_map = {d.document_id: d.language for d in docs}
        for td in all_tokenized:
            lang = doc_lang_map.get(td.document_id, "unknown")
            lang_token_map[lang] = lang_token_map.get(lang, 0) + td.token_count
        tokens_per_language = lang_token_map

        report = self.build_report(
            ingested=ingested,
            retained=docs,
            rejected_reasons=rejected_reasons,
            dedup_stats=dedup_stats,
            token_counts=token_counts,
            tokens_per_source=tokens_per_source,
            tokens_per_language=tokens_per_language,
            splits=splits,
            split_token_counts=split_token_counts,
            quality_scores=qual_info.get("scores", []),
            pii_stats=pii_info.get("stats", {}),
            num_shards=len(shard_infos),
            storage_bytes=storage_bytes,
        )

        return {
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "report": report,
            "shard_infos": shard_infos,
            "splits": splits,
            "packing_result": packing_result,
            "tokenizer_info": tokenizer_info,
        }
