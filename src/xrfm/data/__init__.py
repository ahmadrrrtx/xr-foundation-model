"""
XRFM data subsystem: dataset representation, splitting, packing, manifests,
and full Phase 1 data pipeline.

Public surface:

Phase 0 (backward compat):
    from xrfm.data import TextDataset, DatasetConfig, split_dataset_lines, chunk_token_ids

Phase 1 (new):
    from xrfm.data import Document, DocumentAttributes
    from xrfm.data import DataPipeline, PipelineConfig
    from xrfm.data import SourceRegistry
    from xrfm.data import ShardedTokenDatasetMap, ShardedTokenDatasetIterable

Responsibilities are separated:
* schema          — canonical document representation
* normalization   — conservative text normalization
* language        — language identification and filtering
* quality         — modular quality filtering
* pii             — PII/safety heuristic filtering
* dedup           — exact + near deduplication
* document_split  — document-level deterministic splitting
* contamination   — evaluation contamination controls
* mixing          — explicit data mixture system
* tokenization    — tokenizer integration with provenance
* packing_extended— sequence packing
* sharding        — deterministic sharding for distributed training
* tokenized_dataset — training dataset from shards
* manifest_v2     — dataset manifest v2
* checksums       — checksum and dataset identity
* reporting       — dataset reporting
* pipeline        — end-to-end orchestration
* sources         — source registry
"""

from xrfm.config.schema import DatasetConfig
from xrfm.data.checksums import compute_dataset_id, sha256_file
from xrfm.data.contamination import ContaminationChecker, ContaminationConfig, ContaminationRegistry
from xrfm.data.dataset import IGNORE_INDEX, TextDataset, XRFMTextDataset, verify_text_file
from xrfm.data.dedup import (
    ExactDedupConfig,
    NearDedupConfig,
    deduplicate_documents,
    deduplicate_exact,
    deduplicate_near,
)
from xrfm.data.document_split import SplitConfig, split_documents
from xrfm.data.language import LanguageDetector, LanguageDetectorConfig, get_detector
from xrfm.data.manifest import (
    DatasetManifest,
    build_dataset_manifest,
    build_manifest,
    save_manifest,
)
from xrfm.data.manifest_v2 import DatasetManifestV2, build_manifest_v2
from xrfm.data.mixing import MixtureComponent, MixtureConfig, mix_documents
from xrfm.data.normalization import NormalizationResult, normalize_document
from xrfm.data.packing import PackingError, chunk_text, chunk_token_ids
from xrfm.data.packing_extended import PackedSequence, PackingConfig, pack_tokenized_documents
from xrfm.data.pii import PIIFilter, PIIFilterConfig
from xrfm.data.pipeline import DataPipeline, PipelineConfig
from xrfm.data.quality import QualityFilter, QualityFilterConfig
from xrfm.data.reporting import DatasetReport

# Phase 1 new exports
from xrfm.data.schema import (
    DeduplicationInfo,
    Document,
    DocumentAttributes,
    LanguageScore,
    PIIScore,
    ProcessingDecision,
    QualityScore,
)
from xrfm.data.sharding import ShardedTokenDataset, ShardInfo, ShardingConfig, assign_shards_to_rank, write_shards
from xrfm.data.sources import SourceConfig, SourceRegistry
from xrfm.data.splits import SplitRatiosError, normalize_text, split_dataset, split_dataset_lines
from xrfm.data.tokenization import TokenizationConfig, TokenizerInfo, tokenize_documents
from xrfm.data.tokenized_dataset import ShardedTokenDatasetIterable, ShardedTokenDatasetMap

__all__ = [
    # Phase 0 compat
    "DatasetConfig",
    "DatasetManifest",
    "IGNORE_INDEX",
    "PackingError",
    "SplitRatiosError",
    "TextDataset",
    "XRFMTextDataset",
    "build_dataset_manifest",
    "build_manifest",
    "chunk_text",
    "chunk_token_ids",
    "normalize_text",
    "save_manifest",
    "split_dataset",
    "split_dataset_lines",
    "verify_text_file",
    # Phase 1
    "Document",
    "DocumentAttributes",
    "LanguageScore",
    "QualityScore",
    "PIIScore",
    "DeduplicationInfo",
    "ProcessingDecision",
    "SourceConfig",
    "SourceRegistry",
    "NormalizationResult",
    "normalize_document",
    "LanguageDetectorConfig",
    "LanguageDetector",
    "get_detector",
    "QualityFilterConfig",
    "QualityFilter",
    "PIIFilterConfig",
    "PIIFilter",
    "ExactDedupConfig",
    "NearDedupConfig",
    "deduplicate_documents",
    "deduplicate_exact",
    "deduplicate_near",
    "SplitConfig",
    "split_documents",
    "ContaminationConfig",
    "ContaminationChecker",
    "ContaminationRegistry",
    "MixtureConfig",
    "MixtureComponent",
    "mix_documents",
    "TokenizationConfig",
    "TokenizerInfo",
    "tokenize_documents",
    "PackingConfig",
    "PackedSequence",
    "pack_tokenized_documents",
    "ShardingConfig",
    "ShardInfo",
    "write_shards",
    "ShardedTokenDataset",
    "ShardedTokenDatasetMap",
    "ShardedTokenDatasetIterable",
    "DatasetManifestV2",
    "build_manifest_v2",
    "compute_dataset_id",
    "sha256_file",
    "DatasetReport",
    "PipelineConfig",
    "DataPipeline",
    "assign_shards_to_rank",
]
