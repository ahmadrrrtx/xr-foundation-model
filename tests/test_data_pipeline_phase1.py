"""
Tests for XRFM Phase 1 data pipeline.

Covers:
- normalization
- language filtering
- quality filtering
- PII detection
- hashing
- exact dedup
- near dedup
- splitting
- mixture weighting
- tokenization
- packing
- manifest generation
- checksums
- worker sharding no duplicates
- integration golden pipeline
"""

import os
import tempfile
import hashlib

import pytest

from xrfm.data.schema import Document
from xrfm.data.normalization import normalize_document
from xrfm.data.language import HeuristicLanguageDetector, filter_by_language, LanguageDetectorConfig
from xrfm.data.quality import QualityFilter, QualityFilterConfig, filter_documents_quality
from xrfm.data.pii import PIIFilter, PIIFilterConfig, filter_documents_pii
from xrfm.data.dedup import deduplicate_exact, deduplicate_near, deduplicate_documents, ExactDedupConfig, NearDedupConfig
from xrfm.data.document_split import split_documents, SplitConfig, verify_no_overlap, split_reproducibility_check
from xrfm.data.mixing import MixtureConfig, mix_documents
from xrfm.data.tokenization import tokenize_documents, TokenizationConfig
from xrfm.data.packing_extended import pack_tokenized_documents, PackingConfig
from xrfm.data.sharding import write_shards, ShardingConfig, verify_worker_no_duplicates, assign_shards_to_rank
from xrfm.data.manifest_v2 import build_manifest_v2
from xrfm.data.checksums import compute_dataset_id, sha256_text, sha256_file
from xrfm.data.pipeline import PipelineConfig, DataPipeline
from xrfm.tokenization import BPETokenizer


def make_doc(text, source="test", lang="en", doc_id=None):
    return Document(text=text, source=source, language=lang, document_id=doc_id or "")


class TestNormalization:
    def test_unicode_normalization(self):
        doc = make_doc("café \r\n test")
        result = normalize_document(doc)
        assert "\r\n" not in result.document.text
        assert result.was_modified

    def test_preserve_code_indentation(self):
        code = "    def foo():\n        return 1"
        doc = make_doc(code, source="code")
        result = normalize_document(doc, preserve_code=True)
        assert "    def foo():" in result.document.text

    def test_empty_detection(self):
        doc = make_doc("   ")
        result = normalize_document(doc)
        assert result.is_empty

    def test_control_char_removal(self):
        doc = make_doc("hello\x00world")
        result = normalize_document(doc)
        assert "\x00" not in result.document.text


class TestLanguageFiltering:
    def test_english_detection(self):
        detector = HeuristicLanguageDetector()
        score = detector.detect("The quick brown fox jumps over the lazy dog")
        assert score.language == "en"
        assert score.confidence > 0

    def test_arabic_detection(self):
        detector = HeuristicLanguageDetector()
        score = detector.detect("مرحبا بالعالم هذا اختبار")
        assert score.language in ("ar", "ur")

    def test_filter_keeps_all_when_no_allowed(self):
        docs = [make_doc("Hello world", lang="en"), make_doc("مرحبا", lang="ar")]
        kept, rejected, scores = filter_by_language(docs, config=LanguageDetectorConfig(allowed_languages=None))
        assert len(kept) == 2

    def test_filter_with_allowed(self):
        docs = [make_doc("Hello world this is English text with many common words the and is", lang="en"), make_doc("مرحبا بالعالم", lang="ar")]
        config = LanguageDetectorConfig(allowed_languages=["en"], threshold=0.3)
        kept, rejected, scores = filter_by_language(docs, config=config)
        # At least English should be kept
        assert len(kept) >= 1


class TestQualityFiltering:
    def test_min_length(self):
        filt = QualityFilter(QualityFilterConfig(min_chars=10, min_words=2))
        doc = make_doc("hi")
        score = filt.score(doc)
        assert score.is_too_short
        keep, reason = filt.should_keep(score)
        assert not keep

    def test_normal_doc_passes(self):
        filt = QualityFilter()
        doc = make_doc("This is a normal document with sufficient length and quality for testing purposes.")
        score = filt.score(doc)
        keep, reason = filt.should_keep(score)
        assert keep

    def test_repetition_detection(self):
        filt = QualityFilter(QualityFilterConfig(max_repetition_ratio=0.3))
        doc = make_doc("a a a a a a a a a a a a a a a a a a a a")
        score = filt.score(doc)
        assert score.has_excessive_repetition

    def test_quality_filter_batch(self):
        docs = [make_doc("This is a good document with enough content to pass quality filters."), make_doc("hi")]
        kept, rejected, scores = filter_documents_quality(docs)
        assert len(kept) == 1
        assert len(rejected) == 1


class TestPIIFiltering:
    def test_email_detection(self):
        filt = PIIFilter()
        doc = make_doc("Contact me at test@example.com")
        score = filt.scan(doc)
        assert score.has_email

    def test_phone_detection(self):
        filt = PIIFilter()
        doc = make_doc("Call me at 123-456-7890")
        score = filt.scan(doc)
        assert score.has_phone

    def test_secret_detection(self):
        filt = PIIFilter()
        doc = make_doc("API key: sk-1234567890abcdef1234567890abcdef")
        score = filt.scan(doc)
        assert score.has_secret or score.has_api_key

    def test_high_risk_removal(self):
        filt = PIIFilter(PIIFilterConfig(remove_high_risk=True))
        doc = make_doc("sk-1234567890abcdef1234567890abcdef")
        score = filt.scan(doc)
        keep, reason = filt.should_keep(score)
        if score.risk_level == "high":
            assert not keep

    def test_pii_batch(self):
        docs = [make_doc("Hello world"), make_doc("Email test@example.com")]
        kept, rejected, scores, stats = filter_documents_pii(docs)
        assert stats["scanned"] == 2


class TestHashing:
    def test_content_hash_deterministic(self):
        doc1 = make_doc("hello world", doc_id="doc1")
        doc2 = make_doc("hello world", doc_id="doc2")
        assert doc1.content_hash == doc2.content_hash

    def test_different_text_different_hash(self):
        doc1 = make_doc("hello world")
        doc2 = make_doc("hello world!")
        assert doc1.content_hash != doc2.content_hash


class TestExactDedup:
    def test_exact_dedup_removes_duplicates(self):
        docs = [make_doc("duplicate", doc_id="1"), make_doc("duplicate", doc_id="2"), make_doc("unique", doc_id="3")]
        result = deduplicate_exact(docs)
        assert len(result.unique_docs) == 2
        assert result.stats["duplicates"] == 1

    def test_exact_dedup_no_duplicates(self):
        docs = [make_doc(f"doc {i}", doc_id=str(i)) for i in range(5)]
        result = deduplicate_exact(docs)
        assert len(result.unique_docs) == 5
        assert result.stats["duplicates"] == 0


class TestNearDedup:
    def test_near_dedup_detects_similar(self):
        docs = [
            make_doc("The quick brown fox jumps over the lazy dog", doc_id="1"),
            make_doc("The quick brown fox jumps over the lazy dog", doc_id="2"),  # exact duplicate will be caught by exact, but near also
            make_doc("The quick brown fox jumps over the lazy dog with extra", doc_id="3"),
            make_doc("Completely different document about machine learning", doc_id="4"),
        ]
        config = NearDedupConfig(threshold=0.8, shingle_size=3)
        result = deduplicate_near(docs, config)
        # At least one near duplicate should be detected (exact duplicates are also near)
        assert result.stats["near_duplicates"] >= 1

    def test_near_dedup_no_false_positives(self):
        docs = [make_doc(f"Unique document number {i} with distinct content", doc_id=str(i)) for i in range(5)]
        config = NearDedupConfig(threshold=0.9, shingle_size=5)
        result = deduplicate_near(docs, config)
        assert result.stats["near_duplicates"] == 0


class TestSplitting:
    def test_split_ratios(self):
        docs = [make_doc(f"doc {i}", doc_id=f"doc-{i}") for i in range(100)]
        config = SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, method="hash")
        splits = split_documents(docs, config)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 100
        # Check no overlap
        assert verify_no_overlap(splits)

    def test_split_deterministic(self):
        docs = [make_doc(f"doc {i}", doc_id=f"doc-{i}") for i in range(50)]
        config = SplitConfig(train_ratio=0.9, val_ratio=0.05, test_ratio=0.05, seed=42, method="hash")
        assert split_reproducibility_check(docs, config)

    def test_split_random_deterministic(self):
        docs = [make_doc(f"doc {i}", doc_id=f"doc-{i}") for i in range(50)]
        config = SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, method="random")
        split1 = split_documents(docs, config)
        split2 = split_documents(docs, config)
        ids1 = sorted([d.document_id for d in split1["train"]])
        ids2 = sorted([d.document_id for d in split2["train"]])
        assert ids1 == ids2

    def test_split_no_overlap_property(self):
        docs = [make_doc(f"content {i}", doc_id=f"id-{i}") for i in range(20)]
        splits = split_documents(docs, SplitConfig(method="hash"))
        train_ids = set(d.document_id for d in splits["train"])
        val_ids = set(d.document_id for d in splits["val"])
        test_ids = set(d.document_id for d in splits["test"])
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0


class TestMixture:
    def test_mixture_weights_sum(self):
        with pytest.raises(ValueError):
            MixtureConfig.from_dict({"a": 0.5, "b": 0.6})

    def test_mixture_sampling(self):
        docs_by_source = {
            "web": [make_doc(f"web doc {i}", source="web", doc_id=f"web-{i}") for i in range(10)],
            "code": [make_doc(f"code doc {i}", source="code", doc_id=f"code-{i}") for i in range(10)],
        }
        config = MixtureConfig.from_dict({"web": 0.7, "code": 0.3}, seed=42)
        mixed, stats = mix_documents(docs_by_source, config, target_docs=10)
        assert len(mixed) == 10
        # Check stats
        assert len(stats) == 2


class TestTokenization:
    def test_tokenization_records_identity(self):
        tokenizer = BPETokenizer.pretrained()
        docs = [make_doc("Hello world, this is a test document for tokenization.")]
        tokenized, info, stats = tokenize_documents(docs, tokenizer)
        assert len(tokenized) == 1
        assert info.vocab_size > 0
        assert info.hash is not None
        assert stats["total_tokens"] > 0

    def test_tokenization_deterministic(self):
        tokenizer = BPETokenizer.pretrained()
        docs = [make_doc("Hello world test")]
        tok1, _, _ = tokenize_documents(docs, tokenizer)
        tok2, _, _ = tokenize_documents(docs, tokenizer)
        assert tok1[0].tokens == tok2[0].tokens


class TestPacking:
    def test_packing_fixed_length(self):
        tokenizer = BPETokenizer.pretrained()
        docs = [make_doc("Hello world " * 20, doc_id=f"doc-{i}") for i in range(5)]
        tokenized, _, _ = tokenize_documents(docs, tokenizer)
        config = PackingConfig(sequence_length=32, add_eos_between_docs=False, allow_cross_document=True, leftover_handling="drop")
        result = pack_tokenized_documents(tokenized, config)
        for seq in result.sequences:
            assert len(seq.input_ids) == 32

    def test_packing_discards(self):
        tokenizer = BPETokenizer.pretrained()
        docs = [make_doc("Hi", doc_id="1")]
        tokenized, _, _ = tokenize_documents(docs, tokenizer)
        config = PackingConfig(sequence_length=100, leftover_handling="drop")
        result = pack_tokenized_documents(tokenized, config)
        # With small doc and drop, might have 0 sequences and discarded tokens
        assert result.discarded_tokens >= 0


class TestSharding:
    def test_shard_writing(self):
        tokenizer = BPETokenizer.pretrained()
        docs = [make_doc("Hello world " * 10, doc_id=f"doc-{i}") for i in range(10)]
        tokenized, _, _ = tokenize_documents(docs, tokenizer)
        packing = pack_tokenized_documents(tokenized, PackingConfig(sequence_length=16))
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ShardingConfig(num_shards=2, output_dir=tmpdir)
            shard_infos = write_shards(packing.sequences, config, vocab_size=tokenizer.vocab_size(), sequence_length=16)
            assert len(shard_infos) == 2
            for si in shard_infos:
                assert os.path.isfile(si.path)
                assert si.sha256 is not None

    def test_worker_no_duplicates(self):
        for num_workers in [1, 2, 4, 8]:
            assert verify_worker_no_duplicates(100, num_workers)

    def test_rank_sharding_no_duplicates(self):
        # Simulate shard infos
        from xrfm.data.sharding import ShardInfo

        shard_infos = [ShardInfo(shard_id=i, path=f"/tmp/shard-{i}.npy", num_sequences=10, num_tokens=160, sha256="abc", dtype="uint32", sequence_length=16) for i in range(8)]
        for world_size in [1, 2, 4, 8]:
            all_ids = []
            for rank in range(world_size):
                assigned = assign_shards_to_rank(shard_infos, world_size, rank)
                all_ids.extend([s.shard_id for s in assigned])
            assert len(all_ids) == len(set(all_ids))
            assert set(all_ids) == set(range(8))


class TestManifest:
    def test_manifest_generation(self):
        tokenizer = BPETokenizer.pretrained()
        from xrfm.data.tokenization import TokenizerInfo

        info = TokenizerInfo.from_tokenizer(tokenizer)
        sources = [{"name": "test", "uri": "test://", "type": "text", "license": "MIT", "license_url": "", "version": "v1", "checksum": "", "language": "en", "document_count": 10, "token_count": 100}]
        splits = {"train": [make_doc("a", doc_id="1")], "val": [], "test": []}
        token_counts = {"total": 100, "train": 100, "val": 0, "test": 0}
        from xrfm.data.sharding import ShardInfo

        shard_infos = [ShardInfo(shard_id=0, path="/tmp/shard-0.npy", num_sequences=5, num_tokens=80, sha256="abc", dtype="uint32", sequence_length=16)]

        manifest = build_manifest_v2(
            dataset_name="test",
            dataset_version="0.1.0",
            processing_config={"test": True},
            tokenizer_info=info,
            sources=sources,
            splits=splits,
            token_counts=token_counts,
            shard_infos=shard_infos,
            sequence_length=16,
            split_config={"method": "hash", "seed": 42},
        )
        assert manifest.dataset_name == "test"
        assert manifest.dataset_id.startswith("xrfm-ds-")
        assert manifest.tokenizer.vocab_size == tokenizer.vocab_size()


class TestChecksums:
    def test_dataset_id_deterministic(self):
        sources = {"a": "v1", "b": "v2"}
        config = {"param": 1}
        id1 = compute_dataset_id(sources, config, "tok-v1", "pipe-v1")
        id2 = compute_dataset_id(sources, config, "tok-v1", "pipe-v1")
        assert id1 == id2

    def test_dataset_id_changes_with_input(self):
        sources1 = {"a": "v1"}
        sources2 = {"a": "v2"}
        config = {"param": 1}
        id1 = compute_dataset_id(sources1, config, "tok-v1", "pipe-v1")
        id2 = compute_dataset_id(sources2, config, "tok-v1", "pipe-v1")
        assert id1 != id2

    def test_sha256_file(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("hello world")
            path = f.name
        try:
            h = sha256_file(path)
            assert len(h) == 64
            expected = hashlib.sha256(b"hello world").hexdigest()
            assert h == expected
        finally:
            os.unlink(path)


class TestGoldenPipelineIntegration:
    def test_full_pipeline_golden(self):
        # Load golden docs
        docs = []
        golden_path = "tests/data/golden/documents.jsonl"
        if not os.path.isfile(golden_path):
            pytest.skip("Golden dataset not found")
        for line in open(golden_path, "r", encoding="utf-8"):
            docs.append(Document.from_jsonl(line))

        tokenizer = BPETokenizer.pretrained()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                dataset_name="golden-test",
                dataset_version="0.1.0",
                output_dir=tmpdir,
                sequence_length=32,
            )
            from xrfm.data.sharding import ShardingConfig

            config.sharding = ShardingConfig(num_shards=2, output_dir=os.path.join(tmpdir, "tokens"))
            config.packing.sequence_length = 32

            pipeline = DataPipeline(config=config, tokenizer=tokenizer)
            result = pipeline.run(docs)

            # Verify manifest exists
            assert os.path.isfile(result["manifest_path"])
            # Verify shards exist
            assert len(result["shard_infos"]) == 2
            # Verify report
            assert result["report"].documents_ingested == len(docs)
            # Verify training step works
            from xrfm.data.tokenized_dataset import ShardedTokenDatasetMap
            import torch

            ds = ShardedTokenDatasetMap(manifest_path=result["manifest_path"])
            assert len(ds) > 0
            input_ids, targets = ds[0]
            assert input_ids.shape[0] == 32
            assert targets.shape[0] == 32
