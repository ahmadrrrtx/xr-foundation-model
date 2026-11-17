"""
Phase 1 (`GATE-1`) Regression & Acceptance Test Suite.

Covers all Phase 1 audit findings and GATE-1 exit criteria:
1. P1-02: Strict SPDX license tier enforcement & track isolation in SourceRegistry
   (blocking unknown/varies, eval_only, rag_only, and hold items from training).
2. P1-03 (`F-13`): 7-language (`en`, `ur`, `ur-Latn`, `ar`, `hi`, `es`, `fr`) + `code`
   language identification with >= 95% macro-F1, >= 98% Urdu vs Arabic precision,
   and >= 90% Roman-Urdu vs English discrimination.
3. P1-04 (`F-14`): Banded MinHash-LSH near-deduplication scaling linearly O(N)
   with explicit `num_bands` / `rows_per_band` configuration.
4. P1-04 (`F-03`, `F-15`, `F-16`):
   - `TokenizerInfo.from_tokenizer` uses content-addressed `tokenizer_fingerprint()`
   - `PackingConfig(include_next_token=True)` + `ShardedTokenDatasetMap` produce
     lossless causal LM `(input_ids, targets)` without `-100` boundary token waste
   - `PipelineConfig.from_dict()` hydrates all nested YAML stages in CLI
   - Streaming bounded-memory ingestion (`StreamingPipelineRunner`)
5. P1-05 (`GATE-1`): Frozen benchmark manifest (`configs/eval/frozen_benchmarks_v1.json`)
   and seed `XRFM-MultiCoderSec` benchmark (`configs/eval/multicodersec_seed_v1.jsonl`)
   wired into `ContaminationRegistry` & `DataPipeline.check_contamination`, catching
   100% of synthetic benchmark leaks across all 7 languages + code.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from xrfm.data.contamination import (
    ContaminationChecker,
    ContaminationRegistry,
    load_frozen_benchmarks_into_registry,
)
from xrfm.data.dedup import NearDedupConfig, deduplicate_near
from xrfm.data.language import HeuristicLanguageDetector, LanguageDetectorConfig, filter_by_language
from xrfm.data.packing_extended import PackingConfig, pack_tokenized_documents
from xrfm.data.pipeline import DataPipeline, PipelineConfig
from xrfm.data.schema import Document
from xrfm.data.sharding import ShardingConfig, write_shards
from xrfm.data.sources import LicenseComplianceError, SourceConfig, SourceRegistry
from xrfm.data.tokenization import TokenizerInfo, tokenize_documents
from xrfm.data.tokenized_dataset import ShardedTokenDatasetMap
from xrfm.tokenization import BPETokenizer

# ---------------------------------------------------------------------------
# Representative 8-class LID evaluation benchmark (7 target languages + code)
# ---------------------------------------------------------------------------
LID_BENCHMARK: list[tuple[str, str]] = [
    # English (en)
    (
        "en",
        "Distributed transformer training requires careful numerical stability across mixed precision operations and attention layers.",
    ),
    (
        "en",
        "When authenticating users in a web application, always verify cryptographic signatures before trusting session cookies.",
    ),
    (
        "en",
        "The software engineering team reviewed the pull request and verified that every unit test passed without regressions.",
    ),
    (
        "en",
        "Database query parameterization prevents SQL injection vulnerabilities by separating user data from executable syntax.",
    ),
    # Urdu Nastaliq / Perso-Arabic (ur) — includes ٹ ڈ ڑ ں ہ ے گ چ پ
    (
        "ur",
        "مصنوعی ذہانت اور کمپیوٹر سائنس کے میدان میں اردو زبان کی ترقی کے لیے معیاری ڈیٹا اور محفوظ سافٹ ویئر بہت ضروری ہے۔",
    ),
    (
        "ur",
        "ویب ایپلیکیشن میں صارفین کا ڈیٹا محفوظ رکھنے کے لیے پاس ورڈ کو ہمیشہ مضبوط طریقے سے ہیش کرنا چاہیے تاکہ کوئی حملہ نہ ہو سکے۔",
    ),
    (
        "ur",
        "پائتھون پروگرامنگ سیکھنا بہت آسان ہے اور اس کے ذریعے آپ بہترین سسٹمز اور جدید الگورتھمز تیار کر سکتے ہیں۔",
    ),
    (
        "ur",
        "ہم نے اس پروجیکٹ میں تمام ٹیسٹ مکمل کر لیے ہیں اور اب یہ ماڈل بہترین کارکردگی دکھا رہا ہے۔",
    ),
    # Roman Urdu (ur-Latn)
    (
        "ur-Latn",
        "Agar aap apni web application ko mehfooz banana chahte hain to user input ko hamesha validate aur sanitize zaroor karein.",
    ),
    (
        "ur-Latn",
        "Hum ne is model ki training mukammal kar li hai aur ab yeh Urdu aur English dono mein bohat acha kaam kar raha hai.",
    ),
    (
        "ur-Latn",
        "Kya aap mujhe bata sakte hain ke Python mein database connection kis tarah banaya jata hai taake koi masla na ho?",
    ),
    (
        "ur-Latn",
        "Yeh software bohat tezi se chalta hai lekin is mein security ke kuch ahem masail hain jinhein hal karna zaroori hai.",
    ),
    # Arabic (ar)
    (
        "ar",
        "تعتبر معالجة اللغات الطبيعية وتطوير نماذج الذكاء الاصطناعي من أهم المجالات الحديثة في علوم الحاسوب وأمن المعلومات.",
    ),
    (
        "ar",
        "يجب التحقق من جميع المدخلات في تطبيقات الويب باستخدام الاستعلامات المعيارية لمنع ثغرات حقن قواعد البيانات بشكل كامل.",
    ),
    (
        "ar",
        "قام فريق الهندسة البرمجية بمراجعة الشفرة المصدرية والتأكد من نجاح جميع الاختبارات قبل نشر التحديث الجديد للمستخدمين.",
    ),
    (
        "ar",
        "تساعد خوارزميات التشفير الحديثة على حماية البيانات الحساسة أثناء النقل والتخزين في الأنظمة السحابية الموزعة.",
    ),
    # Hindi Devanagari (hi)
    (
        "hi",
        "कृत्रिम बुद्धिमत्ता और सुरक्षित सॉफ़्टवेयर विकास के लिए उच्च गुणवत्ता वाले बहुभाषी डेटासेट का निर्माण अत्यंत महत्वपूर्ण है।",
    ),
    (
        "hi",
        "वेब एप्लिकेशन में उपयोगकर्ता के डेटा की सुरक्षा के लिए हमेशा पैरामीटराइज्ड क्वेरी और मजबूत प्रमाणीकरण का उपयोग करना चाहिए।",
    ),
    (
        "hi",
        "हमारी इंजीनियरिंग टीम ने सभी परीक्षण सफलतापूर्वक पूरे कर लिए हैं और अब यह प्रणाली उत्पादन के लिए पूरी तरह तैयार है।",
    ),
    (
        "hi",
        "पायथन प्रोग्रामिंग भाषा में डेटा संरचनाओं और एल्गोरिदम को समझना हर सॉफ़्टवेयर डेवलपर के लिए बहुत उपयोगी होता है।",
    ),
    # Spanish (es)
    (
        "es",
        "Para proteger las aplicaciones web contra vulnerabilidades de inyección, siempre debemos utilizar consultas parametrizadas y validar las entradas.",
    ),
    (
        "es",
        "El equipo de ingeniería desarrolló una arquitectura distribuida que permite entrenar modelos de lenguaje con alta eficiencia y precisión.",
    ),
    (
        "es",
        "La seguridad informática requiere una revisión constante del código fuente para detectar errores antes del despliegue en producción.",
    ),
    (
        "es",
        "Los sistemas modernos de inteligencia artificial pueden comprender múltiples idiomas cuando los datos están bien equilibrados.",
    ),
    # French (fr)
    (
        "fr",
        "Pour sécuriser une application web contre les injections SQL, il est indispensable d'utiliser des requêtes préparées et de valider les entrées.",
    ),
    (
        "fr",
        "L'équipe d'ingénierie logicielle a vérifié toutes les fonctionnalités du système distribué avant la mise en production finale.",
    ),
    (
        "fr",
        "Les modèles de langage multilingues nécessitent une tokenisation équilibrée afin de réduire le coût de calcul sur chaque langue.",
    ),
    (
        "fr",
        "La cryptographie moderne permet de protéger la confidentialité et l'intégrité des données sensibles dans les réseaux informatiques.",
    ),
    # Source Code (code)
    (
        "code",
        "def verify_token(token: str, secret: bytes) -> dict:\n    if not hmac.compare_digest(sig, expected):\n        raise ValueError('invalid')\n    return payload",
    ),
    (
        "code",
        "export async function fetchUser(db: Pool, id: number): Promise<User | null> {\n  const res = await db.query('SELECT * FROM users WHERE id = $1', [id]);\n  return res.rows[0] ?? null;\n}",
    ),
    (
        "code",
        '#!/usr/bin/env bash\nset -euo pipefail\nfor shard in data/shards/*.npy; do\n  sha256sum "$shard" >> checksums.txt\ndone',
    ),
    (
        "code",
        "SELECT u.id, u.email, COUNT(o.id) AS order_count\nFROM users u\nLEFT JOIN orders o ON o.user_id = u.id\nWHERE u.is_active = TRUE\nGROUP BY u.id, u.email;",
    ),
]


class TestPhase1GovernanceRegistry:
    def test_rejects_unknown_or_copyleft_for_training(self):
        reg = SourceRegistry()
        reg.register(
            SourceConfig(
                name="bad-unknown",
                type="web",
                uri="hf://bad/unknown",
                license="unknown",
                license_tier="hold",
                allowed_uses=["hold"],
                enabled=True,
            )
        )
        with pytest.raises(LicenseComplianceError):
            reg.validate_for_training(track="track_a")

    def test_blocks_rag_only_and_eval_only_from_weight_training(self):
        reg = SourceRegistry()
        reg.register(
            SourceConfig(
                name="owasp-rag",
                type="security",
                uri="https://owasp.org/cheatsheets",
                license="CC-BY-SA-4.0",
                license_tier="rag_only",
                allowed_uses=["rag_retrieval"],
                revision="a1b2c3d4",
                enabled=True,
            )
        )
        reg.register(
            SourceConfig(
                name="belebele-eval",
                type="eval",
                uri="hf://facebook/belebele",
                license="CC-BY-NC-4.0",
                license_tier="eval_only",
                allowed_uses=["evaluation"],
                revision="e5f6a7b8",
                enabled=True,
            )
        )
        # Training source selection must never return rag_only or eval_only sources
        trainable = reg.list_trainable_sources(track="track_a")
        assert trainable == []
        # Explicit assertion on direct training attempt must raise LicenseComplianceError
        with pytest.raises(LicenseComplianceError):
            reg.assert_source_trainable("owasp-rag", track="track_a")
        with pytest.raises(LicenseComplianceError):
            reg.assert_source_trainable("belebele-eval", track="track_b")

    def test_production_sources_yaml_enforces_dual_track_governance(self):
        reg = SourceRegistry.from_yaml("configs/data/sources.yaml")
        # Must validate cleanly for both Track A and Track B trainable subsets
        track_a = reg.validate_for_training(track="track_a")
        track_b = reg.validate_for_training(track="track_b")
        assert len(track_a) >= 4
        assert len(track_b) >= 4
        # Every trainable source must have a pinned SPDX license, non-empty revision, and tier_a/tier_b
        for src in track_a + track_b:
            assert src.license not in ("unknown", "varies", "")
            assert src.license_tier in ("tier_a", "tier_b")
            assert "train_track_a" in src.allowed_uses or "train_track_b" in src.allowed_uses
            assert src.revision not in ("", "latest", "unknown")


class TestPhase1LanguageIdentification:
    def test_urdu_beats_arabic_with_98_percent_accuracy(self):
        detector = HeuristicLanguageDetector()
        ur_samples = [text for lang, text in LID_BENCHMARK if lang == "ur"]
        ar_samples = [text for lang, text in LID_BENCHMARK if lang == "ar"]

        for text in ur_samples:
            score = detector.detect(text)
            assert score.language == "ur", f"Expected 'ur', got {score.language!r} for: {text[:40]}"
            assert score.all_scores.get("ur", 0.0) > score.all_scores.get("ar", 0.0)

        for text in ar_samples:
            score = detector.detect(text)
            assert score.language == "ar", f"Expected 'ar', got {score.language!r} for: {text[:40]}"

    def test_roman_urdu_distinguished_from_english(self):
        detector = HeuristicLanguageDetector()
        for lang, text in LID_BENCHMARK:
            if lang == "ur-Latn":
                score = detector.detect(text)
                assert score.language == "ur-Latn", f"Expected 'ur-Latn', got {score.language!r} for: {text[:50]}"

    def test_eight_class_macro_f1_at_least_95_percent(self):
        detector = HeuristicLanguageDetector()
        classes = ["en", "ur", "ur-Latn", "ar", "hi", "es", "fr", "code"]
        tp = {c: 0 for c in classes}
        fp = {c: 0 for c in classes}
        fn = {c: 0 for c in classes}

        for expected, text in LID_BENCHMARK:
            pred = detector.detect(text).language
            if pred == expected:
                tp[expected] += 1
            else:
                fn[expected] += 1
                if pred in fp:
                    fp[pred] += 1

        f1_scores: list[float] = []
        for c in classes:
            prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
            rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            f1_scores.append(f1)

        macro_f1 = sum(f1_scores) / len(f1_scores)
        assert macro_f1 >= 0.95, f"Macro-F1 {macro_f1:.3f} < 0.95; per-class={dict(zip(classes, f1_scores))}"

    def test_filter_keeps_all_seven_target_languages_and_code(self):
        allowed = ["en", "ur", "ur-Latn", "ar", "hi", "es", "fr", "code"]
        docs = [Document(text=t, source="bench", document_id=f"d-{i}") for i, (_, t) in enumerate(LID_BENCHMARK)]
        kept, rejected, _ = filter_by_language(
            docs,
            config=LanguageDetectorConfig(allowed_languages=allowed, threshold=0.35),
        )
        assert len(kept) >= int(0.95 * len(docs))


class TestPhase1LSHAndPipelineUpgrades:
    def test_lsh_banded_minhash_subquadratic_candidate_pruning(self):
        # Create 250 genuinely distinct documents (seeded pseudo-random vocabulary) + 10 near-duplicates
        import hashlib

        docs: list[Document] = []
        for i in range(250):
            tokens = [hashlib.sha256(f"doc-{i}-word-{w}".encode()).hexdigest()[:8] for w in range(24)]
            docs.append(
                Document(
                    text=" ".join(tokens),
                    source="synth",
                    document_id=f"uniq-{i}",
                )
            )
        for j in range(10):
            # Exact / 96% overlap near-duplicate of uniq-{j}
            base_tokens = docs[j].text.split()
            base_tokens[-1] = "modifiedtail"
            docs.append(
                Document(
                    text=" ".join(base_tokens),
                    source="synth",
                    document_id=f"dup-{j}",
                )
            )

        cfg = NearDedupConfig(threshold=0.8, num_perm=128, shingle_size=5, num_bands=16)
        res = deduplicate_near(docs, cfg)
        assert res.stats["near_duplicates"] == 10
        assert len(res.unique_docs) == 250
        # LSH must inspect drastically fewer candidate pairs than O(N^2) = 260*259/2 = 33,670
        assert "candidate_pairs_checked" in res.stats
        assert res.stats["candidate_pairs_checked"] < 500

    def test_tokenizer_info_uses_content_fingerprint_for_distinct_vocabs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fa = os.path.join(tmpdir, "a.txt")
            fb = os.path.join(tmpdir, "b.txt")
            with open(fa, "w", encoding="utf-8") as f:
                f.write(("alpha beta alpha beta alpha gamma\n") * 20)
            with open(fb, "w", encoding="utf-8") as f:
                f.write(("delta omega delta omega delta sigma\n") * 20)
            tok_a = BPETokenizer(vocab_size_target=280)
            tok_a.train(fa)
            tok_b = BPETokenizer(vocab_size_target=280)
            tok_b.train(fb)
            assert tok_a.vocab_size() == tok_b.vocab_size()
            info_a = TokenizerInfo.from_tokenizer(tok_a)
            info_b = TokenizerInfo.from_tokenizer(tok_b)
            assert info_a.hash != info_b.hash

    def test_lossless_next_token_packing_eliminates_minus_100_boundary_waste(self):
        tokenizer = BPETokenizer.pretrained()
        docs = [
            Document(text="Hello world secure coding test " * 25, source="test", document_id=f"d-{i}") for i in range(4)
        ]
        tok_docs, tok_info, _ = tokenize_documents(docs, tokenizer)
        pack_cfg = PackingConfig(
            sequence_length=32,
            include_next_token=True,
            add_eos_between_docs=True,
            eos_token_id=tokenizer.eos_token_id,
            leftover_handling="drop",
        )
        packed = pack_tokenized_documents(tok_docs, pack_cfg)
        assert len(packed.sequences) > 0
        # Each stored row has length L + 1 = 33 so all 32 targets are real next tokens!
        assert len(packed.sequences[0].input_ids) == 33

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_cfg = ShardingConfig(num_shards=1, output_dir=tmpdir)
            shard_infos = write_shards(
                packed.sequences,
                shard_cfg,
                vocab_size=tokenizer.vocab_size(),
                sequence_length=33,
            )
            from xrfm.data.manifest_v2 import build_manifest_v2

            manifest = build_manifest_v2(
                dataset_name="lplus1-test",
                dataset_version="1.0.0",
                processing_config={"include_next_token": True},
                tokenizer_info=tok_info,
                sources=[],
                splits={"train": docs, "val": [], "test": []},
                token_counts={"total": packed.total_tokens, "train": packed.total_tokens, "val": 0, "test": 0},
                shard_infos=shard_infos,
                sequence_length=32,
                split_config={"method": "hash"},
            )
            manifest_path = os.path.join(tmpdir, "manifest.json")
            manifest.save(manifest_path)

            ds = ShardedTokenDatasetMap(manifest_path=manifest_path)
            x, y = ds[0]
            assert x.shape == (32,)
            assert y.shape == (32,)
            # Zero wasted -100 tokens at sequence boundary!
            assert int((y == -100).sum().item()) == 0
            assert int(y[-1].item()) == packed.sequences[0].input_ids[32]

    def test_pipeline_config_from_dict_hydrates_all_nested_yaml_sections(self):
        import yaml

        with open("configs/data/pretrain.yaml", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cfg = PipelineConfig.from_dict(raw)
        assert cfg.sequence_length == raw["sequence_length"]
        assert cfg.quality.min_chars == raw["quality"]["min_chars"]
        assert cfg.near_dedup.threshold == raw["near_dedup"]["threshold"]
        assert cfg.sharding.num_shards == raw["sharding"]["num_shards"]
        for lang in ("en", "ur", "ur-Latn", "ar", "hi", "es", "fr", "code"):
            assert lang in (cfg.language.allowed_languages or [])

    def test_streaming_chunk_runner_processes_bounded_batches(self):
        import hashlib

        from xrfm.data.streaming import StreamingFilterState, StreamingIngestConfig, iter_bounded_document_chunks

        def doc_gen():
            for i in range(25):
                words = [hashlib.sha256(f"stream-{i}-{w}".encode()).hexdigest()[:7] for w in range(20)]
                yield Document(
                    text=" ".join(words),
                    source="stream",
                    document_id=f"s-{i}",
                )

        chunks = list(iter_bounded_document_chunks(doc_gen(), StreamingIngestConfig(chunk_size=10, max_documents=23)))
        assert [len(c) for c in chunks] == [10, 10, 3]

        state = StreamingFilterState(NearDedupConfig(threshold=0.85, num_perm=128, num_bands=16))
        retained_all = []
        for ch in chunks:
            kept, stats = state.process_chunk(ch)
            retained_all.extend(kept)
        assert len(retained_all) == 23
        assert stats["ingested_total"] == 23
        assert stats["retained_total"] == 23


class TestPhase1FrozenEvalAndContamination:
    def test_frozen_benchmarks_manifest_and_seed_multicodersec_exist_and_catch_100_percent_leaks(self):
        assert os.path.isfile("configs/eval/frozen_benchmarks_v1.json")
        assert os.path.isfile("configs/eval/multicodersec_seed_v1.jsonl")

        with open("configs/eval/multicodersec_seed_v1.jsonl", encoding="utf-8") as f:
            seed_items = [json.loads(line) for line in f if line.strip()]

        # Must cover all 7 target natural languages across secure coding / CWE prompts
        seed_langs = {item["language"] for item in seed_items}
        for required_lang in ("en", "ur", "ur-Latn", "ar", "hi", "es", "fr"):
            assert required_lang in seed_langs, f"Missing {required_lang} in multicodersec_seed_v1.jsonl"
        assert len(seed_items) >= 21

        reg = ContaminationRegistry()
        summary = load_frozen_benchmarks_into_registry(
            reg,
            manifest_path="configs/eval/frozen_benchmarks_v1.json",
        )
        assert summary["total_protected_items"] >= len(seed_items)

        checker = ContaminationChecker(reg, ngram_size=8, overlap_threshold=0.5)

        # Inject 7 synthetic leaks (1 per language, including whitespace/case perturbations) + 5 clean docs
        leaked_docs = []
        for lang in ("en", "ur", "ur-Latn", "ar", "hi", "es", "fr"):
            sample = next(it for it in seed_items if it["language"] == lang)
            perturbed = "   " + sample["prompt"].replace("\n", "  ") + "   "
            leaked_docs.append(Document(text=perturbed, source="web_scrape", document_id=f"leak-{lang}"))

        clean_docs = [
            Document(
                text=f"Completely unrelated tutorial {i} about building static CSS layouts and responsive typography grids.",
                source="clean",
                document_id=f"clean-{i}",
            )
            for i in range(5)
        ]

        clean_kept, contaminated, _ = checker.filter_contaminated(leaked_docs + clean_docs)
        assert len(contaminated) == 7, f"Expected all 7 synthetic leaks caught, got {len(contaminated)}"
        assert len(clean_kept) == 5

    def test_data_pipeline_check_contamination_uses_frozen_registry_by_default(self):
        with open("configs/eval/multicodersec_seed_v1.jsonl", encoding="utf-8") as f:
            first_seed = json.loads(f.readline())

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = PipelineConfig(
                dataset_name="contam-gate-test",
                output_dir=tmpdir,
                check_contamination=True,
                contamination_manifest_path="configs/eval/frozen_benchmarks_v1.json",
            )
            pipe = DataPipeline(config=cfg, tokenizer=BPETokenizer.pretrained())
            docs = [
                Document(text=first_seed["prompt"], source="untrusted", document_id="evil-leak"),
                Document(
                    text="A clean software engineering note explaining binary search trees and balanced rotations.",
                    source="clean",
                    document_id="good-doc",
                ),
            ]
            clean, info = pipe.check_contamination(docs)
            assert len(info["contaminated"]) == 1
            assert len(clean) == 1
            assert clean[0].document_id == "good-doc"
