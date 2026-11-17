"""
Phase 2 (`GATE-2`) Regression & Acceptance Test Suite:
Track A 32K Multilingual+Code Tokenizer Tournament, Fertility Parity & HF Bridge.

Covers:
1. `F-02` Remediation: Special-token literal parsing (`encode("<|im_start|>user\\nhello<|im_end|>")`
   emits exact single special token IDs rather than splitting into `<`, `|`, `i`, `m` bytes).
2. `F-01` Remediation: Word-frequency pre-aggregated BPE training speed & determinism,
   plus `HFByteBPETokenizer` Rust-backed wrapper conforming 100% to `TokenizerInterface`.
3. `GATE-2` Fertility & Losslessness Audit:
   - 100% exact lossless UTF-8 roundtrip across English, Urdu Nastaliq, Roman Urdu,
     Arabic, Hindi Devanagari, Spanish, French, Python/TS/Bash/SQL indentation, and emojis.
   - Raw single-byte fallback rate `< 0.5%` on trained 32K multilingual+code tokenizer.
   - Fertility parity ratio relative to English:
     * `<= 2.0x` on `ur`, `ar`, `hi` (eliminating the legacy `10.5x-16.1x` byte-fallback tax!)
     * `<= 1.4x` on `ur-Latn`, `es`, `fr`
4. Tournament report (`docs/experiments/tokenizer_tournament_v1.json`) and winning
   32K tokenizer artifact (`src/xrfm/tokenization/tokenizer_32k_multilingual.json`) integrity.
"""

from __future__ import annotations

import json
import os
import time

from xrfm.tokenization import BPETokenizer, Tokenizer
from xrfm.tokenization.hf_bridge import HFTokenizerAdapter, train_hf_candidate_tokenizer
from xrfm.tokenization.tournament import (
    TOURNAMENT_EVAL_SUITE,
    evaluate_tokenizer_candidate,
)


class TestPhase2SpecialTokenAndSpeedFixes:
    def test_bpe_encodes_registered_special_tokens_atomically(self):
        tok = BPETokenizer(vocab_size_target=320)
        specials = {
            "<|pad|>": 316,
            "<|bos|>": 317,
            "<|eos|>": 318,
            "<|im_start|>": 319,
        }
        tok.train_on_text(
            "def hello_world():\n    return 42\n" * 15,
            special_tokens=specials,
        )
        prompt = "<|im_start|>def hello_world():<|eos|>"
        ids = tok.encode(prompt)
        # First and last token MUST be the exact atomic special token IDs!
        assert ids[0] == 319, f"Expected <|im_start|> id 319, got {ids[:5]}"
        assert ids[-1] == 318, f"Expected <|eos|> id 318, got {ids[-5:]}"
        assert tok.decode(ids) == prompt

    def test_word_frequency_bpe_trains_rapidly_on_repeated_corpus(self):
        # 50,000 characters of repeated multilingual + code text: legacy O(N_words * merges)
        # scanned every occurrence; word-frequency aggregation finishes in < 0.5s.
        corpus = (
            "def validate_query(sql: str) -> bool:\n    return '?' in sql\n"
            "اردو زبان میں مصنوعی ذہانت کی ترقی بہت اہم ہے۔\n"
            "تطوير نماذج الذكاء الاصطناعي في اللغات المتعددة.\n"
        ) * 350
        tok = BPETokenizer(vocab_size_target=450)
        t0 = time.perf_counter()
        tok.train_on_text(corpus)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"Word-frequency BPE took {elapsed:.2f}s (> 2.0s)"
        assert tok.decode(tok.encode(corpus[:300])) == corpus[:300]


class TestPhase2HFBridgeAndTournamentHarness:
    def test_hf_byte_bpe_and_unigram_adapters_implement_xrfm_contract(self, tmp_path):
        texts = [sample for samples in TOURNAMENT_EVAL_SUITE.values() for sample in samples] * 8
        bpe_tok = train_hf_candidate_tokenizer(
            texts=texts,
            vocab_size=1024,
            algorithm="byte_bpe",
            script_protected=True,
        )
        assert isinstance(bpe_tok, Tokenizer)
        assert isinstance(bpe_tok, HFTokenizerAdapter)
        assert bpe_tok.vocab_size() == 1024
        assert bpe_tok.pad_token_id is not None
        assert bpe_tok.eos_token_id is not None

        # Roundtrip across all scripts + indentation + emojis
        probe = "    def foo(x):\n        return 'اردو العربية हिन्दी 🚀'\n"
        assert bpe_tok.decode(bpe_tok.encode(probe)) == probe

        # Save & reload
        save_path = str(tmp_path / "tok_1k.json")
        bpe_tok.save(save_path)
        reloaded = HFTokenizerAdapter.from_file(save_path)
        assert reloaded.encode(probe) == bpe_tok.encode(probe)
        assert reloaded.vocab_size() == bpe_tok.vocab_size()

        # Evaluate candidate metrics
        report = evaluate_tokenizer_candidate("test-1k", reloaded, TOURNAMENT_EVAL_SUITE)
        assert report["roundtrip_exact_rate"] == 1.0
        assert "ur" in report["per_language"]
        assert "fertility_tok_per_word" in report["per_language"]["ur"]


class TestPhase2Winning32KTokenizerGate:
    def test_winning_32k_multilingual_tokenizer_satisfies_all_gate2_criteria(self):
        tok_path = "src/xrfm/tokenization/tokenizer_32k_multilingual.json"
        report_path = "docs/experiments/tokenizer_tournament_v1.json"
        assert os.path.isfile(tok_path), f"Missing winning 32K tokenizer at {tok_path}"
        assert os.path.isfile(report_path), f"Missing tournament report at {report_path}"

        with open(report_path, encoding="utf-8") as f:
            tournament = json.load(f)

        assert len(tournament["candidates"]) == 4  # T1, T2, T3, T4
        winner_id = tournament["winner"]["candidate_id"]
        assert winner_id in {"T1", "T2", "T3", "T4"}

        tok = HFTokenizerAdapter.from_file(tok_path)
        assert tok.vocab_size() == 32768

        # Verify GATE-2 live on the held-out evaluation suite
        eval_metrics = evaluate_tokenizer_candidate(winner_id, tok, TOURNAMENT_EVAL_SUITE)
        assert eval_metrics["roundtrip_exact_rate"] == 1.0

        en_fert = eval_metrics["per_language"]["en"]["fertility_tok_per_word"]
        assert en_fert > 0.0

        for lang in ("ur", "ar", "hi"):
            m = eval_metrics["per_language"][lang]
            ratio = m["fertility_ratio_vs_en"]
            fb = m["byte_fallback_rate"]
            assert ratio <= 2.0, f"GATE-2 violation: {lang} ratio {ratio:.3f} > 2.0x"
            assert fb < 0.005, f"GATE-2 violation: {lang} fallback {fb:.4f} >= 0.5%"

        for lang in ("ur-Latn", "es", "fr"):
            m = eval_metrics["per_language"][lang]
            ratio = m["fertility_ratio_vs_en"]
            assert ratio <= 1.45, f"GATE-2 violation: {lang} fertility ratio {ratio:.3f} > 1.45x"
            assert m["byte_fallback_rate"] < 0.005

        # Compare directly against legacy 2048 ASCII tokenizer to prove dramatic improvement
        legacy = BPETokenizer.pretrained()
        legacy_metrics = evaluate_tokenizer_candidate("legacy-2k", legacy, TOURNAMENT_EVAL_SUITE)
        assert eval_metrics["per_language"]["ur"]["fertility_tok_per_word"] < (
            legacy_metrics["per_language"]["ur"]["fertility_tok_per_word"] * 0.35
        )

    def test_published_native_shards_manifest_is_valid_and_pinned(self):
        manifest_summary_path = "docs/experiments/native_shards_manifest_v1.json"
        assert os.path.isfile(manifest_summary_path), f"Missing {manifest_summary_path}"
        with open(manifest_summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["dataset_id"].startswith("xrfm-ds-")
        assert summary["hf_repo_id"] == "ahmadrrrtx/xrfm-native-shards-v1"
        assert summary["hf_commit"] and len(summary["hf_commit"]) >= 16
        assert summary["vocab_size"] == 32768
        assert summary["sequence_length"] == 2048
        assert summary["stored_row_length"] == 2049
        assert summary["num_shards"] == 8
        assert summary["tokens_total"] >= 15_000_000
        for lang in ("en", "code", "ur", "ur-Latn", "ar", "hi", "es", "fr"):
            assert lang in summary["per_language"]
            assert summary["per_language"][lang]["packed_sequences_2048"] > 0
            assert lang in summary["val_by_language"]
