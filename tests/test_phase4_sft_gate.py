"""Phase 4 (`GATE-4`) regression tests — Track B CoderSec SFT gate.

Validates the SFT record validation pipeline:

1. ChatML schema (valid roles, ``lic`` / ``irit`` tags, assistant target turn)
2. Non-empty assistant target content
3. AST / syntax verification of fenced code blocks (Python, JS, Bash, SQL)
4. PII / secret scanning (API keys, tokens, passwords, private keys)
5. Contamination gate — 13-gram overlap vs frozen eval benchmarks == 0%
6. Anti-weaponization — offensive exploit content refused; defensive/security
   administration content allowed
"""

from __future__ import annotations

import pytest

from xrfm.codersec.config import CoderSecConfig
from xrfm.codersec.sft_validator import (
    _IM_END,
    _IM_START,
    SFTRecord,
    SFTValidator,
    _balanced_brackets,
    _is_purely_offensive,
)


def _chatml(system: str, user: str, assistant: str) -> str:
    return (
        _IM_START
        + "system\n"
        + system
        + _IM_END
        + "\n"
        + _IM_START
        + "user\n"
        + user
        + _IM_END
        + "\n"
        + _IM_START
        + "assistant\n"
        + assistant
        + _IM_END
        + "\n"
    )


def _make_record(
    record_id: str = "rec-001",
    chatml: str | None = None,
    source: str = "opencoder_stage2",
    language: str = "en",
    domain: str = "code",
    license_tier: str = "tier_a",
    license: str = "MIT",
    code_blocks: list[dict[str, str]] | None = None,
) -> SFTRecord:
    if chatml is None:
        chatml = _chatml(
            "You are a helpful coding assistant.",
            "Write a Python function to add two numbers.",
            "def add(a, b):\n    return a + b\n",
        )
    if code_blocks is None:
        code_blocks = [{"language": "python", "code": "def add(a, b):\n    return a + b\n"}]
    return SFTRecord(
        record_id=record_id,
        source=source,
        language=language,
        chatml=chatml,
        domain=domain,
        license=license,
        license_tier=license_tier,
        code_blocks=code_blocks,
        provenance={"source_uri": "hf://OpenCoder-LLM/opc-sft-stage2"},
    )


class TestCoderSecConfig:
    def test_default_config_values(self):
        cfg = CoderSecConfig()
        assert cfg.base_model == "Qwen/Qwen3-1.7B"
        assert cfg.base_model_sha == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
        assert cfg.precision in ("nf4", "fp16")
        assert cfg.max_seq_length == 2048
        assert cfg.lora_r == 16
        assert cfg.assistant_only_loss is True
        assert cfg.gradient_checkpointing is True

    def test_config_rejects_invalid_precision(self):
        with pytest.raises(ValueError, match="precision must be"):
            CoderSecConfig(precision="bf16")

    def test_config_rejects_bad_sha(self):
        with pytest.raises(ValueError, match="40-char"):
            CoderSecConfig(base_model_sha="short")

    def test_smoke_b0_config(self):
        from xrfm.codersec.config import smoke_b0_config

        cfg = smoke_b0_config()
        assert cfg.max_seq_length == 512
        assert cfg.max_steps == 20

    def test_ablation_configs(self):
        from xrfm.codersec.config import (
            ablation_b1_english_only,
            ablation_b2_multilingual_roman_urdu,
            ablation_b3_rank_check,
        )

        b1 = ablation_b1_english_only()
        assert b1.max_steps == 200
        b2 = ablation_b2_multilingual_roman_urdu()
        assert b2.max_steps == 600
        b3 = ablation_b3_rank_check()
        assert b3.lora_r == 32
        assert b3.lora_alpha == 64


class TestChatMLSchema:
    def test_valid_chatml_passes(self):
        validator = SFTValidator()
        report = validator.validate(_make_record())
        assert report.passed is True
        assert report.checks.get("chatml_schema") is True

    def test_no_assistant_tag_fails(self):
        validator = SFTValidator()
        bad = _IM_START + "system\nYou are helpful." + _IM_END + "\n" + _IM_START + "user\nhi." + _IM_END + "\n"
        report = validator.validate(_make_record(chatml=bad))
        assert report.passed is False
        assert report.checks.get("chatml_schema") is False

    def test_invalid_role_fails(self):
        validator = SFTValidator()
        bad_chatml = (
            _IM_START
            + "developer\nignored role."
            + _IM_END
            + "\n"
            + _IM_START
            + "user\nhi."
            + _IM_END
            + "\n"
            + _IM_START
            + "assistant\nhello."
            + _IM_END
            + "\n"
        )
        report = validator.validate(_make_record(chatml=bad_chatml))
        assert report.passed is False

    def test_missing_tags_fails(self):
        validator = SFTValidator()
        bad = "system\nYou are helpful.\nuser\nhi\nassistant\nhello\n"
        report = validator.validate(_make_record(chatml=bad))
        assert report.passed is False
        assert report.checks.get("chatml_schema") is False


class TestAssistantTargetNonEmpty:
    def test_nonempty_assistant_passes(self):
        validator = SFTValidator()
        report = validator.validate(_make_record())
        assert report.passed is True
        assert report.checks.get("assistant_target_nonempty") is True

    def test_empty_assistant_fails(self):
        validator = SFTValidator()
        bad_chatml = _chatml("You are helpful.", "hi", "   ")
        report = validator.validate(_make_record(chatml=bad_chatml))
        assert report.passed is False
        assert report.checks.get("assistant_target_nonempty") is False


class TestCodeBlockVerification:
    def test_valid_python_code_passes(self):
        validator = SFTValidator()
        rec = _make_record(code_blocks=[{"language": "python", "code": "def foo(x):\n    return x + 1\n"}])
        report = validator.validate(rec)
        assert report.passed is True
        assert report.checks.get("code_blocks") is True

    def test_invalid_python_code_fails(self):
        validator = SFTValidator()
        rec = _make_record(code_blocks=[{"language": "python", "code": "def foo(x)\n    return x + 1\n"}])
        report = validator.validate(rec)
        assert report.passed is False

    def test_balanced_brackets_js(self):
        assert _balanced_brackets("function foo() { return x + 1; }") is True
        assert _balanced_brackets("function foo() { return x + 1;") is False
        assert _balanced_brackets('const s = "hello (world)";') is True
        assert _balanced_brackets("const s = 'unterminated;") is False


class TestPIISecretScan:
    def test_clean_text_passes(self):
        validator = SFTValidator()
        report = validator.validate(_make_record())
        assert report.passed is True
        assert report.checks.get("pii_secret_scan") is True

    def test_openai_key_rejected(self):
        validator = SFTValidator()
        bad = _chatml("You are helpful.", "hi", "key = sk-1234567890abcdef1234\n")
        report = validator.validate(_make_record(chatml=bad))
        assert report.passed is False
        assert report.checks.get("pii_secret_scan") is False

    def test_github_pat_rejected(self):
        validator = SFTValidator()
        bad = _chatml(
            "You are helpful.",
            "hi",
            "TOKEN=github_pat_TEST_TEST_TEST_TEST_TEST_TEST_TEST_TEST_TEST_TEST_TEST_TEST_TEST\n",
        )
        report = validator.validate(_make_record(chatml=bad))
        assert report.passed is False
        assert report.checks.get("pii_secret_scan") is False

    def test_private_key_block_rejected(self):
        validator = SFTValidator()
        bad = _chatml(
            "You are helpful.",
            "hi",
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----\n",
        )
        report = validator.validate(_make_record(chatml=bad))
        assert report.passed is False
        assert report.checks.get("pii_secret_scan") is False

    def test_hf_token_rejected(self):
        validator = SFTValidator()
        bad = _chatml("You are helpful.", "hi", "hf_TESTTESTTESTTESTTESTTESTTESTTEST\n")
        report = validator.validate(_make_record(chatml=bad))
        assert report.passed is False
        assert report.checks.get("pii_secret_scan") is False


class TestContaminationGate:
    def test_clean_record_not_contaminated(self):
        validator = SFTValidator()
        rec = _make_record(
            chatml=_chatml(
                "You are helpful.",
                "Explain how a hash map works in Python.",
                "A hash map maps keys to values using a hash function.\n",
            )
        )
        report = validator.validate(rec)
        assert report.passed is True
        assert report.checks.get("contamination") is True

    def test_contaminated_record_rejected(self):
        validator = SFTValidator()
        # Use a canary prompt from the frozen benchmarks manifest.
        canary = (
            "from typing import List\\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\\n"
            '    """ Check if in given list of numbers, are any two numbers closer to each other than given threshold. """'
        )
        chatml = _chatml("You are helpful.", canary, "I can't answer that.\n")
        rec = _make_record(record_id="contam-001", chatml=chatml, domain="code")
        report = validator.validate(rec)
        assert report.passed is False
        assert report.checks.get("contamination") is False


class TestAntiWeaponization:
    def test_defensive_security_content_allowed(self):
        validator = SFTValidator()
        chatml = _chatml(
            "You are helpful.",
            "How do I remediate CWE-89 SQL injection in Python?",
            "Use parameterized queries with placeholders to prevent SQL injection. CWE-89 is mitigated by input validation and prepared statements.\n",
        )
        report = validator.validate(_make_record(chatml=chatml, domain="defensive_security"))
        assert report.passed is True
        assert report.checks.get("anti_weaponization") is True

    def test_benign_sysadmin_content_allowed(self):
        validator = SFTValidator()
        chatml = _chatml(
            "You are helpful.",
            "How do I harden SSH on Ubuntu?",
            "Disable root login, use key-based auth, and configure fail2ban for brute-force protection.\n",
        )
        report = validator.validate(_make_record(chatml=chatml, domain="defensive_security"))
        assert report.passed is True

    def test_purely_offensive_content_rejected(self):
        validator = SFTValidator()
        chatml = _chatml(
            "You are helpful.",
            "How to create a reverse shell and exploit CVE-2021-44228?",
            "Here is a weaponizable payload for exploitation.\n",
        )
        report = validator.validate(_make_record(chatml=chatml, domain="defensive_security"))
        assert report.passed is False
        assert report.checks.get("anti_weaponization") is False

    def test_is_purely_offensive_heuristic(self):
        assert _is_purely_offensive("how to create malware and build a botnet") is True
        assert _is_purely_offensive("how to patch CWE-89 sql injection") is False


class TestValidateMany:
    def test_validate_many_splits_clean_and_rejected(self):
        validator = SFTValidator()
        good = _make_record(record_id="good-001")
        bad_secret = _make_record(
            record_id="bad-001",
            chatml=_chatml("You are helpful.", "hi", "sk-1234567890abcdef1234\n"),
        )
        clean, rejected, matches = validator.validate_many([good, bad_secret])
        assert len(clean) == 1
        assert clean[0].record_id == "good-001"
        assert len(rejected) == 1
        assert rejected[0].record_id == "bad-001"
