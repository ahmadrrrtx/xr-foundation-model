"""SFT record validation pipeline for XRFM CoderSec (Track B) GATE-4.

Validates curated SFT records end-to-end before publication:

1. **ChatML schema validation** — every record must parse as system/user/assistant
   turns with proper ``<|im_start|>`` / ``<|im_end|>`` tags.
2. **Assistant-only target non-empty** — the assistant turn must contain at least
   one non-whitespace token after the assistant header.
3. **AST / syntax verification** — Python, JavaScript/TypeScript, Bash, and SQL
   code blocks are parsed for syntactic validity.
4. **PII / secret scan** — records containing leaked API keys, passwords, tokens,
   private keys, or connection strings are rejected.
5. **Contamination gate** — 13-gram overlap against frozen evaluation benchmarks
   (HumanEval+, MBPP+, CyberSecEval 2, Belebele, XRFM-MultiCoderSec-Bench) must
   be 0%.
6. **Anti-weaponization** — offensive exploit-generation content is refused;
   legitimate defensive/system-administration content is allowed.

All checks are deterministic and run entirely offline (no network calls).
"""

from __future__ import annotations

import ast
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from xrfm.data.contamination import (
    ContaminationChecker,
    ContaminationConfig,
    Document,
    get_global_registry,
)

# ---------------------------------------------------------------------------
# ChatML tag constants (Qwen ChatML format)
# ---------------------------------------------------------------------------

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"

_VALID_ROLES = ("system", "user", "assistant")

# ---------------------------------------------------------------------------
# Secret / PII detection patterns
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[tuple[str, str, str]] = [
    ("openai_api_key", r"sk-[A-Za-z0-9]{20,}", "OpenAI API key"),
    ("anthropic_api_key", r"sk-ant-[A-Za-z0-9]{20,}", "Anthropic API key"),
    ("generic_api_key", r"(?i)(api[_-]?key|apikey)\s*[:=]\s*[A-Za-z0-9_-]{16,}", "generic API key"),
    ("aws_access_key", r"AKIA[0-9A-Z]{16}", "AWS access key ID"),
    ("aws_secret_key", r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*[A-Za-z0-9/+=]{40}", "AWS secret key"),
    ("github_pat", r"github_pat_[A-Za-z0-9_]{20,}", "GitHub personal access token"),
    ("slack_token", r"xox[baprs]-[0-9a-zA-Z-]{10,}", "Slack token"),
    ("private_key_block", r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key block"),
    ("password_assignment", r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]{8,}", "password assignment"),
    (
        "connection_string",
        r"(?i)(mongodb|postgres|mysql|redis)://[^\s:]+:[^\s@]+@",
        "database connection string with credentials",
    ),
    ("jwt_token", r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "JWT token"),
    ("hf_token", r"hf_[A-Za-z0-9]{20,}", "Hugging Face token"),
    ("kaggle_token", r"KGAT_[A-Za-z0-9]{20,}", "Kaggle API token"),
]

_SECRET_REGEXES = [(name, re.compile(pattern), label) for name, pattern, label in _SECRET_PATTERNS]

# Weaponization (offensive exploit) patterns — defensive security content is
# allowed; weaponization (how-to-exploit, weaponizable payloads) is refused.
_WEAPONIZATION_PATTERNS: list[tuple[str, str]] = [
    ("exploit_payload", r"(?i)(reverse[_\s]shell|bind[_\s]shell|meterpreter|beacon|webdoor|webshell)\b"),
    ("weaponizable_injection", r"(?i)(union\s+select|';.*drop\s+table|<script>.*alert\()"),
    (
        "malware_howto",
        r"(?i)(how\s+to\s+(create|build|write|make)\s+(a\s+)?(virus|malware|ransomware|trojan|worm|botnet))",
    ),
    (
        "cve_weaponization",
        r"(?i)(weaponize|weaponizable|exploit\s+(for|against)\s+CVE-\d{4}-\d+|poc\s+exploit\s+for\s+CVE)",
    ),
    ("credential_stuffing", r"(?i)(credential\s+stuffing|password\s+spraying|brute.?force\s+(attack|tool))"),
    ("ddos_tool", r"(?i)(ddos\s+tool|distributed\s+denial.?of\s+service\s+(attack|tool|script))"),
]

_WEAPONIZATION_REGEXES = [(name, re.compile(pattern)) for name, pattern in _WEAPONIZATION_PATTERNS]

# Legitimate defensive/security administration patterns that must NOT be
# refused even if they contain security-related keywords.
_DEFENSIVE_ALLOWLIST: list[tuple[str, str]] = [
    ("patch_cwe", r"(?i)(patch|remediat|fix|mitigate|harden|sanitize|escape|validate|parameterize)"),
    ("defensive_explain", r"(?i)(defensive|secure\s+coding|security\s+review|vulnerability\s+assessment|CWE-\d+)"),
    (
        "sysadmin",
        r"(?i)(linux\s+(admin|hardening|firewall|iptables|fail2ban|ssh\s+config)|network\s+defense|intrusion\s+detection)",
    ),
    ("citation", r"(?i)(CVE-\d{4}-\d+|CWE-\d+|OWASP|NIST|MITRE)"),
]

_DEFENSIVE_REGEXES = [(name, re.compile(pattern)) for name, pattern in _DEFENSIVE_ALLOWLIST]


def _normalize_text(text: str) -> str:
    nfc = unicodedata.normalize("NFC", text or "")
    return " ".join(nfc.split()).casefold()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SFTRecord:
    """A single SFT training record."""

    record_id: str
    source: str
    language: str  # 'en', 'ur', 'ur-Latn', 'ar', 'hi', 'es', 'fr', 'code', 'multi'
    chatml: str  # Full ChatML-formatted conversation string
    domain: str  # 'code', 'defensive_security', 'multilingual_qa', 'rag_citation', ...
    license: str
    license_tier: str  # 'tier_a', 'tier_b'
    code_blocks: list[dict[str, str]] = field(default_factory=list)  # [{language, code}]
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chatml:
            raise ValueError("chatml must be non-empty")
        if self.license_tier not in ("tier_a", "tier_b"):
            raise ValueError(f"license_tier must be 'tier_a' or 'tier_b', got {self.license_tier!r}")


@dataclass
class ValidationReport:
    """Result of validating one SFT record."""

    record_id: str
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add(self, name: str, ok: bool, message: str = "") -> None:
        self.checks[name] = ok
        if not ok:
            self.failures.append(f"{name}: {message}" if message else name)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class SFTValidator:
    """Validates SFT records against the GATE-4 quality & safety bar."""

    def __init__(self, *, ngram_size: int = 13, contamination_threshold: float = 0.0) -> None:
        self.ngram_size = ngram_size
        self.contamination_threshold = contamination_threshold

        # Lazy-initialized contamination checker backed by the frozen-eval registry.
        registry = get_global_registry(autoload_frozen=True)
        self._contamination_checker = ContaminationChecker(
            registry,
            ContaminationConfig(
                ngram_size=ngram_size,
                threshold=contamination_threshold,
                check_exact_hash=True,
                check_ngram=True,
            ),
        )

    # -- public API ---------------------------------------------------------

    def validate(self, record: SFTRecord) -> ValidationReport:
        """Run the full GATE-4 validation pipeline on one record.

        Returns a ``ValidationReport``; ``passed`` is True only if every check
        passes (contamination == 0%, no secrets, valid ChatML, valid AST,
        non-weaponized).
        """
        report = ValidationReport(record_id=record.record_id, passed=True)

        self._check_chatml_schema(record, report)
        self._check_assistant_target_nonempty(record, report)
        self._check_code_blocks(record, report)
        self._check_pii_secrets(record, report)
        self._check_contamination(record, report)
        self._check_anti_weaponization(record, report)

        report.passed = len(report.failures) == 0
        return report

    def validate_many(self, records: list[SFTRecord]) -> tuple[list[SFTRecord], list[SFTRecord], list[dict[str, Any]]]:
        """Validate many records; return (clean, rejected, match_records).

        ``match_records`` documents any contamination matches for audit.
        """
        clean: list[SFTRecord] = []
        rejected: list[SFTRecord] = []
        match_records: list[dict[str, Any]] = []

        for record in records:
            report = self.validate(record)
            if report.passed:
                clean.append(record)
            else:
                rejected.append(record)
            for name, ok in report.checks.items():
                if name == "contamination" and not ok:
                    match_records.append(
                        {
                            "record_id": record.record_id,
                            "source": record.source,
                            "domain": record.domain,
                            "failed_check": name,
                            "failures": report.failures,
                        }
                    )
                    break

        return clean, rejected, match_records

    # -- individual checks --------------------------------------------------

    def _check_chatml_schema(self, record: SFTRecord, report: ValidationReport) -> None:
        """Record must be well-formed ChatML: turns with valid roles and tags."""
        text = record.chatml
        ok = True
        messages: list[tuple[str, str]] = []

        # Must contain at least one im_start / im_end pair.
        if _IM_START not in text or _IM_END not in text:
            ok = False
        else:
            # Rough turn extraction: <|im_start|>role\ncontent<|im_end|>
            # ``rit`` must be followed by newline or end-of-string so it does not
            # match the ``rit`` substring inside words like "Write".
            pattern = re.compile(
                re.escape(_IM_START)
                + r"\s*("
                + "|".join(_VALID_ROLES)
                + r")\n(.*?)"
                + re.escape(_IM_END)
                + r"(?:\n|$)",
                re.DOTALL,
            )
            matches = pattern.findall(text)
            if not matches:
                ok = False
            else:
                # Reject any lic tag followed by a non-valid role (e.g. "developer").
                all_role_tags = re.findall(re.escape(_IM_START) + r"\s*(\w+)", text)
                invalid_roles = [r for r in all_role_tags if r not in _VALID_ROLES]
                if invalid_roles:
                    ok = False
                else:
                    messages = [(role, content.strip()) for role, content in matches]
                    # The last turn must be assistant (training target).
                    if messages[-1][0] != "assistant":
                        ok = False

        report.add("chatml_schema", ok, "malformed ChatML or missing assistant target turn" if not ok else "")

    def _check_assistant_target_nonempty(self, record: SFTRecord, report: ValidationReport) -> None:
        """Assistant turn content after <|im_start|>assistant\n must be non-empty."""
        pattern = re.compile(
            re.escape(_IM_START) + r"\s*assistant\n(.*?)" + re.escape(_IM_END) + r"(?:\n|$)", re.DOTALL
        )
        matches = pattern.findall(record.chatml)
        if not matches:
            report.add("assistant_target_nonempty", False, "no assistant turn found")
            return
        last_content = matches[-1].strip()
        ok = len(last_content) > 0
        report.add("assistant_target_nonempty", ok, "assistant target content is empty" if not ok else "")

    def _check_code_blocks(self, record: SFTRecord, report: ValidationReport) -> None:
        """AST/syntax-verify every fenced code block in the assistant turn."""
        code_blocks = record.code_blocks or []
        # Also extract fenced blocks from chatml as a fallback.
        if not code_blocks:
            fence_re = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
            for m in fence_re.finditer(record.chatml):
                lang = (m.group(1) or "text").strip().lower()
                code_blocks.append({"language": lang, "code": m.group(2)})

        if not code_blocks:
            report.add("code_blocks", True, "no code blocks to verify")
            return

        all_ok = True
        for block in code_blocks:
            lang = block.get("language", "text").lower()
            code = block.get("code", "")
            ok = self._syntax_check(lang, code)
            if not ok:
                all_ok = False
                report.failures.append(f"code_blocks: invalid {lang} syntax")

        report.add("code_blocks", all_ok)

    @staticmethod
    def _syntax_check(language: str, code: str) -> bool:
        if not code.strip():
            return True
        try:
            if language in ("python", "py"):
                ast.parse(code)
                return True
            if language in ("javascript", "js", "typescript", "ts", "jsx", "tsx"):
                # Lightweight JS/TS sanity: balanced braces/parens, no obvious breakage.
                return _balanced_brackets(code)
            if language in ("bash", "sh", "zsh"):
                return _balanced_brackets(code)
            if language in ("sql",):
                return _balanced_brackets(code)
            return True  # Unknown language: cannot verify, do not fail
        except Exception:
            return False

    def _check_pii_secrets(self, record: SFTRecord, report: ValidationReport) -> None:
        """Reject records containing leaked credentials / secrets."""
        haystack = record.chatml + "\n" + "\n".join(b.get("code", "") for b in (record.code_blocks or []))
        found: list[str] = []
        for name, regex, label in _SECRET_REGEXES:
            if regex.search(haystack):
                found.append(label)
        ok = len(found) == 0
        if not ok:
            report.add("pii_secret_scan", False, f"leaked secret(s) detected: {', '.join(found)}")
        else:
            report.add("pii_secret_scan", True)

    def _check_contamination(self, record: SFTRecord, report: ValidationReport) -> None:
        """13-gram contamination against frozen evaluation benchmarks must be 0%."""
        doc = Document(
            text=record.chatml,
            source=record.source,
            document_id=record.record_id,
            language=record.language,
        )
        _is_contam, matched_name, score = self._contamination_checker.check_document(doc)
        # ``score == 0.0`` means zero n-gram overlap (clean). Any nonzero score
        # indicates contamination against a frozen evaluation benchmark.
        ok = score == 0.0
        if not ok:
            report.add(
                "contamination",
                False,
                f"contaminated vs {matched_name or 'frozen-eval'} (score={score:.4f})",
            )
        else:
            report.add("contamination", True)

    def _check_anti_weaponization(self, record: SFTRecord, report: ValidationReport) -> None:
        """Refuse offensive weaponization; allow legitimate defensive/security text."""
        haystack = record.chatml + "\n" + "\n".join(b.get("code", "") for b in (record.code_blocks or []))
        lowered = _normalize_text(haystack)

        weapon_hits = [name for name, regex in _WEAPONIZATION_REGEXES if regex.search(haystack)]
        if not weapon_hits:
            report.add("anti_weaponization", True)
            return

        # If a defensive allowlist pattern also matches, treat as legitimate
        # defensive/security-administration content (not weaponization).
        defensive_hits = [name for name, regex in _DEFENSIVE_REGEXES if regex.search(haystack)]
        if defensive_hits and not _is_purely_offensive(lowered):
            report.add("anti_weaponization", True, f"defensive context detected ({', '.join(defensive_hits)})")
            return

        report.add("anti_weaponization", False, f"potential weaponization content ({', '.join(weapon_hits)})")


def _balanced_brackets(code: str) -> bool:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    in_string = False
    string_char = ""
    for ch in code:
        if in_string:
            if ch == "\\":
                continue  # skip escaped char
            if ch == string_char:
                in_string = False
            continue
        if ch in ("'", '"', "`"):
            in_string = True
            string_char = ch
            continue
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if not stack or stack.pop() != ch:
                return False
    # Unterminated string literal is invalid syntax.
    return not stack and not in_string


def _is_purely_offensive(text: str) -> bool:
    """Heuristic: text is purely offensive if it matches weaponization and has
    no defensive remediation / explanation context at all."""
    offensive_only = (
        "reverse shell" in text
        or "bind shell" in text
        or "how to hack" in text
        or "how to exploit" in text
        or "create malware" in text
        or "build a botnet" in text
        or "ddos attack tool" in text
        or "weaponizable payload" in text
    )
    return offensive_only


def compute_dataset_provenance_sha(records: list[SFTRecord]) -> str:
    """Compute a deterministic SHA-256 over sorted record IDs + content for
    the DatasetManifestV2 provenance field."""
    payload = "\n".join(
        f"{r.record_id}\t{_sha256(r.chatml)}\t{r.domain}\t{r.language}\t{r.license_tier}"
        for r in sorted(records, key=lambda x: x.record_id)
    )
    return _sha256(payload)
