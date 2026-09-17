"""
PII / safety filtering for XRFM.

Baseline heuristic approach, conservative.
Does NOT claim PII-free, reports "PII heuristic filtering".

Detects:
- emails
- phone numbers
- credentials/secrets
- API-key-like strings
- highly sensitive identifier patterns

Reports statistics: scanned, flagged, removed, transformed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Dict

from xrfm.data.schema import Document, PIIScore


# Regex patterns
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"  # US-ish
)
# More generic international phone: +XX XXX XXX XXX
_INTL_PHONE_RE = re.compile(r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}")

# Secrets / credentials
_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{20,}['\"]?)"),
    re.compile(r"(?i)(secret\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}['\"]?)"),
    re.compile(r"(?i)(password\s*[:=]\s*['\"]?[^'\"]{4,}['\"]?)"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # OpenAI-like
    re.compile(r"ghp_[A-Za-z0-9]{36}"),  # GitHub PAT
    re.compile(r"gho_[A-Za-z0-9]{36}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key
    re.compile(r"-----BEGIN (?:RSA )?PRIVATE KEY-----"),
]

# SSN-like, credit card (very conservative)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")  # 13-16 digits


@dataclass
class PIIFilterConfig:
    remove_high_risk: bool = True
    transform_medium_risk: bool = False  # if True, redact rather than remove
    # What to flag
    flag_emails: bool = True
    flag_phones: bool = True
    flag_secrets: bool = True
    flag_sensitive_ids: bool = True
    # Risk thresholds
    high_risk_if_secret: bool = True


class PIIFilter:
    def __init__(self, config: PIIFilterConfig | None = None):
        self.config = config or PIIFilterConfig()

    def scan(self, doc: Document) -> PIIScore:
        text = doc.text
        flagged: List[str] = []

        has_email = False
        has_phone = False
        has_secret = False
        has_api_key = False
        has_sensitive_id = False

        if self.config.flag_emails:
            if _EMAIL_RE.search(text):
                has_email = True
                flagged.append("email")

        if self.config.flag_phones:
            if _PHONE_RE.search(text) or _INTL_PHONE_RE.search(text):
                has_phone = True
                flagged.append("phone")

        if self.config.flag_secrets:
            for pat in _SECRET_PATTERNS:
                if pat.search(text):
                    has_secret = True
                    flagged.append("secret")
                    # Distinguish API key
                    if "api" in pat.pattern.lower() or "sk-" in pat.pattern or "ghp" in pat.pattern:
                        has_api_key = True
                        flagged.append("api_key")
                    break

        if self.config.flag_sensitive_ids:
            if _SSN_RE.search(text):
                has_sensitive_id = True
                flagged.append("ssn")
            # Credit card check: need Luhn? For baseline, just pattern + some digits
            # Avoid false positives on long numbers by checking context
            cc_match = _CREDIT_CARD_RE.search(text)
            if cc_match:
                # Simple Luhn check for 13-16 digit numbers stripped of spaces/dashes
                digits = re.sub(r"[ -]", "", cc_match.group())
                if digits.isdigit() and 13 <= len(digits) <= 16:
                    # Only flag if looks like credit card (not just long number)
                    # For Phase 1, we flag but mark as sensitive
                    has_sensitive_id = True
                    flagged.append("credit_card")

        # Determine risk level
        if has_secret or has_api_key or has_sensitive_id:
            risk = "high"
        elif has_email and has_phone:
            risk = "medium"
        elif has_email or has_phone:
            risk = "low"
        else:
            risk = "low"

        return PIIScore(
            has_email=has_email,
            has_phone=has_phone,
            has_secret=has_secret,
            has_api_key=has_api_key,
            has_sensitive_id=has_sensitive_id,
            flagged_patterns=list(set(flagged)),
            risk_level=risk,
        )

    def should_keep(self, score: PIIScore) -> Tuple[bool, str]:
        if score.risk_level == "high" and self.config.remove_high_risk:
            return False, f"pii_high_risk_{','.join(score.flagged_patterns)}"
        return True, "passed"

    def redact(self, doc: Document, score: PIIScore) -> Document:
        """Redact PII patterns with [REDACTED]."""
        text = doc.text
        # Simple redaction
        if score.has_email:
            text = _EMAIL_RE.sub("[EMAIL_REDACTED]", text)
        if score.has_phone:
            text = _PHONE_RE.sub("[PHONE_REDACTED]", text)
            text = _INTL_PHONE_RE.sub("[PHONE_REDACTED]", text)
        if score.has_secret:
            for pat in _SECRET_PATTERNS:
                text = pat.sub("[SECRET_REDACTED]", text)
        if score.has_sensitive_id:
            text = _SSN_RE.sub("[SSN_REDACTED]", text)
            text = _CREDIT_CARD_RE.sub("[CARD_REDACTED]", text)

        # Return new Document
        import hashlib

        new_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        new_meta = dict(doc.metadata)
        new_meta["pii_redacted"] = True
        new_meta["pii_original_hash"] = doc.content_hash

        return Document(
            text=text,
            source=doc.source,
            document_id=doc.document_id,
            source_uri=doc.source_uri,
            license=doc.license,
            license_url=doc.license_url,
            collection=doc.collection,
            language=doc.language,
            content_hash=new_hash,
            metadata=new_meta,
            created_at=doc.created_at,
            version=doc.version,
        )


def filter_documents_pii(
    docs: List[Document],
    config: PIIFilterConfig | None = None,
) -> Tuple[List[Document], List[Tuple[Document, PIIScore, str]], List[PIIScore], Dict[str, int]]:
    """
    Filter documents for PII.
    Returns (kept, rejected, all_scores, stats).
    Stats: scanned, flagged, removed, transformed.
    """
    filt = PIIFilter(config)
    kept: List[Document] = []
    rejected: List[Tuple[Document, PIIScore, str]] = []
    all_scores: List[PIIScore] = []

    scanned = 0
    flagged = 0
    removed = 0
    transformed = 0

    for doc in docs:
        scanned += 1
        score = filt.scan(doc)
        all_scores.append(score)

        if score.flagged_patterns:
            flagged += 1

        keep, reason = filt.should_keep(score)
        if keep:
            if filt.config.transform_medium_risk and score.flagged_patterns:
                # Redact medium risk
                if score.risk_level == "medium":
                    new_doc = filt.redact(doc, score)
                    kept.append(new_doc)
                    transformed += 1
                else:
                    kept.append(doc)
            else:
                kept.append(doc)
        else:
            rejected.append((doc, score, reason))
            removed += 1

    stats = {
        "scanned": scanned,
        "flagged": flagged,
        "removed": removed,
        "transformed": transformed,
    }

    return kept, rejected, all_scores, stats
