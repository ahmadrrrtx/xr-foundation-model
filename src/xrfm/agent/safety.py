"""Authorization, credential redaction, and bounded audit retention."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from .protocols import ToolSpec


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    reason: str
    requires_confirmation: bool = False


@dataclass
class PermissionPolicy:
    """Least-privilege policy evaluated outside the model."""

    allowed_permissions: set[str] = field(default_factory=set)
    denied_tools: set[str] = field(default_factory=set)
    confirmation_for_risk: set[str] = field(default_factory=lambda: {"high", "critical"})
    auto_confirm: bool = False

    def authorize(self, spec: ToolSpec) -> AuthorizationDecision:
        if spec.name in self.denied_tools:
            return AuthorizationDecision(False, f"tool {spec.name!r} is denied by policy")
        missing = set(spec.permissions) - self.allowed_permissions
        if missing:
            return AuthorizationDecision(False, f"missing permissions: {', '.join(sorted(missing))}")
        needs = spec.requires_confirmation or spec.risk in self.confirmation_for_risk
        if needs and not self.auto_confirm:
            return AuthorizationDecision(True, "explicit confirmation required", True)
        return AuthorizationDecision(True, "authorized")


_PRIVATE_KEY = re.compile(r"-----BEGIN [^-]+-----.*?-----END [^-]+-----", re.DOTALL)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{6,}")
_KEY_VALUE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret|authorization)"
    r"(\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_AWS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "token",
    "password",
    "secret",
    "authorization",
    "private_key",
}


def _redact_text(text: str) -> str:
    text = _PRIVATE_KEY.sub("[REDACTED PRIVATE KEY]", text)
    text = _BEARER.sub("Bearer [REDACTED]", text)
    text = _KEY_VALUE.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", text)
    return _AWS_KEY.sub("[REDACTED]", text)


def sanitize_observation(value: Any, max_chars: int = 12_000) -> Any:
    """Redact common credential shapes and bound recursive tool observations."""
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or not 0 <= max_chars <= 1_000_000:
        raise ValueError("max_chars must be an integer in [0, 1000000]")
    items = 0

    def sanitize(current: Any, depth: int, key: str | None = None) -> Any:
        nonlocal items
        if depth > 32:
            raise ValueError("observation exceeds maximum nesting depth")
        if key is not None and key.lower() in _SENSITIVE_KEYS:
            return "[REDACTED]"
        if isinstance(current, dict):
            items += len(current)
            if items > 1_000:
                raise ValueError("observation exceeds item bound")
            return {str(k): sanitize(v, depth + 1, str(k)) for k, v in current.items()}
        if isinstance(current, (list, tuple)):
            items += len(current)
            if items > 1_000:
                raise ValueError("observation exceeds item bound")
            return [sanitize(item, depth + 1) for item in current]
        text = _redact_text(str(current))
        return text[:max_chars] + ("…[TRUNCATED]" if len(text) > max_chars else "")

    return sanitize(value, 0)


@dataclass
class AuditRecord:
    event: str
    tool: str | None
    details: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class AuditLog:
    """Bounded in-memory audit records; details are redacted at ingestion."""

    records: list[AuditRecord] = field(default_factory=list)
    max_records: int = 1_000
    max_detail_chars: int = 2_000

    def __post_init__(self) -> None:
        if not isinstance(self.max_records, int) or isinstance(self.max_records, bool) or self.max_records <= 0:
            raise ValueError("max_records must be a positive integer")
        if not isinstance(self.max_detail_chars, int) or self.max_detail_chars <= 0:
            raise ValueError("max_detail_chars must be positive")

    def record(self, event: str, tool: str | None = None, **details: Any) -> None:
        safe_details = sanitize_observation(details, max_chars=self.max_detail_chars)
        if self.max_records > 0 and len(self.records) >= self.max_records:
            del self.records[: len(self.records) - self.max_records + 1]
        self.records.append(AuditRecord(event, tool, safe_details))

    def export(self) -> list[dict[str, Any]]:
        return [
            {"event": r.event, "tool": r.tool, "details": r.details, "timestamp": r.timestamp} for r in self.records
        ]
