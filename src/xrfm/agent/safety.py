"""Deterministic authorization, sanitization, and audit primitives."""
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


_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]{6,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{12,}"),
)


def sanitize_observation(value: Any, max_chars: int = 12000) -> Any:
    """Bound and redact tool data before it reaches the model context.

    This is not a content-security filter. It is a containment boundary: data is
    marked as untrusted by the caller, bounded, and common credential shapes are
    redacted. It never grants instructions or permissions.
    """
    if isinstance(value, dict):
        return {str(k): sanitize_observation(v, max_chars) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_observation(v, max_chars) for v in value]
    text = str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: m.group(0).split(":")[0] + ": [REDACTED]", text)
    return text[:max_chars] + ("…[TRUNCATED]" if len(text) > max_chars else "")


@dataclass
class AuditRecord:
    event: str
    tool: str | None
    details: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class AuditLog:
    records: list[AuditRecord] = field(default_factory=list)

    def record(self, event: str, tool: str | None = None, **details: Any) -> None:
        self.records.append(AuditRecord(event, tool, details))

    def export(self) -> list[dict[str, Any]]:
        return [{"event": r.event, "tool": r.tool, "details": r.details, "timestamp": r.timestamp} for r in self.records]
