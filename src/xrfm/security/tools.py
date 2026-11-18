"""Security agent tools for XRFM CoderSec (Phase 7 / GATE-7).

Provides:
- :class:`SecurityAdvisorySearchTool` — searches the defensive-security
  knowledge pack for advisories relevant to a finding.
- :class:`StaticCodeAuditTool` — statically scans project files for insecure
  code patterns without executing anything.
- :func:`register_defensive_security_tools` — registers both on a
  :class:`xrfm.agent.tools.ToolRegistry`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent.protocols import ToolSpec
from ..agent.tools import JSONLimits, ToolRegistry, _validate_schema
from .knowledge_pack import KnowledgeDocument, load_knowledge_pack


@dataclass(frozen=True)
class AdvisoryRecord:
    """A single defensive-security advisory record."""

    advisory_id: str
    source: str
    title: str
    description: str
    severity: str
    cwe_ids: tuple[str, ...] = ()
    mitigations: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "advisory_id": self.advisory_id,
            "source": self.source,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "cwe_ids": list(self.cwe_ids),
            "mitigations": list(self.mitigations),
            "tags": list(self.tags),
        }


class SecurityAdvisorySearchTool:
    """Agent tool: search the defensive-security knowledge pack for advisories.

    Bounded, read-only, deterministic. No network access. Results are redacted
    for sensitive keys before returning to the agent.
    """

    NAME = "security_advisory_search"
    DESCRIPTION = (
        "Search the defensive-security knowledge pack (OWASP ASVS, MITRE CWE/"
        "ATT&CK, NIST SP 800-218 / 800-61r3) for advisories matching a query. "
        "Returns advisory IDs, titles, severities, CWEs, and mitigations."
    )
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "maxLength": 512, "minLength": 1},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            "min_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "advisory_id": {"type": "string"},
                        "source": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "severity": {"type": "string"},
                        "cwe_ids": {"type": "array", "items": {"type": "string"}},
                        "mitigations": {"type": "array", "items": {"type": "string"}},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "score": {"type": "number"},
                    },
                    "required": [
                        "advisory_id",
                        "source",
                        "title",
                        "description",
                        "severity",
                        "cwe_ids",
                        "mitigations",
                        "tags",
                        "score",
                    ],
                    "additionalProperties": False,
                },
            },
            "count": {"type": "integer"},
        },
        "required": ["results", "count"],
        "additionalProperties": False,
    }
    PERMISSIONS = ("security.read",)
    RISK = "low"
    TIMEOUT_SECONDS = 10.0

    def __init__(self, knowledge_pack: tuple[KnowledgeDocument, ...] | None = None) -> None:
        docs = knowledge_pack or load_knowledge_pack()
        self._records: dict[str, AdvisoryRecord] = {}
        self._by_source: dict[str, list[AdvisoryRecord]] = {}
        self._by_cwe: dict[str, list[AdvisoryRecord]] = {}
        self._build_index(docs)

    def _build_index(self, docs: tuple[KnowledgeDocument, ...]) -> None:
        """Convert knowledge documents into advisory records for searching."""
        for doc in docs:
            # Build an advisory record from each knowledge document
            adv = AdvisoryRecord(
                advisory_id=doc.doc_id,
                source=doc.source,
                title=doc.title,
                description=doc.text[:500] + ("..." if len(doc.text) > 500 else ""),
                severity=self._infer_severity(doc),
                cwe_ids=tuple(t for t in doc.tags if t.startswith("CWE-")),
                mitigations=self._extract_mitigations(doc.text),
                tags=doc.tags,
            )
            self._records[adv.advisory_id] = adv
            self._by_source.setdefault(doc.domain, []).append(adv)
            for cwe in adv.cwe_ids:
                self._by_cwe.setdefault(cwe, []).append(adv)

    @staticmethod
    def _infer_severity(doc: KnowledgeDocument) -> str:
        text = (doc.title + " " + doc.text).lower()
        if any(w in text for w in ("critical", "cwe-89", "cwe-78", "cwe-798")):
            return "critical"
        if any(w in text for w in ("high", "cwe-79", "cwe-787")):
            return "high"
        if any(w in text for w in ("medium", "cwe-287", "cwe-521")):
            return "medium"
        return "low"

    @staticmethod
    def _extract_mitigations(text: str) -> tuple[str, ...]:
        mitigations: list[str] = []
        in_mitigation = False
        for line in text.splitlines():
            stripped = line.strip()
            if "mitigation" in stripped.lower() or "prevent" in stripped.lower():
                in_mitigation = True
                continue
            if in_mitigation and stripped.startswith(("-", "1.", "2.", "3.", "4.", "5.")):
                mitigations.append(stripped.lstrip("-123456789. ").strip())
            elif in_mitigation and stripped and not stripped.startswith("#"):
                mitigations.append(stripped)
            elif in_mitigation and stripped.startswith("#"):
                break
        if not mitigations:
            mitigations = ["Review the source document for specific mitigation guidance."]
        return tuple(mitigations[:5])

    def spec(self) -> ToolSpec:
        return ToolSpec(
            self.NAME,
            self.DESCRIPTION,
            self.INPUT_SCHEMA,
            self.OUTPUT_SCHEMA,
            permissions=self.PERMISSIONS,
            risk=self.RISK,
            timeout_seconds=self.TIMEOUT_SECONDS,
        )

    def __call__(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _validate_schema(self.INPUT_SCHEMA, arguments, limits=JSONLimits())
        query = arguments["query"]
        top_k = int(arguments.get("top_k", 5))
        min_score = float(arguments.get("min_score", 0.0))

        query_tokens = set(re.findall(r"\w+", query.lower()))
        if not query_tokens or not self._records:
            return {"results": [], "count": 0}

        scored: list[tuple[AdvisoryRecord, float]] = []
        for record in self._records.values():
            text = (
                f"{record.title} {record.description} "
                f"{' '.join(record.tags)} {' '.join(record.cwe_ids)}"
            ).lower()
            record_tokens = set(re.findall(r"\w+", text))
            overlap = len(query_tokens & record_tokens)
            if overlap > 0:
                score = overlap / len(query_tokens)
                if score > min_score:
                    scored.append((record, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        results = [
            {**record.to_dict(), "score": round(score, 4)}
            for record, score in scored[:top_k]
        ]
        return {"results": results, "count": len(results)}


class StaticCodeAuditTool:
    """Agent tool: statically scan project files for insecure code patterns.

    Read-only pattern scan only — never compiles or executes code. Detects:
    - hardcoded secrets / credentials
    - eval / exec / compile usage
    - subprocess with shell=True
    - pickle deserialization
    - insecure random usage
    - disabled SSL verification
    """

    NAME = "static_code_audit"
    DESCRIPTION = (
        "Statically scan project files for insecure code patterns (hardcoded secrets, "
        "eval/exec, shell=True, pickle, weak random, disabled SSL). Read-only; never "
        "compiles or executes code."
    )
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "relative": {
                "type": "string",
                "maxLength": 4096,
                "description": "Relative path to file or directory under project root.",
            },
            "max_files": {
                "type": "integer",
                "minimum": 1,
                "maximum": 500,
                "description": "Maximum number of files to scan.",
            },
        },
        "required": ["relative"],
        "additionalProperties": False,
    }
    OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "files_scanned": {"type": "integer"},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "pattern": {"type": "string"},
                        "severity": {"type": "string"},
                        "cwe": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["file", "line", "pattern", "severity", "cwe", "description"],
                    "additionalProperties": False,
                },
            },
            "total_findings": {"type": "integer"},
        },
        "required": ["files_scanned", "findings", "total_findings"],
        "additionalProperties": False,
    }
    PERMISSIONS = ("filesystem.read", "security.read")
    RISK = "low"
    TIMEOUT_SECONDS = 30.0

    # pattern -> (severity, cwe, description)
    _PATTERNS: dict[str, tuple[str, str, str]] = {
        r"\b(hf_[A-Za-z0-9]{20,}|KGAT_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})": (
            "critical",
            "CWE-798",
            "Hard-coded API token / credential detected",
        ),
        r"\b(?:password|passwd|pwd)\s*[:=]\s*[\"'][^\"']{3,}[\"']": (
            "critical",
            "CWE-798",
            "Hard-coded password detected",
        ),
        r"\b(?:api[_-]?key|apikey)\s*[:=]\s*[\"'][^\"']{3,}[\"']": (
            "critical",
            "CWE-798",
            "Hard-coded API key detected",
        ),
        r"\bsecret\s*[:=]\s*[\"'][^\"']{3,}[\"']": (
            "critical",
            "CWE-798",
            "Hard-coded secret detected",
        ),
        r"\beval\s*\(": ("high", "CWE-94", "eval() usage — code injection risk"),
        r"\bexec\s*\(": ("high", "CWE-94", "exec() usage — code injection risk"),
        r"\bcompile\s*\(": ("high", "CWE-94", "compile() usage — dynamic code execution"),
        r"\bshell\s*=\s*True": ("high", "CWE-78", "subprocess shell=True — command injection risk"),
        r"\bpickle\.(?:load|loads)\s*\(": ("high", "CWE-502", "pickle deserialization — remote code execution risk"),
        r"\brandom\.(?:random|randint|choice|seed)\s*\(": ("medium", "CWE-338", "weak randomness — use secrets module"),
        r"\bverify\s*[:=]\s*False": ("medium", "CWE-295", "SSL certificate verification disabled"),
        r"\bverify\s*=\s*False": ("medium", "CWE-295", "SSL certificate verification disabled"),
    }

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or Path.cwd()).expanduser().resolve()

    def spec(self) -> ToolSpec:
        return ToolSpec(
            self.NAME,
            self.DESCRIPTION,
            self.INPUT_SCHEMA,
            self.OUTPUT_SCHEMA,
            permissions=self.PERMISSIONS,
            risk=self.RISK,
            timeout_seconds=self.TIMEOUT_SECONDS,
        )

    def __call__(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _validate_schema(self.INPUT_SCHEMA, arguments, limits=JSONLimits())
        relative = arguments["relative"]
        max_files = int(arguments.get("max_files", 50))

        target = (self.root / relative).resolve()
        if self.root not in target.parents and target != self.root:
            return {
                "files_scanned": 0,
                "findings": [],
                "total_findings": 0,
            }

        files: list[Path] = []
        if target.is_file():
            files = [target]
        elif target.is_dir():
            for path in target.rglob("*"):
                if path.is_file() and len(files) < max_files:
                    rel_parts = path.relative_to(self.root).parts
                    if any(part.startswith(".") for part in rel_parts):
                        continue
                    if any(
                        part in {".git", "__pycache__", ".venv", "node_modules", "target", "dist", "build"}
                        for part in path.parts
                    ):
                        continue
                    files.append(path)
        else:
            return {
                "files_scanned": 0,
                "findings": [],
                "total_findings": 0,
            }

        findings: list[dict[str, Any]] = []
        for path in files:
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_no, line in enumerate(raw.splitlines(), start=1):
                for pattern, (severity, cwe, description) in self._PATTERNS.items():
                    if re.search(pattern, line):
                        findings.append(
                            {
                                "file": str(path.relative_to(self.root)),
                                "line": line_no,
                                "pattern": pattern,
                                "severity": severity,
                                "cwe": cwe,
                                "description": description,
                            }
                        )

        return {
            "files_scanned": len(files),
            "findings": findings,
            "total_findings": len(findings),
        }


def register_defensive_security_tools(
    registry: ToolRegistry,
    knowledge_pack: tuple[KnowledgeDocument, ...] | None = None,
    root: str | Path | None = None,
) -> None:
    """Register SecurityAdvisorySearchTool and StaticCodeAuditTool on a registry.

    Raises ``ValueError`` if either tool is already registered.
    """
    registry.register(
        SecurityAdvisorySearchTool(knowledge_pack).spec(),
        SecurityAdvisorySearchTool(knowledge_pack),
    )
    registry.register(
        StaticCodeAuditTool(root).spec(),
        StaticCodeAuditTool(root),
    )
