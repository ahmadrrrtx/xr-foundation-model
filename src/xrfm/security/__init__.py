"""Defensive Security RAG module for XRFM CoderSec (Phase 7 / GATE-7).

Provides:
- Defensive Security Knowledge Pack (OWASP, MITRE CWE, MITRE ATT&CK, NIST)
- Citation-grounded RAG pipeline for security Q&A
- SecurityAdvisorySearchTool and StaticCodeAuditTool agent tools
"""

from xrfm.security.knowledge_pack import (
    KNOWLEDGE_DOCUMENTS,
    KnowledgeDocument,
    load_knowledge_pack,
    search_by_tag,
    search_by_domain,
    search_text,
)
from xrfm.security.rag import CitationGroundedRAG, RAGResult, compare_grounded_vs_ungrounded
from xrfm.security.tools import (
    SecurityAdvisorySearchTool,
    StaticCodeAuditTool,
    AdvisoryRecord,
    register_defensive_security_tools,
)

__all__ = [
    "KNOWLEDGE_DOCUMENTS",
    "KnowledgeDocument",
    "AdvisoryRecord",
    "CitationGroundedRAG",
    "RAGResult",
    "SecurityAdvisorySearchTool",
    "StaticCodeAuditTool",
    "load_knowledge_pack",
    "search_by_tag",
    "search_by_domain",
    "search_text",
    "compare_grounded_vs_ungrounded",
    "register_defensive_security_tools",
]
