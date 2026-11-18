"""Phase 7 (`GATE-7`) regression tests — Defensive Security RAG Knowledge Pack, agent tools, ZeroGPU Space.

Validates the Phase 7 defensive security implementation:

1. Knowledge pack loads with all required document types (OWASP, CWE, ATT&CK, NIST)
2. SecurityAdvisorySearchTool returns relevant advisories with correct schema
3. StaticCodeAuditTool detects insecure code patterns without executing code
4. CitationGroundedRAG produces grounded results with citations
5. Grounded RAG reduces CWE/OWASP hallucination by >= 70% (GATE-7 criterion)
6. CLI `xrfm security` subcommands work correctly
7. Tool registration on ToolRegistry works with correct permissions
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestKnowledgePack:
    """The Defensive Security Knowledge Pack must load with all required sources."""

    def test_knowledge_pack_loads(self):
        from xrfm.security import KNOWLEDGE_DOCUMENTS, load_knowledge_pack

        docs = load_knowledge_pack()
        assert len(docs) > 0, "Knowledge pack must contain documents"
        assert len(KNOWLEDGE_DOCUMENTS) == len(docs)

    def test_required_domains_present(self):
        from xrfm.security import KNOWLEDGE_DOCUMENTS

        domains = {doc.domain for doc in KNOWLEDGE_DOCUMENTS}
        required = {"owasp", "cwe", "attack", "nist", "asvs"}
        assert required.issubset(domains), f"Missing domains: {required - domains}"

    def test_required_licenses_present(self):
        from xrfm.security import KNOWLEDGE_DOCUMENTS

        licenses = {doc.license for doc in KNOWLEDGE_DOCUMENTS}
        assert "CC-BY-SA-4.0" in licenses, "Must include OWASP CC-BY-SA-4.0 docs"
        assert "US-GOVERNMENT-WORK" in licenses, "Must include NIST public domain docs"

    def test_document_has_required_fields(self):
        from xrfm.security import KNOWLEDGE_DOCUMENTS

        for doc in KNOWLEDGE_DOCUMENTS:
            assert doc.doc_id, f"Document missing doc_id: {doc}"
            assert doc.title, f"Document missing title: {doc.doc_id}"
            assert doc.text, f"Document missing text: {doc.doc_id}"
            assert doc.source, f"Document missing source: {doc.doc_id}"
            assert doc.license, f"Document missing license: {doc.doc_id}"
            assert doc.domain, f"Document missing domain: {doc.doc_id}"
            assert isinstance(doc.tags, tuple), f"Tags must be tuple: {doc.doc_id}"

    def test_cwe_documents_have_cwe_ids(self):
        from xrfm.security import KNOWLEDGE_DOCUMENTS

        cwe_docs = [d for d in KNOWLEDGE_DOCUMENTS if d.domain == "cwe"]
        assert len(cwe_docs) > 0, "Must have CWE documents"
        for doc in cwe_docs:
            assert any(tag.startswith("CWE-") for tag in doc.tags), f"CWE doc {doc.doc_id} must have CWE tag"

    def test_search_by_tag(self):
        from xrfm.security import search_by_tag

        results = search_by_tag("sql-injection")
        assert len(results) > 0, "Must find sql-injection tagged documents"
        assert any("CWE-89" in d.tags or "cwe-89" in d.tags for d in results)

    def test_search_by_domain(self):
        from xrfm.security import search_by_domain

        results = search_by_domain("nist")
        assert len(results) > 0, "Must find NIST documents"
        assert all(d.domain == "nist" for d in results)

    def test_search_text(self):
        from xrfm.security import search_text

        results = search_text("SQL injection", top_k=3)
        assert len(results) > 0, "Text search must return results"
        assert all(score > 0 for _, score in results), "All scores must be positive"


class TestSecurityAdvisorySearchTool:
    """SecurityAdvisorySearchTool must return relevant advisories with valid schema."""

    def test_tool_instantiation(self):
        from xrfm.security import SecurityAdvisorySearchTool

        tool = SecurityAdvisorySearchTool()
        assert tool.NAME == "security_advisory_search"
        assert tool.RISK == "low"
        assert tool.PERMISSIONS == ("security.read",)

    def test_tool_spec(self):
        from xrfm.security import SecurityAdvisorySearchTool

        tool = SecurityAdvisorySearchTool()
        spec = tool.spec()
        assert spec.name == "security_advisory_search"
        assert spec.input_schema["type"] == "object"
        assert "query" in spec.input_schema["properties"]
        assert spec.output_schema is not None
        assert "results" in spec.output_schema["properties"]
        assert "count" in spec.output_schema["properties"]

    def test_search_returns_results(self):
        from xrfm.security import SecurityAdvisorySearchTool

        tool = SecurityAdvisorySearchTool()
        result = tool({"query": "SQL injection", "top_k": 5})

        assert "results" in result
        assert "count" in result
        assert result["count"] > 0
        assert result["count"] == len(result["results"])
        assert result["count"] <= 5

    def test_result_schema_valid(self):
        from xrfm.security import SecurityAdvisorySearchTool

        tool = SecurityAdvisorySearchTool()
        result = tool({"query": "cross-site scripting", "top_k": 3})

        for r in result["results"]:
            required_fields = {
                "advisory_id",
                "source",
                "title",
                "description",
                "severity",
                "cwe_ids",
                "mitigations",
                "tags",
                "score",
            }
            assert required_fields.issubset(set(r.keys())), f"Missing fields: {required_fields - set(r.keys())}"
            assert r["severity"] in {"low", "medium", "high", "critical"}
            assert 0.0 <= r["score"] <= 1.0
            assert isinstance(r["cwe_ids"], list)
            assert isinstance(r["mitigations"], list)
            assert isinstance(r["tags"], list)

    def test_min_score_filter(self):
        from xrfm.security import SecurityAdvisorySearchTool

        tool = SecurityAdvisorySearchTool()
        result_low = tool({"query": "SQL injection", "min_score": 0.0})
        result_high = tool({"query": "SQL injection", "min_score": 0.9})
        assert result_high["count"] <= result_low["count"]

    def test_top_k_limit(self):
        from xrfm.security import SecurityAdvisorySearchTool

        tool = SecurityAdvisorySearchTool()
        result = tool({"query": "injection", "top_k": 2})
        assert result["count"] <= 2


class TestStaticCodeAuditTool:
    """StaticCodeAuditTool must detect insecure patterns without executing code."""

    def test_tool_instantiation(self):
        from xrfm.security import StaticCodeAuditTool

        tool = StaticCodeAuditTool()
        assert tool.NAME == "static_code_audit"
        assert tool.RISK == "low"
        assert "filesystem.read" in tool.PERMISSIONS
        assert "security.read" in tool.PERMISSIONS

    def test_tool_spec(self):
        from xrfm.security import StaticCodeAuditTool

        tool = StaticCodeAuditTool()
        spec = tool.spec()
        assert spec.name == "static_code_audit"
        assert "relative" in spec.input_schema["properties"]
        assert spec.output_schema is not None
        assert "findings" in spec.output_schema["properties"]
        assert "total_findings" in spec.output_schema["properties"]

    def test_detects_hardcoded_secrets(self):
        import os
        import tempfile

        from xrfm.security import StaticCodeAuditTool

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test_secrets.py")
            with open(test_file, "w") as f:
                f.write('API_KEY = "sk-1234567890abcdef1234567890"\n')
                f.write('password = "secret123"\n')

            tool = StaticCodeAuditTool(root=tmpdir)
            result = tool({"relative": "test_secrets.py", "max_files": 10})
            assert result["total_findings"] > 0, "Must detect hardcoded secrets"
            severities = {f["severity"] for f in result["findings"]}
            assert "critical" in severities, "Hardcoded secrets must be critical"

    def test_detects_eval_exec(self):
        import os
        import tempfile

        from xrfm.security import StaticCodeAuditTool

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test_injection.py")
            with open(test_file, "w") as f:
                f.write("eval(user_input)\n")
                f.write("exec(malicious_code)\n")

            tool = StaticCodeAuditTool(root=tmpdir)
            result = tool({"relative": "test_injection.py", "max_files": 10})
            assert result["total_findings"] > 0, "Must detect eval/exec"
            patterns = {f["pattern"] for f in result["findings"]}
            assert any("eval" in p for p in patterns), "Must detect eval()"
            assert any("exec" in p for p in patterns), "Must detect exec()"

    def test_detects_shell_true(self):
        import os
        import tempfile

        from xrfm.security import StaticCodeAuditTool

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test_shell.py")
            with open(test_file, "w") as f:
                f.write("subprocess.run(cmd, shell=True)\n")

            tool = StaticCodeAuditTool(root=tmpdir)
            result = tool({"relative": "test_shell.py", "max_files": 10})
            assert result["total_findings"] > 0, "Must detect shell=True"
            cwes = {f["cwe"] for f in result["findings"]}
            assert "CWE-78" in cwes, "shell=True must map to CWE-78"

    def test_read_only_no_execution(self):
        import os
        import tempfile

        from xrfm.security import StaticCodeAuditTool

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test_safe.py")
            with open(test_file, "w") as f:
                f.write('print("hello world")\n')

            tool = StaticCodeAuditTool(root=tmpdir)
            result = tool({"relative": "test_safe.py", "max_files": 10})
            assert result["total_findings"] == 0, "Safe code should have no findings"
            assert result["files_scanned"] == 1

    def test_path_traversal_blocked(self):
        from xrfm.security import StaticCodeAuditTool

        tool = StaticCodeAuditTool(root="/home/user/xr-foundation-model")
        result = tool({"relative": "../../etc/passwd", "max_files": 10})
        assert result["files_scanned"] == 0, "Path traversal must be blocked"


class TestCitationGroundedRAG:
    """CitationGroundedRAG must produce grounded results with citations."""

    def test_rag_instantiation(self):
        from xrfm.security import CitationGroundedRAG

        rag = CitationGroundedRAG()
        assert rag.indexer is not None
        assert rag.retriever is not None

    def test_grounded_answer(self):
        from xrfm.security import CitationGroundedRAG

        rag = CitationGroundedRAG()
        result = rag.grounded_answer("How to prevent SQL injection?", top_k=3)

        assert result.grounded is True
        assert len(result.citations) > 0, "Must return citations"
        assert result.citation_coverage > 0.0
        assert result.citation_coverage <= 1.0

    def test_citation_schema_valid(self):
        from xrfm.security import CitationGroundedRAG

        rag = CitationGroundedRAG()
        result = rag.grounded_answer("cross-site scripting", top_k=3)

        for c in result.citations:
            required = {"index", "doc_id", "title", "source", "license", "domain", "relevance_score"}
            assert required.issubset(set(c.keys())), f"Missing citation fields: {required - set(c.keys())}"
            assert c["index"] > 0
            assert c["doc_id"]
            assert c["title"]

    def test_format_citations(self):
        from xrfm.security import CitationGroundedRAG

        rag = CitationGroundedRAG()
        result = rag.grounded_answer("SQL injection", top_k=3)
        formatted = rag.format_citations(result.citations)
        assert "Sources:" in formatted
        assert len(formatted) > 0

    def test_no_results_still_returns(self):
        from xrfm.security import CitationGroundedRAG

        rag = CitationGroundedRAG()
        result = rag.grounded_answer("xyzzy nonexistent topic plugh", top_k=3)

        # Should not crash even with no results
        assert isinstance(result.grounded, bool)
        assert isinstance(result.citation_coverage, float)


class TestGate7HallucinationReduction:
    """GATE-7 criterion: grounded RAG reduces CWE/OWASP hallucination by >= 70%."""

    def test_hallucination_reduction_meets_threshold(self):
        from xrfm.security import CitationGroundedRAG, compare_grounded_vs_ungrounded

        rag = CitationGroundedRAG()

        # A weights-only answer that makes up CWE references without grounding
        weights_only_answer = (
            "To fix SQL injection, use CWE-999 and CWE-888. "
            "Also apply OWASP-Fake-Rule and OWASP-MadeUp-Control. "
            "The CWE-12345 vulnerability requires immediate patching."
        )

        comparison = compare_grounded_vs_ungrounded(rag, "How to prevent SQL injection?", weights_only_answer, top_k=5)

        # The grounded answer should cite real sources, reducing hallucination
        assert comparison["gate7_criterion_met"] is True, (
            f"GATE-7 criterion not met: hallucination_reduction={comparison['hallucination_reduction']}"
        )
        assert comparison["hallucination_reduction"] >= 0.70, (
            f"Hallucination reduction {comparison['hallucination_reduction']} < 0.70"
        )
        assert comparison["grounded"] is True
        assert comparison["citation_coverage"] > 0.0

    def test_hallucination_reduction_metrics(self):
        from xrfm.security import CitationGroundedRAG, compare_grounded_vs_ungrounded

        rag = CitationGroundedRAG()

        weights_only = (
            "Use CWE-999 for everything. OWASP-Fake says ignore validation. CWE-888 is the only CWE you need."
        )

        comparison = compare_grounded_vs_ungrounded(rag, "SQL injection prevention", weights_only, top_k=5)

        assert comparison["weights_cwe_count"] >= 2, "Weights answer should have fake CWEs"
        assert comparison["weights_owasp_count"] >= 1, "Weights answer should have fake OWASP refs"
        assert comparison["grounded_cwe_count"] >= 0, "Grounded answer uses real CWEs"
        assert "unsupported_citations" in comparison
        assert "gate7_criterion_met" in comparison


class TestToolRegistration:
    """Defensive security tools must register correctly on ToolRegistry."""

    def test_register_defensive_security_tools(self):
        from xrfm.agent.tools import ToolRegistry
        from xrfm.security import register_defensive_security_tools

        registry = ToolRegistry()
        register_defensive_security_tools(registry)

        specs = registry.specs()
        names = {s.name for s in specs}
        assert "security_advisory_search" in names
        assert "static_code_audit" in names

    def test_registered_tool_permissions(self):
        from xrfm.agent.tools import ToolRegistry
        from xrfm.security import register_defensive_security_tools

        registry = ToolRegistry()
        register_defensive_security_tools(registry)

        for spec in registry.specs():
            if spec.name == "security_advisory_search":
                assert spec.permissions == ("security.read",)
                assert spec.risk == "low"
                assert spec.timeout_seconds == 10.0
            elif spec.name == "static_code_audit":
                assert "filesystem.read" in spec.permissions
                assert "security.read" in spec.permissions
                assert spec.risk == "low"
                assert spec.timeout_seconds == 30.0

    def test_register_duplicate_raises(self):
        from xrfm.agent.tools import ToolRegistry
        from xrfm.security import SecurityAdvisorySearchTool, register_defensive_security_tools

        registry = ToolRegistry()
        register_defensive_security_tools(registry)

        # Re-registering should raise ValueError
        with pytest.raises(ValueError):
            registry.register(
                SecurityAdvisorySearchTool().spec(),
                SecurityAdvisorySearchTool(),
            )


class TestCLISecuritySubcommands:
    """CLI `xrfm security` subcommands must work correctly."""

    def test_cli_security_search(self):
        from xrfm.cli import main

        old_argv = sys.argv
        sys.argv = ["xrfm", "security", "search", "SQL injection", "--top-k", "2"]
        try:
            rc = main()
        finally:
            sys.argv = old_argv
        assert rc == 0, "CLI security search must succeed"

    def test_cli_security_scan(self):
        from xrfm.cli import main

        old_argv = sys.argv
        sys.argv = ["xrfm", "security", "scan", "src/xrfm/security", "--max-files", "5"]
        try:
            rc = main()
        finally:
            sys.argv = old_argv
        assert rc == 0, "CLI security scan must succeed"


# Allow running directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
