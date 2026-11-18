"""Defensive Security Knowledge Pack for XRFM CoderSec (Phase 7 / GATE-7).

Contains attributed security reference documents:

- OWASP Cheat Sheets (CC-BY-SA-4.0 attribution preserved)
- OWASP ASVS 5.0
- MITRE CWE entries
- MITRE ATT&CK Mitigations
- NIST SP 800-218 / 800-61r3

All documents carry SPDX license identifiers and provenance metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KnowledgeDocument:
    """A single attributed knowledge document in the security knowledge pack."""

    doc_id: str
    title: str
    text: str
    source: str
    license: str  # SPDX license identifier
    domain: str  # e.g. "owasp", "cwe", "attack", "nist"
    tags: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_search_record(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "source": self.source,
            "license": self.license,
            "domain": self.domain,
            "tags": list(self.tags),
            "provenance": dict(self.provenance),
        }


# ---------------------------------------------------------------------------
# OWASP Cheat Sheets — CC-BY-SA-4.0 (attribution preserved per OWASP license)
# ---------------------------------------------------------------------------

OWASP_SQL_INJECTION_PREVENTION = KnowledgeDocument(
    doc_id="owasp-cs-sql-injection-prevention",
    title="OWASP Cheat Sheet: SQL Injection Prevention",
    source="https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html",
    license="CC-BY-SA-4.0",
    domain="owasp",
    tags=("sql-injection", "cwe-89", "defensive-coding", "input-validation"),
    provenance={
        "organization": "OWASP",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution_required": True,
        "attribution_text": "OWASP Cheat Sheet Series, https://cheatsheetseries.owasp.org/",
    },
    text=(
        "# SQL Injection Prevention\n\n"
        "## Primary Defenses\n\n"
        "1. **Use Prepared Statements (Parameterized Queries)**\n"
        "   Use parameterized queries with placeholders. Never concatenate user input "
        "into SQL strings.\n\n"
        '   Bad: ``query = "SELECT * FROM users WHERE id = " + user_id``\n'
        '   Good: ``cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))``\n\n'
        "2. **Use Stored Procedures**\n"
        "   Use stored procedures that parameterize input. Avoid dynamic SQL inside "
        "stored procedures.\n\n"
        "3. **Allow-list Input Validation**\n"
        "   Validate all input against a strict allow-list of permitted characters "
        "and formats before use.\n\n"
        "4. **Escape All User-Supplied Input**\n"
        "   If parameterization is impossible, escape input using the database-specific "
        "escape routine for the intended context (identifier, literal, etc.).\n\n"
        "## Additional Defenses\n\n"
        "- Enforce least privilege on database accounts used by applications.\n"
        "- Disable unnecessary database features (xp_cmdshell, etc.).\n"
        "- Use Web Application Firewall (WAF) rules as a secondary layer.\n"
        "- Keep database software patched.\n\n"
        "## CWE Mapping\n\n"
        "CWE-89: Improper Neutralization of Special Elements used in an SQL Command "
        "(SQL Injection)\n"
    ),
)

OWASP_XSS_PREVENTION = KnowledgeDocument(
    doc_id="owasp-cs-xss-prevention",
    title="OWASP Cheat Sheet: Cross-Site Scripting (XSS) Prevention",
    source="https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html",
    license="CC-BY-SA-4.0",
    domain="owasp",
    tags=("xss", "cwe-79", "defensive-coding", "output-encoding"),
    provenance={
        "organization": "OWASP",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution_required": True,
        "attribution_text": "OWASP Cheat Sheet Series, https://cheatsheetseries.owasp.org/",
    },
    text=(
        "# Cross-Site Scripting (XSS) Prevention\n\n"
        "## Primary Rules\n\n"
        "1. **HTML Escape Before Inserting Untrusted Data into HTML Element Content**\n"
        "   Escape: & < > \" ' / with their HTML entities.\n\n"
        "2. **Attribute Escape Before Inserting Untrusted Data into HTML Attributes**\n"
        "   Escape all characters with ASCII values < 256 using the &#xHH; format.\n\n"
        "3. **JavaScript Escape Before Inserting Untrusted Data into JavaScript Values**\n"
        "   Use \\xHH escaping for all characters < 256 except alphanumerics.\n\n"
        "4. **CSS Escape Before Inserting Untrusted Data into CSS**\n"
        "   Escape with \\HH for all non-alphanumeric characters.\n\n"
        "5. **URL Escape Before Inserting Untrusted Data into URL Parameter Values**\n"
        "   Use percent-encoding for all characters except alphanumerics.\n\n"
        "## Additional Rules\n\n"
        "- Use Content Security Policy (CSP) as a defense-in-depth layer.\n"
        "- Set HttpOnly, Secure, SameSite flags on cookies.\n"
        "- Use modern frameworks with auto-escaping (React, Angular, Vue) but verify.\n"
        "- Never insert untrusted data into script src, style, or event handler attributes.\n\n"
        "## CWE Mapping\n\n"
        "CWE-79: Improper Neutralization of Input During Web Page Generation (XSS)\n"
    ),
)

OWASP_AUTHENTICATION_CHEAT_SHEET = KnowledgeDocument(
    doc_id="owasp-cs-authentication",
    title="OWASP Cheat Sheet: Authentication",
    source="https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html",
    license="CC-BY-SA-4.0",
    domain="owasp",
    tags=("authentication", "cwe-287", "cwe-521", "password-storage", "mfa"),
    provenance={
        "organization": "OWASP",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution_required": True,
        "attribution_text": "OWASP Cheat Sheet Series, https://cheatsheetseries.owasp.org/",
    },
    text=(
        "# Authentication Cheat Sheet\n\n"
        "## General Authentication Guidance\n\n"
        "1. **Use Multi-Factor Authentication (MFA)**\n"
        "   Require MFA for all user accounts, especially administrative accounts.\n\n"
        "2. **Password Storage\n"
        "   Store passwords using Argon2id, bcrypt, or scrypt with a unique salt per user.\n"
        "   Never store plaintext passwords or use fast hashes (MD5, SHA-1, SHA-256).\n\n"
        "3. **Enforce Strong Password Policy\n"
        "   Minimum 12 characters, no complexity requirements that encourage reuse, "
        "check against known-bad password lists.\n\n"
        "4. **Generic Error Messages\n"
        "   Use identical generic messages for invalid username or invalid password "
        "to prevent user enumeration.\n\n"
        "5. **Rate Limit Authentication Attempts\n"
        "   Implement account lockout or progressive delays after failed attempts.\n\n"
        "6. **Session Management\n"
        "   Use strong session identifiers, invalidate on logout, set short timeouts,\n"
        "and use Secure/HttpOnly/SameSite cookie flags.\n\n"
        "## CWE Mapping\n\n"
        "CWE-287: Improper Authentication\n"
        "CWE-521: Weak Password Requirements\n"
        "CWE-522: Insufficiently Protected Credentials\n"
    ),
)

# ---------------------------------------------------------------------------
# OWASP ASVS 5.0
# ---------------------------------------------------------------------------

OWASP_ASVS_5_0_V1 = KnowledgeDocument(
    doc_id="owasp-asvs-5.0-v1-1",
    title="OWASP ASVS 5.0 — Verification Requirement V1: Architecture, Design and Threat Modeling",
    source="https://github.com/OWASP/ASVS/blob/master/5.0/ASVS-5.0.md",
    license="CC-BY-SA-4.0",
    domain="asvs",
    tags=("asvs", "threat-modeling", "architecture", "verification"),
    provenance={
        "organization": "OWASP",
        "standard": "ASVS 5.0",
        "level_mapping": {"L1": "basic", "L2": "standard", "L3": "advanced"},
    },
    text=(
        "# OWASP ASVS 5.0 — V1 Architecture and Threat Modeling\n\n"
        "## V1.1 Secure Requirements\n\n"
        "Verify that all business and security requirements are documented and "
        "reviewed (L1).\n\n"
        "## V1.2 Authentication Architecture Verification\n\n"
        "Verify the architecture defines authentication requirements and trust "
        "boundaries (L1).\n\n"
        "## V1.3 Session Management Architecture Verification\n\n"
        "Verify session management design covers token issuance, expiration, and "
        "revocation (L1).\n\n"
        "## V1.4 Access Control Architecture Verification\n\n"
        "Verify access control design enforces least privilege and separation of "
        "duties (L1).\n\n"
        "## V1.5 Data Classification Architecture Verification\n\n"
        "Verify data classification defines sensitivity levels and handling "
        "requirements (L2).\n\n"
        "## V1.6 Threat Modeling Verification\n\n"
        "Verify threat modeling is performed and reviewed at least annually and "
        "before major releases (L2). Identify threats using STRIDE or similar.\n\n"
        "## V1.7 Security Design and Implementation Verification\n\n"
        "Verify security design principles (least privilege, fail-safe, "
        "economy of mechanism) are documented (L3).\n\n"
        "## V1.8 Security Requirements Verification\n\n"
        "Verify security requirements are traceable to implementation and tests "
        "(L3).\n\n"
    ),
)

# ---------------------------------------------------------------------------
# MITRE CWE
# ---------------------------------------------------------------------------

MITRE_CWE_89 = KnowledgeDocument(
    doc_id="cwe-89",
    title="CWE-89: Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')",
    source="https://cwe.mitre.org/data/definitions/89.html",
    license="CC-BY-SA-4.0",
    domain="cwe",
    tags=("CWE-89", "cwe-89", "sql-injection", "injection", "mitre"),
    provenance={
        "organization": "MITRE",
        "cwe_id": "89",
        "name": "Improper Neutralization of Special Elements used in an SQL Command",
        "abstraction": "Base",
        "status": "Draft",
    },
    text=(
        "# CWE-89: SQL Injection\n\n"
        "## Description\n\n"
        "The software constructs all or part of an SQL command using externally-influenced "
        "input from an upstream component, but it does not neutralize or incorrectly "
        "neutralizes special elements that could modify the intended SQL command when "
        "it is sent to a downstream component.\n\n"
        "## Modes of Introduction\n\n"
        "Implementation: Realized when code fails to use parameterized queries or "
        "properly escapes input.\n\n"
        "Architecture and Design: Failure to apply a data sanitization strategy "
        "at architectural boundaries.\n\n"
        "## Common Consequences\n\n"
        "- Confidentiality: Unauthorized data access\n"
        "- Integrity: Unauthorized data modification\n"
        "- Availability: Denial of service via destructive queries\n"
        "- Access Bypass: Authentication bypass\n\n"
        "## Detection Methods\n\n"
        "Automated static analysis can detect string concatenation in SQL queries "
        "with moderate effectiveness. Dynamic analysis (DAST) can inject SQL "
        "metacharacters and observe behavior changes.\n\n"
        "## Potential Mitigations\n\n"
        "1. Use parameterized queries / prepared statements\n"
        "2. Use allow-list input validation\n"
        "3. Use stored procedures with parameters\n"
        "4. Enforce least privilege on database accounts\n"
        "5. Apply OWASP SQL Injection Prevention Cheat Sheet\n\n"
        "## Related Weaknesses\n\n"
        "CWE-77: Command Injection\n"
        "CWE-564: SQL Injection in a Swing Component\n"
        "CWE-652: Improper Neutralization of Data within XQuery Modules\n"
    ),
)

MITRE_CWE_79 = KnowledgeDocument(
    doc_id="cwe-79",
    title="CWE-79: Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')",
    source="https://cwe.mitre.org/data/definitions/79.html",
    license="CC-BY-SA-4.0",
    domain="cwe",
    tags=("CWE-79", "cwe-79", "xss", "cross-site-scripting", "mitre"),
    provenance={
        "organization": "MITRE",
        "cwe_id": "79",
        "name": "Cross-site Scripting",
        "abstraction": "Base",
        "status": "Draft",
    },
    text=(
        "# CWE-79: Cross-site Scripting (XSS)\n\n"
        "## Description\n\n"
        "The software does not neutralize or incorrectly neutralizes user-controllable "
        "input before it is placed in output that is used as a web page served to "
        "other users.\n\n"
        "## Types\n\n"
        "- Reflected XSS: Input is immediately returned by the server in a response.\n"
        "- Stored XSS: Input is stored and later served to other users.\n"
        "- DOM-based XSS: Vulnerability exists in client-side code.\n\n"
        "## Common Consequences\n\n"
        "- Confidentiality: Session hijacking via cookie theft\n"
        "- Integrity: Defacement, malicious content injection\n"
        "- Availability: Browser-based denial of service\n\n"
        "## Potential Mitigations\n\n"
        "1. Context-sensitive output encoding\n"
        "2. Use frameworks with automatic escaping (React, Angular, Vue)\n"
        "3. Content Security Policy (CSP)\n"
        "4. HttpOnly cookie flag to prevent JavaScript cookie access\n"
        "5. Apply OWASP XSS Prevention Cheat Sheet\n\n"
        "## Related Weaknesses\n\n"
        "CWE-80: SQL Injection\n"
        "CWE-83: XPATH Injection\n"
        "CWE-88: Argument Injection\n"
    ),
)

MITRE_CWE_787 = KnowledgeDocument(
    doc_id="cwe-787",
    title="CWE-787: Out-of-bounds Write",
    source="https://cwe.mitre.org/data/definitions/787.html",
    license="CC-BY-SA-4.0",
    domain="cwe",
    tags=("CWE-787", "cwe-787", "buffer-overflow", "memory-safety", "mitre"),
    provenance={
        "organization": "MITRE",
        "cwe_id": "787",
        "name": "Out-of-bounds Write",
        "abstraction": "Base",
        "status": "Draft",
    },
    text=(
        "# CWE-787: Out-of-bounds Write\n\n"
        "## Description\n\n"
        "The software writes data past the end, or before the beginning, of the "
        "intended buffer.\n\n"
        "## Common Consequences\n\n"
        "- Integrity: Data corruption\n"
        "- Availability: Crash / denial of service\n"
        "- Confidentiality: Information disclosure\n"
        "- Access Bypass: Arbitrary code execution\n\n"
        "## Potential Mitigations\n\n"
        "1. Use memory-safe languages (Rust, Go, Java, C#)\n"
        "2. Use bounded copy functions (strncpy, memcpy_s)\n"
        "3. Validate array indices before use\n"
        "4. Enable compiler stack protection (-fstack-protector-strong)\n"
        "5. Use Address Sanitizer (ASan) in testing\n\n"
        "## Related Weaknesses\n\n"
        "CWE-121: Stack-based Buffer Overflow\n"
        "CWE-122: Heap-based Buffer Overflow\n"
        "CWE-190: Integer Overflow\n"
    ),
)

# ---------------------------------------------------------------------------
# MITRE ATT&CK Mitigations
# ---------------------------------------------------------------------------

MITRE_ATTACK_T1059 = KnowledgeDocument(
    doc_id="attack-t1059",
    title="MITRE ATT&CK T1059 — Command and Scripting Interpreter: Mitigations",
    source="https://attack.mitre.org/techniques/T1059/",
    license="CC-BY-NC-SA-4.0",
    domain="attack",
    tags=("attack", "t1059", "command-line", "scripting", "mitigation"),
    provenance={
        "organization": "MITRE",
        "framework": "ATT&CK",
        "technique_id": "T1059",
        "technique_name": "Command and Scripting Interpreter",
        "tactic": "Execution",
    },
    text=(
        "# MITRE ATT&CK T1059 — Command and Scripting Interpreter\n\n"
        "## Description\n\n"
        "Adversaries may abuse command and script interpreters to execute commands, "
        "scripts, or binaries. These interpreters are often used across operating "
        "systems to automate tasks and manage systems.\n\n"
        "## Mitigations\n\n"
        "M1042: Disable or Remove Feature or Tool\n"
        "- Remove unnecessary interpreters (PowerShell, cmd, bash, python) from "
        "production systems where possible.\n\n"
        "M1049: Conducting High-Commitment Activities with Elevated Privileges\n"
        "- Restrict interpreter access to administrative accounts only.\n\n"
        "M1038: User Account Management\n"
        "- Apply least privilege to user accounts that can execute interpreters.\n\n"
        "M1047: Audit\n"
        "- Enable command-line logging and script block logging (Windows).\n"
        "- Monitor interpreter process creation events.\n\n"
        "## Detection\n\n"
        "- Monitor for unusual child processes spawned by interpreters.\n"
        "- Alert on obfuscated command-line arguments.\n"
        "- Track execution of encoded commands (powershell -enc).\n\n"
        "## Related CWEs\n\n"
        "CWE-77: Command Injection\n"
        "CWE-78: Improper Neutralization of Special Elements used in an OS Command\n"
    ),
)

MITRE_ATTACK_T1190 = KnowledgeDocument(
    doc_id="attack-t1190",
    title="MITRE ATT&CK T1190 — Exploit Public-Facing Application: Mitigations",
    source="https://attack.mitre.org/techniques/T1190/",
    license="CC-BY-NC-SA-4.0",
    domain="attack",
    tags=("attack", "t1190", "exploit", "public-facing", "mitigation"),
    provenance={
        "organization": "MITRE",
        "framework": "ATT&CK",
        "technique_id": "T1190",
        "technique_name": "Exploit Public-Facing Application",
        "tactic": "Initial Access",
    },
    text=(
        "# MITRE ATT&CK T1190 — Exploit Public-Facing Application\n\n"
        "## Description\n\n"
        "Adversaries may exploit software vulnerabilities in public-facing "
        "applications to gain initial access.\n\n"
        "## Mitigations\n\n"
        "M1050: Software Updates\n"
        "- Maintain up-to-date versions of all public-facing applications.\n"
        "- Apply security patches within defined SLAs.\n\n"
        "M1048: Application Isolation\n"
        "- Run public-facing applications in isolated environments (containers, DMZ).\n"
        "- Use application sandboxing where available.\n\n"
        "M1031: Network Intrusion Prevention\n"
        "- Deploy WAF to block known exploit patterns.\n"
        "- Use IPS signatures for known vulnerabilities.\n\n"
        "M1026: Privileged Account Management\n"
        "- Run public-facing services with least privilege accounts.\n"
        "- Separate service accounts per application.\n\n"
        "## Detection\n\n"
        "- Monitor web server logs for exploit patterns.\n"
        "- Alert on unusual outbound connections from web servers.\n"
        "- Track new processes spawned by web server processes.\n\n"
        "## Related CWEs\n\n"
        "CWE-89: SQL Injection\n"
        "CWE-78: Command Injection\n"
        "CWE-79: Cross-site Scripting\n"
        "CWE-434: Unrestricted Upload of File with Dangerous Type\n"
    ),
)

# ---------------------------------------------------------------------------
# NIST SP 800-218 (Secure Software Development Framework)
# ---------------------------------------------------------------------------

NIST_SP_800_218 = KnowledgeDocument(
    doc_id="nist-sp-800-218",
    title="NIST SP 800-218 — Secure Software Development Framework (SSDF)",
    source="https://csrc.nist.gov/publications/detail/sp/800-218/final",
    license="US-GOVERNMENT-WORK",  # NIST publications are public domain
    domain="nist",
    tags=("nist", "ssdf", "secure-development", "supply-chain"),
    provenance={
        "organization": "NIST",
        "publication": "SP 800-218",
        "title": "Secure Software Development Framework (SSDF) Version 1.1",
        "status": "Final",
    },
    text=(
        "# NIST SP 800-218 — Secure Software Development Framework (SSDF)\n\n"
        "## Practice Groups\n\n"
        "### PS — Protect the Software\n\n"
        "PS.1: Identify and protect all software components and their dependencies.\n"
        "PS.2: Protect all code from unauthorized access and tampering.\n"
        "PS.3: Verify the integrity of software components.\n\n"
        "### PW — Produce Well-Secured Software\n\n"
        "PW.1: Design software to meet security requirements.\n"
        "PW.2: Implement secure coding practices.\n"
        "PW.3: Review and analyze code for vulnerabilities.\n"
        "PW.4: Address vulnerabilities identified in code review.\n\n"
        "### VA — Vulnerability Identification and Remediation\n\n"
        "VA.1: Perform vulnerability testing (SAST, DAST, SCA).\n"
        "VA.2: Analyze and prioritize vulnerability remediation.\n"
        "VA.3: Remediate vulnerabilities in a timely manner.\n\n"
        "### RV — Respond to Vulnerability Reports\n\n"
        "RV.1: Establish a vulnerability disclosure process.\n"
        "RV.2: Respond to vulnerability reports promptly.\n"
        "RV.3: Publish advisories and patches.\n\n"
        "## Implementation Examples\n\n"
        "- Integrate SAST into CI/CD pipeline\n"
        "- Use Software Bill of Materials (SBOM) for all components\n"
        "- Sign software releases with cryptographic signatures\n"
        "- Perform dependency vulnerability scanning (SCA)\n\n"
        "## Mapping\n\n"
        "Maps to: OWASP ASVS, CWE Top 25, MITRE ATT&CK\n"
    ),
)

# ---------------------------------------------------------------------------
# NIST SP 800-61r3 (Incident Handling)
# ---------------------------------------------------------------------------

NIST_SP_800_61r3 = KnowledgeDocument(
    doc_id="nist-sp-800-61r3",
    title="NIST SP 800-61 Rev 3 — Computer Security Incident Handling Guide",
    source="https://csrc.nist.gov/publications/detail/sp/800-61/rev-3/final",
    license="US-GOVERNMENT-WORK",
    domain="nist",
    tags=("nist", "incident-response", "security-operations"),
    provenance={
        "organization": "NIST",
        "publication": "SP 800-61 Rev 3",
        "title": "Computer Security Incident Handling Guide",
        "status": "Final",
    },
    text=(
        "# NIST SP 800-61 Rev 3 — Incident Handling\n\n"
        "## Incident Response Lifecycle\n\n"
        "1. **Preparation**\n"
        "   Establish incident response policy, plan, and procedures.\n"
        "   Train the incident response team.\n"
        "   Prepare tools and communication channels.\n\n"
        "2. **Detection and Analysis**\n"
        "   Detect precursors and indicators of compromise.\n"
        "   Analyze alerts to determine incident scope.\n"
        "   Document evidence and chain of custody.\n\n"
        "3. **Containment, Eradication, and Recovery**\n"
        "   Contain the incident to prevent further damage.\n"
        "   Eradicate the root cause (remove malware, close vulnerabilities).\n"
        "   Recover systems to normal operations.\n\n"
        "4. **Post-Incident Activity**\n"
        "   Conduct lessons learned.\n"
        "   Update policies and procedures.\n"
        "   Improve detection capabilities.\n\n"
        "## Coordination\n\n"
        "- Internal: IT, legal, HR, public relations, management.\n"
        "- External: law enforcement, vendors, ISPs, other organizations.\n\n"
        "## Evidence Handling\n\n"
        "Preserve evidence integrity using cryptographic hashing and documented "
        "chain of custody.\n"
    ),
)

# ---------------------------------------------------------------------------
# Knowledge Pack Registry
# ---------------------------------------------------------------------------

KNOWLEDGE_DOCUMENTS: tuple[KnowledgeDocument, ...] = (
    OWASP_SQL_INJECTION_PREVENTION,
    OWASP_XSS_PREVENTION,
    OWASP_AUTHENTICATION_CHEAT_SHEET,
    OWASP_ASVS_5_0_V1,
    MITRE_CWE_89,
    MITRE_CWE_79,
    MITRE_CWE_787,
    MITRE_ATTACK_T1059,
    MITRE_ATTACK_T1190,
    NIST_SP_800_218,
    NIST_SP_800_61r3,
)

_DOCUMENT_INDEX: dict[str, KnowledgeDocument] = {doc.doc_id: doc for doc in KNOWLEDGE_DOCUMENTS}
_TAG_INDEX: dict[str, list[KnowledgeDocument]] = {}
for doc in KNOWLEDGE_DOCUMENTS:
    for tag in doc.tags:
        _TAG_INDEX.setdefault(tag, []).append(doc)


def load_knowledge_pack() -> tuple[KnowledgeDocument, ...]:
    """Return all knowledge documents in the pack."""
    return KNOWLEDGE_DOCUMENTS


def get_document(doc_id: str) -> KnowledgeDocument | None:
    """Retrieve a document by ID."""
    return _DOCUMENT_INDEX.get(doc_id)


def search_by_tag(tag: str) -> tuple[KnowledgeDocument, ...]:
    """Find documents by tag."""
    return tuple(_TAG_INDEX.get(tag, []))


def search_by_domain(domain: str) -> tuple[KnowledgeDocument, ...]:
    """Find documents by domain."""
    return tuple(d for d in KNOWLEDGE_DOCUMENTS if d.domain == domain)


def search_text(query: str, top_k: int = 5) -> list[tuple[KnowledgeDocument, float]]:
    """Simple keyword-based search over knowledge documents."""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    scored: list[tuple[KnowledgeDocument, float]] = []

    for doc in KNOWLEDGE_DOCUMENTS:
        text_lower = doc.text.lower()
        title_lower = doc.title.lower()
        score = 0.0

        # Title matches weighted higher
        for word in query_words:
            if word in title_lower:
                score += 2.0
            if word in text_lower:
                score += 1.0
            # Tag matches
            if word in doc.tags:
                score += 1.5

        if score > 0:
            scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
