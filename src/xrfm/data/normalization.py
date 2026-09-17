"""
Document normalization for XRFM.

Conservative normalization:
- Unicode NFC normalization
- Newline normalization (\r\n, \r -> \n)
- Invalid byte handling (already str, but handle surrogates)
- Control character removal (except \n, \t)
- Whitespace normalization (preserve meaningful indentation for code)
- Empty / malformed detection

Every transformation explicit and testable.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Tuple

from xrfm.data.schema import Document

# Control characters to remove (except \n \t)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
# Multiple blank lines -> max 2
_BLANK_LINES_RE = re.compile(r"\n{3,}")
# Trailing whitespace per line
_TRAILING_WS_RE = re.compile(r"[ \t]+$")


@dataclass
class NormalizationResult:
    document: Document
    was_modified: bool
    modifications: list[str]
    is_empty: bool
    is_malformed: bool


def normalize_unicode(text: str, form: str = "NFC") -> Tuple[str, bool]:
    """Unicode normalization, returns (normalized_text, changed)."""
    normalized = unicodedata.normalize(form, text)
    return normalized, normalized != text


def normalize_newlines(text: str) -> Tuple[str, bool]:
    """Normalize \r\n and \r to \n."""
    new_text = text.replace("\r\n", "\n").replace("\r", "\n")
    return new_text, new_text != text


def remove_control_chars(text: str) -> Tuple[str, bool]:
    """Remove control chars except \n and \t."""
    new_text = _CONTROL_CHARS_RE.sub("", text)
    return new_text, new_text != text


def normalize_whitespace(text: str, preserve_code: bool = True) -> Tuple[str, bool]:
    """
    Conservative whitespace normalization.
    - Removes trailing whitespace per line
    - Collapses 3+ newlines to 2
    - Does NOT collapse spaces inside lines (preserves code indentation if preserve_code)
    - Strips leading/trailing blank lines
    """
    original = text
    # Remove trailing whitespace per line
    lines = text.split("\n")
    cleaned_lines = [line.rstrip(" \t") for line in lines]
    text = "\n".join(cleaned_lines)
    # Collapse excessive blank lines
    text = _BLANK_LINES_RE.sub("\n\n", text)
    # Strip leading/trailing whitespace (but not internal)
    text = text.strip("\n")
    # If not preserving code, collapse multiple spaces? We preserve by default.
    if not preserve_code:
        # Collapse 2+ spaces to single, but keep leading indentation?
        # Simple implementation: collapse multiple spaces not at line start
        text = re.sub(r"(?<!^)(?<!\n)[ ]{2,}", " ", text, flags=re.MULTILINE)
    return text, text != original


def detect_malformed(text: str) -> Tuple[bool, str]:
    """
    Detect obviously malformed text.
    Returns (is_malformed, reason).
    """
    if not text or not text.strip():
        return True, "empty"
    if len(text) < 5:
        return True, "too_short"
    # Check for excessive null bytes or replacement characters
    if "\ufffd" in text and text.count("\ufffd") / max(len(text), 1) > 0.1:
        return True, "excessive_replacement_chars"
    # Check if text is mostly non-printable (after our cleaning)
    # Simple heuristic: if >50% of chars are not alphanumeric and not common punctuation
    # We'll skip aggressive check to avoid destroying code.
    return False, ""


def normalize_document(
    doc: Document,
    unicode_form: str = "NFC",
    preserve_code: bool = True,
) -> NormalizationResult:
    """
    Normalize a single document, preserving provenance.
    Returns NormalizationResult with modified document.
    """
    text = doc.text
    modifications: list[str] = []

    # Unicode
    new_text, changed = normalize_unicode(text, form=unicode_form)
    if changed:
        modifications.append(f"unicode_{unicode_form}")
        text = new_text

    # Newlines
    new_text, changed = normalize_newlines(text)
    if changed:
        modifications.append("newline_normalization")
        text = new_text

    # Control chars
    new_text, changed = remove_control_chars(text)
    if changed:
        modifications.append("control_char_removal")
        text = new_text

    # Whitespace
    new_text, changed = normalize_whitespace(text, preserve_code=preserve_code)
    if changed:
        modifications.append("whitespace_normalization")
        text = new_text

    # Malformed detection
    is_malformed, reason = detect_malformed(text)
    is_empty = not text or not text.strip()

    # Create new Document with normalized text
    # Preserve original hash in metadata for traceability
    new_metadata = dict(doc.metadata)
    if modifications:
        new_metadata["normalization_modifications"] = modifications
        new_metadata["original_content_hash"] = doc.content_hash
    if is_malformed:
        new_metadata["malformed_reason"] = reason

    # Recompute hash after normalization
    import hashlib

    new_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    normalized_doc = Document(
        text=text,
        source=doc.source,
        document_id=doc.document_id,  # Keep same ID for traceability? Or regenerate? Keep same.
        source_uri=doc.source_uri,
        license=doc.license,
        license_url=doc.license_url,
        collection=doc.collection,
        language=doc.language,
        content_hash=new_hash,
        metadata=new_metadata,
        created_at=doc.created_at,
        version=doc.version,
    )

    return NormalizationResult(
        document=normalized_doc,
        was_modified=len(modifications) > 0,
        modifications=modifications,
        is_empty=is_empty,
        is_malformed=is_malformed,
    )


def normalize_documents(docs: list[Document], **kwargs) -> list[NormalizationResult]:
    """Batch normalization."""
    return [normalize_document(d, **kwargs) for d in docs]
