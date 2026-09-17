"""
Modular quality filtering for XRFM.

Instead of one giant filter_document(), we have independently testable filters:
- minimum length
- maximum repetition
- symbol/number ratio
- boilerplate detection
- language confidence
- quality score
- malformed detection

Each filter produces explicit decision or score.
Preserves metadata to answer why document was excluded.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Callable, Optional

from xrfm.data.schema import Document, QualityScore, ProcessingDecision


# Common boilerplate phrases (low quality)
_BOILERPLATE_PATTERNS = [
    r"lorem ipsum",
    r"click here",
    r"subscribe now",
    r"all rights reserved",
    r"terms and conditions",
    r"cookie policy",
    r"this page is intentionally left blank",
]


@dataclass
class QualityFilterConfig:
    min_chars: int = 50
    max_chars: int = 100000  # 100k chars max for Phase 1
    min_words: int = 10
    max_symbol_ratio: float = 0.3  # if >30% symbols, low quality
    max_number_ratio: float = 0.5
    max_repetition_ratio: float = 0.3  # max duplicate n-gram ratio
    boilerplate_threshold: float = 0.5
    # Overall quality threshold (0-1)
    overall_threshold: float = 0.3
    # Whether to allow code
    allow_code: bool = True


def _symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    # Symbols: non-alphanumeric, non-space, non-common punctuation?
    # Count characters that are not alphanumeric, not whitespace, not . , ! ? ; : ' " ( ) - etc.
    # Simple: count of characters in set of symbols
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return symbols / max(len(text), 1)


def _number_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(1 for c in text if c.isdigit())
    return digits / max(len(text), 1)


def _repetition_score(text: str, n: int = 3) -> float:
    """
    Detect excessive repetition via duplicate n-grams.
    Returns ratio of duplicate n-grams to total.
    """
    words = text.split()
    if len(words) < n * 2:
        return 0.0
    ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    unique = set(ngrams)
    # Ratio of duplicate occurrences
    return 1.0 - len(unique) / len(ngrams)


def _boilerplate_score(text: str) -> float:
    lower = text.lower()
    hits = 0
    for pat in _BOILERPLATE_PATTERNS:
        if re.search(pat, lower):
            hits += 1
    return min(1.0, hits / 2.0)


def _malformed_score(text: str) -> float:
    """Heuristic malformed detection."""
    if not text.strip():
        return 1.0
    # Excessive replacement chars
    if "\ufffd" in text:
        return min(1.0, text.count("\ufffd") / max(len(text) * 0.1, 1))
    # Very long words (no spaces) could be malformed
    words = text.split()
    if words:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len > 50:
            return 0.8
    return 0.0


class QualityFilter:
    """Base class for quality filters."""

    def __init__(self, config: QualityFilterConfig | None = None):
        self.config = config or QualityFilterConfig()

    def score(self, doc: Document) -> QualityScore:
        text = doc.text
        char_len = len(text)
        words = text.split()
        word_count = len(words)

        sym_ratio = _symbol_ratio(text)
        num_ratio = _number_ratio(text)
        rep_score = _repetition_score(text)
        boil_score = _boilerplate_score(text)
        mal_score = _malformed_score(text)

        is_too_short = char_len < self.config.min_chars or word_count < self.config.min_words
        is_too_long = char_len > self.config.max_chars
        has_excessive_repetition = rep_score > self.config.max_repetition_ratio
        has_excessive_symbols = sym_ratio > self.config.max_symbol_ratio
        is_boilerplate = boil_score > self.config.boilerplate_threshold
        is_malformed = mal_score > 0.5

        # Overall score heuristic: weighted combination, higher is better
        # Start at 1.0, subtract penalties
        overall = 1.0
        if is_too_short:
            overall -= 0.5
        if is_too_long:
            overall -= 0.2
        overall -= rep_score * 0.5
        overall -= sym_ratio * 0.3
        overall -= num_ratio * 0.2
        overall -= boil_score * 0.5
        overall -= mal_score * 0.7
        overall = max(0.0, min(1.0, overall))

        return QualityScore(
            char_length=char_len,
            word_count=word_count,
            symbol_ratio=sym_ratio,
            number_ratio=num_ratio,
            repetition_score=rep_score,
            boilerplate_score=boil_score,
            malformed_score=mal_score,
            overall_score=overall,
            is_too_short=is_too_short,
            is_too_long=is_too_long,
            has_excessive_repetition=has_excessive_repetition,
            has_excessive_symbols=has_excessive_symbols,
            is_boilerplate=is_boilerplate,
            is_malformed=is_malformed,
        )

    def should_keep(self, score: QualityScore) -> Tuple[bool, str]:
        """Decide keep/reject with reason."""
        cfg = self.config
        if score.is_too_short:
            return False, "too_short"
        if score.is_too_long:
            return False, "too_long"
        if score.is_malformed:
            return False, "malformed"
        if score.has_excessive_repetition:
            return False, "excessive_repetition"
        if score.has_excessive_symbols and not cfg.allow_code:
            return False, "excessive_symbols"
        if score.is_boilerplate:
            return False, "boilerplate"
        if score.overall_score < cfg.overall_threshold:
            return False, f"low_quality_score_{score.overall_score:.2f}"
        return True, "passed"


def filter_documents_quality(
    docs: List[Document],
    config: QualityFilterConfig | None = None,
) -> Tuple[List[Document], List[Tuple[Document, QualityScore, str]], List[QualityScore]]:
    """
    Filter documents by quality.
    Returns (kept, rejected_with_score_and_reason, all_scores).
    """
    filt = QualityFilter(config)
    kept: List[Document] = []
    rejected: List[Tuple[Document, QualityScore, str]] = []
    all_scores: List[QualityScore] = []

    for doc in docs:
        score = filt.score(doc)
        all_scores.append(score)
        keep, reason = filt.should_keep(score)
        if keep:
            kept.append(doc)
        else:
            rejected.append((doc, score, reason))

    return kept, rejected, all_scores


# Individual testable filters (for unit tests)
def filter_min_length(doc: Document, min_chars: int = 50, min_words: int = 10) -> bool:
    return len(doc.text) >= min_chars and len(doc.text.split()) >= min_words


def filter_max_repetition(doc: Document, threshold: float = 0.3) -> bool:
    return _repetition_score(doc.text) <= threshold


def filter_symbol_ratio(doc: Document, threshold: float = 0.3) -> bool:
    return _symbol_ratio(doc.text) <= threshold
