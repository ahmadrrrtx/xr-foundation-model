"""
Language identification for XRFM.

Architecture:
- LanguageDetector interface
- Heuristic detector (default, no heavy deps)
- Optional fastText detector if available (documented dependency)
- Configurable thresholds
- Records detected language + confidence

Design: English-focused initially but infrastructure supports multilingual.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from xrfm.data.schema import Document, LanguageScore


# Simple heuristic: character n-gram / common words for English vs others
# This is intentionally lightweight for Phase 1 golden dataset.
# Production can plug in fastText or GlotLID.

_ENGLISH_COMMON = {
    "the", "and", "is", "in", "to", "of", "a", "that", "it", "with", "for", "as",
    "was", "on", "are", "be", "this", "have", "from", "or", "had", "by", "not",
    "but", "what", "all", "were", "we", "when", "your", "can", "said", "there",
    "use", "each", "which", "she", "do", "how", "their", "if", "will", "up",
    "other", "about", "out", "many", "then", "them", "these", "so", "some",
    "her", "would", "make", "like", "him", "into", "time", "has", "look",
    "two", "more", "write", "go", "see", "number", "no", "way", "could",
}

# Very rough script detection
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]{3,}")
_CJK_RE = re.compile(r"[\u4E00-\u9FFF]{2,}")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]{3,}")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]{3,}")


@dataclass
class LanguageDetectorConfig:
    """Configuration for language detection."""

    default_language: str = "en"
    threshold: float = 0.6  # confidence threshold to accept
    detector: str = "heuristic"  # heuristic | fasttext | glotlid
    allowed_languages: Optional[List[str]] = None  # None = all allowed, or list like ["en"]
    # For fastText path
    fasttext_model_path: Optional[str] = None


class LanguageDetector:
    """Interface for language detectors."""

    def __init__(self, config: LanguageDetectorConfig | None = None):
        self.config = config or LanguageDetectorConfig()

    def detect(self, text: str) -> LanguageScore:
        raise NotImplementedError

    def detect_document(self, doc: Document) -> LanguageScore:
        score = self.detect(doc.text)
        # If doc already has language hint, we could use it, but we record detected
        return score


class HeuristicLanguageDetector(LanguageDetector):
    """
    Lightweight heuristic detector.
    No external dependencies, deterministic.
    Suitable for golden dataset and CI.
    """

    def detect(self, text: str) -> LanguageScore:
        if not text or not text.strip():
            return LanguageScore(language="unknown", confidence=0.0, detector="heuristic", detector_version="v1")

        lower = text.lower()
        words = re.findall(r"\b[a-z]+\b", lower)
        total_words = len(words) if words else 1

        # English score via common words overlap
        english_hits = sum(1 for w in words if w in _ENGLISH_COMMON)
        english_ratio = english_hits / total_words

        # Script detection
        scores: Dict[str, float] = {}
        scores["en"] = english_ratio

        # Arabic
        if _ARABIC_RE.search(text):
            # Count arabic chars
            arabic_chars = len(_ARABIC_RE.findall(text))
            scores["ar"] = min(0.9, arabic_chars * 0.1 + 0.3)
        # Urdu uses Arabic script but with extra chars; we approximate
        # If text contains Urdu-specific characters
        if re.search(r"[\u0600-\u06FF]", text):
            # Could be ur or ar, boost both
            scores["ur"] = scores.get("ar", 0.0) * 0.9

        # CJK
        if _CJK_RE.search(text):
            scores["zh"] = 0.8
            scores["ja"] = 0.5

        if _CYRILLIC_RE.search(text):
            scores["ru"] = 0.8

        if _DEVANAGARI_RE.search(text):
            scores["hi"] = 0.8

        # Determine best
        best_lang = max(scores, key=lambda k: scores[k]) if scores else "unknown"
        best_conf = scores.get(best_lang, 0.0)

        # If no strong signal and text is ascii-heavy, assume English
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
        if ascii_ratio > 0.9 and best_conf < 0.3:
            # Fallback: if ascii and some english words, boost en
            if english_hits > 0:
                best_lang = "en"
                best_conf = max(best_conf, 0.5)

        # Normalize confidence 0-1
        best_conf = min(1.0, max(0.0, best_conf))

        return LanguageScore(
            language=best_lang,
            confidence=best_conf,
            scores=scores,
            detector="heuristic",
            detector_version="v1",
        )


class FastTextLanguageDetector(LanguageDetector):
    """
    Wrapper for fastText detector if available.
    Falls back to heuristic if not installed.
    Documents dependency and version.
    """

    def __init__(self, config: LanguageDetectorConfig | None = None):
        super().__init__(config)
        self._model = None
        self._version = "unknown"
        if config and config.fasttext_model_path:
            try:
                import fasttext

                self._model = fasttext.load_model(config.fasttext_model_path)
                self._version = getattr(fasttext, "__version__", "unknown")
            except Exception as e:
                # Log and fallback
                print(f"fastText model load failed: {e}, falling back to heuristic")
                self._model = None

    def detect(self, text: str) -> LanguageScore:
        if self._model is None:
            # Fallback
            return HeuristicLanguageDetector(self.config).detect(text)

        # FastText prediction
        try:
            # fastText expects single line, no newline
            clean = text.replace("\n", " ")[:10000]  # truncate for speed
            labels, probs = self._model.predict(clean, k=3)
            # labels like __label__en
            scores = {}
            for label, prob in zip(labels, probs):
                lang = label.replace("__label__", "")
                scores[lang] = float(prob)
            best = max(scores, key=lambda k: scores[k]) if scores else "unknown"
            conf = scores.get(best, 0.0)
            return LanguageScore(
                language=best,
                confidence=conf,
                scores=scores,
                detector="fasttext",
                detector_version=self._version,
            )
        except Exception:
            return HeuristicLanguageDetector(self.config).detect(text)


def get_detector(config: LanguageDetectorConfig | None = None) -> LanguageDetector:
    cfg = config or LanguageDetectorConfig()
    if cfg.detector == "fasttext":
        return FastTextLanguageDetector(cfg)
    # Default heuristic
    return HeuristicLanguageDetector(cfg)


def filter_by_language(
    docs: List[Document],
    detector: LanguageDetector | None = None,
    config: LanguageDetectorConfig | None = None,
) -> Tuple[List[Document], List[Tuple[Document, LanguageScore]], List[LanguageScore]]:
    """
    Filter documents by language.
    Returns (kept, rejected_with_score, all_scores).
    - If allowed_languages is None, keeps all but records score.
    - If threshold is set, requires confidence >= threshold for allowed languages.
    Does NOT silently delete low-confidence without recording.
    """
    cfg = config or (detector.config if detector else LanguageDetectorConfig())
    det = detector or get_detector(cfg)

    kept: List[Document] = []
    rejected: List[Tuple[Document, LanguageScore]] = []
    all_scores: List[LanguageScore] = []

    allowed = set([l.lower() for l in cfg.allowed_languages]) if cfg.allowed_languages else None

    for doc in docs:
        score = det.detect_document(doc)
        all_scores.append(score)

        # Decision
        if allowed is None:
            kept.append(doc)
        else:
            # If language in allowed and confidence >= threshold, keep
            # If language unknown but allowed contains en and confidence low, we still reject? Configurable.
            if score.language in allowed and score.confidence >= cfg.threshold:
                kept.append(doc)
            else:
                # Special case: if doc's declared language is in allowed and detection is uncertain,
                # we keep but with low confidence? For now, respect threshold.
                if doc.language in allowed and score.confidence < cfg.threshold:
                    # Keep if declared language matches and confidence not too low (<0.2)
                    # To avoid silent deletion, we require explicit reason.
                    if score.confidence >= 0.2:
                        kept.append(doc)
                    else:
                        rejected.append((doc, score))
                else:
                    rejected.append((doc, score))

    return kept, rejected, all_scores
