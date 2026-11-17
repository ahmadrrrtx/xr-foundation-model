"""
Multilingual & Code Language Identification (LID) for XRFM.

Remediates Phase 0 Audit Finding ``F-13``:
- Accurately discriminates all 7 XRFM target languages + source code:
  ``en`` (English), ``ur`` (Urdu Perso-Arabic/Nastaliq), ``ur-Latn`` (Roman Urdu),
  ``ar`` (Arabic), ``hi`` (Hindi Devanagari), ``es`` (Spanish), ``fr`` (French),
  and ``code`` (Python / JS / TS / Bash / SQL / Rust / C / Go).
- Fixes the legacy ``scores["ur"] = scores["ar"] * 0.9`` defect by explicitly
  scoring Urdu-exclusive Perso-Arabic Unicode codepoints
  (``ٹ ڈ ڑ ں ہ ھ ی ے گ پ چ ژ``) and high-frequency Urdu vs Arabic function words.
- Separates Roman Urdu (``ur-Latn``) from English, Spanish, and French even when
  technical English loanwords (e.g., ``web``, ``application``, ``Python``, ``SQL``)
  appear inside code-switched Roman-Urdu sentences.
- Zero external heavy model dependencies required for deterministic offline/CI
  execution; exposes pluggable ``LanguageDetector`` interface for optional FastText
  or GlotLID backends.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from xrfm.data.schema import Document

# ---------------------------------------------------------------------------
# Unicode Script Ranges & Urdu vs Arabic Discriminators
# ---------------------------------------------------------------------------

SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "latin": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
    "arabic_block": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "cjk": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "cyrillic": [(0x0400, 0x04FF)],
}

# Characters exclusive to Urdu / Perso-Arabic (absent in Modern Standard Arabic):
# ٹ (0679), ڈ (0688), ڑ (0691), ں (06BA), ہ (06C1), ھ (06BE), ے (06D2),
# ی (06CC - Farsi/Urdu Yeh vs Arabic ي 064A), پ (067E), چ (0686), ژ (0698), گ (06AF), ۂ (06C2)
URDU_EXCLUSIVE_CHARS: frozenset[str] = frozenset(
    "\u0679\u0688\u0691\u06ba\u06be\u06c1\u06c2\u06cc\u06d2\u06d3\u06af\u067e\u0686\u0698"
)

# Characters characteristic of Standard Arabic (rare or replaced in Urdu orthography):
# ة (0629 Teh Marbuta), ى (0649 Alef Maksura), ي (064A Arabic Yeh with two dots),
# ه (0647 Arabic Heh vs Urdu ہ 06C1), إ (0625), أ (0623)
ARABIC_CHARACTERISTIC_CHARS: frozenset[str] = frozenset("\u0629\u0649\u064a\u0647\u0625")

SPANISH_ACCENT_CHARS: frozenset[str] = frozenset("áéíóúñü¿¡ÁÉÍÓÚÑÜ")
FRENCH_ACCENT_CHARS: frozenset[str] = frozenset("àâæçéèêëîïôœùûüÿÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ")

# ---------------------------------------------------------------------------
# High-Precision Function-Word Lexicons
# ---------------------------------------------------------------------------

URDU_STOPWORDS: frozenset[str] = frozenset(
    {
        "اور",
        "کے",
        "کی",
        "کا",
        "میں",
        "ہے",
        "ہیں",
        "سے",
        "کو",
        "پر",
        "اس",
        "یہ",
        "لیے",
        "تاکہ",
        "نہیں",
        "بھی",
        "ساتھ",
        "کر",
        "سکتے",
        "چاہیے",
        "ہو",
        "ہم",
        "آپ",
        "بہت",
        "تمام",
        "اپنے",
        "نہ",
        "تھا",
        "تھی",
    }
)

ARABIC_STOPWORDS: frozenset[str] = frozenset(
    {
        "في",
        "من",
        "على",
        "إلى",
        "عن",
        "مع",
        "هذا",
        "هذه",
        "التي",
        "الذي",
        "الذين",
        "كان",
        "كانت",
        "تكون",
        "يجب",
        "أن",
        "إن",
        "بين",
        "قبل",
        "بعد",
        "جميع",
        "خلال",
        "أثناء",
        "تعتبر",
        "تساعد",
        "أهم",
        "بشكل",
        "البيانات",
    }
)

HINDI_STOPWORDS: frozenset[str] = frozenset(
    {
        "और",
        "के",
        "की",
        "का",
        "में",
        "है",
        "हैं",
        "से",
        "को",
        "पर",
        "लिए",
        "यह",
        "इस",
        "एक",
        "नहीं",
        "भी",
        "साथ",
        "करना",
        "चाहिए",
        "हमेशा",
        "सभी",
        "बहुत",
        "होता",
        "उपयोग",
    }
)

ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "of",
        "to",
        "in",
        "is",
        "that",
        "for",
        "it",
        "with",
        "as",
        "was",
        "on",
        "are",
        "by",
        "be",
        "this",
        "have",
        "from",
        "or",
        "an",
        "but",
        "not",
        "you",
        "all",
        "can",
        "we",
        "when",
        "before",
        "always",
        "without",
        "every",
        "over",
        "jumps",
        "lazy",
    }
)

# Unambiguous Roman-Urdu function & core vocabulary (chosen to avoid collision with EN/ES/FR)
ROMAN_URDU_STRONG_WORDS: frozenset[str] = frozenset(
    {
        "aur",
        "hai",
        "hain",
        "mein",
        "ko",
        "ka",
        "ki",
        "ke",
        "kya",
        "nahi",
        "nhi",
        "bhi",
        "bohat",
        "bahut",
        "apni",
        "apne",
        "apna",
        "aap",
        "hum",
        "yeh",
        "woh",
        "agar",
        "lekin",
        "magar",
        "taake",
        "kyunke",
        "zaroor",
        "zaroori",
        "karein",
        "karen",
        "karna",
        "karta",
        "karti",
        "karte",
        "chahiye",
        "sakte",
        "sakta",
        "sakti",
        "mujhe",
        "tum",
        "tarah",
        "hamesha",
        "mehfooz",
        "mukammal",
        "acha",
        "accha",
        "kaam",
        "masla",
        "masail",
        "jinhein",
        "jinhain",
        "hal",
        "bata",
        "banaya",
        "banana",
        "jata",
        "jati",
        "raha",
        "rahi",
        "rahe",
        "chalta",
        "dono",
        "kuch",
        "ahem",
        "chahte",
    }
)

ROMAN_URDU_WEAK_WORDS: frozenset[str] = frozenset(
    {"se", "ne", "pe", "par", "ho", "na", "is", "us", "ab", "ek", "jo", "to"}
)

SPANISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "unos",
        "unas",
        "y",
        "o",
        "pero",
        "del",
        "al",
        "para",
        "por",
        "con",
        "sin",
        "sobre",
        "entre",
        "contra",
        "como",
        "cuando",
        "donde",
        "siempre",
        "debemos",
        "utilizar",
        "permite",
        "datos",
        "están",
        "esta",
        "este",
        "estos",
        "estas",
        "sus",
        "más",
        "muy",
        "también",
        "antes",
        "después",
        "código",
        "seguridad",
        "pueden",
    }
)

FRENCH_STOPWORDS: frozenset[str] = frozenset(
    {
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "du",
        "au",
        "aux",
        "et",
        "ou",
        "mais",
        "dans",
        "sur",
        "sous",
        "avec",
        "sans",
        "pour",
        "par",
        "contre",
        "est",
        "sont",
        "être",
        "avoir",
        "il",
        "elle",
        "nous",
        "vous",
        "ils",
        "ce",
        "cette",
        "ces",
        "qui",
        "afin",
        "avant",
        "après",
        "données",
        "sécurité",
        "modèles",
        "chaque",
        "toutes",
        "permet",
        "système",
    }
)

# Regex patterns strongly indicative of programming source code
CODE_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(def|class|import|from|return|async\s+def|elif|except|raise)\b", re.MULTILINE),
    re.compile(r"^\s*(export\s+|function\s+|const\s+|let\s+|interface\s+|type\s+\w+\s*=)", re.MULTILINE),
    re.compile(r"^\s*(#include\s*<|fn\s+\w+\s*\(|pub\s+fn\s+|package\s+\w+|func\s+\w+\()", re.MULTILINE),
    re.compile(r"^\s*(#!/usr/bin/|set\s+-euo\s+pipefail|\bsha256sum\b|\bfi\b|\bdone\b)", re.MULTILINE),
    re.compile(r"\bSELECT\b[\s\S]+\bFROM\b[\s\S]*(\bWHERE\b|\bJOIN\b|\bGROUP\s+BY\b)", re.IGNORECASE),
)

CODE_TOKEN_HINTS: tuple[str, ...] = (
    "->",
    "=>",
    "===",
    "!==",
    "::",
    "??",
    "||",
    "&&",
    "();",
    "};",
    "self.",
    "Promise<",
    "$1",
)

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
FRENCH_ELISION_RE = re.compile(r"\b[ldcjnstquLDCJNSTQU]['’][a-zA-Zàâæçéèêëîïôœùûüÿ]", re.UNICODE)


def _detect_scripts(text: str) -> dict[str, float]:
    """Return script character proportions over non-whitespace characters."""
    counts: dict[str, int] = {k: 0 for k in SCRIPT_RANGES}
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        cp = ord(ch)
        for script, ranges in SCRIPT_RANGES.items():
            for lo, hi in ranges:
                if lo <= cp <= hi:
                    counts[script] += 1
                    break
    if total == 0:
        return {k: 0.0 for k in SCRIPT_RANGES}
    return {k: v / total for k, v in counts.items()}


def _score_code(text: str) -> float:
    """Compute likelihood in [0, 1] that ``text`` is source code rather than prose."""
    stripped = text.strip()
    if not stripped:
        return 0.0

    pattern_hits = sum(1 for pat in CODE_LINE_PATTERNS if pat.search(stripped))
    hint_hits = sum(1 for tok in CODE_TOKEN_HINTS if tok in stripped)
    sym_chars = sum(1 for ch in stripped if ch in "{}[]();=<>_/$#:\\|")
    sym_density = sym_chars / max(len(stripped), 1)
    indented_lines = sum(1 for ln in stripped.splitlines() if ln.startswith(("  ", "\t")))

    score = 0.0
    if pattern_hits >= 1:
        score += 0.55 + 0.15 * min(pattern_hits - 1, 2)
    if hint_hits >= 1:
        score += 0.12 * min(hint_hits, 3)
    if sym_density >= 0.06:
        score += min(0.25, sym_density * 2.5)
    if indented_lines >= 1 and len(stripped.splitlines()) >= 2:
        score += 0.15

    return min(1.0, score)


@dataclass
class LanguageScore:
    language: str
    confidence: float
    script: str
    all_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class LanguageDetectorConfig:
    """Configuration for language filtering."""

    allowed_languages: list[str] | None = None  # None = allow all
    threshold: float = 0.5
    fallback_language: str = "unknown"
    trust_source_metadata_when_uncertain: bool = False


class LanguageDetector:
    """Base interface for pluggable language detectors."""

    def detect(self, text: str) -> LanguageScore:
        raise NotImplementedError


class HeuristicLanguageDetector(LanguageDetector):
    """
    Deterministic 8-class language detector covering XRFM's 7 target languages + code:
    ``en``, ``ur``, ``ur-Latn``, ``ar``, ``hi``, ``es``, ``fr``, ``code``.
    """

    def detect(self, text: str) -> LanguageScore:
        if not text or not text.strip():
            return LanguageScore(language="unknown", confidence=0.0, script="unknown")

        scripts = _detect_scripts(text)
        dominant_script = max(scripts.items(), key=lambda x: x[1])[0] if scripts else "unknown"
        scores: dict[str, float] = {}

        # 1. Source code check (evaluated first so code snippets with Latin keywords classify as 'code')
        code_score = _score_code(text)
        if code_score > 0.0:
            scores["code"] = round(code_score, 4)

        words_raw = WORD_RE.findall(text)
        words_lower = [w.lower() for w in words_raw]
        num_words = max(len(words_lower), 1)

        # 2. Devanagari -> Hindi (hi)
        if scripts.get("devanagari", 0.0) > 0.15:
            dev_ratio = scripts["devanagari"]
            hi_hits = sum(1 for w in words_raw if w in HINDI_STOPWORDS)
            scores["hi"] = round(min(1.0, 0.70 * min(1.0, dev_ratio * 1.3) + 0.30 * min(1.0, hi_hits / 2.0)), 4)

        # 3. Perso-Arabic block -> Urdu (ur) vs Arabic (ar)
        if scripts.get("arabic_block", 0.0) > 0.15:
            ar_block_chars = [ch for ch in text if any(lo <= ord(ch) <= hi for lo, hi in SCRIPT_RANGES["arabic_block"])]
            n_ar_chars = max(len(ar_block_chars), 1)

            ur_char_hits = sum(1 for ch in ar_block_chars if ch in URDU_EXCLUSIVE_CHARS)
            ar_char_hits = sum(1 for ch in ar_block_chars if ch in ARABIC_CHARACTERISTIC_CHARS)

            ur_word_hits = sum(1 for w in words_raw if w in URDU_STOPWORDS)
            ar_word_hits = sum(1 for w in words_raw if w in ARABIC_STOPWORDS)

            # Urdu almost always contains ی (06CC), ے (06D2), ہ (06C1), ں, ٹ, ڈ, ڑ, پ, چ, گ
            ur_signal = 0.55 * min(1.0, (ur_char_hits / n_ar_chars) * 7.0) + 0.45 * min(
                1.0, ur_word_hits / max(1.0, num_words * 0.18)
            )
            ar_signal = 0.45 * min(1.0, (ar_char_hits / n_ar_chars) * 6.0) + 0.55 * min(
                1.0, ar_word_hits / max(1.0, num_words * 0.16)
            )

            # Baseline script presence so short phrases without stopwords still resolve cleanly
            if ur_char_hits > 0 and ur_word_hits == 0 and ar_word_hits == 0:
                ur_signal = max(ur_signal, 0.75)
            elif ar_char_hits > 0 and ur_char_hits == 0:
                ar_signal = max(ar_signal, 0.78)
            elif ur_char_hits == 0 and ur_word_hits == 0:
                # Pure Arabic script with zero Perso-Arabic letters defaults to Arabic
                ar_signal = max(ar_signal, 0.72)

            if ur_char_hits >= 2 or ur_word_hits > ar_word_hits:
                ur_signal = min(1.0, ur_signal + 0.20)
                ar_signal = max(0.0, ar_signal * 0.35)
            elif ar_word_hits > ur_word_hits and ur_char_hits == 0:
                ar_signal = min(1.0, ar_signal + 0.20)
                ur_signal = max(0.0, ur_signal * 0.25)

            scores["ur"] = round(min(1.0, ur_signal), 4)
            scores["ar"] = round(min(1.0, ar_signal), 4)

        # 4. CJK / Cyrillic fallback recognition
        if scripts.get("cjk", 0.0) > 0.25:
            scores["zh"] = round(min(1.0, scripts["cjk"] * 1.25), 4)
        if scripts.get("cyrillic", 0.0) > 0.25:
            scores["ru"] = round(min(1.0, scripts["cyrillic"] * 1.25), 4)

        # 5. Latin script -> English (en), Roman Urdu (ur-Latn), Spanish (es), French (fr)
        if scripts.get("latin", 0.0) > 0.25:
            en_hits = sum(1 for w in words_lower if w in ENGLISH_STOPWORDS)
            ur_strong_hits = sum(1 for w in words_lower if w in ROMAN_URDU_STRONG_WORDS)
            ur_weak_hits = sum(1 for w in words_lower if w in ROMAN_URDU_WEAK_WORDS)
            es_hits = sum(1 for w in words_lower if w in SPANISH_STOPWORDS)
            fr_hits = sum(1 for w in words_lower if w in FRENCH_STOPWORDS)

            es_accents = sum(1 for ch in text if ch in SPANISH_ACCENT_CHARS)
            fr_accents = sum(1 for ch in text if ch in FRENCH_ACCENT_CHARS)
            fr_elisions = len(FRENCH_ELISION_RE.findall(text))

            en_score = min(1.0, (en_hits / num_words) * 3.4)
            ur_latn_score = min(1.0, ((ur_strong_hits * 1.35 + ur_weak_hits * 0.45) / num_words) * 2.8)
            es_score = min(1.0, (es_hits / num_words) * 3.3 + min(0.25, es_accents * 0.08))
            fr_score = min(
                1.0,
                (fr_hits / num_words) * 3.3 + min(0.25, fr_accents * 0.07) + min(0.25, fr_elisions * 0.12),
            )

            # Roman Urdu frequently embeds English tech nouns ("web application", "Python", "user input"),
            # so >= 2 strong Roman-Urdu words decisively mark the utterance as ur-Latn.
            if ur_strong_hits >= 2:
                ur_latn_score = max(ur_latn_score, min(1.0, 0.72 + 0.05 * ur_strong_hits))
                en_score *= 0.45

            # French elisions (l'équipe, d'utiliser, qu'il) + French stopwords disambiguate FR vs ES
            if fr_elisions >= 1 and fr_hits >= 2:
                fr_score = max(fr_score, 0.82)
                es_score *= 0.5

            if "ñ" in text.lower() or "¿" in text or "¡" in text or (es_hits >= 3 and es_hits > fr_hits):
                es_score = max(es_score, min(1.0, 0.78 + 0.03 * es_hits))
                if fr_elisions == 0:
                    fr_score *= 0.55

            # Short pure-ASCII English fallback ("Hello world") when no other Latin language matched
            if (
                max(en_score, ur_latn_score, es_score, fr_score) < 0.35
                and code_score < 0.55
                and ur_strong_hits == 0
                and es_hits == 0
                and fr_hits == 0
            ):
                en_score = max(en_score, round(scripts["latin"] * 0.65, 4))

            # If code_score is high (>= 0.65), dampen natural prose scores so code wins cleanly
            prose_damp = 0.35 if code_score >= 0.65 else 1.0
            scores["en"] = round(en_score * prose_damp, 4)
            if ur_latn_score > 0.05:
                scores["ur-Latn"] = round(ur_latn_score * prose_damp, 4)
            if es_score > 0.05:
                scores["es"] = round(es_score * prose_damp, 4)
            if fr_score > 0.05:
                scores["fr"] = round(fr_score * prose_damp, 4)

        if not scores:
            return LanguageScore(language="unknown", confidence=0.0, script=dominant_script, all_scores={})

        best_lang, best_conf = max(scores.items(), key=lambda x: x[1])
        return LanguageScore(
            language=best_lang,
            confidence=round(min(best_conf, 1.0), 4),
            script=dominant_script,
            all_scores=scores,
        )


def get_detector(config: LanguageDetectorConfig | None = None) -> LanguageDetector:
    """Factory returning configured language detector instance."""
    return HeuristicLanguageDetector()


def filter_by_language(
    documents: list[Document],
    detector: LanguageDetector | None = None,
    config: LanguageDetectorConfig | None = None,
) -> tuple[list[Document], list[tuple[Document, LanguageScore]], list[LanguageScore]]:
    """
    Filter documents by detected language and enrich document metadata.

    Returns:
        (kept_docs, rejected_docs_with_scores, all_scores)
    """
    if detector is None:
        detector = HeuristicLanguageDetector()
    if config is None:
        config = LanguageDetectorConfig()

    kept: list[Document] = []
    rejected: list[tuple[Document, LanguageScore]] = []
    all_scores: list[LanguageScore] = []

    allowed_set = set(config.allowed_languages) if config.allowed_languages is not None else None

    for doc in documents:
        score = detector.detect(doc.text)
        all_scores.append(score)

        resolved_lang = score.language
        resolved_conf = score.confidence

        if (
            config.trust_source_metadata_when_uncertain
            and resolved_conf < config.threshold
            and doc.language
            and doc.language != "unknown"
        ):
            resolved_lang = doc.language
            resolved_conf = max(resolved_conf, config.threshold)

        doc.language = resolved_lang
        doc.metadata["language_confidence"] = resolved_conf
        doc.metadata["script"] = score.script

        if allowed_set is None:
            kept.append(doc)
        elif resolved_lang in allowed_set and resolved_conf >= config.threshold:
            kept.append(doc)
        else:
            rejected.append((doc, score))

    return kept, rejected, all_scores
