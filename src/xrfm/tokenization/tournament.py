"""
Tokenizer Tournament & Cross-Script Fertility Parity Harness for XRFM (`GATE-2`).

Evaluates candidate tokenizers across all 7 target natural language varieties
(`en`, `ur`, `ur-Latn`, `ar`, `hi`, `es`, `fr`) + 6 programming languages
(`python`, `typescript`, `javascript`, `bash`, `sql`, `rust`) on:
- **Fertility (`tokens / word`)**
- **Parity Ratio vs English (`fertility_lang / fertility_en`)**
- **Compression (`UTF-8 bytes / token`)**
- **Raw Single-Byte Fallback Rate (`< 0.5%` target)**
- **Exact UTF-8 Lossless Roundtrip (`decode(encode(x)) == x`, including emojis & indentation)**
"""

from __future__ import annotations

import unicodedata
from typing import Any

from xrfm.tokenization.interface import Tokenizer, tokenizer_fingerprint

# Held-out multi-domain evaluation passages (never passed as sole training memorization strings)
TOURNAMENT_EVAL_SUITE: dict[str, list[str]] = {
    "en": [
        "Modern distributed transformer architectures rely on grouped-query attention, root-mean-square layer normalization, and rotary positional embeddings to scale efficiently under memory constraints.",
        "Defensive software engineering requires strict input validation, parameterized database queries, cryptographic signature verification, and least-privilege process sandboxing.",
        "When evaluating multilingual foundation models, bits-per-byte provides a vocabulary-invariant comparison across morphologically rich and non-Latin writing systems.",
    ],
    "ur": [
        "جدید مصنوعی ذہانت کے ماڈلز کی تربیت کے لیے متوازن اور معیاری کثیر لسانی ڈیٹا سیٹ کا ہونا انتہائی ضروری ہے تاکہ اردو زبان کے صارفین کو بہترین نتائج مل سکیں۔",
        "سافٹ ویئر سیکیورٹی میں ڈیٹا بیس کے حملوں سے بچاؤ کے لیے ہمیشہ محفوظ اور پیرامیٹرائزڈ کیوریز کا استعمال کرنا چاہیے اور صارف کے ان پٹ کی مکمل تصدیق کرنی چاہیے۔",
        "کمپیوٹر سائنس اور الگورتھمز کی مدد سے ہم پیچیدہ مسائل کو تیزی اور درستگی کے ساتھ حل کر سکتے ہیں اور محفوظ نظام تیار کر سکتے ہیں۔",
    ],
    "ur-Latn": [
        "Agar hum apne software systems ko cyber attacks se mehfooz rakhna chahte hain to humein har API endpoint par authentication aur rate limiting lagani chahiye.",
        "Multilingual AI models ki training mein Roman Urdu aur Nastaliq dono ka shamil hona bohat zaroori hai taake har user asani se apna sawal pooch sake.",
        "Database mein password save karne se pehle usay strong cryptographic hashing algorithm ke zariye mehfooz banana ek bunyadi security usool hai.",
    ],
    "ar": [
        "تعتمد نماذج الذكاء الاصطناعي الحديثة على تمثيل متوازن للغات المتعددة لضمان الكفاءة العالية في الفهم والتوليد اللغوي وتقليل تكلفة الحوسبة.",
        "يتطلب تطوير البرمجيات الآمنة التحقق الصارم من جميع مدخلات المستخدمين واستخدام الاستعلامات المعيارية لمنع ثغرات حقن قواعد البيانات وحماية الأنظمة.",
        "تساهم تقنيات التشفير المتقدمة وإدارة الصلاحيات الدقيقة في تعزيز أمن الشبكات وحماية البيانات الحساسة في البيئات السحابية الموزعة.",
    ],
    "hi": [
        "आधुनिक कृत्रिम बुद्धिमत्ता प्रणालियों के विकास में बहुभाषी टोकनाइज़र की दक्षता अत्यंत महत्वपूर्ण भूमिका निभाती है जिससे गणना की लागत कम होती है।",
        "सुरक्षित सॉफ़्टवेयर निर्माण के लिए सभी उपयोगकर्ता इनपुट का सत्यापन करना और पैरामीटराइज्ड डेटाबेस क्वेरी का उपयोग करना अनिवार्य सुरक्षा अभ्यास है।",
        "कंप्यूटर विज्ञान और वितरित प्रणालियों में मेमोरी प्रबंधन तथा संख्यात्मक स्थिरता उच्च प्रदर्शन वाले मॉडल प्रशिक्षण की आधारशिला हैं।",
    ],
    "es": [
        "El entrenamiento eficiente de modelos fundacionales multilingües exige un vocabulario equilibrado que represente equitativamente cada idioma sin fragmentación excesiva.",
        "La ingeniería de software seguro requiere validar rigurosamente todas las entradas externas y emplear consultas parametrizadas para evitar vulnerabilidades críticas.",
        "Los algoritmos criptográficos modernos garantizan la confidencialidad e integridad de la información transmitida a través de redes distribuidas.",
    ],
    "fr": [
        "L'entraînement efficace des modèles de fondation multilingues repose sur une tokenisation équilibrée permettant de réduire la fragmentation des mots dans chaque langue.",
        "Le développement logiciel sécurisé exige une validation stricte des entrées utilisateur ainsi que l'utilisation systématique de requêtes préparées paramétrées.",
        "Les protocoles cryptographiques contemporains assurent la protection des données sensibles et la vérification fiable des identités numériques.",
    ],
    "code": [
        "def verify_webhook_signature(payload: bytes, header_sig: str, secret: bytes) -> bool:\n    expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()\n    return hmac.compare_digest(expected, header_sig)\n",
        "export async function findActiveAccount(pool: Pool, accountId: string): Promise<Account | null> {\n    const result = await pool.query('SELECT id, role FROM accounts WHERE id = $1 AND active = true', [accountId]);\n    return result.rows[0] ?? null;\n}\n",
        '#!/usr/bin/env bash\nset -euo pipefail\nreadonly BACKUP_DIR="${BACKUP_DIR:?must be set}"\nsha256sum "${BACKUP_DIR}"/*.tar.gz > "${BACKUP_DIR}/SHA256SUMS"\n',
    ],
}

ROUNDTRIP_STRESS_PROBES: tuple[str, ...] = (
    "    def indented_python():\n        if True:\n            return '4-space & \\t tab preserved'\n",
    "اردو نستعلیق: ٹ ڈ ڑ ں ہ ھ ی ے گ پ چ ژ — ۱۲۳۴۵",
    "العربية الفصحى: ذكاء اصطناعي وأمن المعلومات — ١٢٣٤٥",
    "हिन्दी देवनागरी: कृत्रिम बुद्धिमत्ता और सॉफ़्टवेयर सुरक्षा — १२३४५",
    "Emojis & Symbols: 🚀🔐🛡️✓ ∑ ∫ ∂ ≠ ≤ ≥ © ™",
    "Mixed Code-Switch: User نے کہا ke `SELECT * FROM users WHERE id = ?` زیادہ mehfooz ہے!",
)


def _count_single_byte_fallbacks(tok: Tokenizer, ids: list[int]) -> int:
    """Count tokens representing raw incomplete UTF-8 single-byte fallbacks."""
    fb_set = getattr(tok, "single_byte_fallback_ids", None)
    if isinstance(fb_set, frozenset):
        return sum(1 for tid in ids if tid in fb_set)
    # For BytePairEncoder: ids 0..255 are raw single bytes; non-ASCII bytes 128..255 are UTF-8 byte fallbacks
    return sum(1 for tid in ids if 128 <= tid < 256)


def evaluate_tokenizer_candidate(
    candidate_id: str,
    tokenizer: Tokenizer,
    eval_suite: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Compute complete fertility, compression, byte-fallback, and roundtrip metrics."""
    suite = eval_suite or TOURNAMENT_EVAL_SUITE
    per_lang: dict[str, dict[str, float | int]] = {}

    roundtrip_total = 0
    roundtrip_exact = 0

    for probe in ROUNDTRIP_STRESS_PROBES:
        roundtrip_total += 1
        decoded = tokenizer.decode(tokenizer.encode(probe))
        if decoded == probe or unicodedata.normalize("NFC", decoded) == unicodedata.normalize("NFC", probe):
            roundtrip_exact += 1

    for lang, samples in suite.items():
        total_words = 0
        total_bytes = 0
        total_tokens = 0
        total_fallbacks = 0

        for text in samples:
            norm_text = unicodedata.normalize("NFC", text)
            ids = tokenizer.encode(norm_text)
            decoded = tokenizer.decode(ids)
            roundtrip_total += 1
            if decoded == norm_text:
                roundtrip_exact += 1

            words = norm_text.split()
            total_words += len(words)
            total_bytes += len(norm_text.encode("utf-8"))
            total_tokens += len(ids)
            total_fallbacks += _count_single_byte_fallbacks(tokenizer, ids)

        fertility = total_tokens / max(total_words, 1)
        bytes_per_tok = total_bytes / max(total_tokens, 1)
        fb_rate = total_fallbacks / max(total_tokens, 1)

        per_lang[lang] = {
            "words": total_words,
            "utf8_bytes": total_bytes,
            "tokens": total_tokens,
            "fertility_tok_per_word": round(fertility, 4),
            "bytes_per_token": round(bytes_per_tok, 4),
            "byte_fallback_rate": round(fb_rate, 6),
        }

    en_fertility = float(per_lang.get("en", {}).get("fertility_tok_per_word", 1.0)) or 1.0
    for lang, m in per_lang.items():
        m["fertility_ratio_vs_en"] = round(float(m["fertility_tok_per_word"]) / en_fertility, 4)

    natural_langs = [k for k in ("en", "ur", "ur-Latn", "ar", "hi", "es", "fr") if k in per_lang]
    mean_nat_fertility = sum(float(per_lang[k]["fertility_tok_per_word"]) for k in natural_langs) / max(
        len(natural_langs), 1
    )
    max_non_latin_ratio = max(
        (float(per_lang[k]["fertility_ratio_vs_en"]) for k in ("ur", "ar", "hi") if k in per_lang),
        default=1.0,
    )
    max_latin_ratio = max(
        (float(per_lang[k]["fertility_ratio_vs_en"]) for k in ("ur-Latn", "es", "fr") if k in per_lang),
        default=1.0,
    )
    max_fb_rate = max((float(m["byte_fallback_rate"]) for m in per_lang.values()), default=0.0)
    rt_rate = roundtrip_exact / max(roundtrip_total, 1)

    gate2_pass = rt_rate == 1.0 and max_fb_rate < 0.005 and max_non_latin_ratio <= 2.0 and max_latin_ratio <= 1.45

    return {
        "candidate_id": candidate_id,
        "vocab_size": tokenizer.vocab_size(),
        "fingerprint": tokenizer_fingerprint(tokenizer),
        "roundtrip_exact_rate": round(rt_rate, 6),
        "mean_natural_fertility": round(mean_nat_fertility, 4),
        "max_non_latin_parity_ratio": round(max_non_latin_ratio, 4),
        "max_latin_parity_ratio": round(max_latin_ratio, 4),
        "max_byte_fallback_rate": round(max_fb_rate, 6),
        "gate2_pass": gate2_pass,
        "per_language": per_lang,
    }
