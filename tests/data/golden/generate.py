import json
import hashlib
from pathlib import Path

docs = []

def add_doc(text, source="golden", lang="en", license="MIT", doc_id=None):
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if doc_id is None:
        doc_id = f"{source}:{h[:16]}"
    docs.append({
        "document_id": doc_id,
        "source": source,
        "source_uri": f"golden://{doc_id}",
        "license": license,
        "license_url": "https://opensource.org/licenses/MIT",
        "collection": "golden",
        "language": lang,
        "text": text,
        "content_hash": h,
        "metadata": {"golden": True},
        "created_at": "2026-09-18T00:00:00Z",
        "version": "xrfm-doc-v1"
    })

# Normal prose
add_doc("The quick brown fox jumps over the lazy dog. This is a simple English sentence used for testing.", source="web", lang="en")
add_doc("Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data.", source="web", lang="en")
add_doc("In the beginning, there was code. And the code was with the programmer, and the code was the programmer. This is a test of normal prose.", source="books", lang="en")

# Code
add_doc("def hello_world():\n    print('Hello, world!')\n\nif __name__ == '__main__':\n    hello_world()", source="code", lang="en")
add_doc("import torch\nimport torch.nn as nn\n\nclass SimpleModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.linear = nn.Linear(10, 1)\n    def forward(self, x):\n        return self.linear(x)", source="code", lang="en")

# Unicode
add_doc("Unicode test: café, naïve, résumé, π, λ, ∑, 你好, مرحبا, اردو", source="web", lang="en")
add_doc("Emojis: 😀 😃 😄 😁 😆 😅 😂 🤣 — testing Unicode handling", source="web", lang="en")

# Whitespace
add_doc("  This document has leading and trailing whitespace.   \n\n  And multiple blank lines.\n\n\n\nEnd.", source="web", lang="en")
add_doc("Line1\nLine2\nLine3\n\nLine5 after blank", source="web", lang="en")

# Numbers
add_doc("Numbers: 123, 456.789, 0.001, 1e10, 3.14159, 42. Testing number ratio handling.", source="web", lang="en")
add_doc("Math: 1+2=3, 2*3=6, 10/2=5, 5-3=2. Equations and symbols.", source="educational", lang="en")

# Empty / near-empty (should be filtered)
add_doc("", source="web", lang="en", doc_id="empty-1")
add_doc("   ", source="web", lang="en", doc_id="empty-2")
add_doc("a", source="web", lang="en", doc_id="near-empty-1")
add_doc("Hi", source="web", lang="en", doc_id="near-empty-2")

# Duplicate documents (exact)
add_doc("This is a duplicate document. It appears twice in the corpus.", source="web", lang="en", doc_id="dup-1")
add_doc("This is a duplicate document. It appears twice in the corpus.", source="web", lang="en", doc_id="dup-2")  # same text, different id but same hash

# Near duplicates
add_doc("The quick brown fox jumps over the lazy dog. This is a test document for near-duplicate detection.", source="web", lang="en", doc_id="near-dup-1")
add_doc("The quick brown fox jumps over the lazy dog. This is a test document for near duplicate detection!", source="web", lang="en", doc_id="near-dup-2")
add_doc("The quick brown fox jumps over the lazy dog. This is a test document for near-duplicate detection with extra words at the end to make it slightly different.", source="web", lang="en", doc_id="near-dup-3")

# Non-English
add_doc("مرحبا بالعالم، هذا اختبار للغة العربية. يجب أن يكتشف النظام هذه اللغة.", source="web", lang="ar")
add_doc("اردو میں یہ ایک ٹیسٹ دستاویز ہے۔ یہ زبان کی شناخت کی جانچ کے لیے ہے۔", source="web", lang="ur")
add_doc("Bonjour le monde, ceci est un test en français.", source="web", lang="fr")
add_doc("Hola mundo, esta es una prueba en español.", source="web", lang="es")

# Malformed
add_doc("lorem ipsum " * 50, source="web", lang="en", doc_id="boilerplate-1")
add_doc("a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a", source="web", lang="en", doc_id="repetition-1")
add_doc("!@#$%^&*()!@#$%^&*()!@#$%^&*()!@#$%^&*()", source="web", lang="en", doc_id="symbols-1")

# Very long
long_text = "This is a very long document. " * 1000
add_doc(long_text, source="books", lang="en", doc_id="long-1")

# PII examples (should be flagged)
add_doc("Contact me at test@example.com for more info.", source="web", lang="en", doc_id="pii-email-1")
add_doc("My phone number is 123-456-7890, call me.", source="web", lang="en", doc_id="pii-phone-1")
add_doc("API key: sk-1234567890abcdef1234567890abcdef", source="code", lang="en", doc_id="pii-secret-1")

# Code with indentation (must preserve)
add_doc("    def indented():\n        x = 1\n        if x:\n            print(x)\n        return x", source="code", lang="en")

# Markdown structure
add_doc("# Heading\n\nThis is a paragraph with **bold** and *italic*.\n\n- List item 1\n- List item 2\n\n```python\nprint('code block')\n```", source="web", lang="en")

# Save
out_path = Path(__file__).parent / "documents.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for d in docs:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print(f"Wrote {len(docs)} docs to {out_path}")
