# XRFM datasets

| File | Contents | Size | License / provenance | Purpose |
|---|---|---|---|---|
| `corpus.txt` | 11 public-domain books (Project Gutenberg: Swift, Carroll, Dickens, Austen, Doyle, Grimm, Shelley, Twain, Stoker, Poe, Wells) + a slice of the Python 3.13 standard library (code) | ~5.5 MB (~1.5–2 M BPE tokens) | Gutenberg texts: public domain (US). Code: Python Software Foundation License. | Real pretraining corpus for XRFM-SMALL experiments. |
| `code_slice.py` | Standalone copy of the PSF-licensed stdlib code slice | ~0.4 MB | PSF License | Code-only experiments / tokenizer diversity check. |
| `sample.txt` | The same short paragraph repeated 100 times | ~30 KB | XRFM project | **Smoke-test toy only.** Explicitly NOT a training corpus (see `docs/audit/FORENSIC_AUDIT.md` F-17). |

## Notes

- Gutenberg boilerplate (headers/footers) was stripped before concatenation.
- The corpus is small and English/European-language dominated. It is suitable for demonstrating that XRFM **learns, overfits, generalizes, and evaluates** — it is not a foundation-model-scale corpus (see `docs/training/COMPUTE_PLAN.md`).
- To rebuild `corpus.txt`: `git show HEAD:data/datasets/gut_*.txt` is not applicable — the individual `gut_*.txt` files were removed after processing; the raw Gutenberg files are reproducible via `https://www.gutenberg.org/cache/epub/{11,1342,1661,2591,84,74,98,345,1080,2148,19033}/pg{id}.txt`.
