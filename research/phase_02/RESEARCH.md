# Phase 2 — Tokenizer: Fresh Research Report

Date: 2026-07-24
Module: Tokenizer (Byte Pair Encoding — original implementation)
Status: Research complete — awaiting architecture validation approval before implementation.

---

## 1. Official Sources Consulted

- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. *Proceedings of ACL*.
- Kudo, T. (2018). Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. *Proceedings of ACL*.
- Raschka, S. (2024). *Build a Large Language Model (From Scratch)*. Manning Publications. (Conceptual reference — tokenizer pipeline, no code copied.)
- Karpathy, A. (2023). *nanoGPT* (GitHub). (Conceptual reference for dataset/token pipeline — no code copied.)
- OpenAI Documentation: `tiktoken` library (conceptual reference for optimized BPE — no dependency adopted for XRFM).
- Hugging Face `tokenizers` documentation (conceptual comparison — no dependency adopted).

---

## 2. Current Best Practices (BPE Tokenization)

Byte Pair Encoding (BPE) is the standard tokenization method for decoder-only language models (GPT family, Llama, Mistral, DeepSeek). Key practices:

### 2.1 Vocabulary Building
- Initialize vocabulary with all individual bytes or characters.
- Count adjacent pair frequencies in the training corpus.
- Iteratively merge the most frequent pair, adding the merged token to vocabulary.
- Stop when vocabulary reaches target size (e.g., 32,768; Llama 3 uses 128,288; Qwen uses ~150,000).

### 2.2 Encoding Process
- Given input text, split into initial tokens (bytes/characters).
- Apply learned merge rules in order (most frequent first, as learned during training).
- Produce integer token IDs based on final vocabulary.

### 2.3 Decoding Process
- Map integer IDs back to vocabulary tokens.
- Concatenate tokens to reconstruct text.
- Note: BPE is approximately lossless for text within the vocabulary; out-of-vocabulary characters may produce `<unk>` or byte-level fallbacks.

---

## 3. Alternative Tokenization Methods (Evaluated, Not Implemented in Phase 2)

### 3.1 SentencePiece (Unigram Model)
- **Approach:** Trains a unigram language model over subword units; selects subword vocabulary to maximize likelihood.
- **Strengths:** Better for multilingual text; handles whitespace naturally; used by T5, Llama 3 (partial), and modern multilingual models.
- **Weaknesses:** More complex training algorithm; vocabulary selection is probabilistic rather than frequency-based.
- **XRFM Classification:** OPTIONAL (post-v0.5.0). Stable `TokenizerInterface` allows future adoption without dataset loader rewrites.

### 3.2 WordPiece (Used by BERT)
- **Approach:** Greedy subword selection based on likelihood improvement; similar to BPE but with different scoring.
- **Strengths:** Good for encoder-only models (BERT); handles unknown words via subword decomposition.
- **Weaknesses:** Less common for decoder-only generative models; vocabulary selection can be less intuitive.
- **XRFM Classification:** OPTIONAL (future encoder-decoder branch, v2.0+).

### 3.3 Unigram (Used by SentencePiece Alternative)
- **Approach:** Direct unigram language model optimization; selects vocabulary and tokenization that maximize corpus likelihood.
- **Strengths:** Theoretically optimal vocabulary selection; handles rare words well.
- **Weaknesses:** More complex to implement correctly; requires probabilistic modeling.
- **XRFM Classification:** RESEARCH-ONLY (future investigation).

### 3.4 TikToken-Style (Optimized BPE)
- **Approach:** BPE with optimized merge rules, special token handling, and faster encoding/decoding (used by OpenAI's modern models).
- **Strengths:** Production-grade performance; handles special tokens (`<|endoftext|>`, chat markers) cleanly.
- **Weaknesses:** Requires careful special token management; optimization adds complexity.
- **XRFM Classification:** OPTIONAL (post-v0.5.0). The `TokenizerInterface` (`save`/`load`) supports special token persistence.

---

## 4. Design Alternatives for XRFM Tokenizer

### Option A: Simple BPE Only (No Interface)
- **Pros:** Fastest implementation; no abstraction overhead.
- **Cons:** Dataset loader and training loop must be rewritten if switching to SentencePiece or optimized BPE later.
- **Rejected:** Violates XRFM principle of future-proof interfaces.

### Option B: BPE + Stable `TokenizerInterface` (Chosen)
- **Pros:** Original BPE implementation; dataset loader depends only on interface (`encode`/`decode`); future algorithms swap without rewrites; supports future special token strategies.
- **Cons:** Slight overhead of abstract base class (minimal performance impact for educational/small-scale training).
- **Chosen:** Aligns with architecture freeze requirements (`tokenizer/DESIGN.md`).

### Option C: Use `tiktoken` Directly (External Dependency)
- **Pros:** Production-grade, optimized, handles special tokens well.
- **Cons:** Violates XRFM principle of full ownership; adds external dependency; harder to customize for research (e.g., custom vocabulary, special dataset formats); does not support easy algorithm swap.
- **Rejected:** Not adopted as core. May be evaluated as optional enhancement for production serving only.

---

## 5. Performance and Scalability Considerations

### 5.1 Vocabulary Size
- 32,768: Standard for small-scale models (GPT-2, early models).
- 50,304: Used by some modern small models (Qwen 1.8B, Phi-3).
- 128,288: Llama 3 (large vocabulary improves multilingual performance but increases embedding layer size).
- 150,000+: Very large vocabulary; requires larger embedding matrix; increases memory footprint.
- **XRFM Recommendation:** Start with 50,304 (`config/config.yaml` default). Configurable via `ConfigLoader.get("model.vocab_size")`. Can scale to 128,288 or larger without code changes.

### 5.2 Training Time for Vocabulary Building
- Small corpus (Tiny Shakespeare, ~1MB): Seconds.
- Medium corpus (WikiText, ~100MB): Minutes.
- Large corpus (OpenWebText, 10GB+): Hours.
- **XRFM Design:** Vocabulary training is a one-time preprocessing step. Trained vocabulary saved to `tokenizer/` (not rebuilt during training). Dataset loader loads pre-trained vocabulary.

### 5.3 Memory Impact
- Vocabulary matrix: `vocab_size × d_model`. For 50,304 vocab and 256 `d_model`: ~51MB (FP32). For 1B model (`d_model` = 2048): ~410MB. This is part of the model, not an additional overhead.
- Token sequences: Longer sequences (larger `max_seq_len`) increase activation memory but do not affect tokenizer size.
- **XRFM Design:** Configurable `vocab_size` allows trade-off between vocabulary richness and embedding layer memory.

---

## 6. Compatibility with XRFM Architecture

### 6.1 Interface Compatibility
- `TokenizerInterface.encode(text: str) -> List[int]` matches dataset loader requirement.
- `TokenizerInterface.decode(List[int]) -> str` supports inference output conversion.
- `vocab_size()` provides vocabulary size to model embedding layer (`nn.Embedding`).
- `save(path)` and `load(path)` support checkpoint persistence (vocabulary saved with checkpoints for reproducibility).

### 6.2 Config Integration
- `ConfigLoader.get_model_config()` returns `vocab_size` which tokenizer uses.
- Changing vocabulary size requires only YAML config change (`model.vocab_size`) and retraining tokenizer — no code rewrites.

### 6.3 Scaling Path Compatibility
- `XRFM-10M`: 50,304 vocab (default) — fine.
- `XRFM-100M`: Same vocab or expanded to 100,000 — configurable.
- `XRFM-1B`: May use larger vocab (128,288) or custom vocabulary — interface unchanged.
- `XRFM-MoE`: Tokenizer interface unchanged; MoE architecture uses same token sequences.
- `XRFM-Multimodal`: Text tokenizer interface unchanged; vision data handled separately (future design, not implemented in Phase 2).

---

## 7. Potential Risks and Mitigations

### Risk: Vocabulary Size Mismatch Between Tokenizer and Model
- **Mitigation:** `ConfigLoader` provides single source of truth. Model embedding layer (`nn.Embedding`) uses `config.model.vocab_size`. Tokenizer saves vocabulary size in metadata. Loading tokenizer validates vocab size against config.

### Risk: Special Token Handling (Chat Markers, End-of-Text, System Prompts)
- **Mitigation:** `TokenizerInterface` design reserves special token support. BPE implementation will include basic special tokens (`<|endoftext|>`, `<|startoftext|>`). Future expansion to chat markers (`<|im_start|>user`, etc.) does not break interface — only requires vocabulary extension.

### Risk: Training Vocabulary on Very Small Corpus Produces Poor Tokenization
- **Mitigation:** This is expected. Small corpora (Tiny Shakespeare) produce basic vocabulary. As dataset scales (WikiText → OpenWebText), vocabulary quality improves. Vocabulary retraining is a preprocessing step, not part of the training loop.

---

## 8. Final Recommendation

**Proceed with Phase 2: Implement original Byte Pair Encoding tokenizer with stable `TokenizerInterface`.**

**Classification:**
- BPE implementation = CORE (required for Phase 2).
- `TokenizerInterface` = CORE (required for future scalability).
- Vocabulary persistence = CORE.
- SentencePiece / Unigram / TikToken = OPTIONAL (post-v0.5.0, interface supports swap).
- Special token expansion = OPTIONAL (post-v0.2.0, interface supports extension).

**No implementation code written yet.** Awaiting approval to proceed to Step 2 (Architecture Validation) and Step 3 (Design Review) before writing `tokenizer/bpe.py`.
