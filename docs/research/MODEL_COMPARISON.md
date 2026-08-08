# XRFM — Comparative Model Research (Phase 9)

**Purpose:** Compare XRFM against relevant open foundation models using primary sources (official papers, technical reports, official repos), then classify which techniques are A) essential, B) useful-but-optional, C) premature optimization, D) irrelevant at XRFM's scale.

Sources cited: Llama 3 model card & Meta blog; Llama 3 technical report details (via primary report summary); OLMo 2 paper (arXiv:2501.00656); Pythia (EleutherAI, HF model cards); Mistral 7B paper (arXiv:2310.06825); plus established facts for GPT-2, Llama 2, Qwen2.5, Gemma 2.

---

## 1. Comparison Table

| Model | Params | Arch highlights | Vocab (tokenizer) | Context | Training tokens | Optimizer / LR / Batch | Data strategy |
|---|---|---|---|---|---|---|---|
| **GPT-2** (Radford et al. 2019) | 117M–1.5B | Decoder-only, LayerNorm post-norm, learned pos emb, dropout 0.1 | 50,257 byte-level BPE | 1024 | WebText ~40 GB (~8–9 B tokens) | Adam, lr 2.5e-4, batch 512, grad clip 1.0 | Web scrape, quality filtering |
| **Llama 2** (Touvron et al. 2023) | 7B–70B | Pre-RMSNorm, RoPE, SwiGLU, no bias | 32 K SentencePiece | 4096 | 2 T | AdamW β(0.9,0.95), lr 3e-4, wd 0.1, clip 1.0, warmup 2000, cosine, batch 4 M tokens | Public data mix, dedup, ~90%× more data than LLaMA 1 |
| **Llama 3** (Meta 2024) | 8B, 70B, 405B | GQA, RoPE (θ=500k), no dropout, masked document boundaries | 128 K tiktoken-style BPE | 8192 (→128 K after RoPE scaling) | 15 T+ (8B/70B); 15.6 T (405B) | Cosine, peak lr 8e-5 (405B), 8000 warmup steps; batch 4M→8M→16M tokens; 3.9% tokens = code | New public mix; URL/doc/line dedup; n-gram coverage, dirty-word, token-distribution KL, fastText language filters; **attention mask prevents cross-document attention** |
| **Mistral 7B** (2023) | 7B | GQA (8 KV heads), **sliding window 4096**, rolling buffer cache, no bias, 32 layers, d=4096 | 32 K SentencePiece | 8192 (SWA) | unknown (~8 T est.) | AdamW | Public data, no SWA masking in practice for most benchmarks |
| **Mixtral 8x7B** (2024) | ~47B (12.9B active) | MoE, 8 experts, top-2 routing, GQA, SWA | 32 K | 32 K | ~9 T est. | AdamW | Same data class as Mistral |
| **Qwen2.5** (2024–25) | 0.5B–72B | GQA, SwiGLU, QKV bias in attention, RoPE, up to 128 K context | 151,936 (tiktoken-like) | 32 K/128 K | 18 T (72B) | AdamW, cosine, wd 0.1 | Massive web mix incl. multi-language |
| **Gemma 2** (2024) | 2B, 9B, 27B | GeGLU, **logit soft-capping**, alternating local(1024)/global attention, RoPE, RMSNorm, **final-norm pre-lm-head + post-norm** | 256 K (SentencePiece) | 8192 | 13 T (9B) | AdamW, cosine | Filtered web + code, multilingual |
| **OLMo 2** (AI2 2024) | 1B, 7B, 13B, 32B | MHA (7B/13B), GQA (32B), pre-RMSNorm, SwiGLU, RoPE; **no weight decay on embeddings**; stability via data filtering (DCLM) | 100 K BPE | 4096 | 5 T (7B), 4 T (1B), 6.5 T (32B) | AdamW, peak lr 3e-4 (7B)/4e-4 (1B)/6e-4 (32B), warmup 2000, cosine→linear anneal, grad clip 1.0, batch 512–2048 × 4096 | **Fully open: data, code, logs, 1000s of checkpoints**; quality filtering + dedup |
| **Pythia** (EleutherAI 2023) | 70M–12B | GPT-NeoX-style: RoPE, no weight tying, zero dropout | 50 K BPE (GPT-NeoX tokenizer) | 2048 | ~300 B (all sizes, identical data order) | Adam, lr 1e-3 (70M)→1.2e-4 (12B), batch 2 M tokens, 143 k steps | The Pile; deduped & non-deduped variants for science |
| **XRFM (current)** | 19.2 M | MHA, RoPE (half-split), SwiGLU, RMSNorm pre-norm, weight tying, dropout 0.1, biases everywhere | 408-token char BPE (committed); config claims 50,304 | 512 | Toy (one sentence ×100) | AdamW lr 1e-3, wd 0.01, cosine+warmup; **no bf16, no seed, no val loop** | One repeated paragraph; char-range splits |

## 2. What the Research Actually Shows (for XRFM)

1. **Causal masking is fundamental and explicit everywhere.** Every production decoder (GPT-2 → Llama 3) applies a causal mask; Llama 3 additionally masks cross-document attention within packed sequences. XRFM's implicit masking (via SDPA import success) does not meet this bar (F-01).
2. **Byte-level BPE with full Unicode coverage is the industry default** (GPT-2 byte BPE; Llama 3 tiktoken; Qwen2.5 tiktoken-like; OLMo BPE). Whitespace-preserving tokens (`Ġ`/`▁` conventions) are essential. XRFM's char-level latin-1 tokenizer (F-11/F-12) is the single biggest obstacle to real training.
3. **Hyperparameters converge on a narrow range for small models:** AdamW β=(0.9,0.95) (Llama 2/3, Qwen), wd 0.1, grad clip 1.0, cosine decay with ~2000-step warmup, peak LR 3–4e-4 (100M–1B scale), batch ≈ 0.5–2 M tokens at 1B+ scales (proportionally smaller for tiny models). XRFM's lr=1e-3/wd=0.01 are plausible only for very small models; clip 1.0 matches.
4. **Dropout is typically 0 for pretraining** (Pythia, Llama 3); GPT-2 used 0.1. XRFM's 0.1 is acceptable for small-scale but should be config-tunable.
5. **Weight tying**: GPT-2 ties embeddings+unembeddings; Llama/Mistral/Qwen do NOT tie; Pythia does not tie. For a 19 M model with a 50 K vocab, tying is the right call (embedding is 67% of params). Keep it.
6. **Initialization**: GPT-2 uses scaled residual init; Llama uses small std (~0.02) linear inits. XRFM's pure Xavier is fine at 6 layers but is a documented risk at depth (F-05).
7. **Data > architecture at fixed compute** (Llama 3: "quality data remains the king"; OLMo: DCLM filtering alone reproduced big gains; Chinchilla/Kaplan: token-budget trade-offs). XRFM's single repeated sentence is the antithesis; even a "scientific toy" run should use real diverse text (e.g., a slice of a public corpus) to be interpretable.
8. **Reproducibility is a first-class artifact in open science**: OLMo releases data, code, logs, and thousands of checkpoints; Pythia releases 154 checkpoints per model with identical data order. XRFM checkpoints lack config/seed/data version (F-33) — below even toy-grade scientific standards.
9. **GQA, MoE, sliding window, speculative decoding, quantization** are efficiency techniques for large models; at 19 M params on CPU they are **premature** (C-class) except quantization, which is harmless.
10. **bf16/AMP**: every modern run uses bf16 or fp16 with autocast; XRFM's "mixed precision" does nothing (F-23). On CPU-only hardware this is moot (fp32 is fine), but the config/doc claims must be corrected.

## 3. Classification for XRFM (given its actual scale: ≤ ~100 M params, CPU/1-GPU)

| Technique | Class | Rationale |
|---|---|---|
| Explicit causal mask, built-in, tested | **A — Essential** | Correctness of the LM task; current implicit mechanism is fragile (F-01) |
| Byte-level BPE, whitespace-preserving, Unicode-safe | **A — Essential** | Tokenizer cannot represent text otherwise (F-11/F-12) |
| Real data (diverse text, dedup, split at boundaries) | **A — Essential** | Otherwise every metric is meaningless (F-17–F-20) |
| Loss masking (padding ignore_index; optional pack boundaries) | **A — Essential** | Padding currently trains on garbage (F-15/F-39) |
| Seeding + reproducible dataloader + config-in-checkpoint | **A — Essential** | Reproducibility rule (F-25/F-33) |
| Scheduler state in checkpoints / resume | **A — Essential** | Resume currently breaks LR schedule (F-24) |
| Validation loss + perplexity during training | **A — Essential** | Need real signal; currently absent (F-27) |
| AdamW β(0.9,0.95), wd 0.1, cosine, warmup, clip 1.0 | **B — Useful but optional** | Adopt for consistency with research; lr depends on scale |
| Weight tying | **A (keep)** | Correct at this vocab/param ratio |
| RoPE half-split → keep; optionally switch to interleaved | **B** | Internally correct; interleaved only matters for HF compat |
| Dropout=0.1 pretraining | **B** | Keep configurable; modern pretraining uses 0 |
| Scaled residual init | **B** | Cheap stability insurance at depth |
| No-bias attention/FFN | **B** | Marginal at this scale |
| KV cache (already present) | **A** | Verified correct |
| bf16/autocast on GPU | **B** | Only relevant when GPU available; fix claims now |
| GQA, MoE, SWA, speculative decoding, torch.compile | **C — Premature** | No measurable benefit at 19 M on CPU; keep code but mark research-only |
| FlashAttention | **C** | SDPA fallback is fine; keep wrapper but make masking explicit |
| SentencePiece/tiktoken/Unigram | **C/D** | Current BPE is sufficient once made byte-level |
| RoPE θ scaling to 500k (long context) | **D — Irrelevant** | Context 512–2048 |
| Multi-GPU FSDP tuning, throughput MFU work | **D** | No GPU available; document as future |
