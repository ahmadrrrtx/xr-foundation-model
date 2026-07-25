# Phase 4 — Transformer Architecture: Fresh Research Report

Date: 2026-07-24
Module: Transformer Architecture (Embedding, Positional Encoding, Attention, FFN, Normalization, Residual, Initialization, Output)
Status: Research in progress — design review pending — implementation not yet started.

---

## 1. Official Papers and Sources Consulted

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. *NeurIPS*.
- Su, J., Ahmed, M., Lu, Y., et al. (2023). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv:2104.09864*.
- Touvron, H., Lavril, T., Izacard, G., et al. (2023). Llama: Open and Efficient Foundation Language Models. Meta AI.
- Touvron, H., Martin, L., Stone, K., et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. Meta AI.
- Jiang, A.Q., Sablayrolles, A., Mensch, A., et al. (2023). Mistral 7B. *arXiv:2310.06825*.
- Team, DeepSeek-AI (2024). DeepSeek-V3 Technical Report.
- Team, Qwen (2024). Qwen2.5 Technical Report.
- Team, Gemma (2024). Gemma 2 Technical Report.
- Team, OLMo (2024). OLMo: Accelerating the Science of Language Models.
- Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. (GQA / MQA concepts — reference only, not copied.)
- Press, O., & Wolf, L. (2016). ALiBi: Attention with Linear Biases for Efficient Long-Context Inference.
- Peng, B., Quesnelle, J., et al. (2024). YaRN: Efficient Context Window Extension of Large Language Models.
- Dao, T. (2024). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. *ICML*.
- Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. *NeurIPS*.
- Shazeer, N. (2020). GLU Variants Improve Transformer. (SwiGLU reference — not copied.)
- Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). Training Compute-Optimal Large Language Models. *Chinchilla* (DeepMind).
- Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). Scaling Laws for Neural Language Models. OpenAI.

---

## 2. Component-by-Component Research and Recommendations

### 2.1 Embedding Layer

**What it does:** Maps integer token IDs (`tokenizer.encode()` output) to dense continuous vectors (`d_model` dimensions). This is the first layer of the model.

**Why it exists:** Neural networks cannot process discrete integers directly; embeddings convert discrete tokens into continuous representations that the attention mechanism and feed-forward layers can process using linear algebra.

**Mathematical intuition:** Given vocabulary size `V` and model dimension `d`, the embedding matrix `E ∈ R^(V × d)` maps token index `i` to vector `E[i, :] ∈ R^d`. The embedding layer is essentially a lookup table initialized randomly and learned during training.

**Best practices (official sources):**
- Initialize with scaled normal or uniform distributions (standard practice).
- Tie weights between embedding layer and output projection (`lm_head.weight = embedding.weight`). This reduces parameter count by `V × d` and improves training stability (standard in GPT, Llama, Mistral, DeepSeek).
- Vocabulary size (`V`) configurable via config (`ConfigLoader.get_model_config()["vocab_size"]`).

**Trade-offs:**
- Large vocabulary (128k+) improves multilingual/token coverage but increases embedding layer memory (`V × d`). For `XRFM-10M` (`d=256`, `V=50304`), embedding memory ≈ 51MB (FP32). For `XRFM-1B` (`d=2048`, `V=100000`), ≈ 820MB. Acceptable for modern GPUs but significant.
- Small vocabulary (32k) reduces memory but may split common words into multiple tokens, reducing efficiency.

**XRFM Recommendation:** CORE — Implement original embedding layer with weight tying. Configurable `vocab_size` supports 32k–150k+ without rewrites.

---

### 2.2 Positional Encoding

**What it does:** Injects sequence position information into token embeddings. Without positional information, the transformer would be permutation-invariant (would treat a sentence in any order the same way), which is incorrect for language.

**Why it exists:** Self-attention computes relationships between all token pairs simultaneously; it has no inherent concept of "before" or "after". Positional encoding adds this temporal/spatial structure.

**Options evaluated:**

**A. Learned Positional Embeddings (BERT, early GPT)**
- Learnable parameter matrix `P ∈ R^(max_seq_len × d_model)` added to token embeddings.
- Simple, works well for fixed-length sequences.
- Does not naturally extend to longer sequences than `max_seq_len` (requires interpolation or fine-tuning).
- Used in BERT and early transformer variants.

**B. Sinusoidal Positional Encodings (Original Transformer)**
- Fixed sinusoidal functions of different frequencies.
- No learnable parameters; naturally extends to any sequence length.
- Less effective for very long sequences compared to modern methods.

**C. Rotary Positional Embedding (RoPE) — Llama, Qwen, Mistral, DeepSeek**
- Encodes relative position via rotation matrices applied to query/key vectors.
- Mathematical property: attention score depends on relative distance `(m - n)`, not absolute positions.
- Excellent for long sequences; naturally supports length extension (with scaling adjustments like YaRN or NTK-aware scaling).
- Standard in modern LLMs (Llama 3, Qwen 2.5, Mistral 7B, DeepSeek-V3).
- **XRFM Recommendation:** CORE. Implement original RoPE. Configurable (`use_rope: true/false` in config) for future comparison with ALiBi or learned embeddings.

**D. ALiBi (Attention with Linear Biases)**
- Adds a linear bias to attention scores based on relative distance (`-m × |i - j|`).
- Simple, no rotation matrices; naturally supports arbitrary sequence lengths.
- Used in some BLOOM variants and newer research (MPT-7B, some Mistral variants).
- **XRFM Recommendation:** RESEARCH-ONLY. RoPE is standard and sufficient for Phase 4; ALiBi can be evaluated as an optional replacement in future versions (post-v1.0) without interface changes.

**E. YaRN / LongRoPE (Length Extension)**
- Modifications to RoPE to extend context length without fine-tuning (YaRN uses temperature scaling; LongRoPE uses interpolation and rescaling).
- Essential for 128k+ context models (Llama 3.1 128k, Qwen 2.5 128k, DeepSeek 64k).
- **XRFM Recommendation:** RESEARCH-ONLY for Phase 4. Context extension is a Phase 9+ optimization (after basic model works). The RoPE interface supports future extension via configuration adjustments.

**Trade-offs summary:**
- RoPE = best balance of modern performance, long-sequence support, and implementation simplicity. Standard choice.
- Learned embeddings = simpler but limited to fixed lengths; not recommended for scalable LLM.
- Sinusoidal = robust but outperformed by RoPE in modern benchmarks.
- ALiBi = interesting alternative; deferred.
- YaRN/LongRoPE = optimization for very long contexts; deferred.

**XRFM Recommendation:** CORE — Original RoPE implementation (`xrfm/model/attention/rope.py` or integrated into attention module). Configurable (`use_rope` in `config/config.yaml`).

---

### 2.3 Multi-Head Attention

**What it does:** Computes attention scores between all token pairs in parallel, splits into multiple "heads" (each learning different relationship types), and concatenates results.

**Why it exists:** Self-attention captures both local and long-range dependencies; multi-head design allows the model to attend to different types of relationships (syntax, semantics, pronoun resolution, etc.) simultaneously.

**Options evaluated:**

**A. Standard Multi-Head Attention (Manual Implementation)**
- Full control over Q/K/V projections, attention score computation, softmax, and output projection.
- Required for research modifications (GQA, MQA, sliding window, sparse patterns).
- **XRFM Recommendation:** CORE. Implement original manual multi-head attention (`xrfm/model/attention/multi_head.py`).

**B. `nn.MultiheadAttention` (PyTorch Native)**
- Optimized, well-tested, but hides internal details.
- Harder to extend with GQA, sliding window, or custom masking patterns.
- **XRFM Recommendation:** RESEARCH-ONLY (optional enhancement — could be used for comparison or performance benchmarks, but original manual implementation is preferred for control and extensibility).

**C. Grouped Query Attention (GQA) — Llama 2/3, Mistral, DeepSeek**
- Reduces number of KV heads while keeping Q heads high. Reduces KV cache memory by sharing KV projections across query groups.
- Essential for very large models (70B+) and long-context inference.
- **XRFM Recommendation:** OPTIONAL (post-v0.5.0 or v1.0+). The standard multi-head attention interface (`MultiHeadAttention`) should be designed so GQA can replace the attention mechanism without changing model block interfaces (`TransformerBlock`). The architecture freeze (`ARCHITECTURE_REVIEW.md`) confirms this: GQA requires only attention module modifications, not full rewrites.

**D. Multi-Query Attention (MQA) — Earlier variant**
- Single KV head shared across all Q heads. More memory-efficient than GQA but potentially lower quality.
- Used in some earlier efficient models.
- **XRFM Recommendation:** RESEARCH-ONLY (optional replacement; GQA preferred over MQA for quality).

**E. FlashAttention / FlashAttention-2**
- Reorders attention computation (tiling + recomputation) to achieve `O(N)` memory instead of `O(N²)`.
- Mathematically identical result to standard attention; pure optimization.
- Integrated in PyTorch 2.0+ (`torch.nn.functional.scaled_dot_product_attention`).
- **XRFM Recommendation:** OPTIONAL (Phase 9 optimization). Manual multi-head attention in Phase 4; FlashAttention can replace the score computation internally without changing the `MultiHeadAttention` interface (drop-in optimization).

**F. Sliding Window Attention (Mistral, Longformer)**
- Limits attention to a local window (`w` tokens) plus a few global tokens. Reduces complexity from `O(N²)` to `O(N × w)`.
- Essential for extremely long contexts (1M+ tokens, though rarely needed for standard LLMs).
- **XRFM Recommendation:** RESEARCH-ONLY. Sliding window requires modifications to attention masking patterns. The interface supports custom masks; sliding window is a future optional enhancement.

**Trade-offs summary:**
- Manual multi-head = best for control, extensibility, research modifications.
- FlashAttention = pure performance optimization (optional, Phase 9).
- GQA = scalability optimization for large models (optional, post-v1.0).
- Sliding window = extreme long-context optimization (research-only for standard LLM).

**XRFM Recommendation:** CORE — Manual multi-head attention (`xrfm/model/attention/multi_head.py`). FlashAttention = OPTIONAL (Phase 9, drop-in). GQA = OPTIONAL (post-v1.0). Sliding window = RESEARCH-ONLY.

---

### 2.4 Feed-Forward Network (FFN)

**What it does:** Applies a non-linear transformation to each token's representation independently (after attention). Expands dimension (typically `4 × d_model`), applies activation, then projects back.

**Why it exists:** Attention captures relationships between tokens; FFN captures complex non-linear patterns within each token's representation. The combination of attention + FFN allows the model to learn both relational and representational features.

**Options evaluated:**

**A. Standard GELU (`nn.GELU`)**
- Standard activation in original Transformer and many modern models.
- Smooth approximation to ReLU; performs well empirically.
- **XRFM Recommendation:** Not adopted as default. SwiGLU preferred over GELU for modern architectures.

**B. ReLU (`nn.ReLU`)**
- Simple, fast, but less smooth; some studies (Shazeer 2020, SwiGLU) show SwiGLU outperforms ReLU/GELU in language modeling.
- **XRFM Recommendation:** RESEARCH-ONLY (optional for comparison; not recommended as default).

**C. SwiGLU (Swish + Gated Linear Unit) — Llama, Mistral, Qwen, DeepSeek**
- Uses `Swish` (or `SiLU`) activation with a gating mechanism (`W_1`, `W_2`, `W_3`).
- More parameters than standard FFN (`3 × d_model × d_ff` vs `2 × d_model × d_ff`) but empirically improves quality.
- Standard in modern open-source LLMs (Llama 3, Qwen 2.5, Mistral, DeepSeek-V3).
- **XRFM Recommendation:** CORE. Implement original SwiGLU (`xrfm/model/layers/swiglu.py`) with configurable use (`use_swiglu` in config).

**D. GEGLU (Gated GELU)**
- Gated variant of GELU; less common than SwiGLU in modern LLMs.
- **XRFM Recommendation:** RESEARCH-ONLY (optional; SwiGLU preferred).

**Trade-offs summary:**
- SwiGLU = best empirical performance for modern LLMs; slightly more parameters; standard choice.
- GELU = simpler; acceptable but outperformed by SwiGLU.
- ReLU = simplest; not recommended for state-of-the-art LLM.
- GEGLU = less common alternative; not recommended as default.

**XRFM Recommendation:** CORE — Original SwiGLU (`xrfm/model/layers/swiglu.py`). Configurable (`use_swiglu` in `config/config.yaml`).

---

### 2.5 Normalization

**What it does:** Normalizes activations to stabilize training (prevent exploding/vanishing gradients) and improve convergence speed.

**Why it exists:** Deep networks (32+ layers) suffer from gradient instability without normalization. Normalization ensures that inputs to each layer have consistent mean and variance.

**Options evaluated:**

**A. LayerNorm (Post-Norm)**
- Original Transformer: normalization applied after the sub-layer (attention or FFN) and before the residual connection.
- More stable for shallow networks; can cause gradient issues in very deep networks.
- **XRFM Recommendation:** Not adopted as default. Pre-norm preferred.

**B. LayerNorm (Pre-Norm) — Modern Standard**
- Normalization applied before the sub-layer (attention or FFN), then added to the residual connection.
- More stable for deep networks (32+ layers); standard in modern LLMs (Llama, Mistral, Qwen, DeepSeek).
- **XRFM Recommendation:** CORE — Pre-norm architecture with LayerNorm. Configurable (`use_rmsnorm` selects RMSNorm variant).

**C. RMSNorm (Root Mean Square Normalization)**
- Removes mean-centering step (`(x - mean) / sqrt(var + eps)` → `x / sqrt(mean(x²) + eps)`).
- Slightly faster (no mean computation); empirically similar performance to LayerNorm.
- Used in Llama 3, Qwen 2.5, DeepSeek-V3, Mistral.
- **XRFM Recommendation:** CORE. Implement original RMSNorm (`xrfm/model/layers/rmsnorm.py`). Configurable (`use_rmsnorm` in config). Fallback to standard LayerNorm if `use_rmsnorm` is false (optional, for comparison/research).

**Trade-offs summary:**
- RMSNorm = standard for modern LLMs; slightly faster; similar quality to LayerNorm.
- LayerNorm (pre-norm) = robust; slightly more computation; widely tested.
- Post-norm = not recommended for deep networks.

**XRFM Recommendation:** CORE — Original RMSNorm (`use_rmsnorm: true` by default); LayerNorm available as fallback (`use_rmsnorm: false`, optional for comparison/research).

---

### 2.6 Residual Connections

**What it does:** Adds the input of a sub-layer to its output (`x + Attention(x)` or `x + FFN(x)`). This allows gradients to flow directly through the network, preventing vanishing gradients in deep architectures.

**Why it exists:** Essential for training very deep networks (100+ layers). Without residuals, gradients decay exponentially through layers.

**Options evaluated:**
- Standard residual (`x + SubLayer(x)`) — universal standard.
- Pre-activation residual (`SubLayer(LayerNorm(x)) + x`) — modern standard (pre-norm + residual).
- Post-activation residual (`LayerNorm(x + SubLayer(x))`) — original Transformer.

**XRFM Recommendation:** CORE — Pre-activation residual (`SubLayer(LayerNorm(x)) + x`). Matches modern architecture (Llama, Mistral, DeepSeek, Qwen). Config not required (always enabled); architecture design document confirms this decision.

---

### 2.7 Initialization Strategy

**What it does:** Determines how model weights are initialized before training. Good initialization prevents early divergence and improves training speed.

**Why it exists:** Random initialization with incorrect scaling causes loss divergence or very slow convergence. Proper scaling ensures gradients have appropriate magnitudes at the start of training.

**Options evaluated:**

**A. Xavier / Glorot Initialization**
- Scales weights by `sqrt(6 / (fan_in + fan_out))`. Standard for linear and embedding layers.
- **XRFM Recommendation:** CORE for embedding layer and linear projections.

**B. Kaiming / He Initialization**
- Scales weights by `sqrt(2 / fan_in)`. Optimized for ReLU/GELU activations.
- **XRFM Recommendation:** CORE for FFN layers (`SwiGLU` uses `SiLU` activation; Kaiming/He appropriate).

**C. GPT-Style Initialization (Scaled Normal)**
- Initializes weights with `N(0, σ²)` where `σ` is scaled by `1 / sqrt(N_layers)` or similar factor for very deep networks.
- Used in GPT-2, GPT-3, some modern variants.
- **XRFM Recommendation:** OPTIONAL. The architecture includes enough layers (6 for 10M, 12 for 100M, 24 for 1B) that standard Xavier/Kaiming works; GPT-style scaled initialization can be adopted for very deep networks (70B+) but is not required for Phase 4.

**D. DeepNorm**
- Special initialization for very deep networks (100+ layers) that scales weights and biases differently.
- **XRFM Recommendation:** RESEARCH-ONLY. Only needed for extremely deep models; current depth (6–24 layers) handled by standard initialization.

**Trade-offs summary:**
- Xavier/Kaiming = robust standard; sufficient for 10M–7B scale.
- GPT-style scaled initialization = optional optimization for very deep models.
- DeepNorm = research-level; deferred.

**XRFM Recommendation:** CORE — Xavier initialization for embeddings and projections; Kaiming/He for SwiGLU layers. Configurable initialization methods reserved (`use_deepnorm` optional, `use_gpt_init` optional) for future versions without rewrites.

---

### 2.8 Output Head (Unembedding / Projection Layer)

**What it does:** Maps the final hidden state (`d_model`) back to vocabulary space (`vocab_size`) to produce token prediction logits.

**Why it exists:** The model's internal representation (`d_model`) is different from the vocabulary space (`vocab_size`). A projection layer converts hidden representations into probabilities over the vocabulary.

**Options evaluated:**

**A. Tied Weights (`lm_head.weight = embedding.weight`) — Standard in GPT, Llama, Mistral, Qwen**
- Shares the same weight matrix between embedding and output projection.
- Reduces parameter count by `V × d` (significant for large vocabularies).
- Improves training stability by ensuring embedding and projection learn consistent representations.
- **XRFM Recommendation:** CORE. Implement weight tying (`xrfm/model/gpt.py`: `self.lm_head.weight = self.embedding.weight`). Configurable (`tie_weights` optional for research comparison).

**B. Separate Projection (`nn.Linear(d_model, vocab_size)`)**
- Independent projection layer; no weight sharing.
- More parameters; sometimes used in very large models (some GPT-4 variants reportedly use separate projections, though not confirmed officially).
- **XRFM Recommendation:** OPTIONAL (`tie_weights: false` for comparison/research only). Weight tying is standard and preferred.

**Trade-offs summary:**
- Tied weights = standard practice; reduces memory; improves stability; no known quality degradation.
- Separate projection = optional for very large models; increases memory.

**XRFM Recommendation:** CORE — Weight tying enabled by default (`tie_weights: true`). Separate projection available as optional (`tie_weights: false`) for research comparison.

---

### 2.9 Numerical Stability

**What it does:** Ensures computations remain numerically stable during training and inference (avoiding NaN, Inf, or extreme gradient values).

**Why it exists:** Deep networks with softmax attention and large matrices are prone to numerical instability (overflow in softmax, underflow in gradients).

**Best practices:**
- **Softmax scaling:** Scale attention scores by `1 / sqrt(d_k)` before softmax (standard; prevents saturation).
- **Precision handling:** Mixed precision (`torch.cuda.amp`) for faster training; master weights in FP32 for stability.
- **Gradient clipping:** `torch.nn.utils.clip_grad_norm_` with `grad_clip` value (Phase 5 training loop).
- **Masking:** Proper masking for padding tokens (prevents attention to padded positions).
- **Initialization:** Xavier/Kaiming initialization (prevents initial gradient explosion).
- **Layer normalization:** Pre-norm with RMSNorm (stabilizes activations through deep networks).

**XRFM Recommendation:** CORE — All numerical stability practices implemented (attention score scaling, gradient clipping in training loop, pre-norm normalization, proper masking, initialization). No shortcuts.

---

## 3. Component Classification for Phase 4 Implementation

Based on the research above, the following components will be implemented in Phase 4:

**CORE (Required — Implement in Phase 4):**
- Embedding layer (`embedding`) with weight tying (default) and Xavier initialization.
- RoPE (`rope`) — original implementation.
- Manual multi-head attention (`multi_head_attention`) — original implementation.
- RMSNorm (`rmsnorm`) — original implementation (pre-norm).
- SwiGLU (`swiglu`) — original implementation.
- Transformer block (`transformer_block`) — original implementation (pre-norm + residual).
- GPT model architecture (`gpt_model`) — original implementation (embedding + RoPE + blocks + output projection with weight tying).
- Numerical stability practices (softmax scaling, masking, gradient clipping hooks for Phase 5).

**OPTIONAL (Can be added later — Interface designed but not activated):**
- FlashAttention (`flash_attention` integration — Phase 9 optimization; manual attention interface supports drop-in replacement).
- GQA (`grouped_query_attention` — Phase 8+; attention interface supports future GQA without full rewrites).
- ALiBi (`alibi` — research-only; attention masking interface supports future ALiBi patterns).
- Separate output projection (`tie_weights: false` — optional for research comparison; weight tying is default).
- GPT-style scaled initialization (`gpt_init` option in config) — optional; Xavier/Kaiming is default.

**RESEARCH-ONLY (Future investigation — Not implemented in Phase 4):**
- DeepNorm (`deepnorm`) — only needed for extremely deep networks (100+ layers); deferred.
- YaRN / LongRoPE (`longrope`, `yarn`) — context extension; deferred to Phase 9+.
- Sliding window attention (`sliding_window_attention`) — deferred; attention masking interface supports future patterns.
- Triton custom kernels (`triton_kernels`) — maximum performance customization; deferred.
- Mamba / State Space (`mamba`, `state_space`) — alternative architecture branch; deferred to v3.0.

---

## 4. Architecture Validation (Phase 4 Readiness)

**Config compatibility:** `ConfigLoader.get_model_config()` provides all architecture parameters (`vocab_size`, `d_model`, `n_layers`, `n_heads`, `d_ff`, `max_seq_len`, `dropout`, `use_rope`, `use_rmsnorm`, `use_swiglu`). The model architecture (`gpt_model`) will use these parameters directly.

**Interface stability:** The `MultiHeadAttention` interface is designed so GQA (`grouped_query_attention`) can replace it without changing `TransformerBlock` or `GPTModel` interfaces. RoPE interface supports future YaRN/LongRoPE extensions via configuration adjustments (`rope_scaling` parameter, optional).

**No circular dependencies:** Embedding layer depends only on vocabulary size; attention depends only on `d_model` and `n_heads`; transformer block combines attention and FFN; GPT model combines embedding, blocks, and output projection. Each layer is independent.

**Future scalability confirmed:**
- `XRFM-10M` (`d_model=256`, `n_layers=6`): Works with Phase 4 architecture.
- `XRFM-100M` (`d_model=768`, `n_layers=12`): Works by changing config only.
- `XRFM-1B` (`d_model=2048`, `n_layers=24`): Works by changing config; FSDP (optional, Phase 8) handles memory.
- `XRFM-7B` (`d_model=4096`, `n_layers=32`): Works with FSDP; GQA (optional) reduces KV cache for long contexts.
- `XRFM-MoE` (`d_model=2048`, `n_layers=32`, `experts=8`): Requires MoE layer module (`model/moe.py`, v3.0); dataset loader and training loop interfaces unchanged.
- `XRFM-Multimodal`: Requires vision encoder (`model/vision_encoder.py`); text tokenizer (`tokenizer/interface`) unchanged; dataset loader (`data/loader.py`) can handle multimodal metadata in future versions.
- `XRFM-Reasoning`: Requires reasoning dataset (`data/datasets/reasoning/`) and RL-based training loop extensions; model architecture (`attention`, `transformer_block`) unchanged; dataset loader interface unchanged.

---

## 5. Implementation Plan (Phase 4 — Step-by-Step)

Given the engineering execution protocol (10 steps per phase), Phase 4 implementation follows this sequence:

**Step 4.1: Embedding Layer (`model/embedding.py`)**
- Original `nn.Embedding` with Xavier initialization.
- Weight tying (`lm_head.weight = embedding.weight`) — implemented in `GPTModel`.
- Type hints, docstrings, error handling.

**Step 4.2: Positional Encoding (`model/attention/rope.py`)**
- Original RoPE implementation (rotation matrices, frequency scaling).
- Configurable (`use_rope` in config; can disable for comparison/research).

**Step 4.3: Multi-Head Attention (`model/attention/multi_head.py`)**
- Original multi-head attention (manual Q/K/V projections, attention score computation, softmax with `sqrt(d_k)` scaling, output projection).
- Masking support (`attention_mask` parameter) for padding tokens.
- Stable interface for future GQA/sliding window replacements.

**Step 4.4: RMSNorm (`model/layers/rmsnorm.py`)**
- Original RMSNorm implementation (removes mean-centering, uses `sqrt(mean(x²) + eps)`).
- Configurable (`use_rmsnorm` in config; standard LayerNorm as optional fallback).

**Step 4.5: SwiGLU (`model/layers/swiglu.py`)**
- Original SwiGLU implementation (`W_1`, `W_2`, `W_3`, `SiLU` activation, gating mechanism).
- Configurable (`use_swiglu` in config; standard GELU/ReLU as optional fallbacks).

**Step 4.6: Transformer Block (`model/layers/transformer_block.py`)**
- Original pre-norm transformer block (`norm -> attention + residual -> norm -> FFN + residual`).
- Residual connections included by design (no config required; always enabled).
- Stable interface (`TransformerBlock`) for future modifications (e.g., MoE replacement of FFN).

**Step 4.7: GPT Model (`model/gpt.py`)**
- Original decoder-only model (embedding + positional encoding + stacked blocks + final normalization + output projection with weight tying).
- Config-driven architecture (reads `ConfigLoader.get_model_config()` for all hyperparameters).
- Stable public interface (`forward`, `config`).

**Step 4.8: Numerical Stability (`model/numerical_stability.py` — optional helper)**
- Original documentation/module explaining softmax scaling, masking, initialization choices, gradient flow.
- Not a separate runtime module; concepts integrated into attention and model code.

**After all sub-steps:**
- Tests (`tests/test_model_*.py`) — shape validation, gradient flow verification, numerical stability checks, masking correctness, initialization checks.
- Documentation (`docs/model/ARCHITECTURE.md`) — updated with Phase 4 architecture details, tensor shapes, mathematical derivations, design decisions.
- Benchmark (`benchmark/model_forward.py`) — basic forward pass timing and memory usage for small model.
- Self-review (correctness, readability, maintainability, extensibility, performance, documentation, tests).
- Git commit proposal (Phase 4 complete).
- `DECISIONS.md` update (if any new architecture decisions — e.g., confirmation of RoPE over ALiBi, SwiGLU over GELU, manual attention over native multihead).

---

## 6. Final Readiness Statement (Phase 4 Pre-Implementation)

**Status:** Design freeze complete. Implementation ready.

**What is ready:**
- All research completed (official papers, engineering blogs, repository references).
- All 7 architecture components researched with trade-offs and recommendations.
- Implementation plan defined (8 sub-steps from embedding to full model).
- Stable interfaces designed (`MultiHeadAttention`, `TransformerBlock`, `GPTModel`).
- Config integration confirmed (`ConfigLoader.get_model_config()` provides all required parameters).
- Original implementation confirmed (no copied source; all concepts cited; independent coding from first line).
- Future scalability confirmed (MoE, Multimodal, Reasoning all supported by current interface design).
- Testing framework ready (`tests/` directory exists; pytest configured).
- Benchmark framework reserved (`benchmark/` directory exists; basic benchmark planned).
- Documentation framework ready (`docs/model/ARCHITECTURE.md` reserved; design notes documented in this file).
- Security checks confirmed (input validation for mask parameters, numerical stability safeguards).
- Performance philosophy confirmed (correctness > maintainability > optimization).

**What requires approval before implementation:**
- Phase 4 implementation authorization.
- No additional architecture changes required (design validated against freeze docs).
- Implementation can begin with embedding layer (sub-step 4.1) and proceed sequentially.

**What will not change during Phase 4 implementation:**
- Tokenizer interface (`tokenizer/interface.py`) — stable.
- Dataset loader (`xrfm/data/loader.py`) — stable (uses tokenizer interface only).
- Config loader (`xrfm/config/loader.py`) — stable.
- Professional open-source files (`LICENSE`, `CONTRIBUTING.md`, etc.) — stable.
- `DECISIONS.md` — will receive updates only if new architecture decisions arise during implementation (e.g., confirmation of RoPE over ALiBi, SwiGLU over GELU).

**Recommendation:** Proceed with Phase 4 implementation. Begin with embedding layer (`model/embedding.py`) and proceed sequentially through attention, normalization, FFN, block, and full model. Each sub-step includes its own mini self-review before moving to the next.
