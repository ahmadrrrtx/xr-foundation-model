# XRFM — Target Specification (Phase 11)

**Design rule:** parameter counts are chosen from *measured* compute (COMPUTE_PLAN.md), not hype. All configs share one codebase; only YAML differs.

## 1. Common Architectural Contract (all sizes)

| Component | Specification |
|---|---|
| Architecture | Decoder-only causal transformer, pre-norm residual blocks |
| Normalization | RMSNorm (eps 1e-6), pre-norm placement (norm → sublayer → + residual) |
| Attention | Multi-head self-attention, **explicit causal mask**, RoPE applied to Q/K |
| RoPE | base 10,000; half-split rotation (kept — verified correct; documented) |
| FFN | SwiGLU: `silu(W_g x) ⊗ W_u x` → `W_d`; no bias (matches Llama-family) |
| Residual | `x = x + dropout(sublayer(norm(x)))`, dropout 0.1 default |
| Final norm | RMSNorm before LM head |
| LM head | Linear(vocab) **weight-tied** with embedding |
| Initialization | Embedding/std: normal(0, 0.02) (or xavier for embed at small vocab); linear layers normal(0,0.02)-ish; **optional GPT-2-style residual scaling 1/√(2·n_layers)** — config flag |
| Attention biases | None (matches modern practice) |
| Embedding padding | Dedicated PAD token id (not byte 0) with zero row; loss `ignore_index=PAD` |
| Precision | fp32 on CPU; real bf16 autocast when GPU available (fixed MixedPrecisionLoader) |
| Checkpoint | `model` + `optimizer` + `scheduler` + config + seed + dataset/tokenizer version + step/loss; `map_location` on load |

## 2. Sizes

### XRFM-TINY — unit/integration tests
- vocab 1024 (byte-level BPE), d_model 64, n_layers 2, n_heads 4, d_ff 256, max_seq_len 128
- Params ≈ **0.15 M** (embedding 65 K + per-layer ~45 K×2) → runs in <100 MB RAM, tests in seconds.
- Purpose: unit/integration/overfit tests.

### XRFM-SMALL — local development / default CPU experiments
- vocab 2048, d_model 128, n_layers 4, n_heads 4, d_ff 512, max_seq_len 256
- Params ≈ **0.9–1.1 M** (embedding 262 K + 4 layers × ~160 K) → the workhorse for this CPU sandbox.
- Throughput measured-ish: ≥ 1500 tokens/s → 10 M tokens ≈ 2 h.

### XRFM-MEDIUM — serious experiments (free cloud T4 / 1 GPU)
- vocab 8192, d_model 384, n_layers 8, n_heads 6, d_ff 1536, max_seq_len 1024
- Params ≈ **~15–20 M** (embedding 3.1 M dominates; non-embedding ~11 M). bf16 on GPU; fp32 on CPU fallback for smoke only.

### XRFM-LARGE — available GPU/cloud experiments (24 GB-class)
- vocab 16384, d_model 768, n_layers 12, n_heads 12, d_ff 3072, max_seq_len 2048
- Params ≈ **~70–90 M**. Requires GPU or long CPU runs; NOT for this sandbox (documented, not executed here).

## 3. Training Protocol (all sizes)

| Hyperparameter | TINY/SMALL | MEDIUM | LARGE | Notes |
|---|---|---|---|---|
| Optimizer | AdamW β=(0.9, 0.95) | same | same | research-backed |
| Weight decay | 0.1 (no decay on norms/biases; optional no-decay embeddings) | same | same | Llama/OLMo |
| LR | 6e-4 (TINY) / 3e-4 (SMALL) | 3e-4 | 3e-4 | OLMo-1B/7B-style |
| Warmup | 200–500 steps | 2000 | 2000 | cosine |
| Schedule | cosine → 10% floor | cosine | cosine | |
| Grad clip | 1.0 | 1.0 | 1.0 | |
| Batch | 8–32 (CPU) | 32 × accum | 64 × accum | effective batch documented |
| Seed | fixed per run, stored in checkpoint | same | same | |
| Eval | val loss + PPL every N steps | same | same | |

## 4. Evaluation Protocol
- **Intrinsic:** val loss, token PPL (padding ignored), top-1 next-token accuracy on held-out split (document-boundary split, no leakage).
- **Sanity:** overfit test (must reach train loss < 0.5 on tiny set); generation coherence (greedy + temperature samples); checkpoint resume equivalence (loss at step k == resumed loss).
- **External (optional, when corpus available):** WikiText-2 test PPL (reported with full protocol table: model/params/tokens/data/context/compute/eval-set/metric — RULE in mission).
- Every comparison in the final report will include the full protocol row; XRFM will not be compared against models trained on radically different data/scale without the protocol table.

## 5. Reproducibility Contract (RULE 6)
Every run artifact bundle: commit SHA + config hash + tokenizer file + dataset file hash + seed + full hyperparameters + training log (JSONL) + checkpoint. This is enforced by the new trainer (see REMEDIATION_PLAN).
