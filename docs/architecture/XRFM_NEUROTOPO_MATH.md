# XRFM-NeuroTopo (XRFM-NT) — Mathematical Specification

**Status:** PROPOSAL (pre-implementation)
**Companion to:** `docs/architecture/XRFM_NEUROTOPO_SPEC.md`
**Date:** 2026-08-12
**Notation key.** Scalars lowercase; vectors bold lowercase `h`; matrices bold capitals `M`;
`t` = token/time index; `l` = layer; module index `i ∈ [N]`; d = module dim.
Equations marked **[v1]** are in the first implementation; **[staged]** are later.

---

## 1. Input embedding and module injection  [v1]

Token ids `x_t ∈ V` map to embeddings `e_t = E[x_t] ∈ R^d` with `E` shared (weight-tied)
with the output head. A causal convolution over a small window `w` (Mamba-style) gives
local token context:

```
u_t = Conv1D_causal(e_{t-w+1:t})           ∈ R^d
```

Modules are initialised/updated by injecting `u_t`. A learned injection matrix
`P ∈ R^{d × (N·d)}` (or per-module gains `g_i`) distributes the token signal:

```
h^{(l)}_{t,0,i} = h^{(l)}_{t-1,i} + g_i · P_i u_t        i = 1..N
```

At sequence start, `h^{(l)}_{0,i} = 0`.

---

## 2. Per-module local dynamics  [v1]

Each module runs a cheap recurrent update. **v1 default: gated recurrence (diagonal,
parallel-scan trainable):**

```
f_t = σ(W_f h_{t,i} + U_f u_t + b_f)          (forget)
i_t = σ(W_i h_{t,i} + U_i u_t + b_i)          (input)
c_t = tanh(W_c h_{t,i} + U_c u_t + b_c)       (candidate)
h_{t,i} ← f_t ⊙ h_{t-1,i} + i_t ⊙ c_t
```

**v1.1 alternative: diagonal/selective SSM (Mamba-2/SSD-class).** A discretized linear
system with input-dependent `Δ, B, C`:

```
h_t = Ā(Δ_t) h_{t-1} + B̄(Δ_t, B_t) u_t
y_t = C_t h_t
```

with `Ā = exp(Δ A)`, `B̄ = (Δ A)^{-1}(exp(Δ A) − I) · Δ B` (ZOH), and `Δ,B,C` produced
from the input (selectivity). This is the content-dependent-gating path. Chosen by config
`neurons.local_dynamics ∈ {gru, ssm}`.

---

## 3. Topology generator:  A_t = f(H_t, M_t, context)  [v1]

### 3.1 Local prior
A fixed or slowly-learned neighborhood defines always-on edges:

```
E_local = { (i, j) : j ∈ kNN-lattice(i, k_local) }
```

### 3.2 Long-range candidate scores
Low-rank bilinear score plus a memory term, for every ordered pair (i, j):

```
s_{ij} = (1/√r) (U h_i)^T (V h_j) + b
            + γ · (P m_i)^T (Q m_j)                [memory term, optional]
```

where `U,V,P,Q ∈ R^{r×d}`, `r ≪ d` (low rank), `m_i` is module i's slice of episodic
memory (§5), and `γ` a learned scalar.

### 3.3 Sparsification
Keep the top-`k_long` outgoing edges per module:

```
N^long(i) = TopK_j( s_{ij}, k = k_long )
```

Gates are `g_{ij} = σ(s_{ij})` for selected edges (and 0 otherwise). Local edges always
survive with gate `σ(s^local_{ij})`.

### 3.4 Plasticity / temporal decay  [v1]
Edges persist across tokens with gated decay (analogous to DeltaNet-2 retention):

```
A_{t,ij} = α_{ij} · A_{t-1,ij} + (1 − α_{ij}) · g_{ij}   for (i,j) ∈ E_local ∪ N^long(i)
α_{ij}    = σ(w_α^T [h_i; h_j; m_i] + b_α)
```

`A_t` is reset to zero at each new sequence. (A sparse edge list + gates is stored, not
an N×N dense matrix.)

### 3.5 Edge features  [v1]
Each edge has a small learned embedding `e_{ij} = Emb_edge(type_{ij}) ∈ R^{d_e}`, where
`type` is a soft mixture over {neutral, support, contradict} produced from the sign/
alignment of `h_i, h_j` — these are the "evidence edges" used by the confidence head.

---

## 4. Sparse message passing  [v1]

Messages and aggregation over the dynamic neighborhood `N_t(i) = E_local ∪ N^long(i)`:

```
msg_{j→i} = W_msg [ h_j ; e_{ij} ]
â_i        = Σ_{j ∈ N_t(i)} A_{t,ij} · msg_{j→i}
h_i        ← GRU_cell(h_i, â_i)             (or residual + LayerScale)
```

With `|N_t(i)| = k_local + k_long = O(1)` per module, message passing is `O(N k d)` per
layer per token — independent of sequence length.

---

## 5. Episodic associative memory (gated delta rule)  [v1]

A fixed-capacity memory matrix `M ∈ R^{d_k × d_v}` (per layer or shared), read and
written every step.

### 5.1 Key/query/value
```
k_t = φ_k(W_k pool(H_t)) ∈ R^{d_k}
q_t = φ_q(W_q pool(H_t)) ∈ R^{d_k}
v_t = φ_v(W_v pool(H_t)) ∈ R^{d_v}
```
where `pool` aggregates module states (attention over modules or mean) and φ is L2 norm
for k/q (following Gated DeltaNet) / identity for v.

### 5.2 Surprise-gated, channel-wise erase/write (Gated Delta Rule-2)
```
read_t      = M_t^T k_t                              ∈ R^{d_v}
err_t       = v_t − read_t                           (prediction error / surprise)
erase_t     = σ(W_e [h_t; k_t; err_t]) ∈ [0,1]^{d_k} (key-axis channel gate)
write_t     = σ(W_w [h_t; v_t; err_t]) ∈ [0,1]^{d_v} (value-axis channel gate)
decay_t     = σ(W_d [h_t; k_t]) ∈ [0,1]              (retention α)
β_t         = σ(W_β [err_t])                         (step size, surprise-modulated)

M_{t+1} = diag(decay_t) M_t
          + β_t · ( write_t ⊙ ( k_t ⊗ (err_t ⊙ erase_t) ) )
```
This unifies: Hebbian association (`k ⊗ v`), error-correcting delta rule (subtract current
read before writing), and channel-wise erase/write (DeltaNet-2), with Titan-style
surprise modulation of the step size. Memory is bounded: `M` has fixed size; no growing
KV cache.

### 5.3 Read into hidden state
```
c_t = M_{t+1}^T q_t          (optionally L2-normalised)
H_t ← H_t + W_c c_t broadcast over modules
```

### 5.4 Persistent memory  [v1]
A frozen-after-pretraining (but learned during pretraining) matrix
`C ∈ R^{K × d}` contributes a persistent context:
```
p_t = SoftMax( q_t C^T /√d ) C          (Titans persistent-memory branch)
H_t ← H_t + W_p p_t
```

---

## 6. Feed-forward sublayer  [v1]
Standard pre-norm SwiGLU (reuse XRFM's implementation for a matched control):
```
h ← h + SwiGLU(RMSNorm(h))
```

---

## 7. Output readout and language modeling  [v1]

After L layers, a learned **module-read query** `r ∈ R^d` attends over modules:
```
α_i = SoftMax_i( r^T h^{(L)}_{t,i} / √d )
z_t = Σ_i α_i h^{(L)}_{t,i} ∈ R^d
```
Logits use the weight-tied embedding:
```
p(x_{t+1}) = SoftMax( z_t E^T )
L_lm = − log p(x_{t+1} = target_t)
```

---

## 8. Uncertainty / evidence head  [v1]

A graph-evidence vector summarises the step's reasoning state:
```
g_t = [ mean(A_t), var(A_t),                  # edge-mass / dispersion
        support_w_t, contradict_w_t,            # Σ typed-edge weights
        margin_t = top1 − top2 (memory read),  # retrieval certainty
        surprise_t = ‖err_t‖,                  # memory prediction error
        effrank(H_t),                          # effective rank of module states
        entropy(module attn α) ]               # routing entropy
```
A 2-layer MLP maps `[z_t; c_t; g_t]` to confidence logits `u_t ∈ R^5` over
{ANSWER, QUALIFY, ABSTAIN, REQUEST_CONTEXT, REQUEST_EVIDENCE}, plus a scalar
calibration logit `κ_t`.

### 8.1 Confidence/abstention loss
With labels `y^u ∈ {0..4}` (from answerable/unanswerable/ambiguous/conflicting/OOD data):
```
L_conf = CE(u_t, y^u_t) + Brier(softmax(u_t), onehot(y^u))
```
plus a calibration penalty:
```
L_cal = ECE-style bin loss OR focal confidence penalty on over-confident errors.
```
Generation uses a decision rule: ANSWER if `p_ANSWER > τ` and `κ calibrated`; else emit
the corresponding qualify/abstain/request token/action.

---

## 9. Auxiliary objectives  [v1, weights set experimentally]

Do **not** stack 15 losses. v1 uses three, all ablatable:

### 9.1 Sparsity / degree control
Targets a desired mean degree `k̄ = k_local + k_long` and prevents hub collapse:
```
L_sparse = ( mean_i |N_t(i)| − k̄ )^2
          + λ_ent · H( routing distribution over modules )      (load balance)
```

### 9.2 Connectivity (spectral)
Penalise graph fragmentation (from STBAM evidence). For the (symmetrised) normalised
Laplacian `L_sym = I − D^{-1/2} A D^{-1/2}` of the module graph:
```
L_connect = ( #(eigenvalues of L_sym near 0) − 1 )^2
```
i.e. one connected component ⇒ exactly one zero eigenvalue. Approximated cheaply with a
few power iterations or a Fiedler-vector proxy (full eigendecomposition only in diagnostics).

### 9.3 Memory / edge prediction (self-supervised, optional)
```
L_mem   = ‖ v_t − M_t^T k_t ‖^2                (associative recall)
L_edge  = BCE( edge predictors, next-step co-activation )   [staged]
```

### 9.4 Total loss
```
L_total = L_lm
        + λ_sparse  L_sparse
        + λ_connect L_connect
        + λ_conf    (L_conf + L_cal)
        + λ_mem     L_mem          (optional)
```
All λ determined by ablation; start with λ_conf = 1 (on labeled-batch steps, 0 otherwise),
λ_sparse, λ_connect small (1e-3–1e-2), λ_mem off until memory alone is validated.

---

## 10. Concept graph (semantic memory)  [staged — NOT v1]

A latent typed graph `G_c = (V_c, E_c)`:
- Nodes `v_a` are concept/entity embeddings produced by a learned read-in from `z_t`.
- Edge logits for types `τ ∈ {supports, contradicts, related, temporal, uncertainty}`:
  ```
  ψ_{ab,τ} = MLP_τ([v_a; v_b; z_t])
  p(edge)  = sigmoid / softmax over τ
  ```
- Neural state reads the concept graph via attention over incident edges:
  `r_a = Σ_{(b,τ)} p_{ab,τ} W_τ v_b`, and writes back via `L_edge`/`edge prediction`.
- The concept graph is **internal and differentiable** (not an external RAG KG).

---

## 11. Computational complexity summary

| Component | Per token, per layer | Sequence dependence |
|---|---|---|
| Embedding/conv | O(d·w) | — |
| Local dynamics | O(N·d) | O(n) total via scan |
| Topology scores (low-rank) | O(N·r·d) | none over n |
| Top-k selection | O(N² log k) over **modules only** (N small) | none |
| Message passing | O(N·k·d) | none |
| Memory read/write | O(d_k·d_v) fixed | O(n·d_k·d_v) total (linear) |
| Persistent memory | O(K·d) | none |
| FFN | O(N·d·d_ff) | — |
| **Total per sequence** | **O(L·(n·(N·k·d + d_k·d_v)))** | **linear in n** |

vs Transformer attention `O(L·n²·d)`. Inference state is fixed-size (module states +
edge list + `M` + `C`), i.e. **constant memory per sequence** — no growing KV cache.

---

## 12. Initialization and stability  [v1]
- All linear/embedding layers: small normal init `std = 0.02/√(2L)` residual scaling
  (GPT-2 style) to stabilise the recurrent+message stack (lesson from XRFM audit F-05).
- `decay_init = 0.9` so edges/memory persist at start; gates initialised near identity.
- RMSNorm pre-norm everywhere; gradient clipping 1.0; bf16 on GPU, fp32 on CPU.
- Topology logits scaled `1/√r`; softmax/top-k numerically stable.
- NaN/Inf guards: nonfinite loss skips the batch (existing XRFM behaviour), plus an
  `error_if_nonfinite` config option.

---

## 13. Determinism & checkpointing  [v1]
- Seed global RNGs + dataloader generator + worker seeds (reuse XRFM's `_set_seed`).
- Checkpoint stores: module states, edge lists/gates, `M`, `C`, optimizer, scheduler,
  config, seed, dataset/tokenizer versions, and the `λ_*` weights — so a resumed run is
  bitwise-equivalent where determinism is supported.
- `torch.compile`/Triton kernels are acceleration only; reference path must produce
  matching numerics within tolerance.

---

## 14. Falsifiable predictions (derived from this spec)
1. **P1 (long context):** constant inference memory → NT's perplexity/needle retrieval
   degrades more slowly than Transformer beyond trained context length.
2. **P2 (memory):** gated delta memory improves multi-key retrieval vs no-memory ablation.
3. **P3 (graph):** dynamic topology beats static topology and no-topology (plain
   recurrent) at matched params.
4. **P4 (uncertainty):** graph-evidence head beats token-entropy/max-prob baselines on
   ECE, selective accuracy, and OOD/conflicting-evidence abstention.
5. **P5 (cost):** at equal quality, NT uses less inference memory and (for n large) less
   compute than the Transformer control.

Each prediction maps directly to an experiment in the training/experiment plan. If a
prediction fails, the corresponding component is removed or redesigned — not defended.
