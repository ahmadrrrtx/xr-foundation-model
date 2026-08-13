# XRFM Frontier — Dynamic Graphs, Sparse Topology & Routing

**Date:** 2026-08-12
**Purpose:** Investigate the central mechanism — a learned, dynamic, sparse connectivity
(graph) that is *part of computation*: `A_t = f(H_t, memory, context)`.

## 1. What "dynamic topology" can mean (distinct mechanisms)

1. **Learned static adjacency.** A parameter/generator produces one graph shared across
   inputs (e.g., node-embedding outer product). Cheap, but not context-dependent.
2. **Input-dependent adjacency.** `A_t = g(H_t)` — the graph changes per token/context.
   This is what the mission targets.
3. **Temporal rewiring.** The graph evolves over sequence time with memory/plasticity
   (edges persist, decay, grow). Dynamic GNN lineage.
4. **Conditional computation / routing.** A sparse subgraph is *activated* per input
   (MoE token-to-expert routing; neuron-level sparse execution).
5. **Training-time topology evolution.** Connectivity changes during training via
   dynamic sparse training (RigL), then (optionally) fixed at inference.

XRFM-NT should combine **(2)+(3)+(4)** at inference and optionally **(5)** during
training. Pure dense `A_t` over all d neurons is O(d²) and is explicitly rejected.

## 2. Evidence from the literature

- **Learnable adjacency from features:** STBAM uses a Transformer encoder over a block
  adjacency to infer missing temporal links, regularized so the Laplacian has a single
  zero eigenvalue (graph stays connected) [1](https://arxiv.org/html/2310.02606v1).
  ADLGNN/AAGCN build dynamic dependency graphs from data with no pre-defined graph,
  initializing adjacency statistically and updating it with attention
  [2](https://www.sciencedirect.com/science/article/abs/pii/S002002552201581X)
  [3](https://www.nature.com/articles/s41598-024-60598-2).
- **Dynamic GNN survey (2024):** integrated GNN+RNN layers, discrete/continuous-time
  dynamic GNNs; spatio-temporal message passing [4](https://arxiv.org/html/2404.18211v1).
- **Dynamic sparse training (RigL/SRigL):** periodically prune low-magnitude and grow
  high-gradient connections; ERK sparsity distribution; matches dense at high sparsity
  [5](https://arxiv.org/pdf/1911.11134) [6](https://arxiv.org/pdf/2305.02299).
  DST ensembles improve OoD robustness *and uncertainty*
  [7](https://openreview.net/pdf?id=RLtqs6pzj1-).
- **MoE routing (DeepSeek-V3):** fine-grained experts, top-k routing, shared expert,
  auxiliary-loss-free bias balancing; 37B active of 671B [8](https://blog.prompt20.com/posts/mixture-of-experts-serving/).
- **Hypergraphs:** HGraphormer injects hypergraph Laplacian into attention;
  one-stage node→node message passing unifies two-stage methods
  [9](https://arxiv.org/abs/2312.00336). EHNN gives maximally expressive equivariant
  hypergraph layers [10](https://www.ecva.net/ecva/papers/eccv_2022/papers_ECCV/papers/136810086.pdf).

## 3. Design constraints derived from the evidence

- **Sparsity must be structured to run fast.** Unstructured sparse adjacency does not
  accelerate stock GPUs. Use block/top-k/local-neighborhood sparsity and (later) N:M
  sparsity or custom Triton gather-scatter.
- **Keep the graph connected.** A graph that fragments into components loses gradient
  flow and expressivity; a spectral/connectivity regularizer (Fiedler value, zero-eigenvalue
  count) is cheap and evidence-backed [1].
- **Edges need gating, not just existence.** Edge weights should be gated (sigmoid) and
  decayed over time; otherwise the graph saturates. This mirrors retention gates in
  linear attention and erase/write gates in DeltaNet-2.
- **Modularity/hierarchy.** Enforce local neighborhoods (cheap, dense) + a small number
  of learned long-range "hub" links — the cortical principle and the practical way to
  get both locality and long-range without O(n²).
- **Routing = conditional computation.** Per-context activation of expert/subgraph
  modules gives real sparsity (MoE-style) and is the most proven path to "activate only
  relevant neural subgraphs."

## 4. Candidate mechanism: Dynamic Topology Generator

For a layer/block with N neural *units* (units = grouped neurons/modules, not individual
scalars), given state H_t ∈ R^{N×d}, memory S, context c_t:

1. **Local edges:** fixed or slowly-learned kNN/local grid (cheap, dense within module).
2. **Long-range candidates:** score unit pairs via low-rank bilinear form
   `score(i,j) = (U h_i)^T (V h_j) / √d` plus a memory term `M[:,i]^T M[:,j]`.
3. **Top-k selection:** keep top-k outgoing edges per unit (sparse, blockable), plus
   always-on local edges. Produces `A_t` with gating values in (0,1).
4. **Plasticity/decay:** `A_t = σ(decay ⊙ A_{t-1} + new_scores)` with erase/write gates
   (DeltaNet-2 analogy); reset between sequences.
5. **Message passing:** `H'_i = update(H_i, Σ_{j∈N_t(i)} edge(i,j) · message(H_j, e_ij))`.
6. **Regularizers:**
   - `L_sparse`: entropy/L1 on edge usage (target mean degree k).
   - - `L_connect`: spectral — penalize more than one near-zero Laplacian eigenvalue
     (connectivity) [1].
   - `L_diversity`: routing entropy / load balance across modules (MoE-style).
7. **Uncertainty coupling:** edge *agreement* (support vs contradict evidence) feeds the
   confidence head (UNCERTAINTY.md).

This makes the graph genuinely part of computation: it routes messages, gates memory
writes, and supplies uncertainty signals — not a visualization.

## 5. Scaling & compute

- N must be modest (e.g., 64–1024 *modules*, each a small neuron group), not d individual
  neurons, to keep top-k/edge ops tractable.
- O(L · N · k · d) per layer instead of O(L · n² · d) attention, where n is sequence
  length; orthogonal to sequence length (good for long context).
- A Triton fused top-k + gather-scatter kernel is a later optimization; a pure-PyTorch
  reference using `torch.topk` + index_add works for correctness/overfit at small scale.

## 6. Risks
- Top-k routing instability / collapse (mitigate with MoE load balancing, noise,
  expert dropout).
- The graph may learn to be trivially uniform (collapse to attention-like) or identity
  (no communication); regularizers and local-prior initialization prevent this.
- Dynamic graphs complicate batching/KV-style inference; design a recurrent state
  (edge set + node states) from the start.

## 7. What NOT to do
- Do not build a dense N×N adjacency over hidden units.
- Do not claim unstructured sparsity gives GPU speedups.
- Do not treat the graph as an attention mask around an otherwise-unchanged Transformer
  (mission rule: graph must be part of computation).
