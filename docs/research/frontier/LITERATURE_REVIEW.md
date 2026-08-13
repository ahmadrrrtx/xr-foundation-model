# XRFM Frontier — Literature Review

**Mission:** XRFM Frontier Architecture Research
**Date:** 2026-08-12
**Scope:** Ten research communities from neuron topology to topological data analysis.
**Method:** Targeted web search (arXiv, OpenReview, NeurIPS/ICML/ICLR, technical blogs,
GitHub, Hugging Face). This is a *breadth-first* map of the landscape; each entry is
a starting point, not an exhaustive reading. Depth is recommended before implementation.

> **Honesty note.** "Brain-inspired" below means *computational principle extracted*,
> not biological fidelity. None of these systems are claimed to work "like the brain."
> Citations use `[id](url)` to the search results that surfaced them.

---

## A. Neuron topology & learned connectivity

The core question: can the *connectivity* of neurons be learned or adapted rather than
fixed by a layer's weight matrix?

- **Learnable adjacency / dynamic graphs.** "Learning Adjacency Matrix for Dynamic GNN"
  uses a Transformer encoder over a block-adjacency to infer missing temporal links,
  with a loss that drives the graph Laplacian toward a single zero eigenvalue
  (one connected component) [1](https://arxiv.org/html/2310.02606v1). ADLGNN and AAGCN
  learn dependencies and an adaptive adjacency jointly with GCN propagation for
  multivariate data where no graph is given
  [2](https://www.sciencedirect.com/science/article/abs/pii/S002002552201581X)
  [3](https://www.nature.com/articles/s41598-024-60598-2). **Take-away:** an adjacency
  can be *generated from node states* (`A_t = f(H_t)`) and regularized by spectral
  properties (connectivity, Fiedler value) — directly relevant to the mission's
  `A_t = f(H_t, memory, context)`.
- **Dynamic graph neural networks (survey).** A 2024 survey categorizes integrated
  GNN+RNN layers, discrete-time and continuous-time dynamic GNNs
  [4](https://arxiv.org/html/2404.18211v1). These are mostly for spatio-temporal
  graph data, not language, but the *message-passing over a time-varying graph*
  mechanism transfers.
- **Neuron2Graph / neuron-as-graph ideas** were searched conceptually; the directly
  relevant, well-evidenced lineage is dynamic sparse training (Section I) rather than
  a single "Neuron2Graph" language model.
- **Caution.** Most "learnable adjacency" work operates on *instance graphs*
  (nodes = entities/time series) with O(n²) adjacency. For language, neurons are not
  naturally "nodes" with pairwise semantics — naively making every hidden unit a node
  is O(d²) and has no structural prior. The graph must be **sparse and structured**
  (modules, local neighborhoods, a few long-range links) to be tractable.

## B. Brain / neuroscience-inspired computation

Useful *principles*, not imitation:

- **Local computation & sparse connectivity.** Cortical circuits are dominated by
  local (nearby) connections with a small number of long-range projections; this
  motivates local-neighborhood message passing + sparse long-range links rather than
  dense all-to-all attention.
- **Synaptic plasticity / Hebbian / STDP.** "Fire together, wire together" update rules
  are the historical root of *associative memory* (Hopfield) and modern linear-attention
  fast-weight updates (see Section E/D). Gated DeltaNet's `S += k ⊗ β(v − Sᵀk)` is
  essentially an error-correcting Hebbian/delta rule [5](https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15).
- **Predictive coding / active inference.** Hierarchical prediction-error minimization;
  provides a principled *objective* (predict inputs/latents, minimize surprise) and
  maps onto the Titans "surprise-based" memory gating (Section E).
- **Liquid time-constant / neural ODE networks.** LTCs are continuous-time recurrent
  nets with state-dependent time constants, formulated as ODEs and integrated with
  fused solvers; they show stable bounded dynamics and strong temporal performance
  with sparse, compact networks [6](https://ar5iv.labs.arxiv.org/html/2006.04439).
  **Take-away:** per-neuron *adaptive timescales* (some fast, some slow) are a useful
  mechanism for multi-scale memory and can be implemented without ODE solvers via
  closed-form continuous-depth (CfC) approximations.
- **Spiking neural networks for LM.** Spiking/neuromorphic LMs exist but are
  predominantly efficiency/edge plays; training (surrogate gradients) is finicky and
  there is no evidence of a quality advantage for text. **Verdict: monitor, do not
  adopt for v1.**

## C. Graph / hypergraph computation

- **Message-passing GNNs** aggregate neighbor features along edges; the expressive
  bottleneck is over-squashing and the locality bias (need many layers for long range).
- **Hypergraph neural networks** model multi-way (higher-than-pairwise) relations.
  HGraphormer injects hypergraph structure (Laplacian) into Transformer attention and
  unifies node→hyperedge→node (two-stage) and direct node→node (one-stage) message
  passing [7](https://arxiv.org/abs/2312.00336). Equivariant Hypergraph NNs (EHNN) use
  hypernetworks/self-attention for maximal expressivity over general hypergraphs
  [8](https://www.ecva.net/ecva/papers/eccv_2022/papers_ECCV/papers/136810086.pdf).
- **Relevance to XRFM.** A *hyperedge* can express "these k concepts/neurons are
  jointly involved in this inference," which is more natural than pairwise edges for
  representing multi-concept composition. Hypergraphs are more expensive; a pragmatic
  design uses pairwise sparse edges for routing and optional hyperedges for
  concept-binding. **Caution:** most GNN/HGNN work is on *relational data*, not
  sequential text; causality and token-order must be engineered in.

## D. Post-Transformer sequence models (most directly competitive)

This is the richest, fastest-moving area and the natural source of mechanisms for XRFM.

| Model | Mechanism | Complexity (train / infer) | Memory | Strengths | Weaknesses |
|---|---|---|---|---|---|
| **Transformer** | Softmax self-attention | O(n²) / O(n²) w/ KV cache | grows w/ n | Exact retrieval, mature, best at scale | Quadratic cost |
| **Mamba (S6)** [9](https://arxiv.org/pdf/2312.00752) | Selective SSM; input-dependent Δ,B,C; hardware-aware scan | O(n) / O(n) state | fixed | Linear scaling, long context, 5× infer throughput | Weaker in-context recall than attention; needs custom CUDA kernels |
| **Mamba-2 (SSD)** [10](https://arxiv.org/abs/2405.21060) | Structured state-space duality; matmul-friendly | O(n) | fixed | 2–8× faster than Mamba, tensor-parallel, GVA heads | Still a fixed-state bottleneck |
| **RWKV (→v7 "Goose")** [11](https://arxiv.org/html/2305.13048v2) | Linear attention as RNN; time-mix + channel-mix, learned decay | O(n) / O(1) per token | fixed | 100% RNN, constant-time decode, scaled to 14B | Single-vector state limits very-long-range fine recall |
| **Gated DeltaNet / DeltaNet-2** [5](https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15) [12](https://arxiv.org/abs/2605.22791) | Delta-rule associative memory matrix; gated/channel-wise erase+write | O(n) chunkwise / O(1) | fixed matrix S | Error-correcting memory; strong retrieval; Triton kernels | Newer; kernel-dependent |
| **RetNet** | Multi-scale retention (linear attention + decay) | O(n) / O(1) | fixed | Parallel train + recurrent infer | Less adoption than Mamba/RWKV |
| **Hyena/S4/S5** | Long-conv / LTI state spaces | O(n log n)–O(n) | fixed | Good on non-text modalities | LTI = not content-selective; weaker on language |
| **Titans (LMM/MAC)** [13](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) | Deep MLP neural long-term memory + attention (short-term) + persistent memory; surprise-gated, test-time learning | near-linear | fixed + persistent | Outperforms Mamba-2/Gated DeltaNet/Transformer++; scales >2M tokens | New (Dec 2025); memory-module cost; needs validation |
| **Hybrids (Jamba, Samba, Zamba, Qwen3-Next)** | Interleave SSM/linear-attn with sparse/sliding-window attention | mostly linear + bounded attention | fixed + small KV | Best of both: SSM memory + exact local/periodic attention | More complex |

**Key cross-cutting insight (MIRAS / Titans)** [13](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/):
a unified view treats sequence models as **online optimization over an associative
memory** — linear attention/Hebbian, RetNet, DeltaNet, and Titans are different
*memory objectives* (dot-product vs MSE) with different *retention gates* and
*memory architectures* (vector/matrix/deep MLP). This is directly the conceptual
vocabulary XRFM needs: **memory as a learned, updatable object, not just a KV cache.**

## E. Memory architectures

- **Working memory** = the current sequence window (attention) and/or a small recurrent
  state (SSM). Precise but bounded.
- **Episodic / long-term memory** = Titans' neural memory module: a deep MLP storing a
  key→value mapping, updated with a momentum+forgetting (weight-decay) rule driven by
  "surprise," learned at test time, parallelizable [13](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/).
  Persistent memory = fixed (frozen-after-training) weights encoding task knowledge.
- **Modern Hopfield / associative memory** [14](https://www.emergentmind.com/topics/modern-hopfield-network-mhn):
  continuous-state, log-sum-exp energy, softmax retrieval; the update rule is
  *mathematically equivalent to one attention step*. Exponential storage capacity;
  capacity depends on pattern separation, improved by encoded representations (HEN).
  This gives a principled basis for a content-addressable concept store.
- **Memory-augmented NNs (NTM, DNC, differentiable memory)** established external
  read/write memory with addressing; mostly superseded by the associative-memory and
  long-context lineage but the read/write/erase interface remains useful.
- **Design conclusion for XRFM.** Do *not* bolt on RAG. Build **three neural memories**:
  (1) working = recurrent state / local attention, (2) episodic/associative =
  gated fast-weight matrix or small neural memory updated during the sequence,
  (3) semantic/persistent = frozen concept embeddings + (optionally) a structured
  concept graph (Section G). Make capacity configurable, not hard-coded.

## F. Knowledge / hallucination / uncertainty

- **Epistemic vs aleatoric.** Epistemic = "I don't know because I wasn't trained on
  it" (should abstain); aleatoric = "genuinely ambiguous/multiple answers" (should
  qualify). "To Believe or Not to Believe Your LLM" derives an information-theoretic
  (mutual-information) lower bound on *epistemic* uncertainty from multiple samples,
  enabling hallucination detection even for multi-answer queries
  [15](https://arxiv.org/html/2406.02543v2).
- **Effective-rank uncertainty** measures the spectral rank of hidden states across
  samples/layers as an OOD/hallucination signal — no extra module required
  [16](https://arxiv.org/html/2510.08389).
- **Abstention architectures.** A composite of instruction-level refusal + a structural
  abstention gate gave 96–98% accuracy / 0–4% hallucination; the *gate alone* missed
  "confidently confabulated" conflicting-evidence cases — motivating explicit
  **conflict detection** [17](https://arxiv.org/html/2604.06195).
- **Incentives/calibration.** I-CALM shows reward-shaped prompts move the
  abstention–hallucination frontier; DCS scores the full belief distribution
  (penalizing error-hedging vs abstention-hedging)
  [18](https://arxiv.org/html/2604.03904) [19](https://arxiv.org/html/2510.04302).
- **Architecture-level lesson for XRFM.** Token probability (max prob, entropy) is
  *necessary but insufficient* — it cannot distinguish "confidently wrong" from
  "confidently right." XRFM needs an **explicit evidence/confidence head** fed by
  hidden + memory state, trained with abstention labels, outputting states
  {ANSWER, QUALIFY, ABSTAIN, REQUEST_CONTEXT, REQUEST_EVIDENCE}, and evaluated with ECE,
  Brier, selective accuracy, abstention rate, OOD AUROC, false-confidence rate. This is
  a *head + objective*, not a system prompt.

## G. Neuro-symbolic / concept graphs

- Differentiable neuro-symbolic reasoning over knowledge graphs (e.g., DiffLogic)
  combines KG embeddings with weighted, differentiable logical rules and a grounding
  filter to keep it scalable [20](https://openreview.net/pdf?id=bETvUctiTR).
- **Relevance to XRFM.** Symbolic structure should live *inside* the cognitive
  architecture as a **latent concept graph** (entity/concept nodes;
  supports/contradicts/related/uncertainty/temporal edges) that neural state reads
  from and writes to — not only as an external KG. Differentiable edge weights let
  gradients teach the graph; discrete/structured edges give interpretability. This is
  high-risk (graph induction from text is unsolved in general) and should be staged.

## H. Topological data analysis (TDA)

- Persistent homology tracks connected components (β₀), loops (β₁), voids across
  filtrations. "Neural persistence" (ICLR 2019) applies TDA to a network's *weighted
  graph* as a structural-complexity/regularization measure
  [21](https://openreview.net/pdf?id=ByxkijC5FQ).
- ANNs can *predict* Betti numbers faster than classical PH pipelines and are more
  noise-robust [22](https://arxiv.org/html/2509.09140).
- **Six candidate uses for XRFM (mission list):**
  1. Architectural constraint — keep the dynamic graph connected (spectral loss).
  2. Routing — use topology/community structure to route messages.
  3. Regularizer — neural-persistence/topology penalty on activation graphs.
  4. Memory structure — track component/loop structure of the concept graph.
  5. Uncertainty signal — topological change in hidden states as OOD signal (early).
  6. Diagnostic — Betti/persistence of representations across layers/training.
- **Caution.** TDA is expensive and currently strongest as a *diagnostic/regularizer*,
  not a runtime compute path. Start with (6) and (1); defer the rest.

## I. Dynamic sparsity

- **RigL** (Evci et al., ICML/ICLR) starts from a random sparse net and periodically
  *prunes* low-magnitude weights and *grows* new ones at high-gradient locations,
  keeping FLOPs proportional to density; matches dense accuracy at high sparsity on
  CNNs/Transformers [23](https://arxiv.org/pdf/1911.11134).
- **SRigL** learns structured, constant-fan-in N:M sparsity (ICLR 2024)
  [24](https://arxiv.org/pdf/2305.02299).
- **Sparse Evolutionary Training (SET)** is the simpler predecessor (magnitude prune +
  random grow). DST ensembles improve accuracy, OoD robustness, *and uncertainty
  estimation* [25](https://openreview.net/pdf?id=RLtqs6pzj1-).
- **Caution for XRFM.** Unstructured sparsity does **not** yield wall-clock speedups on
  stock GPUs; RigL at very high sparsity can ablate neurons / create fan-in imbalance.
  Use sparsity for (a) the *dynamic topology* mechanism itself (context-dependent
  subgraph activation = conditional computation), and (b) structured/block sparsity for
  real speedups — not as a generic "sparse is faster" claim.
- **MoE routing** (DeepSeek-V3) is the most successful form of *conditional* sparsity:
  fine-grained experts + top-k routing + auxiliary-loss-free bias balancing + a shared
  expert; 671B total / 37B active [26](https://blog.prompt20.com/posts/mixture-of-experts-serving/).
  This is the template for "activate only relevant subgraphs."

## J. Evolutionary / self-organizing networks

- **NEAT/HyperNEAT** evolve topology and weights together; historically small-scale,
  expensive, GPU-unfriendly.
- **Plastic neural networks / learned rewiring** overlap heavily with dynamic sparsity
  (Section I). The credible modern version is **DST during training** (RigL-style
  topology evolution) plus **context-dependent routing at inference** (MoE-style).
  Full neuroevolution of an LM topology is not competitive at LM scale and is
  **not recommended for v1**; it can inform the *rewiring criterion* (gradient +
  magnitude + surprise), not the outer-loop optimizer.

---

## Cross-cutting research gaps (opportunities for XRFM)

1. **Graph/topology as *compute*, not visualization, for language.** Most "graph LM"
   work either (a) runs GNNs on external knowledge graphs, or (b) uses attention which
   is a dense complete graph. A genuinely *sparse, dynamically rewired, message-passing*
   sequence mixer for text — where edges are generated from neural state and memory —
   is not a settled, dominant architecture.
2. **Memory + uncertainty + topology in one coherent system.** Titans/MIRAS solve
   memory; abstention papers solve uncertainty; DST/MoE solve conditional sparsity.
   They are not integrated. Combining a (fixed-capacity, updatable) associative memory
   with a context-routed sparse subgraph and an explicit evidence/abstention head is
   a novel *composition* even if each part is known.
3. **Brain principles at LM scale done honestly.** Most "brain-like LMs" are marketing.
   The defensible subset — local+sparse connectivity, modularity, multi-timescale
   dynamics, plasticity as fast-weight update, separate memory systems — has partial
   realizations (SSMs, linear attention, MoE) but no single coherent architecture.

## What XRFM should NOT do (evidence-based)

- Do not build a dense O(d²) "neuron graph" — no tractable, no structural prior.
- Do not claim SNNs/neuroevolution give a quality advantage at scale (no evidence).
- Do not use unstructured sparsity and claim wall-clock speedups on stock GPUs.
- Do not treat TDA as a runtime compute path initially (use as diagnostic/regularizer).
- Do not bolt on RAG and call it "memory."
- Do not rely on token-probability alone for uncertainty/abstention.

---

## Searches performed (breadth log)

Post-Transformer: Mamba/Mamba-2/Mamba-3, RWKV, Gated DeltaNet/DeltaNet-2, Titans/MIRAS,
RetNet/linear attention, liquid/LTC, neural ODE.
Memory: modern Hopfield/associative, Titans persistent/long-term, memory-augmented NN.
Graphs: learnable adjacency, dynamic GNN survey, hypergraph/HGraphormer/EHNN.
Sparsity: RigL, SRigL, SET/DST, lottery tickets, DeepSeek MoE routing.
Uncertainty: epistemic/aleatoric MI, effective-rank, abstention gates, I-CALM, DCS.
TDA: persistent homology, neural persistence, Betti estimation.
Neuro-symbolic: differentiable KG reasoning, latent/concept graphs.
Neuroevolution: NEAT/HyperNEAT, plastic/self-organizing nets.
Datasets: FineWeb/FineWeb-Edu, SlimPajama, Dolma, Zyda, C4, The Pile.
OSS: mamba-ssm/causal-conv1d (Apache, CUDA), HF tokenizers/datasets/datatrove.

*Each area above warrants a focused deep-dive (with primary-paper reads) before code is
frozen. This review establishes the map and the design constraints.*
