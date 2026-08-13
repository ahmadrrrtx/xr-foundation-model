# XRFM-NeuroTopo — Frontier Architecture Research Report

**Mission:** XRFM / XR Foundation Model — discover whether a genuinely new
language-model architecture can emerge from *dynamic neural topology, learned
connectivity, sparse message passing, structured memory, and explicit uncertainty*
rather than dense token-to-token self-attention.
**Date:** 2026-08-12
**Status:** RESEARCH COMPLETE — **STOP for approval before implementation**
(mission Section 25). No production code has been written; the existing Transformer
baseline is untouched and remains the CONTROL.

---

## 1. Current XRFM audit (summary)

XRFM today is a small but **scientifically reproducible decoder-only GPT-Transformer**
in pure PyTorch. A prior forensic audit (2026-08-08) fixed 49 issues (causal masking,
byte-level BPE tokenizer, fake mixed precision, non-resumable checkpoints, API import
failure, toy data, vocabulary split-brain). Verified in this audit:

- **Architecture:** pre-norm decoder blocks — RMSNorm + multi-head attention (explicit
  causal mask, RoPE half-split) + SwiGLU; weight-tied embedding/lm_head; KV cache.
- **Tokenizer:** byte-level BPE v2; lossless Unicode round-trip (verified:
  `"Hello, world! 你好"` round-trips exactly).
- **Training:** AdamW β=(0.9,0.999), cosine+warmup, grad accum/clip, seeded, resumable,
  bf16 on GPU (fp32 CPU), validation hook, JSONL metrics.
- **Data:** line-boundary deduped splits; padding with `-100` loss masking;
  `corpus.txt` ≈1.77 M tokens (public-domain books + code).
- **Eval:** padding-corrected PPL, strided PPL, top-k accuracy; KV-cached inference with
  temp/top-k/top-p/repetition penalty/stop sequences.
- **Tests:** **227/227 passing** (222 core + 5 API), including ground-truth math tests.
- **Baseline (SMALL, executed):** 1,318,528 params; 5,000 steps / 10.24 M tokens;
  val loss 7.24→5.71; **val PPL 301.55** (random ≈2048); ~2,300 tok/s on 2 vCPU.
- **Honest weaknesses:** tiny scale, English-only small corpus, distributed training
  unverified on GPU, no external benchmarks, dense O(n²) attention, no memory/uncertainty/
  graph machinery.

Full audit: **`docs/research/XRFM_CURRENT_STATE.md`**. The existing Transformer is the
**CONTROL MODEL and must not be deleted.**

## 2. Existing research (breadth map)

Across ten communities:
- **Post-Transformer mixers:** Mamba (selective S6), Mamba-2 (SSD — SSM/attention
  duality), RWKV (linear-attention-as-RNN), RetNet/GLA, Gated DeltaNet/DeltaNet-2
  (error-correcting gated associative memory), Titans/MIRAS (deep neural long-term memory
  with test-time learning), and hybrids (Jamba/Samba/Zamba/Qwen3-Next).
- **Memory:** modern Hopfield (softmax retrieval = one attention step; exponential
  capacity), NTM/DNC lineage, Titans' three-tier (short/long/persistent) memory.
- **Dynamic graphs/topology:** learnable-adjacency GNNs (STBAM with spectral
  connectivity loss), dynamic GNN surveys, adaptive topology (ADLGNN/AAGCN).
- **Sparsity/routing:** RigL/SRigL dynamic sparse training, SET, lottery tickets;
  MoE (DeepSeek-V3 auxiliary-loss-free balancing, fine-grained experts).
- **Hypergraphs:** HGraphormer (hypergraph Laplacian injected into attention), EHNN.
- **Uncertainty:** epistemic-vs-aleatoric MI estimation, effective-rank hidden-state
  signals, composite abstention gates, I-CALM/DCS calibration.
- **TDA:** neural persistence (ICLR 2019), Betti-number estimation; strongest as a
  diagnostic/regularizer.
- **Neuro-symbolic:** differentiable KG reasoning (DiffLogic); native-in-LM graph
  processing (NAG).
- **Neuro-inspired:** liquid time-constant / neural-ODE / CfC (adaptive timescales).
- **Datasets/OSS:** FineWeb/FineWeb-Edu (ODC-By), Dolma (AI2 ImpACT), SlimPajama
  (Apache), The Pile (MIT + subset terms); `mamba-ssm`/`fla`/RWKV (Apache/MIT), PyG/DGL,
  Triton, HF tokenizers/datasets/datatrove.

**The field's 2024–26 consensus:** no single "Transformer killer" — the winning shape is
**hybrid, memory-centric, sparsely-activated**, with attention for exact retrieval,
SSM/linear-recurrence for constant-memory mixing, and (newly, Titans) a learned
updatable memory. Full detail: `docs/research/frontier/LITERATURE_REVIEW.md`,
`POST_TRANSFORMER.md`, `MODEL_LANDSCAPE.md`, `DYNAMIC_GRAPHS.md`, `NEUROTOPOLOGY.md`.

## 3. Existing architectures (comparison)

| Family | Train | Inference memory | Long context | LM quality | Maturity |
|---|---|---|---|---|---|
| Transformer (control) | O(n²) | KV grows n | costly but exact | best | production |
| Mamba-2/SSD | O(n) | constant | excellent | competitive | growing |
| RWKV | O(n) | constant | good | competitive | mature |
| Gated DeltaNet-2 | O(n) chunked | constant matrix | excellent (RULER) | SOTA linear-recurrent | emerging |
| RetNet/GLA | O(n) | constant | good | good | mature |
| Titans/MIRAS | ~O(n) | constant+persistent | >2M tokens | beats Mamba-2/DeltaNet/Transformer++ | new (Dec 2025) |
| Hybrid SSM+Attn | mixed | constant+small KV | excellent | SOTA-ish (Jamba etc.) | growing |

## 4. Open-source implementations
- **Mamba/Mamba-2:** `state-spaces/mamba` + `causal-conv1d`, Apache-2.0 (NVIDIA CUDA).
- **Linear/recurrent mixers:** `fla` (flash-linear-attention, MIT) implements GLA,
  DeltaNet/Gated DeltaNet, RetNet, RWKV, Linear Attention, Mamba2 in Triton.
- **RWKV:** `BlinkDL/RWKV-LM`, Apache-2.0, strong CPU support.
- **Graph/sparse:** PyTorch Geometric (MIT), DGL (Apache-2.0); native `torch.sparse`.
- **TDA:** Gudhi (MIT); avoid giotto-tda in core (AGPL).
- **Data:** HF Datasets/datatrove/tokenizers (Apache-2.0).
- **Eval:** `lm-evaluation-harness` (MIT); custom ECE/Brier.
- All core deps are BSD/MIT/Apache; license hygiene documented in `OPEN_SOURCE_STACK.md`.

## 5. Failed / rejected approaches (evidence-based)
- **Dense O(d²) neuron graph** — intractable, no structural prior; rejected.
- **Spiking neural networks for LM quality** — no evidence of advantage; surrogate-gradient
  training is finicky; monitor only.
- **Full neuroevolution (NEAT/HyperNEAT) of LM topology** — not competitive at LM scale,
  GPU-unfriendly; its useful kernel (gradient+magnitude rewiring) is absorbed via DST.
- **"Just add RAG" as memory** — external, non-differentiable, not part of computation;
  explicitly rejected by the mission.
- **Unstructured sparsity claimed as GPU speedup** — does not accelerate stock GPUs; use
  structured/block sparsity or conditional compute (MoE).
- **TDA as a runtime compute path** — too expensive; use as diagnostic/regularizer.
- **giotto-tda in core** — AGPL license; isolated analysis only.
- **Over-complex v1 (full concept graph + hyperedges + TDA + MoE all at once)** — violates
  "strongest falsifiable hypothesis, not maximal complexity"; staged behind flags.

## 6. Research gaps (opportunities)
1. **Graph as compute for language over neural units** — most "graph LM" work uses
   token/external graphs or dense graph-Transformers; a sparse, plastic, *neural-module*
   graph that is the primary mixer is not a dominant architecture.
2. **Memory + uncertainty + topology integrated** — Titans solves memory; abstention
   papers solve uncertainty; DST/MoE solve conditional sparsity. They are not unified.
3. **Brain-principles done honestly at LM scale** — local+sparse connectivity, modularity,
   multi-timescale dynamics, plasticity as fast-weights, separate memory systems —
   partially realized, no single coherent architecture.
4. **Evidence-grounded abstention from graph signals** — abstention gates exist but are
   not driven by structural support/conflict evidence in a latent concept graph.

## 7. Novelty matrix (honest summary)
- **Not novel (borrowed, attributed):** self-attention; SSMs/SSD; linear/delta-rule
  associative memory; modern Hopfield; learned/dynamic adjacency; GNN message passing;
  MoE routing; dynamic sparse training; uncertainty/abstention heads; epistemic/aleatoric
  estimation; TDA/neural persistence; three-tier memory (Titans).
- **Plausibly distinctive (composition, not a new primitive):** a language model whose
  token mixer is **sparse message passing over a context-generated graph of learned neural
  modules**, whose states persist via a **gated associative memory**, and whose
  **uncertainty head reads graph-evidence** (support/contradict edges) to abstain — with
  the graph routing messages, gating memory, and driving confidence.
- **Confidence:** Medium as of 2026-08-12. No directly matching implementation was found in
  the searched literature/codebase, but Graph Transformers, NAG, and Titans are adjacent.
  A deeper prior-art + patent search is **required before any publication claim**.
- Full matrix: `docs/research/frontier/NOVELTY_MATRIX.md`.

## 8. Candidate architectures
Six candidates were scored on novelty, trainability, GPU/memory efficiency, long context,
LM/reasoning potential, hallucination control, interpretability, scalability, implementation
complexity, and falsifiable-hypothesis strength:
- **A** Sparse Dynamic Neural-Graph LM · **B** Graph-Coupled SSM · **C** NeuroTopological
  Recurrent LM · **D** Sparse Plastic Neural LM (deep DeltaNet/Titans) · **E**
  Memory-Graph Cognitive LM (full concept graph) · **F** Hybrid SSM+Sparse Attention (baseline).
Detail/scores: `docs/research/frontier/CANDIDATES.md`.

## 9. Selected architecture: XRFM-NeuroTopo (Candidate C, "XRFM-NT")
Selected for the **strongest, most specific falsifiable hypothesis**, not maximal
complexity. v1 is the minimal system testing the thesis: **sparse dynamic neural-module
graph + gated associative memory + graph-grounded uncertainty head**, with all advanced
features (concept graph writes, hyperedges, TDA, MoE) staged behind config flags.

## 10. Mathematical formulation (summary; full spec in `XRFM_NEUROTOPO_MATH.md`)
- **Modules:** `H_t ∈ R^{N×d}` recurrent; local dynamics GRU or diagonal SSM `h_t=Āh_{t-1}+B̄u_t`.
- **Topology:** local edges + top-k long-range from low-rank scores
  `s_ij=(U h_i)^T(V h_j)/√r + γ(P m_i)^T(Q m_j)`; plastic decay
  `A_t = α A_{t-1} + (1−α) σ(s)`.
- **Message passing:** `h_i ← GRU(h_i, Σ_{j∈N(i)} A_ij · W_msg[h_j;e_ij])`, O(N·k·d).
- **Memory (gated delta):**
  `err=v−Mᵀk`; `M ← diag(decay)M + β(write ⊙ k⊗(err ⊙ erase))`, surprise-gated;
  read `c=qM`; plus persistent memory `SoftMax(qCᵀ)C`.
- **Output:** module-readout `z` → weight-tied `zEᵀ` logits.
- **Uncertainty:** evidence vector `g_t` (edge support/conflict, memory margin, surprise,
  effective rank, routing entropy) → 5-state head {ANSWER, QUALIFY, ABSTAIN,
  REQUEST_CONTEXT, REQUEST_EVIDENCE}.
- **Loss:** `L = L_lm + λ_sparse L_sparse + λ_connect L_connect + λ_conf(L_conf+L_cal)
  [+ λ_mem L_mem]` — minimal, weights set experimentally.
- **Complexity:** linear in sequence length; **constant inference memory** (no growing KV).

## 11. Computational graph
`Embedding → causal conv → inject into N modules → [local dynamics → topology gen →
sparse MP → memory write/read → persistent read → SwiGLU]×L → module readout → LM head
(+ confidence head)`. Recurrent `state = (H, A edges/gates, M, C)` analogous to KV cache
but fixed-size. Spec: `XRFM_NEUROTOPO_SPEC.md`.

## 12. Memory design
- **Working** = recurrent module state + bounded local attention (if hybrid variant).
- **Episodic** = gated delta-rule associative matrix (surprise-gated erase/write).
- **Semantic/persistent** = frozen-after-pretraining learned vectors (v1); later a latent,
  differentiable concept graph with supports/contradicts/related/temporal/uncertainty edges.
- **Procedural** = the trained weights/routing policy.
- All differentiable, configurable capacity, no RAG. Detail: `MEMORY_ARCHITECTURES.md`.

## 13. Topology design
- Local dense neighborhoods + sparse learned long-range links; gated, plastic edges;
  load-balancing to prevent collapse; spectral connectivity regularizer to prevent
  fragmentation. The graph routes messages, gates memory writes, and supplies evidence —
  it is part of computation. Detail: `DYNAMIC_GRAPHS.md`.

## 14. Uncertainty design
- Signals: token entropy (aleatoric), effective rank/ensemble disagreement (epistemic),
  memory retrieval margin + surprise, support/conflict edge weights (evidence), OOD
  distance to persistent concepts.
- 5-state head trained with CE + Brier + calibration; decision rule emits answer/qualify/
  abstain/request. Evaluated with ECE, Brier, selective accuracy, abstention frontier,
  OOD AUROC, false-confidence, hallucination rate. Detail: `UNCERTAINTY.md`.

## 15. Training objectives
Start with next-token CE only; add confidence loss when abstention data is ready; add
connectivity loss only if the graph fragments; memory recall loss only if memory
underperforms. Aux losses are ablated, never stacked for effect.

## 16. Dataset strategy (free/open)
- **Primary:** FineWeb-Edu (sample-10BT) / FinerWeb-10BT (ODC-By) + SlimPajama slice
  (Apache); held-out Wikipedia + code.
- **Code:** SlimPajama code split (Apache); StarCoder2 later.
- **Uncertainty:** TruthfulQA, AmbigQA, SQuAD-unanswerable, FEVER, MMLU-OOD + synthetic
  contradictory/insufficient-context examples (license-clean, unlimited).
- Document-boundary splits, dedup, language ID, PII removal; hashes recorded for
  reproducibility. Detail: `DATASET_LANDSCAPE.md`.

## 17. Free/open-source toolchain
PyTorch (BSD), HF tokenizers/datasets/datatrove (Apache), `fla`/mamba/RWKV (Apache/MIT)
for accelerated recurrent kernels, PyG/DGL (MIT/Apache) for sparse ops, Triton (MIT) for
custom fused kernels, Gudhi (MIT) for TDA diagnostics, FastAPI/pytest/ruff/mypy. CPU
reference path always runs in CI; GPU kernels are acceleration only. Detail:
`OPEN_SOURCE_STACK.md`.

## 18. Hardware requirements
- **Stages 1–3 (correctness/tiny):** CPU sandbox (this environment) — minutes.
- **Stage 4 (fair comparison):** 1× T4/L4 16 GB — hours–1 day/run.
- **Stages 5–6 (MEDIUM/LARGE):** A100 40/80 GB (or multi-T4) — 1–several days.
- **Stages 7–9:** reuse checkpoints on 1 GPU.
- CPU-correct/GPU-fast design keeps everything testable without a GPU.

## 19. Scaling strategy
TINY (0.2M) → SMALL (1–2M) → MEDIUM (15–25M) → LARGE (70–100M), each parameter-matched to
its Transformer/recurrent controls; tune N/d/k/memory dims at each size; use GPT-2-style
scaled residual init for depth; validate DDP (gloo CPU then NCCL) before multi-GPU.

## 20. Baselines
- **CONTROL-1:** existing XRFM Transformer.
- **CONTROL-2:** Gated DeltaNet/Mamba-2 (+ optional periodic attention), no neural graph —
  isolates "memory" from "graph."
- Both matched on params, tokens, tokenizer, optimizer, schedule, compute.

## 21. Ablations
Static vs dynamic graph; memory on/off; uncertainty on/off; topology reg on/off; hybrid
attention; local-only vs local+long-range; scalar vs channel-wise gates; surprise gating;
persistent memory; GRU vs SSM; top-k sweep. Full grid in `EXPERIMENT_MATRIX.md`.

## 22. Evaluation
PPL/top-k/strided PPL; throughput & VRAM vs context; RULER/needle/multi-key retrieval;
ECE/Brier/selective accuracy/abstention/OOD AUROC/false-confidence/hallucination; continual-
learning backward transfer; interpretability diagnostics (edge usage, module activation,
offline Betti/persistence). Every number ships with a protocol row.

## 23. Risks
Routing/hub collapse; graph fragmentation/gradient loss; memory saturation/interference;
depth instability; sparse-op GPU slowness; over-abstention; PPL gap to Transformer;
distributed training unverified; novelty overlap with adjacent work; over-complexity.
Mitigations are tabulated in the training plan.

## 24. Expected failure modes (and responses)
- **Cannot overfit (Stage 2):** architecture untrainable → simplify local dynamics/check
  init before any scaling.
- **Dynamic ≈ static graph:** dynamism adds nothing → freeze topology or adopt graph as
  fixed prior.
- **Graph + memory ≈ memory-only:** graph adds nothing → reduce to CONTROL-2 and report.
- **Uncertainty head ≈ token-prob baseline:** architectural uncertainty fails → keep
  token-prob abstention only.
- **NT PPL decisively worse than Transformer even with hybrid attention:** the core
  hypothesis fails for language modeling → pivot; document honestly.
- **Cannot beat Transformer on long context at matched memory:** constant-state premise
  weakened → re-examine memory capacity/gating.
These are pre-registered in `EXPERIMENT_MATRIX.md` §9.

## 25. Implementation roadmap
1. **Approve this report** (current stop point).
2. Scaffold `xrfm/{core,topology,neurons,dynamics,memory,routing,uncertainty,knowledge}`
   with interfaces + the package layout; do **not** touch `model/` (control).
3. Pure-PyTorch v1: modules, topology gen (local+topk), sparse MP, gated delta memory,
   persistent memory, confidence head; config dataclasses; param-budget matcher.
4. Unit tests for every component (topology, sparsity, connectivity, memory r/w,
   uncertainty, gradient flow, numerics, checkpoint/resume, determinism) — CPU green.
5. Stage 1–2 gates (correctness, overfit, synthetic retrieval/conflict).
6. Stage 3 tiny-real-corpus run vs both controls.
7. (GPU) Stage 4 fair comparison with 3 seeds; ablations.
8. Triton/`fla` acceleration only after correctness; then Stage 5+ scaling.
9. Concept graph, hyperedges, TDA reg, MoE only as staged experiments after v1 validates.

---

## Deliverables index (all produced this mission)
- **Audit:** `docs/research/XRFM_CURRENT_STATE.md`
- **Research reviews:** `docs/research/frontier/{LITERATURE_REVIEW, MODEL_LANDSCAPE,
  NEUROTOPOLOGY, MEMORY_ARCHITECTURES, UNCERTAINTY, DYNAMIC_GRAPHS, POST_TRANSFORMER,
  DATASET_LANDSCAPE, OPEN_SOURCE_STACK, NOVELTY_MATRIX, CANDIDATES, EXPERIMENT_MATRIX}.md`
- **Architecture:** `docs/architecture/XRFM_NEUROTOPO_SPEC.md`,
  `docs/architecture/XRFM_NEUROTOPO_MATH.md`
- **Training:** `docs/training/NEUROTOPO_TRAINING_PLAN.md`
- **This report:** `XRFM_NEUROTOPO_RESEARCH_REPORT.md`

## STOP CONDITION (mission Section 25)
This mission ends here. The following have been produced and **no implementation has
begun**: (1) full research report, (2) novelty matrix, (3) architecture proposal,
(4) mathematical spec, (5) implementation plan, (6) dataset plan, (7) training plan,
(8) open-source toolchain, (9) compute plan, (10) experiment matrix.

**Awaiting approval before any code is written or training launched.** The existing
Transformer baseline is intact (227 tests passing), the working tree was cleaned of
regenerable artifacts, and no source was modified.

> **Hypothesis, not claim.** XRFM-NT is a *candidate* architecture with a falsifiable
> hypothesis. It may lose to the Transformer or to a memory-only recurrent control; if
> so, that result will be documented honestly and the design changed. There will be no
> "Transformer killer," "human-brain equivalent," or "hallucination-free" claims without
> executed evidence.
