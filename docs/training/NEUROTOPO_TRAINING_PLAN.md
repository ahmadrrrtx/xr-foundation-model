# XRFM-NeuroTopo — Training & Experiment Plan

**Status:** PROPOSAL (pre-implementation; awaits approval per mission Section 25)
**Date:** 2026-08-12
**Companion specs:** `XRFM_NEUROTOPO_SPEC.md`, `XRFM_NEUROTOPO_MATH.md`
**Golden rule:** every stage has a **measurable hypothesis** and a **stop/fail criterion**.
No expensive training is launched until the prior stage passes.

---

## 0. Controls and fairness constraints

The comparison is apples-to-apples (mission Section 17):

| Held equal across all models | Value |
|---|---|
| Tokenizer | XRFM byte-level BPE (shared) |
| Training data | identical splits/shards/order (same seed) |
| Training tokens | identical budget |
| Optimizer | AdamW β=(0.9,0.95), wd 0.1, clip 1.0 |
| Schedule | linear warmup + cosine (matched steps) |
| Parameter count | ±2% (matched per size via budget script) |
| Compute budget | identical GPU-hours / wall budget |
| Precision | bf16 on GPU, fp32 CPU reference |
| Seeds | 3 seeds per headline comparison |

**Models compared:**
1. **CONTROL-Transformer** — existing `GPTModel` (dense attention).
2. **CONTROL-Recurrent** — Gated DeltaNet/Mamba-2 + optional periodic attention, **no
   neural graph** (isolates "memory" from "graph").
3. **XRFM-NT (full)** — sparse dynamic graph + gated associative memory + uncertainty head.
4. **Ablations** of NT: static graph; dynamic graph; +memory; +uncertainty; +topology reg; full.

---

## 1. Losses and curriculum of objectives

Start minimal (mission Section 15). Turn on auxiliary losses only after the base LM works.

| Phase | Objective |
|---|---|
| Stage 1–3 | `L_lm` only (plus L_sparse only if needed to stabilise routing) |
| Stage 4 | `+ L_conf` (uncertainty) once labeled abstention data is ready |
| Stage 4+ | `+ L_connect` (connectivity) if graph fragments |
| Stage 5+ | `+ L_mem` (associative recall) if memory underperforms |

`λ` values are selected by small sweeps; the headline result must include an ablation with
all aux losses off to attribute gains.

---

## 2. Nine stages (mission Section 16)

### Stage 1 — Tiny architecture correctness
- **Config:** NT-TINY (N=32, d=32, 2 layers), CPU.
- **Test:** forward/backward shapes; finite gradients; deterministic seed; save/resume
  equivalence; KV/state-cache equivalence (recurrent decode == full forward); all unit
  tests (topology, sparsity, connectivity, memory read/write, uncertainty, numerical
  stability, checkpoint, determinism, CPU/GPU, mixed precision).
- **Hypothesis:** the module is mathematically correct and reproducible.
- **Pass:** test suite green; bitwise resume within tolerance; no NaN/Inf.
- **Fail:** do not proceed.

### Stage 2 — Overfit synthetic graph dynamics
- **Data:** (a) tiny fixed corpus (must reach train loss <0.5 like XRFM overfit test);
  (b) **synthetic associative/retrieval tasks**: copy, associative recall (key→value over
  long distance), needle-in-haystack, and a simple "conflict" task (two contradictory
  facts → should abstain).
- **Hypothesis:** NT can memorise and route; memory head retrieves distant keys;
  uncertainty head learns abstention on conflict items.
- **Pass:** train loss <0.5 on overfit; >90% retrieval at the trained distance; abstention
  accuracy >80% on conflict items.
- **Fail:** if it cannot overfit, the architecture cannot train — redesign before scaling.

### Stage 3 — Tiny real corpus
- **Data:** `corpus.txt` (~1.77M tokens) + a held-out slice; NT-SMALL scale (~1–2M params).
- **Hypothesis:** val loss decreases and is competitive with the Transformer control at
  matched params/tokens.
- **Pass:** val PPL decreases monotonically; no routing collapse (entropy healthy);
  graph stays connected.
- **Fail:** if NT is >X% worse than control at tiny scale, diagnose (graph vs memory vs
  optimizer) before scaling.

### Stage 4 — Fair comparison vs controls
- **Data:** 100M–1B tokens of FineWeb-Edu sample / SlimPajama (free license); NT-SMALL/MEDIUM.
- **Run:** CONTROL-Transformer, CONTROL-Recurrent, NT (3 seeds each), matched budget.
- **Metrics:** val loss/PPL, train tok/s, inference tok/s, VRAM, long-context retrieval,
  ECE/Brier/selective accuracy/abstention/OOD AUROC.
- **Hypothesis (H1):** NT matches or beats controls on long-context retrieval and
  calibration at matched compute; is competitive on PPL; uses constant inference memory.
- **Pass/Fail:** pre-register thresholds (e.g., NT PPL within 5% of Transformer; retrieval
  better beyond context; ECE lower). If NT loses decisively, document and pivot.

### Stage 5 — Scale model
- **Config:** NT-MEDIUM (~15–25M) and (if GPU permits) NT-LARGE (~70–100M); tune N, d,
  k_local/k_long, memory dims; parameter-match controls.
- **Hypothesis:** NT follows a healthy scaling trend (loss vs params/tokens) and the
  graph/memory benefits persist or grow.
- **Pass:** smooth scaling curves; no instability at depth (use scaled residual init).

### Stage 6 — Train on real dataset
- **Data:** 1–10B+ tokens FineWeb-Edu (+ code/math slices); bf16 on GPU; DDP/FSDP once
  validated; gradient accumulation for effective batch.
- **Hypothesis:** NT reaches target PPL and long-context performance at MEDIUM scale.
- **Pass:** converge to plan; checkpoint every N steps; full provenance logged.

### Stage 7 — Evaluate uncertainty
- **Data:** TruthfulQA, AmbigQA, SQuAD-unanswerable, FEVER (conflict), MMLU-OOD, plus
  synthetic contradictory-evidence and insufficient-context sets.
- **Metrics:** accuracy-vs-confidence, ECE, Brier, selective accuracy, abstention rate,
  OOD AUROC/AUPR, false-confidence rate, hallucination rate, risk-coverage curves, DCS.
- **Hypothesis (P4):** NT's graph-evidence head beats token-probability and no-head
  baselines.
- **Pass:** meaningful abstention–hallucination frontier; lower ECE; high OOD AUROC.

### Stage 8 — Evaluate long context
- **Tasks:** RULER needle-in-haystack, multi-key retrieval, language modeling at context
  lengths beyond training (extrapolation).
- **Hypothesis (P1/P5):** constant-memory state gives gentler degradation than Transformer
  KV cache; gated memory improves multi-key recall.
- **Pass:** NT wins (or ties at lower memory) at long contexts; report VRAM vs length.

### Stage 9 — Evaluate continual learning
- **Protocol:** sequential streams of domains (books → code → science) without replaying;
  measure backward transfer / forgetting, and whether episodic+persistent memory retains
  prior knowledge.
- **Hypothesis:** separate persistent/episodic memory reduces catastrophic forgetting vs
  controls.
- **Pass:** lower forgetting with memory on vs ablated; document honestly.

---

## 3. Ablation study (mission Section 20)

Every innovation is isolated. All ablations are parameter- and compute-matched.

| # | Variant | Dynamic graph | Memory | Uncertainty | Topo reg | Tests |
|---|---|---|---|---|---|---|
| A | Transformer control | n/a (dense attn) | KV cache | no | no | baseline |
| B | Recurrent control | no | gated delta/SSD | no | no | memory alone |
| C | NT static graph | fixed adjacency | yes | yes | no | does dynamism help? |
| D | NT dynamic graph | **yes** | no | no | no | graph alone |
| E | NT + memory | yes | **yes** | no | no | graph+memory |
| F | NT + uncertainty | yes | yes | **yes** | no | does the head help? |
| G | NT + topology reg | yes | yes | yes | **yes** | does connectivity reg help? |
| H | NT full | yes | yes | yes | yes | the proposed system |
| I | NT hybrid-attn | yes | yes | yes | yes | + periodic bounded attention (is pure graph enough?) |

Additional micro-ablations: local-only edges vs local+long-range; surprise-gating on/off;
channel-wise erase/write vs scalar gate; persistent memory on/off; GRU vs SSM local dynamics.

---

## 4. Hyperparameters (starting points)

| Size | N | d | L | k_local | k_long | d_k/d_v | K (persistent) | ~params |
|---|---|---|---|---|---|---|---|---|
| TINY | 32 | 32 | 2 | 4 | 8 | 64 | 16 | ~0.2 M |
| SMALL | 64 | 64 | 4 | 8 | 16 | 128 | 32 | ~1–2 M |
| MEDIUM | 128 | 96 | 8 | 8 | 32 | 256 | 64 | ~15–25 M |
| LARGE | 256 | 128 | 12 | 16 | 48 | 512 | 128 | ~70–100 M |

Optimizer: AdamW β=(0.9,0.95), lr 3e-4 (SMALL/MEDIUM) / 6e-4 (TINY), wd 0.1, warmup
200–2000, cosine, clip 1.0, dropout 0.0 pretraining / 0.1 fine-tuning. These mirror the
validated XRFM control recipe (see XRFM_CURRENT_STATE.md).

---

## 5. Compute plan (free-first)

| Stage | Hardware | Est. cost |
|---|---|---|
| 1–2 | CPU sandbox (this environment) | minutes |
| 3 | CPU / free T4 | <1 h GPU |
| 4 | 1× T4/L4 (free cloud) | hours–1 day per run × seeds |
| 5–6 | 1× A100 40GB or multi-T4 | 1–several days |
| 7–9 | reuse Stage-6 checkpoints on 1 GPU | hours each |

All stages 1–3 must pass on CPU before any GPU is requested. No multi-node training until
DDP/FSDP is validated (currently untested in XRFM — see current-state audit).

---

## 6. Evaluation metrics (single source of truth)

- **Language modeling:** val loss, token PPL (padding-corrected), top-1/top-5 accuracy,
  strided long-sequence PPL.
- **Efficiency:** train tok/s/GPU, inference tok/s, VRAM vs seq len, parameter count,
  active-FLOP ratio (sparsity).
- **Long context:** RULER/needle retrieval accuracy vs context length; PPL extrapolation.
- **Uncertainty:** ECE, Brier, selective accuracy-coverage AUC, abstention rate, OOD
  AUROC/AUPR, false-confidence rate, hallucination rate on unanswerable/conflict sets.
- **Interpretability:** edge-usage visualisations, module activation patterns, Betti/
  persistence diagnostics (offline), evidence-path traces.
- **Continual:** backward transfer, forgetting, retained accuracy.

Every reported number includes the protocol row (model/params/tokens/data/context/compute/
eval-set/metric) — the fairness rule already used by XRFM.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Routing collapse (all edges to one module) | entropy/load-balance loss, expert dropout, noise on scores |
| Graph fragments (gradient/info loss) | connectivity regularizer, local prior guarantees |
| Memory saturation/interference | channel-wise erase/write, surprise gating, normalization |
| Instability at depth | GPT-2 scaled residual init, RMSNorm, grad clip, warmup |
| Sparse ops slow on GPU | block/top-k sparsity; Triton gather-scatter; dense fallback |
| Uncertainty head over-abstains | tune on abstention–hallucination frontier; reward calibration |
| Cannot match Transformer PPL | hybrid attention block (variant I); honest pivot |
| Distributed unverified | validate DDP gloo (CPU) then NCCL before Stage 6 |

---

## 8. Reproducibility contract (extends XRFM's)
Every run artifact bundle: commit SHA + config hash + tokenizer file + dataset shard hashes
+ seed + all λ + training JSONL log + checkpoint (with module/edge/memory/persistent state)
+ eval results + exact command. CPU reference path is always available for CI.
