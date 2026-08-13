# XRFM Frontier — Experiment Matrix

**Date:** 2026-08-12
**Purpose:** Explicit, pre-registered experiment grid so every claim is backed by a run.
All runs use the same tokenizer, data splits/seeds, optimizer, schedule, param budget, and
compute (fairness rule). 3 seeds for headline comparisons; 1 seed for ablations.

Legend: PPL = perplexity; LC = long-context retrieval; ECE = expected calibration error;
Sel.Acc = selective accuracy; Abst = abstention rate; OOD = OOD AUROC; Hall = hallucination
rate on unanswerable/conflict; Thr = throughput tok/s; Mem = inference memory vs context.

## 1. Primary comparison (Stage 4)

| Exp ID | Model | Dynamic graph | Memory | Uncertainty | Topo reg | Hybrid attn | Metrics |
|---|---|---|---|---|---|---|---|
| E1 | Transformer (CONTROL-1) | — | KV | no | no | dense | PPL, LC, Thr, Mem |
| E2 | Recurrent (CONTROL-2: Gated DeltaNet/Mamba2) | no | gated delta | no | no | periodic | PPL, LC, Thr, Mem |
| E3 | **XRFM-NT full** | yes | gated delta | yes | yes | off | all |

## 2. NT ablation grid (Stage 4, matched params/tokens)

| Exp ID | Variant | Dyn graph | Memory | Unc | TopoReg | Question answered |
|---|---|---|---|---|---|---|
| A1 | NT no-graph (recurrent only) | no | yes | yes | no | is the graph doing anything? |
| A2 | NT static graph | fixed | yes | yes | no | does dynamism help? |
| A3 | NT dynamic graph, no memory | yes | no | no | no | graph alone |
| A4 | NT graph + memory | yes | yes | no | no | memory contribution |
| A5 | NT graph + memory + uncertainty | yes | yes | yes | no | does the head help calibration? |
| A6 | NT full (no topo reg) | yes | yes | yes | no | is connectivity reg needed? |
| A7 | NT full | yes | yes | yes | yes | the proposed system |
| A8 | NT full + hybrid attention | yes | yes | yes | yes | every N layers | is pure graph enough? |

## 3. Component micro-ablations (Stage 2/4)

| ID | Variable | Levels | Metric |
|---|---|---|---|
| M1 | Edges | local-only / local+long-range | PPL, LC |
| M2 | Memory write gate | scalar / channel-wise erase+write | retrieval, interference |
| M3 | Surprise gating | on / off | memory efficiency, PPL |
| M4 | Persistent memory | on / off | long-context, continual |
| M5 | Local dynamics | GRU / diagonal SSM (SSD) | PPL, stability, speed |
| M6 | Top-k (long-range) | 8/16/32/64 | quality vs FLOPs |
| M7 | Sparsity regularizer λ_sparse | sweep | edge mass, routing entropy |
| M8 | Connectivity reg λ_connect | on/off | #components, gradient flow |
| M9 | Confidence loss weight λ_conf | sweep | ECE vs coverage |
| M10 | Number of modules N | sweep at fixed params | scaling |

## 4. Scaling (Stage 5)

| ID | Size | N/d/L | Params | Tokens | Metrics |
|---|---|---|---|---|---|
| S1 | TINY | 32/32/2 | ~0.2M | 10–100M | PPL, overfit |
| S2 | SMALL | 64/64/4 | ~1–2M | 0.5–2B | PPL, LC, calibration |
| S3 | MEDIUM | 128/96/8 | ~15–25M | 1–10B | full suite |
| S4 | LARGE (if GPU) | 256/128/12 | ~70–100M | 10B+ | full suite, external bench |

Each scaling point runs E1/E2/E3 to produce a matched scaling curve.

## 5. Uncertainty / abstention (Stage 7)

| ID | Dataset / task | States tested | Metrics |
|---|---|---|---|
| U1 | TruthfulQA | ANSWER vs ABSTAIN | Hall, Abst, ECE |
| U2 | AmbigQA | ANSWER vs QUALIFY | aleatoric separation |
| U3 | SQuAD-unanswerable | ANSWER vs ABSTAIN | selective accuracy |
| U4 | FEVER / synthetic conflict | ANSWER vs QUALIFY/REQUEST_EVIDENCE | conflict detection |
| U5 | MMLU held-out subjects (OOD) | ANSWER vs ABSTAIN | OOD AUROC |
| U6 | Synthetic insufficient-context | ANSWER vs REQUEST_CONTEXT | abstention precision/recall |
| U7 | Baselines | token max-prob, entropy, effective-rank only | compare vs NT head |

## 6. Long context (Stage 8)

| ID | Task | Context lengths | Metric |
|---|---|---|---|
| L1 | RULER needle-in-haystack | 1k–128k | retrieval accuracy |
| L2 | Multi-key retrieval | 1k–64k | #keys recalled |
| L3 | Strided PPL | up to max_seq | PPL vs length |
| L4 | PPL extrapolation | beyond trained length | degradation slope |
| L5 | Inference memory | 1k–128k | bytes/token vs Transformer |

## 7. Continual learning (Stage 9)

| ID | Stream | Memory on/off | Metric |
|---|---|---|---|
| C1 | books → code → science (no replay) | off vs on | backward transfer, forgetting |
| C2 | domain shifts | off vs on | retained accuracy |

## 8. Correctness / engineering gates (Stage 1, must pass first)

| ID | Test | Pass criterion |
|---|---|---|
| G1 | Unit tests (topology, sparsity, connectivity, memory, uncertainty, grad flow, numerics, checkpoint, determinism) | all green on CPU |
| G2 | Recurrent state equivalence | incremental decode == full forward (≤1e-5) |
| G3 | Resume equivalence | loss at step k after resume == without resume |
| G4 | Gradient flow | no dead modules; finite grads over 100 steps |
| G5 | Sparsity | mean degree within target; graph connected |
| G6 | GPU/mixed precision | bf16 forward finite; matches fp32 within tolerance |

## 9. Decision rules (pre-registered)
- If E3 PPL is >5% worse than E1 at matched budget AND A8 doesn't close the gap →
  the pure-graph mixer fails; adopt hybrid or pivot.
- If A2 (static) ≥ A7 (dynamic) → dynamic topology adds nothing; simplify.
- If A4 ≈ A7 on retrieval → memory isn't helping; revisit write rule.
- If A5's head does not beat token-prob baselines on ECE/OOD → architectural uncertainty
  claim fails; keep only token-prob abstention.
- Every "fails" outcome is documented honestly (mission rule), not spun.
