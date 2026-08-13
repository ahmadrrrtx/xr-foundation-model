# XRFM Frontier — Uncertainty, Hallucination & Abstention

**Date:** 2026-08-12
**Purpose:** Design an *architecture-level* mechanism so XRFM can estimate
{KNOWN, UNKNOWN, AMBIGUOUS, INSUFFICIENT_CONTEXT, CONFLICTING_EVIDENCE, OOD} and
abstain/qualify — without relying on system prompts or token probability alone.

## 1. Why token probability is not enough

- Max-probability/entropy over the next token cannot distinguish **confidently wrong**
  from **confidently right** ("confident confabulation") [1](https://arxiv.org/html/2604.06195).
- Epistemic uncertainty ("I wasn't trained on this") must be separated from aleatoric
  uncertainty ("genuinely multiple valid answers"); an information-theoretic
  mutual-information estimate over multiple samples gives a lower bound specifically on
  *epistemic* uncertainty and detects hallucinations even for multi-answer queries
  [2](https://arxiv.org/html/2406.02543v2).
- Effective rank of hidden states across samples/layers is a cheap, strong
  hallucination/OOD signal that needs no extra module [3](https://arxiv.org/html/2510.08389).

## 2. Mechanisms that work (evidence)

| Mechanism | What it gives | Cost | Architectural? |
|---|---|---|---|
| Token entropy / max-prob | Baseline aleatoric signal | free | yes (existing) |
| MI over samples (epistemic) [2](https://arxiv.org/html/2406.02543v2) | Epistemic UQ, hallucination detection | multiple samples | sampling-time |
| Effective-rank hidden states [3](https://arxiv.org/html/2510.08389) | OOD/hallucination signal | cheap | can be a head |
| **Structural abstention gate** [1](https://arxiv.org/html/2604.06195) | Hard abstain decision; preserves answerable accuracy | small head | **yes** |
| Composite (gate + instruction/refusal) [1](https://arxiv.org/html/2604.06195) | 96–98% accuracy, 0–4% hallucination | gate + prompting | partly |
| Reward-shaped abstention (I-CALM) [4](https://arxiv.org/html/2604.03904) | Tunes abstention–hallucination frontier | data/scoring | objective |
| Distributional Correctness Score [5](https://arxiv.org/html/2510.04302) | Evaluates full belief vs single answer | eval metric | evaluation |

**Key finding [1](https://arxiv.org/html/2604.06195):** neither a structural gate nor
instruction-level refusal alone suffices; the *composite* achieved near-zero
hallucination, but the gate specifically failed on **conflicting-evidence**
confabulation — which motivates explicit conflict detection from XRFM's concept/evidence
graph.

## 3. XRFM uncertainty design (architecture-level)

### 3.1 Signals feeding the confidence estimator
1. **Aleatoric:** next-token distribution entropy / max probability (free).
2. **Epistemic (cheap):** effective rank & norm of hidden states across layers;
   disagreement between a small ensemble of routing paths/subgraphs.
3. **Memory/evidence signals:** retrieval *margin* from associative/concept memory
   (gap between best and next-best match), write "surprise," and number of *supporting*
   vs *contradicting* evidence edges in the concept graph.
4. **OOD signal:** distance/density of the current representation from persistent
   concept embeddings; topological change vs training-time representation topology.

### 3.2 Confidence head + decision states
A small head on top of pooled hidden + memory state predicts, per answer/span:
- `p_known` — evidence supports an answer.
- `p_ambiguous` — multiple plausible answers (aleatoric).
- `p_unknown` — epistemic gap / no evidence.
- `p_ood` — out of distribution.
- `p_conflict` — evidence contradicts.
- `p_insufficient_ctx` — need more context.

Mapped to **output states**:
`ANSWER` (emit), `QUALIFY` (answer + uncertainty caveat), `ABSTAIN` ("I don't know"),
`REQUEST_CONTEXT`, `REQUEST_EVIDENCE`.

### 3.3 Training objective
- Primary LM loss **plus** an explicit *abstention/confidence loss* on labeled data:
  - answerable items → predict ANSWER + correct;
  - unanswerable/OOD/conflicting items → predict the appropriate non-ANSWER state.
- Use a **proper scoring rule** (Brier / negative log-likelihood over the state
  distribution) and reward *calibration*, not just abstention.
- Optionally self-supervised: treat high-MI/high-effective-rank samples as soft
  UNKNOWN targets (bootstrap without hand labels).

### 3.4 Evaluation (required metrics)
- Accuracy vs confidence, **ECE** (expected calibration error), **Brier score**.
- **Selective accuracy** (accuracy on answered) vs **abstention rate** (coverage curve).
- **OOD detection AUROC/AUPR** (in-distribution vs OOD questions).
- **False-confidence rate** (high confidence when wrong).
- **Hallucination rate** on unanswerable/conflicting sets.
- Risk-coverage curves; Distributional Correctness Score [5](https://arxiv.org/html/2510.04302).

## 4. Datasets needed for uncertainty training/eval
- Answerable QA (TriviaQA-style) + unanswerable/ambiguous (AmbigQA, TruthfulQA).
- Contradictory-evidence sets (synthetic: supply conflicting passages).
- OOD questions (domain shift vs training corpus).
- "I don't know" / abstention corpora.
- These must be **free-license** and documented in DATASET_LANDSCAPE.md.

## 5. Honesty constraints (mission rules)
- Do **not** claim "hallucination-free" — report measured rates.
- The gate is a *candidate mechanism*; ablate it vs token-prob-only and no-head.
- If the gate does not beat token-probability baselines experimentally, document it.
