# XRFM Frontier — Dataset Landscape

**Date:** 2026-08-12
**Purpose:** Free/open, legally usable datasets for general language, code, math,
reasoning, science, and (critically) uncertainty/abstention/OOD evaluation. All sizes
are approximate and from dataset cards/papers; **verify license and current terms at
download time** — this is a planning map, not legal advice.

## 1. General web/language pretraining

| Dataset | Tokens | Lang | License | Quality / dedup / filtering | Use case | Storage | Notes |
|---|---|---|---|---|---|---|---|
| **FineWeb** [HF](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | ~15–18.5T | en (multi deriv.) | **ODC-By 1.0** | 96 CommonCrawl dumps; per-dump MinHash dedup; strict filtering; datatrove pipeline | Large pretraining | ~44–55 TB | Sample-10BT/100BT/350BT subsets for free-tier |
| **FineWeb-Edu** | 1.3T (+5.4T "high") | en | **ODC-By 1.0** | Llama-3-70B-classifier educational filter; dramatically boosts knowledge/reasoning | Quality pretraining | several TB | **Recommended primary** for free compute |
| **FinerWeb-10BT** | 10B | en | ODC-By 1.0 | Line-level LLM filtering | Small/sovereign runs | ~27 GB | Good for 10–100 M-param models |
| **FineWeb2** | 20 TB / 5B docs | 1000+ langs | ODC-By | Adaptive multilingual pipeline | Multilingual | huge | For multilingual XRFM later |
| **Dolma** (AI2) | ~3T (v1.6) / ~1.2T (v1.7) | en+mixed | **AI2 ImpACT** (low-risk, attribution) | Mixed web/wiki/code/books; heavy filtering/dedup | Open pretraining | TBs | Fully tooled; check ImpACT terms for redistribution |
| **SlimPajama** (Cerebras) | 627B | en+mixed | **Apache-2.0** | Cleaned/deduped RedPajama (Llama-1 sources) | Mid-scale pretraining | ~2 TB | Apache license is a major plus |
| **RedPajama-V2** | up to 30T raw | en | Apache-2.0 | CC + quality signals; dedup variants | Large pretraining | 10s TB | |
| **C4** | 175B (en) | en | ODC-By 1.0 | Single CC snapshot; known noise/filter issues | Baseline/ablation | ~800 GB | Older; use en.noclean vs realnews variants carefully |
| **RefinedWeb** (TII) | ~600B | en | Apache-2.0 | Strict dedup/filtering | Web pretraining | TBs | Base of Falcon |
| **The Pile** (EleutherAI) | 340B (v1), Pile-II larger | en+mixed | **MIT (overall); subset licenses vary** | 22 diverse sources; academic/science/code | Diverse small/mid pretraining | ~800 GB | Verify per-subset licenses; some non-commercial |
| **Zyda / Zyda-2** | 1.3T / 5T | en+others | permissive | Cross-dataset LSH dedup | Pretraining mix | TBs | Combines RefinedWeb/SlimPajama/C4/Pile/Starcoder |
| **Wikipedia** | ~3–5B words → ~5B+ tokens | many | CC BY-SA (content) | Clean; canonical knowledge | Knowledge/eval | ~25 GB | Use dumps + WikiExtractor; attribution required |
| **Books** (public domain) | small (1–5B tokens) | en | Public Domain | Project Gutenberg etc. | Long-form prose | GBs | Current XRFM corpus is in this class |
| **OpenWebMath** | ~15B | en | ODC-By? (check) | Math web pages, LaTeX preserved | Math pretraining | ~50 GB | License was relaxed; verify |

**Recommendation for XRFM's free-compute stage:** FineWeb-Edu **sample-10BT** (or
FinerWeb-10BT) for the first real run; add a SlimPajama slice for diversity; hold out a
Wikipedia + code split for validation. All ODC-By/Apache → safe to use and to publish
derivatives with attribution.

## 2. Code

| Dataset | Size | License | Notes |
|---|---|---|---|
| **StarCoder / StarCoder2** subsets | 600B+/several T tokens | BigCode OpenRAIL-M (check commercial terms) | permissively-licensed code from GitHub; decontaminated |
| **The Stack v2** | many TB | per-repo licenses (filtered permissive) | Source of StarCoder |
| **Code-Python subset of SlimPajama** | ~35B tokens | Apache-2.0 | Easy, safe code slice |
| **PSF Python source** (already in XRFM) | tiny | PSF | smoke-scale only |

**Recommendation:** SlimPajama code split (Apache) for v1; graduate to StarCoder2 if its
license fits.

## 3. Math & reasoning

| Dataset | Size | License | Use |
|---|---|---|---|
| OpenWebMath | ~15B tokens | verify | Math-language pretraining |
| **MetaMathQA / MathInstruct** | ~400K | MIT | Math instruction/SFT |
| **MATH / GSM8K** | 12.5K / 8.5K | MIT | eval (not training) |
| **ProofPilot / Lean datasets** | varies | Apache/MIT | formal reasoning (later) |

## 4. Instruction / dialogue

- **OpenAssistant / OASST**: Apache-2.0; conversational SFT.
- **UltraFeedback / UltraChat binarized**: MIT; preference/SFT.
- **FLAN collection subsets**: various (check); broad task coverage.
- **Dolly-15k**: CC-BY-SA.
These are small (10^5–10^6 examples) and for fine-tuning/evaluation, not pretraining.

## 5. Uncertainty / abstention / contradiction / OOD (critical for Section 12)

| Dataset | What it gives | License |
|---|---|---|
| **TruthfulQA** | Questions models commonly hallucinate; abstention-relevant | Apache-2.0 |
| **TriviaQA / NaturalQuestions** | Answerable QA (positive class + known answers) | Apache-2.0 / CC-BY |
| **AmbigQA / AmbigNQ** | Genuinely ambiguous questions (aleatoric UQ) | CC-BY |
| **SQuAD-unanswerable (v2)** | Answerable + unanswerable span questions | CC-BY-SA |
| **MMLU / MMLU-Pro** | Multi-domain knowledge; OOD-by-subdomain | MIT |
| **BoolQ / StrategyQA** | boolean reasoning; confidence/abstention | Apache-2.0 |
| **FEVER** | claims supported/refuted by evidence (conflict detection) | CC-BY-SA (research) |
| **SciQ / QuaRTz / WiCE** | evidence-grounded QA; conflicting-evidence eval | varies, check |
| **Self-consistency / semantic uncertainty datasets** | used in MI-based epistemic UQ [paper](https://arxiv.org/html/2406.02543v2) | varies |

**Synthetic option (recommended):** generate **contradictory-evidence** and
**insufficient-context** examples by perturbing retrieved passages (supply two supporting,
one conflicting, or no relevant passage) and label the correct output state. This gives
unlimited, license-clean training data for the abstention head.

## 6. Compute/storage planning (free-first)

| Stage | Data | Tokens | Approx disk |
|---|---|---|---|
| Correctness / overfit | existing corpus.txt | ~1.7M | <10 MB |
| Tiny real | FinerWeb-10BT sample | 10M–100M | <1 GB |
| SMALL/MEDIUM run | FineWeb-Edu sample-10BT | 0.5–2B | ~10–30 GB |
| Serious MEDIUM | FineWeb-Edu sample-100BT | 5–20B | ~100–300 GB |
| LARGE | FineWeb-Edu (+ SlimPajama) | 50B+ | TBs |

Streaming via `datasets`/`datatrove` avoids downloading whole corpora; use a local
tokenized+packed cache (memmap/arrow) for repeated runs.

## 7. Data quality rules (applied uniformly)
- Dedup (MinHash per-source), language ID, URL/quality filters, PII removal.
- Split at **document boundaries**, never mid-sentence (lesson from XRFM forensic audit).
- Preserve whitespace/newlines (tokenizer is byte-level).
- Hold out a contamination-free validation set (Wikipedia + books + code) and check
  benchmark contamination.
- Record dataset version, hash, and license in every checkpoint (reproducibility rule).
