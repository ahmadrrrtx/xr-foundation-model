# Research, Tool, Data, and Evaluation Matrix

**Research date:** 2026-10-22.  
**Purpose:** record candidates and their conditions of use; this is not a list of blanket approvals.

## Verification legend

- **Repo license checked:** public repository metadata/license text was inspected on the research date. The actual dependency must still be pinned and its license rechecked.
- **Tier A candidate:** potentially suitable for intended training after record-level provenance/attribution checks.
- **Tier B / REVIEW:** compilation, web-content, share-alike, privacy, or upstream terms require specific review.
- **HOLD:** do not use for default training until the stated conflict/unknown is resolved.
- **EVAL-ONLY:** reserve for evaluation; exclude all splits from training by default.

No legal conclusion is attempted here. Data stewards must evaluate intended use, redistribution, model training, attribution, and local law. An open-source code license does not clear text, model, or dataset rights.

## 1. Engineering toolchain

| Layer | Official source | License check | Recommended role | Material caveat / decision |
|---|---|---|---|---|
| Scalable data | [DataTrove](https://github.com/huggingface/datatrove) | Apache-2.0; active repo checked | **Default** streaming processing architecture | Pin revision; build XRFM source/provenance manifests around it rather than losing metadata |
| GPU data processing | [NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) | Apache-2.0; active repo checked | Optional | Adopt only after a measured GPU filtering/dedup need; avoid another mandatory stack |
| Language identification | [GlotLID](https://github.com/cisnlp/GlotLID) | Repo describes Apache-2.0 plus notices; GitHub API returns `NOASSERTION` | Initial language/script labels | Training sources have mixed terms; classification is not Roman/code-switch ground truth |
| Urdu NLP | [UrduHack](https://github.com/urduhack/urduhack) | MIT; repo checked | Normalization and linguistic diagnostics | Preserve raw text; do not normalize away natural variation without ablation |
| Tokenization | [HF Tokenizers](https://github.com/huggingface/tokenizers), [SentencePiece](https://google.github.io/sentencepiece/) | Apache-2.0; repos checked | BPE/unigram tournament and export | Byte fallback is for OOV robustness, not a substitute for ordinary Urdu subwords |
| Small training | Existing XRFM | MIT project code | **Default for S0–S2 after repair** | Current GPU/resume/OOM defects block use until `G1` |
| Scaling | [OLMo-core](https://github.com/allenai/OLMo-core) | Apache-2.0; active repo checked | Bake-off candidate | Evaluate reproducibility/recipes and HF export at fixed budget, not project popularity |
| Scaling/parallelism | [TorchTitan](https://github.com/pytorch/torchtitan) | BSD-3-Clause; active repo checked | Bake-off candidate | Select only if correctness/resume/throughput/maintenance beat alternatives |
| Post-training | [TRL](https://github.com/huggingface/trl) | Apache-2.0; active repo checked | Default SFT; optional later preference work | Requires stable HF model/chat contract and loss-mask tests |
| Open post-training recipes | [Open Instruct](https://github.com/allenai/open-instruct) | Apache-2.0; active repo checked | Reference/optional RLVR later | Do not import a large recipe stack before SFT and verifiers are stable |
| General LM evaluation | [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) | MIT; repo checked | Default external harness behind pinned adapter | Custom Urdu tasks still need versioned local definitions and contamination control |
| Alternative LM evaluation | [Lighteval](https://github.com/huggingface/lighteval) | MIT; repo checked | Optional alternative | Replace only after result/output-equivalence tests |
| Structured validation | [jsonschema](https://github.com/python-jsonschema/jsonschema) | MIT; repo checked | **Mandatory candidate** for Draft 2020-12 boundaries | Validation does not establish authorization or semantic safety |
| Constrained decoding | [XGrammar](https://github.com/mlc-ai/xgrammar) | Apache-2.0; active repo checked | Serving-side JSON/grammar constraints | Test supported schema subset and fallback behavior; runtime validation remains mandatory |
| Serving | [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) | vLLM repo Apache-2.0; checked | Conditional production serving | Custom XRFM must first meet HF/server parity; API fields vary by version, so pin docs/version |
| Agent interoperability | [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), [tool specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) | MIT SDK; repo checked | Optional tool-protocol adapter | Tool annotations/descriptions and cross-server outputs are untrusted; user consent/policy still required |
| Agent security | [AgentDojo](https://github.com/ethz-spylab/agentdojo) | MIT; repo checked | Injection/utility-under-attack evaluation reference | Adapt tasks to four language surfaces and local tools; no benchmark proves absolute safety |
| Agent task evaluation | [Gorilla/BFCL](https://github.com/ShishirPatil/gorilla) | Apache-2.0; repo checked | Structured-call/AST reference | Schema correctness is not final-state task success; pin exact benchmark version |
| Interactive agents | [τ-bench](https://github.com/sierra-research/tau2-bench) | MIT; repo checked | Protocol reference / extended evaluation | Built-in language packs do not replace native Urdu/Roman tasks; revisions can change scores |

## 2. Data candidates and license/provenance state

**No external corpus below is approved for training yet.** “Candidate” means worthy of a source review, not permission to ingest.

| Candidate | Research finding | Terms observed | State and required action |
|---|---|---|---|
| [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Reproducible multilingual web pipeline; Urdu subset is a possible broad-background source | HF metadata at checked revision `af9c1333…`: ODC-By; card also invokes [Common Crawl terms](https://commoncrawl.org/terms-of-use) | **Tier B / REVIEW.** Compilation terms do not relicense every page. Source/domain/PII/decontamination audit required |
| [MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400) | Broad audited multilingual document resource | HF metadata at `9d886a76…` says ODC-By; narrative/card material has also presented CC-BY-4.0 wording | **HOLD.** Resolve the exact revision's license text and upstream document terms before use |
| [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) | Multilingual mC4/OSCAR-derived candidate | Checked HF metadata at `6a8734bc…` had no license field; upstream source terms apply | **HOLD.** No license metadata is not permission; review mC4/OSCAR and underlying content |
| [GlotCC](https://github.com/cisnlp/GlotCC) | Minority-language Common Crawl pipeline/corpus | CC0 for packaging/metadata; Common Crawl terms for underlying text | **Tier B / REVIEW.** Same compilation-versus-content distinction as FineWeb2 |
| [UrduLM / AUC reference](https://arxiv.org/html/2601.17664) | Reports 33 GB, ~13M rows, estimated 5–6B Urdu tokens; useful prior art | Paper states usage aligned with original terms and limited strictly to research purposes | **HOLD for default training.** Verify actual artifact/data terms and every source; paper license is not artifact clearance |
| [Alif](https://github.com/traversaal-ai/alif-urdu-llm) / Urdu-Instruct | Useful continued-pretraining/SFT baseline; reported 51,686-example Urdu-Instruct set | Dataset described as CC BY-SA in project materials | **EVAL/BASELINE; separate REVIEW for training.** Analyze attribution/share-alike implications and artifact revision |
| [Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar) | Transliteration candidate relevant to Roman/native-script alignment | Card describes CC-BY for manually collected data and CC0 for mined/existing sources; HF metadata at `e418c1fc…` is generic `cc` | **Tier A/B candidate by subset.** Validate direct files and Urdu coverage. Hub viewer currently reports inconsistent-column generation failure; do not rely on viewer alone |
| [Roman-Urdu-Parl paper](https://dl.acm.org/doi/10.1145/3464424) and [HF mirror](https://huggingface.co/datasets/Mavkif/Roman-Urdu-Parl-split) | Reports 6.37M Urdu/Roman-Urdu pairs; potentially valuable alignment data | HF mirror at `b9ef6706…` says Apache-2.0; a Kaggle distribution is labeled CC BY 4.0; original source/provenance must govern | **HOLD.** Conflicting mirror labels and web/social provenance require original-publisher resolution |
| [UCI Roman Urdu resource](https://archive.ics.uci.edu/ml/datasets/Roman+Urdu+Data+Set) and [GitHub compilation](https://github.com/Smat26/Roman-Urdu-Dataset) | Small manually tagged/sentiment and lexical resources | UCI search/card material says CC BY 4.0; GitHub compilation says GPL-3.0 | **HOLD.** Determine which rights apply to which files; do not merge labels across mirrors |
| [RomanUrdu sentiment corpus](https://huggingface.co/datasets/Khubaib01/RomanUrdu-NLP-Sentiment-Corpus) | 134,052 labeled samples claimed; possible classification/auxiliary source | HF metadata at `7e69620f…` says Apache-2.0 | **REVIEW/HOLD.** Audit original social/chat provenance, duplication, label method, consent and representativeness; not default pretraining data |
| [Makhzan](https://github.com/zeerakahmed/makhzan) | Editorial-oriented Urdu corpus with structural metadata, useful quality reference | Repository distinguishes scripts/stats MIT terms from other source materials | **Tier B / REVIEW.** Inspect per-document rights; do not apply the code license to text |
| **Owned/commissioned native corpus** | Highest-value source for natural Roman/code-switching, agent trajectories, dialect/register coverage | Contributor agreement can explicitly cover training, redistribution, attribution, and withdrawal process | **Preferred Tier A path.** Requires consent, PII minimization, secure raw storage, reviewer compensation and provenance |

### Data admission invariants

1. Preserve original URL/source ID, revision/acquisition time, content hash, license evidence, and transformations.
2. Do not infer “commercially usable” from a Hub tag alone; inspect original and underlying source terms.
3. Keep code, packaging, annotations, and underlying text rights separate.
4. Do not redistribute raw Tier B material merely because training use is under review.
5. Private WhatsApp/Telegram/group content requires explicit consent; public accessibility alone is not consent.
6. Teacher-generated data needs teacher/model/version/prompt/terms and human quality review.
7. Any later discovered critical license/privacy issue quarantines the source and forces descendant dataset/model review.

## 3. Evaluation registry candidates

All benchmark revisions and licenses must be pinned in `EVAL-001`. `EVAL-ONLY` means no pretraining/SFT/RL use of any split by default, including translations or near-duplicates.

| Resource | What it measures | Role | Conditions |
|---|---|---|---|
| [UrduMMLU](https://arxiv.org/html/2606.07167v2) | 26,389 primarily native Urdu MCQs, 26 subjects/five domains | **Core native knowledge/understanding** | Verify data/code revision and terms; report domain/subject and prompt-language results with CIs |
| [UrBLiMP](https://aclanthology.org/2026.findings-acl.29/) | 5,696 grammatical minimal pairs across ten phenomena | **Core linguistic competence** | Use forced-choice likelihood protocol; report per phenomenon, not only average |
| [Belebele](https://huggingface.co/datasets/facebook/belebele) | Multilingual reading comprehension; Urdu Arabic and Latin scripts | **Core script-sensitive reading** | Metadata at `7899cdfa…`: CC BY-SA 4.0; EVAL-ONLY; verify exact `urd_Arab`/`urd_Latn` configurations |
| [UQuAD+](https://doi.org/10.1145/3759455) | 20,000 native human-annotated Urdu MRC pairs, answerable/unanswerable | **Core/extended reading and abstention** | Reported CC BY-NC-SA; EVAL-ONLY and confirm artifact availability/terms |
| [UrduBench](https://github.com/traversaal-ai/urdubench_leaderboard) | Human-validated translated Urdu reasoning tasks | **Extended reasoning** | Label translated origin; do not equate with native-authored cultural competence |
| [UQA](https://github.com/sameearif/UQA) | Large translated Urdu QA resource | **Extended transfer diagnostic** | Keep separate from native UQuAD+; verify revision/terms |
| [Roman Urdu IR](https://aclanthology.org/2025.lowresnlp-1.9/) | Multi-hop translated/transliterated MS MARCO retrieval | **Extended retrieval diagnostic** | Synthetic/multi-hop origin must be explicit; not a substitute for natural Roman text |
| [RUAB](https://aclanthology.org/2026.abjadnlp-1.37/) | Matched English/mixed/transliterated safety inputs; 400 intents/1,200 forms | **Core safety-transfer protocol if artifacts available** | Verify release/license; otherwise recreate the protocol with new native-reviewed, non-duplicated items |
| Custom held-out bpb | Tokenizer-independent language modeling by source/domain/surface | **Core training/scale signal** | Natural/synthetic/transliterated strata reported separately; document-group split |
| Custom four-surface generation suite | Fluency, adequacy, naturalness, instruction following, cultural appropriateness | **Core human evaluation** | Native-authored/double-reviewed, blind model identity, sealed test set |
| `XRFM-UrduAgentBench` | Final-state tool success, policy, clarification, recovery across four surfaces | **Core agent utility** | Build before SFT; no-tool/base/scripted-oracle controls; pass@1 and pass^4 separate |
| [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) | Function-calling structure/AST/execution categories | **Core structured-call reference** | Pin version; do not use leaderboard vendor numbers as local results |
| [AgentDojo](https://github.com/ethz-spylab/agentdojo) | Utility and prompt-injection attacks in tool environments | **Core agent security reference** | Add Urdu/Roman/mixed attacks and secret canaries; publish residual ASR |
| [τ-bench](https://github.com/sierra-research/tau2-bench) | Interactive user/tool/policy tasks and final-world-state evaluation | **Extended protocol reference** | Use native custom tasks for Urdu; record exact benchmark release because task fixes affect comparability |
| English retention tasks via LM Harness | General English knowledge/reasoning regression | **Core retention** | Pin a small stable suite; verify each dataset's terms and contamination status |

## 4. Compute/storage research boundaries

Official [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks) and [dataset documentation](https://www.kaggle.com/docs/datasets) were reviewed during research. Published guidance described up to 20 GB auto-saved `/kaggle/working`, up to 200 GB per dataset, GPU sessions up to 12 hours, P100/T4x2 options, and a variable weekly GPU quota commonly described as 30 hours.

These are **not guaranteed account resources** and must not be converted into a training-time promise. At run time, record the actual hardware/quota/session/disk and measure checkpoint/upload overhead. Kaggle credentials are unnecessary for planning and should never be pasted into chat or committed.

Drive is appropriate for private documents/reports/archives, not active high-throughput training I/O. Final weights should use versioned safetensors on an artifact hub with model/data cards and hashes.

## 5. Research limitations

- Repository/tool licenses were checked through public metadata and official text on 2026-10-22; dependencies are not yet pinned for implementation.
- Dataset tags and paper statements are leads, not legal clearance. Conflicts are deliberately left `HOLD` rather than “resolved” in XRFM's favor.
- No physical CUDA/Kaggle run, two-GPU fault test, large corpus download, external teacher call, or paid evaluation was performed.
- No external benchmark score is presented as an XRFM result. Baselines must be rerun locally at pinned revisions.
- Human native-speaker review is still an unassigned program input; the plan treats it as a gate rather than pretending automated research replaces it.
