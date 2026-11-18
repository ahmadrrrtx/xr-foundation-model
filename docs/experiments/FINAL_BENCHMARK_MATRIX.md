# XRFM Final Benchmark Matrix — Phase 6 (`GATE-6`)

**Generated:** 2026-11-18  
**Gate:** `GATE-6` — Cross-Track Evaluation, Red-Teaming & Stress Tests  
**Status:** `IN_PROGRESS` — Evaluation framework ready; Kaggle T4 execution pending  
**Compute:** Kaggle 1× Tesla T4 (`$0`) — optional HF L4 validation (max `$0.40`)

---

## 1. Models Evaluated

| Model | HF Repo | Commit SHA | Track |
|-------|---------|------------|-------|
| XRFM Native 25M | `ahmadrrrtx/xrfm-native-25m` | `3487b9c8` | Track A |
| XRFM Native 85M | `ahmadrrrtx/xrfm-native-85m` | `dbe5f102` | Track A |
| Qwen3-1.7B (base) | `Qwen/Qwen3-1.7B` | `70d244cc` | Track B base |
| Qwen2.5-Coder-1.5B-Instruct (base) | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `2e1fd397` | Track B base (bake-off runner-up) |
| XRFM CoderSec QLoRA (merged) | `ahmadrrrtx/xrfm-codersec-flagship-merged` | *(pending publication)* | Track B fine-tuned |

---

## 2. Benchmark Suites

| Suite | Sub-suites | Languages | Metric | Status |
|-------|-----------|-----------|--------|--------|
| **EvalPlus** | HumanEval+, MBPP+ | Python (code) | pass@1 | FRAMEWORK_READY |
| **MultiPL-E** | JS, C++, RS, SH | Code | pass@1 | FRAMEWORK_READY |
| **CyberSecEval 2** | Insecure code rate, Prompt injection resilience, False-refusal rate | EN, code | rate (%) | FRAMEWORK_READY |
| **Belebele** | `urd_Arab`, `urd_Latn`, `eng_Latn`, `arb_Arab`, `hin_Deva`, `spa_Latn`, `fra_Latn` | EN, UR, UR-Latn, AR, HI, ES, FR | accuracy (%) | FRAMEWORK_READY |
| **XRFM-MultiCoderSec-Bench** | Secure code review, Defensive hardening, Roman-Urdu debugging, Benign sysadmin anti-overrefusal, Malicious exploit refusal | EN, UR, UR-Latn, AR, HI, ES, FR | compliance rate (%) | FRAMEWORK_READY |

---

## 3. Red-Teaming Tests

### 3.1 Weaponization Compliance (< 2% required)

Tests across 7 languages + Roman-Urdu code-switching, using prompts designed
to elicit weaponizable code/security advice (password exfiltration, SQL
injection payloads, reverse shells, privilege escalation).

| Language | Prompts | Threshold | Status |
|----------|---------|-----------|--------|
| English | 3 | < 2% | FRAMEWORK_READY |
| Urdu (urd_Arab) | 2 | < 2% | FRAMEWORK_READY |
| Urdu (urd_Latn / Roman-Urdu) | 2 | < 2% | FRAMEWORK_READY |
| Arabic | 2 | < 2% | FRAMEWORK_READY |
| Hindi | 2 | < 2% | FRAMEWORK_READY |
| Spanish | 2 | < 2% | FRAMEWORK_READY |
| French | 2 | < 2% | FRAMEWORK_READY |
| Code-switched RU-EN | 2 | < 2% | FRAMEWORK_READY |
| **Total** | **17** | **< 2%** | **FRAMEWORK_READY** |

### 3.2 False-Refusal Rate (< 10% required)

Tests legitimate defensive security prompts that should NOT be refused
(SQL injection prevention, XSS sanitization, rate limiting, etc.).

| Language | Prompts | Threshold | Status |
|----------|---------|-----------|--------|
| English | 3 | < 10% | FRAMEWORK_READY |
| Urdu (urd_Arab) | 2 | < 10% | FRAMEWORK_READY |
| Urdu (urd_Latn / Roman-Urdu) | 2 | < 10% | FRAMEWORK_READY |
| Arabic | 2 | < 10% | FRAMEWORK_READY |
| Hindi | 2 | < 10% | FRAMEWORK_READY |
| Spanish | 2 | < 10% | FRAMEWORK_READY |
| French | 2 | < 10% | FRAMEWORK_READY |
| **Total** | **17** | **< 10%** | **FRAMEWORK_READY** |

### 3.3 Insecure Code Generation Rate

Compares XRFM CoderSec QLoRA (merged) vs Qwen3-1.7B unadapted base on
secure code generation prompts. Criterion: XRFM insecure rate < base rate.

| Comparison | Prompts | Criterion | Status |
|------------|---------|-----------|--------|
| XRFM CoderSec vs Qwen3-1.7B base | 3 | XRFM < base | FRAMEWORK_READY |

### 3.4 Roman-Urdu Orthographic Perturbation Stress Test

Tests answer stability across orthographic variants:
- `kya/kia`, `mujhe/mujy`, `hain/hen`
- Mixed Urdu-English stack traces

| Perturbation | Pairs | Metric | Threshold | Status |
|-------------|-------|--------|-----------|--------|
| kya/kia | 2 | Answer stability | ≥ 0.80 | FRAMEWORK_READY |
| mujhe/mujy | 2 | Answer stability | ≥ 0.80 | FRAMEWORK_READY |
| hain/hen | 1 | Answer stability | ≥ 0.80 | FRAMEWORK_READY |
| Code-switched stack traces | 2 | Answer stability | ≥ 0.80 | FRAMEWORK_READY |

---

## 4. Gate Criteria — GATE-6

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| Malicious weaponization compliance rate | < 2% (all 7 languages) | FRAMEWORK_READY |
| Benign sysadmin/defensive false-refusal rate | < 10% | FRAMEWORK_READY |
| Insecure code generation rate | Lower than unadapted base | FRAMEWORK_READY |

---

## 5. Failure Case Documentation

To be documented after Kaggle T4 evaluation run. Expected failure cases:
- 1.7B model may still hallucinate obscure CWEs without RAG grounding
- 85M native model may hit capacity limits on complex reasoning tasks

---

## 6. Execution Plan

1. **Kaggle T4 Kernel**: `ahmadrrrtx/xrfm-codersec-phase6-eval` — runs all benchmark
   suites against Track A (25M, 85M), Track B base models, and Track B QLoRA
   merged checkpoint
2. **Optional HF L4 validation** (max `$0.40`): 24 GB VRAM single-run validation
   of merged FP16 weights
3. **Results output**: `docs/experiments/FINAL_BENCHMARK_MATRIX.md` (this file)
   + `docs/experiments/phase6_dual_track/P6-01_evaluation_results.json`

---

## 7. Artifacts

| Artifact | Path |
|----------|------|
| Evaluation results JSON | `docs/experiments/phase6_dual_track/P6-01_evaluation_results.json` |
| Evaluation harness | `scripts/run_phase6_evaluation.py` |
| Regression tests | `tests/test_phase6_gate.py` |
| Frozen benchmarks manifest | `configs/eval/frozen_benchmarks_v1.json` |

---

*This matrix is a living document. Results will be populated as Kaggle T4
evaluation kernels complete.*
