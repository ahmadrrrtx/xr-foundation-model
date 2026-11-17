# XR Foundation Model (XRFM) — Risk Register, Governance & Resource Plan

**Document Date:** 2026-11-17  
**Resource Envelope:** Kaggle Free GPUs (`2× T4 16GB` / `1× P100 16GB`, 12h session cap, ~30h/wk quota, 20 GB `/kaggle/working`), Hugging Face PRO subscription (`ZeroGPU` Spaces + Hub storage/bandwidth), and **`~$4.00` remaining Hugging Face compute credits** (strictly protected emergency/validation reserve).

---

## 1. Compute, Storage & `$4.00` Credit Ledger

### 1.1 Golden Financial Rule: Never Burn the `$4.00` HF Credits on Routine Work
At official November 2026 Hugging Face Jobs pricing (`T4 = $0.40/hr`, `L4 = $0.80/hr`, `A10G = $1.00/hr`, `A100 = $2.50/hr`, `ZeroGPU overage = $6.00/hr`), `$4.00` equals only **1.6 hours on A100** or **5.0 hours on L4** — **40 minutes of unmonitored ZeroGPU paid overage would wipe out 100% of your balance!**

**Mandatory Billing Safeguards (Execute Before Phase 1):**
1. In Hugging Face **Settings → Billing**, confirm **spending limit / pre-paid cap** is locked so no credit card auto-reload or ZeroGPU `$6.00/hr` paid overage can trigger.
2. Always pass `--timeout 35m` on any Hugging Face Job CLI invocation (`hf jobs run --flavor l4x1 --timeout 35m ...`) because HF Jobs default to 30m or run until manual cancellation if set higher.
3. Run **100% of Phase 0–2 CPU preprocessing, tokenization, contamination indexing, and unit tests** on local CPU / GitHub Actions / Kaggle CPU (`$0.00` GPU quota, `$0.00` HF credits).

### 1.2 Detailed Compute & Storage Allocation Table

| Workload | Target Platform | Hardware Spec | Sessions × Duration | Weekly Quota Impact | HF Credit Cost | Persistent Storage Target |
|---|---|---|---:|---:|---:|---|
| **Unit Tests, Data Curation & 32K Tokenizer Tournament** | Local / Kaggle CPU | 4–8 vCPU, 29 GB RAM | 2 × 1.5h | `0.0h GPU` | `$0.00` | HF Dataset Repo (`RRRTX/xrfm-native-shards-v1`, ~3.5 GB) |
| **Track A Run A1 (`xrfm-native-25m`, 500M tok)** | Kaggle GPU | `2× T4` (32 GB total VRAM, FP16 DDP) | 1 × 4.0h | `4.0h` (Wk 1) | `$0.00` | HF Model Repo (`~100 MB` safetensors + `.pt` state) |
| **Track A Run A2 (`xrfm-native-85m`, 1.7B tok)** | Kaggle GPU | `2× T4` (32 GB total VRAM, FP16 DDP) | 2 × 9.0h | `18.0h` (Wk 1) | `$0.00` | HF Model Repo (`~340 MB` safetensors; rotate last-2 ckpts in `/kaggle/working`) |
| **Track B Base Bake-Off (`Qwen3-1.7B` vs `Qwen2.5-Coder-1.5B`)** | Kaggle GPU | `1× T4` (16 GB VRAM, NF4/FP16) | 1 × 2.5h | `2.5h` (Wk 1) | `$0.00` | Committed JSON (`docs/experiments/`) |
| **Track B QLoRA Smoke + 3× 1-Epoch Ablations** | Kaggle GPU | `1× T4` (16 GB VRAM, ~4.8 GB peak) | 1 × 3.5h | `3.5h` (Wk 2) | `$0.00` | HF Model Repo (`~80 MB` LoRA adapters each) |
| **Track B Flagship QLoRA (`3 epochs`, 40K packed)** | Kaggle GPU | `1× T4` (16 GB VRAM, ~4.8 GB peak) | 1 × 3.5h | `3.5h` (Wk 2) | `$0.00` | HF Model Repo (`~80 MB` LoRA + `~3.5 GB` merged FP16) |
| **Phase 6 Full Eval + Red-Team + Perturbation Suite** | Kaggle GPU *(+ opt HF L4)* | `1× T4` *(opt `1× L4 24GB`)* | 1 × 4.0h *(+ opt 0.5h)* | `4.0h` (Wk 2) | `$0.00` *(opt `$0.40`)* | Committed JSON (`docs/experiments/`) |
| **Phase 7 Interactive Demo (`CoderSec` + Security RAG)** | HF Spaces (PRO) | **ZeroGPU** (included in PRO) | Hosted | `0.0h` | `$0.00` | HF Space Repo (`<100 MB` code + RAG SQLite/FAISS) |
| **TOTAL PROGRAM** | — | — | **~35.5 GPU-h** | **~1.2 Weeks** | **`$0.00`–`$0.40`** | **`~$3.60–$4.00` Reserve Intact** |

---

## 2. Comprehensive Technical, Scientific & Governance Risk Matrix

| ID | Risk Category | Specific Failure Mode | Likelihood | Impact | Prevention & Engineering Control | Fallback Trigger & Contingency Plan |
|---|---|---|---|---|---|---|
| **R-01** | **Hardware / Precision** | **T4 BF16 Emulation / P100 FP16 Overflow:** Training crawls at `0.15×` speed on T4 due to emulated `bfloat16`, or NaNs on P100/T4 due to FP16 `x.pow(2)` overflow in `RMSNorm` (`> 65,504`). | High *(confirmed in audit)* | Critical | Explicitly configure `fp16=True, bf16=False` on T4/P100 (`compute_capability < 8.0`), fix `RMSNorm` and `RoPE` to compute in `float32` and cast back to `x.dtype`, and enable `GradScaler`. | If loss spikes occur at `lr=3e-4` in FP16, clamp `grad_clip=1.0`, add QK-Norm on Track A, or drop `lr` to `1.5e-4`. |
| **R-02** | **Kaggle Ephemeral Limits** | **12-Hour Session Preemption or `/kaggle/working` 20 GB Disk Full:** Unpushed checkpoints lost on session timeout, or merged FP16 checkpoints (`~3.5 GB × 6`) fill the 20 GB disk mid-run and crash `torch.save`. | High | High | Set `save_total_limit = 2` locally; save lightweight LoRA adapters (`~80 MB`) every 250 steps and push asynchronously to private HF Hub repo (`hub_strategy="checkpoint"`); cap each Kaggle script at `9.5 hours` graceful exit (`max_wall_hours=9.5`). | Resume seamlessly from last Hub checkpoint SHA in next Kaggle session (`resume_from_checkpoint=True`). |
| **R-03** | **Multilingual / Tokenizer** | **Script Starvation or Roman-Urdu Collapse:** `ur`, `ar`, or `hi` degrade to byte-fallback (`>6 tok/word`), or `ur-Latn` gets misclassified as English/Indonesian/Hindi-Latin and dropped during LID filtering. | Medium | High | Enforce Phase 2 `GATE-2` (`<2.0×` fertility parity vs English, `0%` unexpected byte fallback); use dedicated `FineWeb-2` `urd_Latn` shard + `RomanUrduCodeSwitchDetector` + `aya_dataset` human Urdu/Arabic/Hindi anchors. | Switch Track A tokenizer to Candidate `T4` (explicit per-script merge quota) or expand vocab from `32,768` to `49,152`. |
| **R-04** | **Catastrophic Forgetting (Track B)** | **Security/Multilingual SFT Degrades Core Coding:** Fine-tuning `Qwen3-1.7B` on security/multilingual QA causes `HumanEval+` / `MBPP+` pass@1 to drop by `>3%`. | Medium | High | Keep **35% execution-verified multi-language coding examples** (`opc-sft-stage2` + `OpenCodeInstruct`) inside the SFT mixture; use moderate LoRA (`r=16, alpha=32`, `lr=2e-4`, `neftune=5.0`), assistant-only completion loss, and evaluate every epoch. | Reduce LoRA learning rate to `1e-4`, increase coding replay ratio from `35%` to `50%`, or perform linear task-vector adapterinterpolation (`0.7 * sft_lora + 0.3 * code_lora`). |
| **R-05** | **Hallucinated Security Citations** | **Fake CVE / CWE / OWASP Control Numbers:** A 1.7B model invents plausible-sounding `CVE-2025-99999` or misstates OWASP ASVS requirements when answering purely from weights. | High *(inherent at 2B scale)* | High | Never train SFT targets to guess ungrounded CVE IDs; train explicit epistemic calibration (*"Verify exact CVE advisory in NVD; root vulnerability pattern corresponds to CWE-89..."*) + ground exact control citations via the **Phase 7 Offline OWASP/CWE/NIST RAG Pack**. | Enforce post-generation regex validator in `xrfm.agent` checking cited `CWE-\d+` / `ASVS` IDs against the local SQLite KB before displaying! |
| **R-06** | **Dual-Use Safety vs. Over-Refusal** | **Either Unsafe Exploit Assistance OR Paralyzing False Refusals:** Model either outputs weaponized malware/SQLi payloads when prompted in Roman-Urdu (`RUAB` bypass), OR refuses harmless `nmap`/`iptables`/log-parsing questions containing words like `"exploit"` or `"kill process"`. | Medium | Critical | Include **multilingual + Roman-Urdu calibrated safety pairs** (refusing weaponization while explaining defensive patching) AND **benign sysadmin/defender anti-overrefusal pairs** (`10%` Pillar 4 in SFT); evaluate both malicious compliance (`<2%`) and benign false refusal (`<10%`) in `GATE-6`. | Rebalance safety vs. benign-admin SFT ratio and add deterministic system-prompt defensive scope framing in `codersec` chat template. |
| **R-07** | **Licensing & Provenance Contamination** | **Non-Commercial or Eval-Only Leakage:** Accidental inclusion of `Qwen2.5-Coder-3B` (`Qwen Research License`), `MultiPL-E` (BSD-3 + ML training ban), `CC-BY-NC` security blogs, or `CC-BY-SA` text inside Apache-2.0 model weights. | Medium *(caught in Phase 2 research!)* | Critical | Hard-code `license_tier` checks in `src/xrfm/data/sources.py`: reject `license=other/NC/eval_only` at build time; keep `OWASP CheatSheetSeries` (`CC-BY-SA-4.0`) strictly in the attributed external RAG KB rather than SFT weights; pin all HF SHAs. | Immediately purge offending source from `sources.yaml`, rebuild deterministic shards, and invalidate affected downstream checkpoints via `DatasetManifestV2` hash check. |
| **R-08** | **Benchmark Contamination** | **Inflated Scores via Pretrain/SFT Overlap:** SFT coding sources (`opc-sft`, `Magicoder`) accidentally overlap with `HumanEval+`, `MBPP+`, or `CyberSecEval 2`. | Medium | High | Run `src/xrfm/data/contamination.py` (exact normalized SHA-256 + 13-gram token/char overlap + docstring similarity) against **all** frozen Phase 1 benchmarks before writing any training shard; publish contamination audit JSON alongside model cards. | Quarantine flagged examples and re-evaluate on time-split `LiveCodeBench` problems published after base-model cutoff. |
| **R-09** | **Reputation / Authorship Confusion** | **Community Misinterprets Track B as "Trained From Scratch":** Presenting `XRFM CoderSec` alongside `XRFM Native` without crystal-clear lineage tags invites warranted skepticism in the ML community. | Low *(with strict governance)* | Critical | Enforce naming (`XRFM Native` vs `XRFM CoderSec`), separate HF repos, auto-generated `base_model: Qwen/Qwen3-1.7B` metadata in HF YAML frontmatter, and explicit UI badges on every screen and table. | Block `GATE-0` and `GATE-8` release sign-off if any doc, README, or UI omits base-model attribution. |

---

## 3. Dual-Use Defensive Cybersecurity Governance Policy

To ensure **XRFM CoderSec** is both genuinely useful to software/security engineers and responsibly bounded:

### 3.1 Explicitly Supported Defensive & Engineering Capabilities (Green Zone)
- **Vulnerability Root-Cause Analysis & Secure Remediation:** Identifying SQLi (`CWE-89`), XSS (`CWE-79`), SSRF (`CWE-918`), buffer overflows (`CWE-120/787`), command injection (`CWE-78`), insecure deserialization (`CWE-502`), path traversal (`CWE-22`), hardcoded secrets (`CWE-798`), and broken access control, and generating **drop-in patched code + unit tests**.
- **System, Cloud & Network Hardening:** Writing secure `Dockerfile`s, Kubernetes `NetworkPolicy`/RBAC manifests, Linux `systemd`/`ufw`/`sshd_config` hardening configs, CSP/CORS headers, and least-privilege IAM policies.
- **Detection & Incident Response Engineering:** Writing Sigma/YARA/Suricata detection rules, parsing auth/web/cloud logs, triaging stack traces, and mapping defensive controls to **MITRE ATT&CK Mitigations**, **OWASP ASVS 5.0**, and **NIST SP 800-61r3 / SP 800-218**.
- **Multilingual & Code-Switched Accessibility:** Delivering all of the above fluently in **English, Urdu (`اردو`), Roman Urdu (`ur-Latn`), Arabic (`العربية`), Hindi (`हिन्दी`), Spanish, and French**.

### 3.2 Explicitly Excluded Offensive / Dual-Use Harm Capabilities (Red Zone — Refused Across All 7 Languages)
- Functional zero-day or CVE weaponization scripts, ransomware/wiper routines, credential-stealing malware, C2 implants, evasion/obfuscation payloads designed to bypass EDR/AV, social-engineering/phishing lures, or unauthorized targeting instructions against specific hosts/organizations.
- When refusing a Red-Zone prompt (in any language or Roman-Urdu code-switch), the model must respond **calmly, concisely, and in the user's language**, pivoting immediately to the underlying defensive concept or remediation pattern rather than lecturing.
