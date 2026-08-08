# XRFM — Baseline Environment & Repository State (Phase 0)

**Date (recorded):** 2026-08-08
**Recorded by:** XRFM forensic audit (Agent Mode, Arena.ai)
**Purpose:** Exact starting state of the repository and environment before any modification. Nothing in this file is a claim about the code's correctness — it is a point-in-time inventory.

---

## 1. Repository State

| Item | Value |
|---|---|
| Repository URL | https://github.com/ahmadrrrtx/xr-foundation-model.git |
| Clone location | `/home/user/repo` |
| **HEAD commit SHA** | `cff2dc6c6cc231bbdbb3a3a9c7c9ddaf8f022fbd` |
| **Branch at baseline** | `main` (clean; no uncommitted changes) |
| HEAD commit subject | `feat: Add Free Gradio HuggingFace Space entrypoint & dataloader fixes` |
| Other remote branches | `fix/release-v1.0.0`, `fix/release-v1.0.0-v2` (untouched) |
| Tracked files | 137 |
| Working tree size | 143 MB (incl. a 76.8 MB committed checkpoint) |
| `.git` size | 69 MB |

**Audit branch created:** `audit/forensic-v2` (based on `cff2dc6`). `main` is left untouched.

## 2. Environment (sandbox used for the audit and all experiments)

| Item | Value |
|---|---|
| OS | Debian GNU/Linux 13 (trixie), kernel 6.1.158 x86_64 |
| CPU | 2 vCPU — Intel Xeon @ 2.60 GHz (1 physical core, 2 threads) |
| RAM | 1.9 GB total (~1.1–1.5 GB available) |
| Swap | 0 |
| Disk | 25 GB root; ~20 GB available at audit start |
| **GPU** | **NONE — CPU-only environment** (no nvidia-smi, no CUDA) |
| Python | 3.13.14 (`/usr/local/bin/python3`) |
| pip | 26.1.2 |

### Installed Python packages relevant to the project

| Package | Version |
|---|---|
| torch | 2.13.0+cpu (installed for the audit) |
| numpy | 2.3.5 |
| PyYAML | 6.0.3 |
| pytest | 9.0.3 |
| fastapi | 0.141.1 |
| uvicorn | 0.52.1 |
| pydantic | 2.13.4 |
| gradio | 6.22.0 |

> Note: the sandbox's installed packages do not persist across sessions; they are re-installed as needed. All experiment artifacts (checkpoints, logs, docs) are saved in the workspace and persist.

## 3. Repository Inventory (137 tracked files)

```
.dockerignore, .github/workflows/{ci,cd,release}.yml, .gitignore
ARCHITECTURE_REVIEW.md, AUDIT_CLOSURE.md, CHANGELOG.md, CODE_OF_CONDUCT.md,
CONTRIBUTING.md, DECISIONS.md, LICENSE, README.md, ROADMAP.md, SECURITY.md, UPGRADE_PLAN_A_PLUS.md
api/__init__.py, api/main.py, api/schemas.py
api/routes/{__init__,completions,health,metrics,tokenize_endpoints}.py
benchmark/{inference_forward,model_forward,training_forward}.py
checkpoints/checkpoint_step_500.pt          # 76,805,341 bytes, committed to git
config/config.yaml
data/datasets/sample.txt                    # 100 lines, ~29.8 KB
deployment/{Dockerfile,Dockerfile.gpu,docker-compose.yml,gunicorn_conf.py}
deployment/huggingface_space/{Dockerfile,README.md,app.py}
docs/{data,deployment,evaluation,inference,model,optimization,research,scaling,tokenizer,training}/*.md
evaluation/{__init__,benchmarks,perplexity}.py
inference/{__init__,engine,kv_cache,sampling}.py
model/{embedding,gpt}.py
model/attention/{multi_head,rope}.py
model/layers/{rmsnorm,swiglu,transformer_block}.py
notebooks/XRFM_Training_Colab.ipynb
optimization/{__init__,flash_attention,quantization,speculative_decoding}.py
pyproject.toml, requirements.txt
research/{FINAL_READINESS.md, blueprints/*, interface/*, phase_0[2-6]/RESEARCH.md,
         phase_05/SELF_REVIEW.md, phase_10/RESEARCH.md, roadmap/*, tdr/TDR_ALL.md}
scripts/{serve_custom_model.py, torchrun_launch.sh, train_custom_model.py}
tests/test_{attention,config,data_loader,distributed,embedding,evaluation,inference,
           model_architecture,optimization,regression,search_engine,tokenizer_bpe,
           training,transformer_block}.py     # 192 tests total
tokenizer/{DESIGN.md,__init__,bpe,decode,encode,interface}.py, tokenizer/vocab.json
training/{checkpoint,distributed,loop,mixed_precision,optimizer,scheduler}.py
vercel_app/, webui/                          # static JS chat UIs
xrfm/{__init__.py, config/loader.py, data/loader.py, search/{agent,indexer,retriever}.py}
```

### Version indicators found (already inconsistent at baseline)

| Location | Version string |
|---|---|
| `README.md` | v1.0.0 "Production Ready — All Phases Complete" |
| `pyproject.toml` | 1.0.0 |
| `xrfm/__init__.py` | 1.0.0 |
| `config/config.yaml` (header) | v0.7.0 |
| `CHANGELOG.md` (first entry) | v0.6.0 |
| `AUDIT_CLOSURE.md` | v0.5.1 |
| Git log (latest release-style commit) | "Release v1.0.0", plus "Upgrade XRFM v1.1.0" commit |

## 4. Committed Binary Artifacts (unverifiable claims)

| Artifact | Size | Content | Audit status |
|---|---|---|---|
| `checkpoints/checkpoint_step_500.pt` | 76.8 MB | torch zip; `model_state_dict` for 6-layer, d_model=256, **vocab_size=50304** GPT; metadata `step=500, loss=0.0116, best_loss=0.0116`; **optimizer_state_dict EMPTY**; no config, no scheduler state | Cannot be verified against any training log in the repo; **cannot resume training** (no optimizer state); inconsistent with shipped tokenizer (see §5) |
| `tokenizer/vocab.json` | 15 KB | `vocab_size_target=1024`, actual vocab **408 tokens** (256 latin-1 chars + 152 merges), `special_tokens: {}` | Trained on `data/datasets/sample.txt`-like text; far below target; no special tokens |
| `data/datasets/sample.txt` | 29.8 KB | **The same ~297-char paragraph repeated 100 times** (single sentence repeated) | Toy data; not a foundation-model corpus |

## 5. The Three-Way Vocabulary Contradiction (recorded at baseline)

1. `config/config.yaml` → `model.vocab_size: 50304` (embedding is 50304×256; ~12.9 M of the 19.2 M parameters are embedding rows).
2. Committed tokenizer `tokenizer/vocab.json` → 408 usable tokens (IDs 0–407).
3. `scripts/train_custom_model.py` trains a fresh `BytePairEncoder(vocab_size_target=1024)` at runtime and overwrites `tokenizer/vocab.json`.

The committed "pre-trained" checkpoint (vocab 50304) cannot be tokenized/decoded with the committed tokenizer (408 tokens); the training script cannot produce a model coherent with either.

## 6. Baseline Test Suite Result (recorded before any change)

```
$ python -m pytest tests/ -v
192 passed, 1 warning in 32.55s
```

All 192 tests pass at baseline. Audit determination: the suite verifies *shapes, no-NaN, and API behavior of the code against itself*; it contains **no ground-truth numerical reference tests, no causal-masking assertion, no loss-decrease test, no overfit test, no real resume test, and no API import test** (the API does not import — see forensic audit).

## 7. Baseline Empirical Facts (measured 2026-08-08, CPU)

- End-to-end training smoke test (6 steps, batch 4, seq 127, d_model 256, 6 layers, vocab 50304): **~1.19 s/step, ~425 tokens/s**; loss started ≈10.64–10.69 (near random ln(50304)≈10.83) and decreased.
- Attention is causal at baseline **only because** `optimization.flash_attention.flash_attention_forward` imports successfully and routes to `F.scaled_dot_product_attention(is_causal=True)`. The manual fallback path (mask=None) has **no -inf masking anywhere** (verified: scores contain no -inf).
- `api.main` **fails to import**: `ImportError: cannot import name 'search_routes' from 'api.routes'` (the module does not exist in the repo).
- Tokenizer round-trip `decode(encode(x))` is **lossy for every English input tested** (all whitespace removed) and **raises ValueError on any non-Latin-1 character** (e.g. `你`, Arabic, emoji).

## 8. Reproducibility Notes (baseline)

- No seed is set anywhere in the training stack (`training/loop.py`, `training/distributed.py`, scripts).
- `DataLoader(shuffle=True)` uses the unseeded global RNG → non-reproducible batch order.
- Checkpoints do not store config, tokenizer version, dataset version, or code commit.
- `SchedulerLoader` implements no `state_dict`/`load_state_dict` → resume restarts the LR schedule from step 0.
- README references `scripts/validate_training.py`, which **does not exist in any commit** in this repository's history.
