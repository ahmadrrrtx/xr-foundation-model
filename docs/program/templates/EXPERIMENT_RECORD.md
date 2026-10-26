# Experiment Record Template

> Copy to `experiments/<experiment_id>.md` (or the approved append-only store) before launching. Do not put credentials, private raw text, or benchmark answers in this record.

## Identity and ownership

- Experiment ID (`YYYYMMDD-gate-task-short-seed`):
- Ledger task / gate:
- Owner / reviewers:
- Status (`DESIGN`, `RUNNING`, `STOPPED`, `ANALYZED`, `ACCEPTED`, `REJECTED`):
- Start / end timestamps and timezone:
- Hypothesis and explicit success/stop criteria:
- Decision this experiment can support:

## Reproducibility

- Repository URL and exact commit:
- Working tree clean? (`git status --porcelain` evidence):
- Command(s), config files, and config SHA-256:
- Environment/lockfile/container digest:
- Python, PyTorch, CUDA, driver, OS:
- Hardware, GPU count/model/memory:
- Seeds (Python/NumPy/Torch/data/workers):
- Determinism settings and expected numerical tolerance:

## Data and tokenizer

- Raw/source manifest IDs and hashes:
- Frozen training/validation/test manifest hashes:
- Data processing version:
- Data-source review links and admitted license tiers:
- Unique/repeated token counts by language/script/domain:
- Maximum epochs and sampling mixture:
- Tokenizer artifact/revision/hash:
- Benchmark registry revision and contamination-exclusion manifest hash:
- Native-review sample/report link:

## Model and optimization

- Initialization (`random` required for native runs) and initial-weight hash:
- Architecture/config, parameter count (total/non-embedding), context:
- Precision policy and actual device capability result:
- Optimizer, scheduler, warmup, clipping, accumulation:
- Batch size per device and global tokens/batch:
- Token/step budget and 25/50/100% evaluation checkpoints:
- Checkpoint contents and durable archive locations:

## Budget and interruption control

- Maximum GPU-hours / wall time / cost / tokens (hard caps):
- Measured throughput and memory from smoke run:
- Kaggle/session/quota/disk at start (if applicable):
- Checkpoint and upload duration:
- Automatic stop conditions (NaN, OOM, validation plateau, budget, drift, security):
- Resume plan and last known-good checkpoint:

## Results (link immutable machine-readable outputs)

- Training/validation curves and final artifact hashes:
- Held-out bpb by Urdu/English/Roman/mixed and domain:
- Core benchmark results with revisions, prompts, seeds, CIs:
- Agent final-state, pass@1, pass^4, schema, policy and recovery results:
- Security/injection/canary results by surface:
- Throughput, memory, energy/time and interruption overhead:
- Sample-level outputs and error-analysis links:
- Native-speaker blind review and agreement:

## Comparison and decision

- Exact control/baseline and whether compute/data/prompts were equal:
- Hypothesis accepted/rejected and uncertainty:
- Negative results and anomalies:
- Gate/ledger tasks updated (commit link):
- New decision record required?:
- Next action (stop, repeat, scale, rollback, publish):
- Independent reviewer/sign-off and date:
