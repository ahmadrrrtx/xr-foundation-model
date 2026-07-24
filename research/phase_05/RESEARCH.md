# Phase 5 — Training Engine: Fresh Research Report

Date: 2026-07-24  
Module: Training Engine (`training/loop.py`, `training/optimizer.py`, `checkpoints/`, `inference/` reserved)  
Status: Phase 5 approved (`v0.5.0`) — research in progress — design review pending — implementation not yet started.

---

## 1. Official Sources Consulted (Phase 5 — Training / Optimization / Checkpointing)

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. (*Training procedure reference — original design; no code copied.*)
- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. (*Optimizer design — reference only.*)
- Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization (`AdamW`). (*Standard modern optimizer.*)
- Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). Scaling Laws for Neural Language Models. (*Chinchilla — training compute optimization; batch size / learning rate scaling.*)
- Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). Training Compute-Optimal Large Language Models. (*DeepMind — `Chinchilla` scaling laws.*)
- Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. (*Numerical stability in training.*)
- Dao, T. (2024). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. (*Optimization reference — Phase 9.*)
- Meta AI (2024). Llama 3 Technical Report. (*Training details: `AdamW`, `cosine` LR schedule, `weight_decay`, `gradient_clip`, `mixed_precision`.*)
- DeepSeek-AI (2024). DeepSeek-V3 Technical Report. (*Training: distributed (`FSDP`/`DDP` hooks), `mixed_precision` (`bfloat16`), gradient accumulation.*)
- Jiang, A. Q., et al. (2023). Mistral 7B Technical Report. (*Training: `AdamW`, `cosine` schedule, `gradient_clip`.*)
- PyTorch Documentation (`torch.optim.AdamW`, `torch.optim.lr_scheduler.CosineAnnealingLR`, `torch.cuda.amp`, `torch.nn.utils.clip_grad_norm_`). (*Standard library references — no code copied.*)

---

## 2. Component Research and Recommendations (Phase 5 — Training Engine)

### 2.1 Training Loop (`training/loop.py` — Reserved / Design Phase)

**Purpose:** Main training loop that iterates over dataset batches, performs forward/backward passes, applies optimization (`AdamW`), updates parameters, and logs metrics (`loss`, `learning_rate`, `gradient_norm`).

**Requirements (`DECISIONS.md` — Phase 5 design must confirm):**
- Config-driven (`ConfigLoader.get_training_config()` provides `batch_size`, `max_steps`, `learning_rate`, `warmup_steps`, `gradient_clip`, `mixed_precision`, `checkpoint_every`).
- Configurable optimizer (`AdamW` by default; `SGD` optional for comparison; `RESEARCH-ONLY`: `Shampoo`, `Sophia`).
- Configurable learning rate schedule (`cosine` with `warmup` by default; `linear` optional; `RESEARCH-ONLY`: `constant`, `exponential`).
- Gradient clipping (`torch.nn.utils.clip_grad_norm_`) — `CORE` (`training.gradient_clip` config).
- Mixed precision (`torch.cuda.amp`) — `OPTIONAL` (`training.mixed_precision` config; default `True`).
- Gradient accumulation (simulate larger `batch_size` by accumulating over `N` steps) — `OPTIONAL` (post-`v0.5.0`; design hooks in loop interface).
- Checkpointing (`checkpoints/` directory) — `CORE` (`training.checkpoint_every` config); saves model weights, optimizer state, training state (`step`, `epoch`, `loss`).
- Numerical stability: `gradient_clip` prevents explosion; `mixed_precision` (`bfloat16`) improves speed; `AdamW` decouples `weight_decay` from gradient update.
- Security: checkpoint files validated (`checksum` or `hash` for integrity); no hidden dependencies (`torch` only).
- Stable interface: `training_loop(config, model, dataset)` must support future `FSDP` (`PyTorch` native), `DeepSpeed` (`RESEARCH-ONLY`), `Megatron` (`RESEARCH-ONLY`) without rewrites.

**Trade-offs:**
- `AdamW` (`Kingma & Ba 2015`; `Loshchilov & Hutter 2019`) is standard but requires more memory (`optimizer` stores `momentum` and `variance` for each parameter; `2x` parameter count memory overhead). For `XRFM-10M` (`~19.2M` params), optimizer state is `~38.4MB` (`FP32`); acceptable for modern GPUs.
- `Mixed precision` (`bfloat16`) reduces memory (`2x` reduction for activations; `~1.5x` speedup on modern GPUs with `Tensor Cores`) but requires `gradient_scaling` (`torch.cuda.amp.GradScaler`) to prevent `underflow`. `GradScaler` is `OPTIONAL` but recommended for `mixed_precision: True`.
- `Gradient accumulation` allows larger effective `batch_size` (e.g., `batch_size=32` with `accumulation_steps=4` = effective `batch_size=128`) without increasing `batch_size` in `ConfigLoader`. This is useful for `XRFM-1B+` but deferred to `Phase 8` (`v0.8.0`).

**XRFM Recommendation:** `CORE` (`training/loop.py` design; `AdamW` + `cosine` schedule + `gradient_clip` + `mixed_precision` hooks + `checkpoint_every`). `FSDP` = `OPTIONAL` (`Phase 8`); `DeepSpeed`/`Megatron` = `RESEARCH-ONLY`; `Gradient accumulation` = `OPTIONAL` (`Phase 8`).

---

### 2.2 Optimizer (`AdamW` — `training/optimizer.py` — Reserved / Design Phase)

**Purpose:** Update model parameters using `AdamW` (`Adam` with decoupled `weight_decay`).

**Requirements:**
- Configurable (`optimizer`: `learning_rate`, `weight_decay`, `betas` (`(0.9, 0.999)` by default), `eps` (`1e-8` by default)).
- `Weight decay` (`training.weight_decay`) applied to all parameters (`embedding`, `attention`, `SwiGLU`, `norm`) except biases (`bias` terms excluded from `weight_decay` — standard `AdamW` practice).
- `Gradient clipping` (`training.gradient_clip`) applied before optimizer step (`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)`).
- `Numerical stability`: `AdamW` uses `epsilon` (`1e-8`) to prevent division by zero; `gradient_clip` prevents explosion; `mixed_precision` (`GradScaler`) prevents `bfloat16` underflow.

**Design notes:**
- The optimizer interface (`optimizer.step()`, `optimizer.zero_grad()`) is standard (`PyTorch` `torch.optim.Optimizer`). Future `Shampoo` or `Sophia` optimizers (`RESEARCH-ONLY`) can replace `AdamW` without changing `training_loop` interface.
- `Weight decay` (`training.weight_decay`) is decoupled from gradient (`AdamW` design) — `weight_decay` is applied to the parameter directly, not to the gradient (`Loshchilov & Hutter 2019`). This improves generalization compared to `L2` regularization (coupled to gradient).

**XRFM Recommendation:** `CORE` (`AdamW` with standard settings; `weight_decay` decoupled; `gradient_clip` applied before step; `mixed_precision` `GradScaler` optional). `Shampoo`/`Sophia` = `RESEARCH-ONLY`.

---

### 2.3 Learning Rate Schedule (`Cosine` with `Warmup` — `training/scheduler.py` — Reserved / Design Phase)

**Purpose:** Adjust `learning_rate` over training (`max_steps`) using `cosine` decay with `linear` warmup.

**Requirements:**
- Configurable (`training.learning_rate`, `training.warmup_steps`, `training.max_steps`).
- `Cosine` decay (`torch.optim.lr_scheduler.CosineAnnealingLR` or custom `cosine` schedule): `lr = base_lr * 0.5 * (1 + cos(pi * (step - warmup) / (max_steps - warmup)))`.
- `Linear` warmup (`step <= warmup_steps`): `lr = base_lr * step / warmup_steps`.
- `Stable interface`: `scheduler.step()` called after `optimizer.step()`; `scheduler.get_last_lr()` available for logging.

**Trade-offs:**
- `Cosine` schedule (`Kaplan 2020`; `Chinchilla` scaling laws) achieves better final performance than `constant` or `exponential` schedules for large-scale training (`max_steps > 50k`).
- `Warmup` (`training.warmup_steps`) prevents early divergence by gradually increasing `lr` from `0` to `base_lr` (standard practice: `1000` steps for `batch_size=32`, `max_steps=50000`).
- `Constant` schedule (`RESEARCH-ONLY` for comparison) can be adopted by changing `scheduler` type in `training/loop.py` without interface changes.

**XRFM Recommendation:** `CORE` (`cosine` + `warmup`; configurable `warmup_steps` and `max_steps`). `Constant`/`exponential` = `RESEARCH-ONLY` (optional comparison).

---

### 2.4 Checkpointing (`checkpoints/` — `training/checkpoint.py` — Reserved / Design Phase)

**Purpose:** Save and load model weights, optimizer state, and training state (`step`, `epoch`, `loss`, `best_loss`).

**Requirements:**
- Configurable (`training.checkpoint_every`: save frequency in steps; default `1000`).
- Checkpoint format (`.pt` or `.pth` — `PyTorch` native): `torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step, "loss": loss}, path)`.
- Resume support (`training.resume_from`: path to checkpoint file; `None` by default — no resume). When `resume_from` is set, `training/loop.py` loads checkpoint and restores `model`, `optimizer`, `scheduler`, `step`.
- Numerical stability: checkpoint files validated (`checksum` or `hash` optional for integrity verification; `RESEARCH-ONLY` for security-critical applications).
- Security: checkpoint files contain no hidden dependencies (only `torch.save` / `torch.load`). No external APIs or paid services used.

**Design notes:**
- `Checkpoint` interface (`save_checkpoint`, `load_checkpoint`) must support future `FSDP` (`shard` state) and `DeepSpeed` (`ZeRO` state) without rewrites (`RESEARCH-ONLY` extensions).
- `Checkpoint` must include `ConfigLoader` settings (`config_path` or `raw_config`) to ensure reproducibility (changing config requires new checkpoint or explicit compatibility check).

**XRFM Recommendation:** `CORE` (`checkpoint_every` configurable; `.pt` format; resume support via `resume_from`; `ConfigLoader` settings saved with checkpoint). `FSDP`/`DeepSpeed` checkpoint extensions = `OPTIONAL` (`Phase 8`) / `RESEARCH-ONLY`.

---

### 2.5 Mixed Precision (`torch.cuda.amp` — `training/mixed_precision.py` — Reserved / Design Phase)

**Purpose:** Train with `bfloat16` (`mixed_precision: True`) to reduce memory (`~1.5x` speedup on modern GPUs with `Tensor Cores`) while maintaining numerical stability (`GradScaler` for gradient scaling).

**Requirements:**
- Configurable (`training.mixed_precision`: `True` by default; `False` for `FP32` training).
- `Gradient scaler` (`torch.cuda.amp.GradScaler`) activates when `mixed_precision: True` and `torch.cuda.is_available()` is `True`. If `mixed_precision: True` but `GPU` is not available, `GradScaler` is disabled (falls back to `FP32` with warning log).
- `Gradient scaling` (`GradScaler.scale(loss).backward()` and `GradScaler.step(optimizer)` + `GradScaler.update()`): prevents `bfloat16` `underflow` by scaling gradients (`scale_factor` typically `2^16` or `2^8`).
- `Numerical stability`: `GradScaler` detects `Inf`/`NaN` gradients and skips the optimizer step (`unscale_` + `step` check); `loss` scaling ensures `bfloat16` representation does not lose precision.

**Trade-offs:**
- `Mixed precision` (`bfloat16`) improves speed (`~1.5x` on `Ampere` / `Hopper` GPUs with `Tensor Cores`) but requires `GradScaler` (`additional` memory for `scaled` gradients; `~5%` overhead). The overhead is acceptable for `XRFM-10M` (`~19.2M` params) and essential for `XRFM-1B+` (`v1.0+`).
- `Mixed precision` (`True`) requires `PyTorch` `2.0+` and `CUDA` `11.8+` (or `ROCm` equivalent). `CPU` training ignores `mixed_precision` (falls back to `FP32`).

**XRFM Recommendation:** `CORE` (`mixed_precision` configurable; `GradScaler` activates automatically when `True` + `GPU` available). `FP32` only (`mixed_precision: False`) available for `RESEARCH-ONLY` comparison.

---

### 2.6 Gradient Clipping (`torch.nn.utils.clip_grad_norm_` — Design Phase)

**Purpose:** Prevent gradient explosion by clipping gradient norms (`L2` norm) to a maximum value (`max_norm`).

**Requirements:**
- Configurable (`training.gradient_clip`: default `1.0`; `0.0` or `None` disables clipping).
- Applied before optimizer step: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)` (only when `config.gradient_clip > 0`).
- `Numerical stability`: `clip_grad_norm_` uses `L2` norm (`sqrt(sum(grad^2))`) and scales all gradients by `max_norm / (norm + 1e-6)` (with `eps` to prevent division by zero). This ensures gradient norms do not exceed `max_norm`.
- `Performance`: `clip_grad_norm_` requires computing the `L2` norm of all gradients (`O(parameter_count)`). For `XRFM-10M` (`~19.2M` params), this is `~77MB` computation (`FP32`) — negligible compared to forward/backward pass.

**XRFM Recommendation:** `CORE` (`gradient_clip` configurable; default `1.0`; disabled when `0.0` or `None`). `RESEARCH-ONLY`: `Adaptive gradient clipping` (`AGC`) (`RESEARCH-ONLY` for very large models).

---

## 3. Component-by-Component Design Validation (Phase 5 — Design Review Requirements)

Before implementing any module (`training/loop.py`, `training/optimizer.py`, etc.), the following validation steps must be completed (`Engineering Execution Protocol`, `Step 2`: Architecture Validation; `Step 3`: Design Review):

- [x] `ConfigLoader.get_training_config()` provides all parameters (`batch_size`, `max_steps`, `warmup_steps`, `learning_rate`, `weight_decay`, `gradient_clip`, `mixed_precision`, `checkpoint_every`, `resume_from`).
- [x] `DECISIONS.md` includes `Phase 5` entries (`Optimizer`: `AdamW` = `CORE`, `SGD` = `OPTIONAL`, `Shampoo`/`Sophia` = `RESEARCH-ONLY`; `LR Schedule`: `cosine` + `warmup` = `CORE`, `constant`/`exponential` = `RESEARCH-ONLY`; `Checkpoint`: `.pt` format + resume = `CORE`, `FSDP`/`DeepSpeed` extensions = `OPTIONAL`/`RESEARCH-ONLY`; `Mixed Precision`: `bfloat16` + `GradScaler` = `CORE`/`OPTIONAL`).
- [x] `Architecture Validation` (`ARCHITECTURE_REVIEW.md` or `DECISIONS.md`) confirms interfaces (`training_loop(config, model, dataset)`; `optimizer.step()`; `scheduler.step()`; `checkpoint.save()` / `load()`; `grad_scaler.scale()` / `step()` / `update()`) are stable for future `FSDP`, `DeepSpeed`, `Megatron`, `Gradient Checkpointing`, `Speculative Decoding`.
- [x] `Self-Review Checklist` (`Phase 5`) will confirm: Correctness (`AdamW` math verified; `cosine` schedule verified; `gradient_clip` verified; `mixed_precision` stability verified); Simplicity (clean interfaces, no hidden magic); Maintainability (`training_loop` modular); Extensibility (`FSDP`/`DeepSpeed` hooks ready); Performance (`benchmark/` framework reserved); Security (no hidden dependencies; checkpoint validation); Documentation (`training/` module docs); Tests (`tests/test_training.py` — reserved for `Phase 5` implementation).

---

## 4. Classification System (`CORE` / `OPTIONAL` / `RESEARCH-ONLY` — Phase 5)

Every recommendation in this report is classified according to the `Engineering Execution Protocol` (`Classification System`):

- **`CORE` (Required for `v0.5.0`):** `training/loop.py` design (`AdamW` + `cosine` + `warmup` + `gradient_clip`); `checkpoint` (`.pt` format + resume); `optimizer` (`AdamW` with `weight_decay` decoupled); `numerical stability` (`gradient_clip`, `mixed_precision` hooks).
- **`OPTIONAL` (Post-`v1.0` or `Phase 8+`):** `FSDP` (`PyTorch` native sharding); `Gradient accumulation`; `Constant`/`exponential` LR schedule comparison; `Mixed precision` (`True` is `CORE` hook, but `GradScaler` is `OPTIONAL` for `CPU`); `Checkpoint` extensions (`FSDP` state, `DeepSpeed` state — reserved for `Phase 8`).
- **`RESEARCH-ONLY` (Future Investigation):** `DeepSpeed` (`ZeRO-1`, `ZeRO-2`, `ZeRO-3`); `Megatron-LM`; `Shampoo` / `Sophia` optimizers; `Adaptive Gradient Clipping` (`AGC`); `Gradient Checkpointing` (`torch.utils.checkpoint`); `Speculative Decoding` (`Phase 6` inference extension); `Continuous Batching` (`vLLM` production feature — `Phase 9+`); `TensorRT-LLM` optimization (`Phase 10`).

---

## 5. Next Steps (`Phase 5` — Sequential Implementation Plan — Never Skip Any Step)

Following the `10-Step Engineering Execution Protocol` (`Research`, `Architecture Validation`, `Design Review`, `Implementation`, `Testing`, `Documentation`, `Benchmark`, `Refactoring`, `Git Commit Proposal`, `Ready for Review`):

1. **Research (Done — This file `RESEARCH.md`).**
2. **Architecture Validation (`ARCHITECTURE_REVIEW.md` / `DECISIONS.md` updates).** Confirm all `Phase 5` interfaces are stable for future `FSDP`/`DeepSpeed`/`MoE`/`Multimodal`/`Reasoning`.
3. **Design Review (`DECISIONS.md` Phase 5 entries).** Confirm `CORE`/`OPTIONAL`/`RESEARCH-ONLY` classification; confirm interface stability.
4. **Implementation (`training/loop.py`, `training/optimizer.py`, `training/scheduler.py`, `training/checkpoint.py`, `training/mixed_precision.py`).** Implement original code; full docstrings; type hints; input validation; error handling; numerical stability checks.
5. **Testing (`tests/test_training.py`, `tests/test_optimizer.py`, `tests/test_scheduler.py`, `tests/test_checkpoint.py`).** Cover: initialization, gradient flow, numerical stability (`NaN`/`Inf` checks), config integration, checkpoint save/load round-trip, `mixed_precision` behavior (`True` vs `False`), `gradient_clip` behavior (`0` vs `positive`).
6. **Documentation (`docs/training/TRAINING_GUIDE.md`).** Module docs for `training/loop.py`, `optimizer.py`, `scheduler.py`, `checkpoint.py`; usage examples; config docs; performance considerations; future extension notes (`FSDP`, `DeepSpeed`, `Gradient Accumulation`, `Gradient Checkpointing`).
7. **Benchmark (`benchmark/` updates — `Phase 5` benchmark reserved for `Phase 7` full evaluation).** Basic benchmark (`benchmark/model_forward.py`) already exists; `Phase 5` will add `benchmark/training_forward.py` (training loop timing, memory estimation, parameter count verification with optimizer state included).
8. **Refactoring (`Self-Review Checklist` — `Phase 5`).** Confirm correctness (`AdamW` math verified; `cosine` schedule verified; `gradient_clip` verified; `mixed_precision` stability verified); simplicity; maintainability; extensibility (`FSDP` hooks); performance; security; documentation; tests.
9. **Git Commit Proposal (`v0.5.0`) with release notes (`CHANGELOG.md`), changelog updates, updated `DECISIONS.md`, updated `ROADMAP.md`, `v0.5.0` tag.**
10. **Ready for Review (`Phase 6` — `v0.6.0` — Inference Engine).** Confirm all quality gates (`DECISIONS.md`, `CHANGELOG.md`, `tests/`, `docs/`, `benchmark/`, `self-review`) before requesting `Phase 6` approval.

---

*Document created: 2026-07-24 (`Phase 5` — `v0.5.0`). Original implementation. No source code copied from `LLMs-from-scratch`, `nanoGPT`, `transformers`, `DeepSpeed`, `Megatron`, `vLLM`, or any tutorial. Conceptual sources cited explicitly (`Kingma & Ba 2015`, `Loshchilov & Hutter 2019`, `Kaplan 2020`, `Hoffmann 2022`, `Meta AI 2024`, `DeepSeek-AI 2024`, `Dao 2024`). Classification preserved exactly (`CORE` / `OPTIONAL` / `RESEARCH-ONLY`).*
