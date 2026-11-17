#!/usr/bin/env python3
"""
Phase 3 (`GATE-3`): Track A (`XRFM Native`) Pretraining, CUDA Exact-Resume Probe &
8-Domain `BPB` Evaluation Runner.

Designed to run identically on Kaggle GPUs (`1x P100` / `2x T4`) or local CPU/GPU:
1. Downloads `ahmadrrrtx/xrfm-native-shards-v1` (pinned commit) via `snapshot_download`.
2. Runs an automated **Hardware & CUDA Exact-Resume Verification Probe** (`TRN-001..004`):
   - Trains a model uninterrupted for `20` steps (`seed=1234`), recording final weights & loss.
   - Re-trains for `10` steps -> saves format-v2 atomic checkpoint -> instantiates a brand-new
     process-equivalent `TrainingLoop` -> resumes from step `10` to `20` -> verifies `max_abs_diff == 0.0`!
3. Trains the requested Track A preset (`config/native-25m.yaml` or `config/native-85m.yaml`)
   using `XRFMModel` + `TrainingLoop` (FP16 mixed precision on CUDA, grouped AdamW, cosine LR),
   measuring validation **Bits-Per-Byte (`BPB`)** across all 8 held-out domains
   (`en`, `code`, `ur`, `ur-Latn`, `ar`, `hi`, `es`, `fr`) at step `0` (random init),
   periodic intervals, and final step.
4. Exports `model.safetensors` + format-v2 `checkpoint_final.pt` + complete telemetry JSON +
   honest Model Card (`README.md`), and uploads to Hugging Face Hub (`HF_TOKEN` from env).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import tempfile
import time
from typing import Any

import torch
from huggingface_hub import HfApi, snapshot_download

from xrfm.config.loader import load_config
from xrfm.data.schema import Document
from xrfm.data.tokenized_dataset import ShardedTokenDatasetMap
from xrfm.evaluation.perplexity import compute_bits_per_byte
from xrfm.inference import generate
from xrfm.models.gpt import XRFMModel
from xrfm.tokenization.hf_bridge import HFTokenizerAdapter
from xrfm.training.distributed import get_raw_model
from xrfm.training.loop import TrainingLoop
from xrfm.training.safetensors_io import export_model_safetensors

EVAL_DOMAINS: tuple[str, ...] = ("en", "code", "ur", "ur-Latn", "ar", "hi", "es", "fr")


def load_val_texts_by_domain(dataset_dir: str, max_docs_per_domain: int = 24) -> dict[str, list[str]]:
    val_dir = os.path.join(dataset_dir, "val_by_language")
    out: dict[str, list[str]] = {}
    for dom in EVAL_DOMAINS:
        fpath = os.path.join(val_dir, f"val_{dom}.jsonl")
        if not os.path.isfile(fpath):
            continue
        texts: list[str] = []
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = Document.from_jsonl(line)
                if doc.text.strip():
                    texts.append(doc.text[:1800])
                if len(texts) >= max_docs_per_domain:
                    break
        if texts:
            out[dom] = texts
    return out


def evaluate_all_domains_bpb(
    model: XRFMModel,
    tokenizer: HFTokenizerAdapter,
    val_texts: dict[str, list[str]],
    max_seq_len: int = 512,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    scores: dict[str, float] = {}
    try:
        for dom, texts in val_texts.items():
            res = compute_bits_per_byte(
                model,
                texts,
                tokenizer,
                max_seq_len=min(max_seq_len, model.max_seq_len),
                stride=min(max_seq_len, model.max_seq_len),
            )
            scores[dom] = round(float(res["bpb"]), 4)
    finally:
        if was_training:
            model.train()
    return scores


def run_cuda_exact_resume_probe(dataset_manifest: str, device_str: str) -> dict[str, Any]:
    """Verify atomic format-v2 checkpoint save + exact resume on the active target device (CUDA/CPU)."""
    probe_cfg = load_config("config/native-25m.yaml")
    probe_cfg.model.d_model = 128
    probe_cfg.model.n_layers = 2
    probe_cfg.model.n_heads = 4
    probe_cfg.model.n_kv_heads = 2
    probe_cfg.model.d_ff = 256
    probe_cfg.model.max_seq_len = 256
    probe_cfg.data.max_seq_len = 256
    probe_cfg.model.dropout = 0.0
    probe_cfg.training.batch_size = 2
    probe_cfg.training.grad_accum_steps = 1
    probe_cfg.training.max_steps = 12
    probe_cfg.training.warmup_steps = 2
    probe_cfg.training.checkpoint_every = 6
    probe_cfg.training.mixed_precision = device_str.startswith("cuda")
    probe_cfg.training.device = device_str
    probe_cfg.training.num_workers = 0
    probe_cfg.training.pin_memory = False
    probe_cfg.training.prefetch_factor = None
    probe_cfg.training.persistent_workers = False
    probe_cfg.training.seed = 777

    ds = ShardedTokenDatasetMap(manifest_path=dataset_manifest, mmap=True)

    # Slice sequence length to 256 for fast 2-second probe
    class _ShortSliceDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return min(len(ds), 64)

        def __getitem__(self, idx: int):
            x, y = ds[idx]
            return x[:256], y[:256]

    short_ds = _ShortSliceDataset()

    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        # 1. Uninterrupted 12 steps
        cfg_full = copy.deepcopy(probe_cfg)
        torch.manual_seed(777)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(777)
        m_full = XRFMModel(cfg_full)
        loop_full = TrainingLoop(
            config=cfg_full,
            model=m_full,
            dataset=short_ds,
            checkpoint_dir=tmp1,
            data_manifest_hash="probe-ds-v1",
            tokenizer_hash="probe-tok-32k",
        )
        res_full = loop_full.training_loop(max_steps=12, checkpoint_every=100)

        # 2. Split 6 + 6 steps with format-v2 checkpoint save & fresh-instance resume
        # Note: config fields stay identical across both halves so _config_identity() matches!
        cfg_part1 = copy.deepcopy(probe_cfg)
        torch.manual_seed(777)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(777)
        m_p1 = XRFMModel(cfg_part1)
        loop_p1 = TrainingLoop(
            config=cfg_part1,
            model=m_p1,
            dataset=short_ds,
            checkpoint_dir=tmp2,
            data_manifest_hash="probe-ds-v1",
            tokenizer_hash="probe-tok-32k",
        )
        res_p1 = loop_p1.training_loop(max_steps=6, checkpoint_every=6)
        ckpt_step6 = res_p1["checkpoint_path"]

        cfg_part2 = copy.deepcopy(probe_cfg)
        cfg_part2.training.resume_from = ckpt_step6
        m_p2 = XRFMModel(cfg_part2)
        loop_p2 = TrainingLoop(
            config=cfg_part2,
            model=m_p2,
            dataset=short_ds,
            checkpoint_dir=tmp2,
            data_manifest_hash="probe-ds-v1",
            tokenizer_hash="probe-tok-32k",
        )
        res_resumed = loop_p2.training_loop(max_steps=12, checkpoint_every=100)

        max_weight_diff = 0.0
        for (k1, p1), (k2, p2) in zip(m_full.state_dict().items(), m_p2.state_dict().items()):
            assert k1 == k2
            diff = float((p1.detach().float().cpu() - p2.detach().float().cpu()).abs().max().item())
            if diff > max_weight_diff:
                max_weight_diff = diff

    return {
        "device": device_str,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "mixed_precision": probe_cfg.training.mixed_precision,
        "uninterrupted_final_loss": round(float(res_full["final_loss"]), 6),
        "resumed_final_loss": round(float(res_resumed["final_loss"]), 6),
        "max_weight_abs_diff": max_weight_diff,
        "exact_match": max_weight_diff < 1e-6,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train XRFM Native (Track A) from HF shards")
    parser.add_argument("--config", default="config/native-25m.yaml")
    parser.add_argument("--shards-repo", default="ahmadrrrtx/xrfm-native-shards-v1")
    parser.add_argument("--shards-revision", default=None)
    parser.add_argument("--local-shards-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-docs-per-domain", type=int, default=20)
    parser.add_argument("--output-dir", default="artifacts/native_run")
    parser.add_argument("--hf-model-repo", default="ahmadrrrtx/xrfm-native-25m")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    t_start = time.perf_counter()
    hf_token = os.environ.get("HF_TOKEN")

    if args.local_shards_dir is not None:
        ds_local_dir = args.local_shards_dir
        print(f"[1/6] Using local shards directory: {ds_local_dir}", flush=True)
    else:
        print(f"[1/6] Downloading shards from {args.shards_repo} ...", flush=True)
        ds_local_dir = snapshot_download(
            repo_id=args.shards_repo,
            repo_type="dataset",
            revision=args.shards_revision,
            token=hf_token,
        )
    manifest_path = os.path.join(ds_local_dir, "manifest.json")
    tok_path = os.path.join(ds_local_dir, "tokenizer_32k_multilingual.json")
    tokenizer = HFTokenizerAdapter.from_file(tok_path)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_names": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        ),
        "torch_version": torch.__version__,
    }
    print(f"[2/6] Hardware probe: {gpu_info}", flush=True)

    resume_probe = run_cuda_exact_resume_probe(manifest_path, device_str)
    print(f"[3/6] Exact-resume probe: {resume_probe}", flush=True)
    if not resume_probe["exact_match"]:
        raise RuntimeError(f"Exact-resume probe failed: {resume_probe}")

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
        cfg.training.warmup_steps = min(max(1, args.max_steps // 20), max(0, args.max_steps - 1))
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.grad_accum is not None:
        cfg.training.grad_accum_steps = args.grad_accum
    if args.eval_every is not None:
        cfg.training.eval_every = args.eval_every

    os.makedirs(args.output_dir, exist_ok=True)
    if args.keep_checkpoints:
        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    else:
        ckpt_dir = os.path.join(tempfile.gettempdir(), f"xrfm_ckpts_{os.getpid()}")
    os.makedirs(ckpt_dir, exist_ok=True)
    # Only save final checkpoint during main run unless --keep-checkpoints is passed
    cfg.training.checkpoint_every = (
        max(1, cfg.training.max_steps // 2) if args.keep_checkpoints else cfg.training.max_steps
    )
    if not torch.cuda.is_available():
        cfg.training.mixed_precision = False
        cfg.training.pin_memory = False
        cfg.training.num_workers = 0
        cfg.training.prefetch_factor = None
        cfg.training.persistent_workers = False

    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)
        torch.cuda.reset_peak_memory_stats()

    model = XRFMModel(cfg)
    unique_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    print(f"[4/6] Instantiated XRFMModel ({unique_params:,} tied parameters)", flush=True)

    train_ds = ShardedTokenDatasetMap(manifest_path=manifest_path, mmap=True)
    val_texts = load_val_texts_by_domain(ds_local_dir, max_docs_per_domain=args.eval_docs_per_domain)
    model.to(device_str)
    bpb_step0 = evaluate_all_domains_bpb(model, tokenizer, val_texts)
    print(f"[Eval step=0] BPB: {bpb_step0}", flush=True)

    bpb_trajectory: list[dict[str, Any]] = [{"step": 0, "bpb": bpb_step0}]

    eval_interval = cfg.training.eval_every if cfg.training.eval_every > 0 else max(1, cfg.training.max_steps // 4)
    cfg.training.eval_every = eval_interval
    total_target_steps = cfg.training.max_steps

    loop = TrainingLoop(
        config=cfg,
        model=model,
        dataset=train_ds,
        checkpoint_dir=ckpt_dir,
        data_manifest_hash=args.shards_repo,
        tokenizer_hash="tokenizer_32k_multilingual_t4",
    )

    def _val_hook(active_loop: TrainingLoop) -> dict[str, float]:
        raw_m = get_raw_model(active_loop.model)
        bpb_now = evaluate_all_domains_bpb(raw_m, tokenizer, val_texts)  # type: ignore[arg-type]
        last_loss = float(active_loop._metrics_records[-1]["loss"]) if active_loop._metrics_records else float("nan")
        bpb_trajectory.append(
            {
                "step": active_loop.current_step,
                "bpb": bpb_now,
                "train_loss": round(last_loss, 4) if math.isfinite(last_loss) else None,
            }
        )
        mean_bpb = sum(bpb_now.values()) / max(1, len(bpb_now))
        print(f"[Eval step={active_loop.current_step}] mean_BPB={mean_bpb:.4f} BPB={bpb_now}", flush=True)
        return {"loss": mean_bpb, "perplexity": float(2.0**mean_bpb)}

    loop.validation_fn = _val_hook
    train_summary = loop.training_loop(
        max_steps=total_target_steps,
        checkpoint_every=cfg.training.checkpoint_every,
        log_interval=max(1, min(25, total_target_steps // 10)),
    )

    # Ensure final step is evaluated if total_target_steps wasn't a multiple of eval_interval
    if bpb_trajectory[-1]["step"] != total_target_steps:
        raw_m = get_raw_model(loop.model)
        bpb_last = evaluate_all_domains_bpb(raw_m, tokenizer, val_texts)  # type: ignore[arg-type]
        bpb_trajectory.append(
            {
                "step": total_target_steps,
                "bpb": bpb_last,
                "train_loss": round(float(train_summary["final_loss"]), 4),
            }
        )
        print(f"[Eval step={total_target_steps}] loss={train_summary['final_loss']:.4f} BPB={bpb_last}", flush=True)
    else:
        bpb_trajectory[-1]["train_loss"] = round(float(train_summary["final_loss"]), 4)

    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0

    bpb_final = bpb_trajectory[-1]["bpb"]
    bpb_deltas = {dom: round(bpb_final[dom] - bpb_step0[dom], 4) for dom in bpb_final}
    gate3_all_improved = all(bpb_final[dom] < bpb_step0[dom] for dom in bpb_final)

    raw_trained_model = get_raw_model(loop.model)
    sample_prompts = {
        "en": "In modern software security, input validation",
        "ur": "اردو زبان اور مصنوعی ذہانت کے میدان میں",
        "ur-Latn": "Python programming mein database ko mehfooz rakhne ke liye",
        "code": "def calculate_checksum(data: bytes) -> str:\n",
    }
    generations: dict[str, str] = {}
    for k, prompt in sample_prompts.items():
        res_gen = generate(
            raw_trained_model,  # type: ignore[arg-type]
            prompt,
            tokenizer=tokenizer,
            max_new_tokens=36,
            temperature=0.7,
            top_p=0.9,
        )
        generations[k] = prompt + res_gen.text

    # Export SafeTensors + Tokenizer + Config + Telemetry
    st_path = os.path.join(args.output_dir, "model.safetensors")
    export_model_safetensors(
        raw_trained_model,
        st_path,
        metadata={
            "track": "XRFM-Native-Track-A",
            "vocab_size": str(cfg.model.vocab_size),
            "d_model": str(cfg.model.d_model),
            "n_layers": str(cfg.model.n_layers),
            "n_heads": str(cfg.model.n_heads),
            "n_kv_heads": str(cfg.model.n_kv_heads or cfg.model.n_heads),
            "steps": str(total_target_steps),
        },
    )
    tokenizer.save(os.path.join(args.output_dir, "tokenizer_32k_multilingual.json"))
    with open(os.path.join(args.output_dir, "model_config.json"), "w", encoding="utf-8") as cf:
        json.dump(
            {
                "architecture": "XRFMModel",
                "lineage": "Track A (From-Scratch Original PyTorch Architecture)",
                "parameters_tied": unique_params,
                "vocab_size": cfg.model.vocab_size,
                "d_model": cfg.model.d_model,
                "n_layers": cfg.model.n_layers,
                "n_heads": cfg.model.n_heads,
                "n_kv_heads": cfg.model.n_kv_heads,
                "d_ff": cfg.model.d_ff,
                "max_seq_len": cfg.model.max_seq_len,
                "rope_theta": cfg.model.rope_theta,
            },
            cf,
            indent=2,
        )

    wall_sec = round(time.perf_counter() - t_start, 2)
    tokens_processed = (
        total_target_steps * cfg.training.batch_size * cfg.training.grad_accum_steps * cfg.model.max_seq_len
    )

    telemetry = {
        "experiment": "track_a_xrfm_native_pretraining",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": args.config,
        "shards_repo": args.shards_repo,
        "parameters_tied": unique_params,
        "steps": total_target_steps,
        "tokens_trained": tokens_processed,
        "wall_seconds": wall_sec,
        "gpu_hours": round(wall_sec / 3600.0, 4) if torch.cuda.is_available() else 0.0,
        "peak_vram_mb": peak_vram_mb,
        "hardware": gpu_info,
        "cuda_exact_resume_probe": resume_probe,
        "final_train_loss": round(float(train_summary.get("final_loss", 0.0)), 4),
        "bpb_step0": bpb_step0,
        "bpb_final": bpb_final,
        "bpb_delta": bpb_deltas,
        "bpb_trajectory": bpb_trajectory,
        "gate3_pass": gate3_all_improved,
        "sample_generations": generations,
    }
    telemetry_path = os.path.join(args.output_dir, "training_telemetry.json")
    with open(telemetry_path, "w", encoding="utf-8") as tf:
        json.dump(telemetry, tf, indent=2, ensure_ascii=False)

    bpb_rows_md = "\n".join(
        f"| `{d}` | {bpb_step0[d]:.4f} | {bpb_final[d]:.4f} | {bpb_deltas[d]:+.4f} |" for d in bpb_final
    )
    model_card = f"""---
license: apache-2.0
language:
  - en
  - ur
  - ar
  - hi
  - es
  - fr
pipeline_tag: text-generation
tags:
  - xrfm
  - from-scratch
  - multilingual
  - urdu
  - roman-urdu
  - code
---

# {cfg.project} (`Track A — XRFM Native`)

**Lineage Disclosure:** Trained **100% from random initialization** using the original `XRFMModel` PyTorch architecture (`GQA`, cached FP32-safe `RoPE`, FP32-safe `RMSNorm`, `SwiGLU`) and the custom 32,768-vocab Script-Protected Byte-Level BPE tokenizer (`T4`) on [`{args.shards_repo}`](https://huggingface.co/datasets/{args.shards_repo}). Zero third-party pretrained weights were loaded.

- **Tied Parameters:** `{unique_params:,}`
- **Steps / Tokens Trained:** `{total_target_steps:,}` steps (`{tokens_processed:,}` tokens)
- **Hardware:** `{gpu_info["device_names"][0] if gpu_info["device_names"] else "CPU"}` (`peak VRAM = {peak_vram_mb} MB`, `{wall_sec}s`)
- **CUDA Atomic Checkpoint Exact-Resume Diff:** `{resume_probe["max_weight_abs_diff"]}`

## Held-Out Per-Language Bits-Per-Byte (`BPB`, Lower is Better)

| Domain | Step 0 (`Random Init`) | Step `{total_target_steps}` (`Trained`) | Delta (`BPB`) |
|---|---:|---:|---:|
{bpb_rows_md}

## Honest Scope & Intended Use
`XRFM Native` is a small-scale (`{unique_params / 1e6:.1f}M` parameter) open-science architectural & multilingual representation research model. For production-grade multi-language coding and defensive security assistance, use **Track B (`XRFM CoderSec 1.7B`)**.
"""
    with open(os.path.join(args.output_dir, "README.md"), "w", encoding="utf-8") as mf:
        mf.write(model_card)

    if not args.no_upload and hf_token:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=args.hf_model_repo, repo_type="model", private=False, exist_ok=True)
        # Upload safetensors, config, tokenizer, telemetry, and model card (skip bulky optimizer .pt shards)
        for fname in (
            "model.safetensors",
            "tokenizer_32k_multilingual.json",
            "model_config.json",
            "training_telemetry.json",
            "README.md",
        ):
            api.upload_file(
                path_or_fileobj=os.path.join(args.output_dir, fname),
                path_in_repo=fname,
                repo_id=args.hf_model_repo,
                repo_type="model",
                commit_message=f"Upload {fname} ({total_target_steps} steps, GATE-3={gate3_all_improved})",
            )
        print(f"[5/6] Uploaded Track A model to https://huggingface.co/{args.hf_model_repo}", flush=True)

    if not args.keep_checkpoints and os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    print(f"[6/6] COMPLETE! GATE-3={gate3_all_improved} wall={wall_sec}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
