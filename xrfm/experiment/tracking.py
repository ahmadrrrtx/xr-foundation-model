"""
XRFM experiment tracking (Phase 40).

Every training run captures a complete, machine-readable record so no
experiment depends on undocumented terminal state. Written at run start and
updated at run end; stored as JSONL/JSON next to the metrics log.

Fields captured (mission Phase 40 list):
git commit, model config, tokenizer version, dataset version, seed, optimizer,
learning rate, scheduler, batch size, gradient accumulation, sequence length,
precision, GPU, steps, tokens, training loss, validation loss, perplexity,
tokens/sec, peak VRAM, checkpoint path.
"""

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any


def git_commit(path: str | None = None) -> str:
    """Return the short HEAD commit of the repository (or 'unknown')."""
    try:
        cwd = path or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


@dataclass
class ExperimentRecord:
    """Complete record of one training experiment."""

    # identity
    run_id: str
    git_commit: str = ""
    # config
    model_config: dict[str, Any] = field(default_factory=dict)
    tokenizer_version: str = ""
    tokenizer_vocab_size: int = 0
    dataset_name: str = ""
    dataset_version: str = ""
    dataset_manifest: str = ""  # path to Phase-33 manifest
    seed: int = 0
    optimizer: str = "adamw"
    learning_rate: float = 0.0
    scheduler: str = "cosine+warmup"
    batch_size: int = 0
    grad_accum_steps: int = 1
    sequence_length: int = 0
    precision: str = "fp32"  # fp32 | bf16 | fp16
    device: str = ""
    gpu_name: str = ""
    # run
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    steps: int = 0
    tokens: int = 0
    # results
    training_loss: float = float("nan")
    validation_loss: float = float("nan")
    perplexity: float = float("nan")
    tokens_per_sec: float = 0.0
    peak_vram_mb: float = 0.0
    checkpoint_path: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
