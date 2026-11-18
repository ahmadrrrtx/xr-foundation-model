"""Configuration for XRFM CoderSec (Track B) QLoRA training.

Defines the training preset used for GATE-4 base-model evaluation and
GATE-5 QLoRA fine-tuning: NF4 + FP16 mixed precision, assistant-only loss
masking (ChatML), max_seq_length=2048, and LoRA hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CoderSecConfig:
    """Immutable CoderSec QLoRA training configuration.

    Attributes:
        base_model: Hugging Face model id of the base model to adapt.
        base_model_sha: Pinned commit SHA of the base model (verifiable).
        precision: Mixed-precision dtype ("nf4" or "fp16").
        max_seq_length: Maximum sequence length in tokens (2048 per roadmap).
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_dropout: LoRA dropout probability.
        lora_target_modules: Target module names for LoRA adaptation.
        assistant_only_loss: Only compute loss on assistant turns (ChatML).
        gradient_checkpointing: Enable gradient checkpointing to save VRAM.
        batch_size: Per-device batch size.
        grad_accum_steps: Gradient accumulation steps.
        learning_rate: Peak learning rate.
        warmup_steps: Linear warmup steps.
        max_steps: Total training steps.
        save_steps: Checkpoint save interval.
        eval_steps: Evaluation interval.
        seed: Random seed for reproducibility.
    """

    base_model: str = "Qwen/Qwen3-1.7B"
    base_model_sha: str = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    precision: str = "nf4"
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    assistant_only_loss: bool = True
    gradient_checkpointing: bool = True
    batch_size: int = 1
    grad_accum_steps: int = 16
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    max_steps: int = 600
    save_steps: int = 100
    eval_steps: int = 50
    seed: int = 42

    def __post_init__(self) -> None:
        if self.precision not in ("nf4", "fp16"):
            raise ValueError(f"precision must be 'nf4' or 'fp16', got {self.precision!r}")
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be > 0, got {self.max_seq_length}")
        if self.lora_r <= 0:
            raise ValueError(f"lora_r must be > 0, got {self.lora_r}")
        if not (0.0 <= self.lora_dropout <= 1.0):
            raise ValueError(f"lora_dropout must be in [0, 1], got {self.lora_dropout}")
        if not (0.0 < self.learning_rate):
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {self.max_steps}")
        if not isinstance(self.base_model_sha, str) or len(self.base_model_sha) != 40:
            raise ValueError(f"base_model_sha must be a 40-char git SHA, got {self.base_model_sha!r}")


@dataclass(frozen=True)
class AblationConfig:
    """Controlled ablation configuration (GATE-5)."""

    name: str
    description: str
    config_overrides: dict[str, object] = field(default_factory=dict)


def smoke_b0_config() -> CoderSecConfig:
    """Smoke B0 config: minimal run to validate the pipeline end-to-end."""
    return CoderSecConfig(
        base_model="Qwen/Qwen3-1.7B",
        base_model_sha="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        precision="fp16",
        max_seq_length=512,
        lora_r=8,
        lora_alpha=16,
        max_steps=20,
        save_steps=10,
        eval_steps=10,
        seed=42,
    )


def ablation_b1_english_only() -> CoderSecConfig:
    """B1 ablation: English-only training data (baseline for multilingual ablation)."""
    return CoderSecConfig(
        base_model="Qwen/Qwen3-1.7B",
        base_model_sha="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        precision="fp16",
        max_seq_length=2048,
        max_steps=200,
        seed=42,
    )


def ablation_b2_multilingual_roman_urdu() -> CoderSecConfig:
    """B2 ablation: 7-lang + Roman-Urdu training data (flagship mixture)."""
    return CoderSecConfig(
        base_model="Qwen/Qwen3-1.7B",
        base_model_sha="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        precision="fp16",
        max_seq_length=2048,
        max_steps=600,
        seed=42,
    )


def ablation_b3_rank_check() -> CoderSecConfig:
    """B3 ablation: LoRA rank sweep (r=8 vs r=16 vs r=32)."""
    return CoderSecConfig(
        base_model="Qwen/Qwen3-1.7B",
        base_model_sha="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        precision="fp16",
        max_seq_length=2048,
        lora_r=32,
        lora_alpha=64,
        max_steps=200,
        seed=42,
    )
