"""QLoRA fine-tuning for XRFM CoderSec (Track B) — GATE-5.

Implements QLoRA training with NF4 + FP16 mixed precision, assistant-only loss
masking (ChatML), max_seq_length=2048, and LoRA adaptation on Kaggle Tesla T4.

Supports:
- Smoke B0 (minimal pipeline validation)
- Ablation B1: English-only vs B2: 7-lang/Roman-Urdu vs B3: rank check
- Flagship Track B QLoRA training

Outputs LoRA adapter + merged checkpoint for HF Hub publication.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch  # type: ignore[import-not-found]
from peft import (  # type: ignore[import-not-found]
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (  # type: ignore[import-not-found]
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from xrfm.codersec.config import CoderSecConfig  # noqa: E402
from xrfm.codersec.dataset import (  # noqa: E402
    CoderSecSFTDataset,
    load_sft_records_from_jsonl,
)


@dataclass
class TrainResult:
    """Training result record."""

    run_name: str
    config: dict[str, Any]
    train_loss: float | None
    eval_loss: float | None
    steps: int
    seconds: float
    peak_vram_mb: float
    lora_adapter_path: str | None
    merged_checkpoint_path: str | None


def build_qlora_config(cfg: CoderSecConfig) -> LoraConfig:
    """Build LoRA configuration from CoderSecConfig."""
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_base_model(
    model_id: str,
    model_sha: str,
    precision: str = "nf4",
    gradient_checkpointing: bool = True,
    hf_token: str | None = None,
):
    """
    Load base model with NF4/FP16 quantization for QLoRA training.

    Returns (model, tokenizer).
    """
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    info = api.model_info(model_id)
    if info.sha != model_sha:
        raise ValueError(f"Model {model_id} SHA mismatch: expected {model_sha}, got {info.sha}")

    if precision == "nf4" and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=model_sha,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=model_sha,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_sha,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def train_qlora(
    cfg: CoderSecConfig,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]] | None = None,
    output_dir: str = "outputs/codersec_qlora",
    hf_token: str | None = None,
    run_name: str | None = None,
) -> TrainResult:
    """
    Run QLoRA fine-tuning.

    Args:
        cfg: CoderSecConfig with training hyperparameters.
        train_records: Training SFT records.
        eval_records: Optional evaluation SFT records.
        output_dir: Output directory for checkpoints.
        hf_token: Hugging Face API token.
        run_name: Name for this training run.

    Returns:
        TrainResult with metrics and output paths.
    """
    if run_name is None:
        run_name = f"codersec_qlora_{int(time.time())}"

    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load model & tokenizer.
    model, tokenizer = load_base_model(
        model_id=cfg.base_model,
        model_sha=cfg.base_model_sha,
        precision=cfg.precision,
        gradient_checkpointing=cfg.gradient_checkpointing,
        hf_token=hf_token,
    )

    # Apply LoRA.
    lora_config = build_qlora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create datasets.
    train_dataset = CoderSecSFTDataset(
        records=train_records,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
    )
    eval_dataset = None
    if eval_records:
        eval_dataset = CoderSecSFTDataset(
            records=eval_records,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_seq_length,
        )

    # Data collator (MLM=False for causal LM).
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments.
    training_args = TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps if eval_dataset else cfg.max_steps + 1,
        logging_steps=10,
        fp16=(cfg.precision == "fp16"),
        bf16=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=False,
        seed=cfg.seed,
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    # Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train.
    train_result = trainer.train()
    train_seconds = time.time() - t0

    eval_metrics = None
    if eval_dataset:
        eval_metrics = trainer.evaluate()

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    # Save LoRA adapter.
    lora_adapter_path = str(output_path / "lora_adapter")
    model.save_pretrained(lora_adapter_path)
    tokenizer.save_pretrained(lora_adapter_path)

    # Merge LoRA into base model.
    merged_path = str(output_path / "merged")
    merge_lora_adapter(
        base_model_id=cfg.base_model,
        base_model_sha=cfg.base_model_sha,
        lora_adapter_path=lora_adapter_path,
        output_path=merged_path,
        hf_token=hf_token,
    )

    return TrainResult(
        run_name=run_name,
        config=asdict(cfg),
        train_loss=train_result.training_loss,
        eval_loss=eval_metrics["eval_loss"] if eval_metrics else None,
        steps=train_result.global_step,
        seconds=round(train_seconds, 2),
        peak_vram_mb=round(peak_vram_mb, 2),
        lora_adapter_path=lora_adapter_path,
        merged_checkpoint_path=merged_path,
    )


def merge_lora_adapter(
    base_model_id: str,
    base_model_sha: str,
    lora_adapter_path: str,
    output_path: str,
    hf_token: str | None = None,
) -> str:
    """
    Merge LoRA adapter weights into base model and save merged checkpoint.

    Returns the output path.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    info = api.model_info(base_model_id)
    if info.sha != base_model_sha:
        raise ValueError(f"Model {base_model_id} SHA mismatch: expected {base_model_sha}, got {info.sha}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        revision=base_model_sha,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        revision=base_model_sha,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load PEFT model and merge.
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    model = model.merge_and_unload()

    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path_obj))
    tokenizer.save_pretrained(str(output_path_obj))

    return str(output_path_obj)


def run_smoke_b0(hf_token: str | None = None) -> TrainResult:
    """Smoke B0: minimal QLoRA run to validate pipeline end-to-end."""
    from xrfm.codersec.config import smoke_b0_config

    cfg = smoke_b0_config()

    # Generate a small synthetic training set.
    records = []
    for i in range(cfg.max_steps * cfg.batch_size):
        records.append(
            {
                "record_id": f"smoke_b0_{i}",
                "source": "synthetic",
                "language": "en",
                "chatml": (
                    "licsystem\nYou are a helpful coding assistant.irit\n"
                    f"licuser\nWrite a function to add {i} and {i + 1}.irit\n"
                    f"licassistant\ndef add_{i}(a, b):\n    return a + b\nirit\n"
                ),
                "domain": "code",
                "license": "MIT",
                "license_tier": "tier_a",
                "code_blocks": [{"language": "python", "code": f"def add_{i}(a, b):\n    return a + b\n"}],
                "provenance": {"source_uri": "local://smoke_b0"},
            }
        )

    return train_qlora(
        cfg=cfg,
        train_records=records,
        output_dir="outputs/codersec_qlora",
        hf_token=hf_token,
        run_name="smoke_b0",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XRFM CoderSec QLoRA Training")
    parser.add_argument("--smoke-b0", action="store_true", help="Run smoke B0 test")
    parser.add_argument("--config", type=str, default=None, help="Config name: b0, b1, b2, b3")
    parser.add_argument("--train-data", type=str, default=None, help="Path to training JSONL")
    parser.add_argument("--eval-data", type=str, default=None, help="Path to eval JSONL")
    parser.add_argument("--output-dir", type=str, default="outputs/codersec_qlora")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")

    if args.smoke_b0:
        result = run_smoke_b0(hf_token=hf_token)
    elif args.config:
        from xrfm.codersec.config import (
            ablation_b1_english_only,
            ablation_b2_multilingual_roman_urdu,
            ablation_b3_rank_check,
        )

        configs = {
            "b0": lambda: CoderSecConfig(),
            "b1": ablation_b1_english_only,
            "b2": ablation_b2_multilingual_roman_urdu,
            "b3": ablation_b3_rank_check,
        }
        cfg = configs[args.config]()

        train_records = []
        if args.train_data:
            train_records = load_sft_records_from_jsonl(args.train_data)
        else:
            # Synthetic records for ablation testing.
            for i in range(cfg.max_steps * cfg.batch_size):
                train_records.append(
                    {
                        "record_id": f"{args.config}_{i}",
                        "source": "synthetic",
                        "language": "en",
                        "chatml": (
                            "licsystem\nYou are a helpful assistant.irit\n"
                            f"licuser\nTask {i}.irit\n"
                            f"licassistant\nResponse {i}\nirit\n"
                        ),
                        "domain": "code",
                        "license": "MIT",
                        "license_tier": "tier_a",
                        "code_blocks": [],
                        "provenance": {"source_uri": "local://ablation"},
                    }
                )

        result = train_qlora(
            cfg=cfg,
            train_records=train_records,
            eval_records=None,
            output_dir=args.output_dir,
            hf_token=hf_token,
            run_name=args.run_name or args.config,
        )
    else:
        parser.print_help()
        sys.exit(1)

    print(json.dumps(asdict(result), indent=2))
