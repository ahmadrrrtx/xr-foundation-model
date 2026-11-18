"""LoRA adapter merging for XRFM CoderSec (GATE-5).

Merges trained LoRA adapters into base Qwen models and publishes merged
checkpoints to Hugging Face Hub.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch  # type: ignore[import-not-found]
from peft import PeftModel  # type: ignore[import-not-found]
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from xrfm.codersec.config import CoderSecConfig  # noqa: E402


def merge_and_publish(
    base_model_id: str,
    base_model_sha: str,
    lora_adapter_path: str,
    output_repo: str,
    hf_token: str | None = None,
    commit_message: str | None = None,
) -> str:
    """
    Merge LoRA adapter into base model and publish to HF Hub.

    Args:
        base_model_id: Base model repository id.
        base_model_sha: Pinned commit SHA of base model.
        lora_adapter_path: Path to trained LoRA adapter.
        output_repo: Target HF Hub repository id (e.g. 'ahmadrrrtx/xrfm-codersec-qlora').
        hf_token: Hugging Face API token.
        commit_message: Commit message for the upload.

    Returns:
        HF commit SHA of the published model.
    """
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=hf_token)

    # Verify base model SHA.
    info = api.model_info(base_model_id, token=hf_token)
    if info.sha != base_model_sha:
        raise ValueError(f"Model {base_model_id} SHA mismatch: expected {base_model_sha}, got {info.sha}")

    t0 = time.time()

    # Create target repo if needed.
    try:
        create_repo(
            repo_id=output_repo,
            repo_type="model",
            exist_ok=True,
            private=False,
            token=hf_token,
        )
    except Exception:
        pass  # Repo may already exist.

    # Load base model in FP16.
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

    # Save to temporary directory.
    output_dir = Path("/var/tmp/codersec_merged") / output_repo.split("/")[-1]  # nosec B108 — /var/tmp staging (not /tmp tmpfs)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Upload to HF Hub.
    if commit_message is None:
        commit_message = f"feat(codersec-qlora): publish merged checkpoint ({output_repo})"

    commit_info = api.upload_folder(
        repo_id=output_repo,
        repo_type="model",
        folder_path=str(output_dir),
        path_in_repo=".",
        commit_message=commit_message,
    )

    elapsed = time.time() - t0
    print(f"Published merged checkpoint: {output_repo} @ {commit_info.oid} ({elapsed:.1f}s)")

    return commit_info.oid


def merge_from_config(
    cfg: CoderSecConfig,
    lora_adapter_path: str,
    output_repo: str,
    hf_token: str | None = None,
    commit_message: str | None = None,
) -> str:
    """
    Merge LoRA adapter using values from CoderSecConfig.

    Convenience wrapper around merge_and_publish.
    """
    return merge_and_publish(
        base_model_id=cfg.base_model,
        base_model_sha=cfg.base_model_sha,
        lora_adapter_path=lora_adapter_path,
        output_repo=output_repo,
        hf_token=hf_token,
        commit_message=commit_message,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge LoRA adapter and publish")
    parser.add_argument("--base-model", required=True, help="Base model id")
    parser.add_argument("--base-sha", required=True, help="Base model commit SHA")
    parser.add_argument("--lora-adapter", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output-repo", required=True, help="Target HF Hub repo id")
    parser.add_argument("--commit-message", default=None, help="Commit message")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not set.")
        sys.exit(1)

    sha = merge_and_publish(
        base_model_id=args.base_model,
        base_model_sha=args.base_sha,
        lora_adapter_path=args.lora_adapter,
        output_repo=args.output_repo,
        hf_token=hf_token,
        commit_message=args.commit_message,
    )
    print(f"Merge complete. HF commit: {sha}")
