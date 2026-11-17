"""
SafeTensors weight export & import for XRFM Native (`XRFMModel`).

Handles tied embedding/LM-head tensors (`embedding.token_embedding.weight` and
`lm_head.weight` sharing storage) cleanly by deduplicating shared tensor pointers
during `safetensors.torch.save_file` and re-tying upon `load_safetensors_weights`.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from xrfm.training.distributed import get_raw_model


def export_model_safetensors(
    model: torch.nn.Module,
    path: str,
    metadata: dict[str, str] | None = None,
) -> str:
    """Export an `XRFMModel` (or DDP-wrapped module) to `.safetensors`."""
    from safetensors.torch import save_file

    raw_model = get_raw_model(model)
    state = raw_model.state_dict()

    seen_ptrs: dict[int, str] = {}
    clean_tensors: dict[str, torch.Tensor] = {}
    tied_aliases: list[str] = []

    for name, tensor in state.items():
        ptr = tensor.data_ptr()
        if ptr in seen_ptrs:
            tied_aliases.append(f"{name}->{seen_ptrs[ptr]}")
            continue
        seen_ptrs[ptr] = name
        clean_tensors[name] = tensor.detach().cpu().contiguous()

    meta = {"format": "pt", "framework": "xrfm-native-v2"}
    if tied_aliases:
        meta["tied_weights"] = ",".join(tied_aliases)
    if metadata:
        for k, v in metadata.items():
            meta[str(k)] = str(v)

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    save_file(clean_tensors, path, metadata=meta)
    return path


def load_model_safetensors(
    model: torch.nn.Module,
    path: str,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load `.safetensors` weights into `XRFMModel`, restoring tied weights cleanly."""
    from safetensors import safe_open

    if not os.path.isfile(path):
        raise FileNotFoundError(f"SafeTensors file not found: {path}")

    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}
    dev_str = str(device)
    with safe_open(path, framework="pt", device=dev_str) as f:
        metadata = dict(f.metadata() or {})
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    # Restore tied aliases recorded in metadata (e.g. lm_head.weight -> embedding.token_embedding.weight)
    tied_spec = metadata.get("tied_weights", "")
    if tied_spec:
        for pair in tied_spec.split(","):
            if "->" in pair:
                alias_key, src_key = pair.split("->", 1)
                if alias_key not in tensors and src_key in tensors:
                    tensors[alias_key] = tensors[src_key]

    raw_model = get_raw_model(model)
    raw_model.load_state_dict(tensors, strict=strict)

    # Ensure weight tying remains intact on XRFMModel (`embedding.embedding.weight` <-> `lm_head.weight`)
    if hasattr(raw_model, "lm_head") and hasattr(raw_model, "embedding"):
        emb_mod = getattr(raw_model, "embedding", None)
        tok_emb = getattr(emb_mod, "embedding", None) or getattr(emb_mod, "token_embedding", None)
        if tok_emb is not None and hasattr(raw_model.lm_head, "weight"):
            raw_model.lm_head.weight = tok_emb.weight

    return metadata
