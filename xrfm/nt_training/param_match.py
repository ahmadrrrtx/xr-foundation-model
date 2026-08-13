"""XRFM-NeuroTopo — parameter-count matching utilities (Phase 15).

Matches CONTROL-1 (Transformer), CONTROL-2 (memory-only recurrent), and
XRFM-NT within a tolerance by width/depth search.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

import tempfile
from pathlib import Path

import yaml

from model.gpt import GPTModel
from xrfm.core import NeuroTopoModel
from xrfm.neurotopo.config import MemoryConfig, NeuroTopoConfig, TopologyConfig


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_transformer(vocab: int, d_model: int, n_layers: int, n_heads: int = 4,
                      max_seq_len: int = 256, pad_id: int = 0) -> GPTModel:
    """Build a CONTROL-1 Transformer from a temporary YAML config (the existing
    GPTModel is config-path based)."""
    cfg = {
        "model": {
            "vocab_size": vocab, "d_model": d_model, "n_layers": n_layers,
            "n_heads": n_heads, "d_ff": d_model * 4, "max_seq_len": max_seq_len,
            "dropout": 0.0, "use_rope": True, "use_rmsnorm": True,
            "use_swiglu": True,
        }
    }
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    try:
        return GPTModel(config_path=tmp.name, vocab_size=vocab)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def build_nt(vocab: int, d_model: int, n_layers: int, n_modules: int,
             module_dim: int, memory: bool = True, pad_id: int = 0) -> NeuroTopoModel:
    cfg = NeuroTopoConfig(
        vocab_size=vocab, d_model=d_model, n_layers=n_layers, pad_id=pad_id,
        topology=TopologyConfig(n_modules=n_modules, module_dim=module_dim,
                                local_degree=2, longrange_topk=4, rank=8,
                                connectivity_reg=False),
        memory=MemoryConfig(enabled=memory, d_k=16, d_v=16, persistent_slots=8),
    )
    return NeuroTopoModel(cfg)


def match_nt_to_transformer(vocab: int, target_params: int,
                            tolerance: float = 0.05) -> dict[str, Any]:
    """Search NT widths to match a target parameter count."""
    best = None
    for n_layers in (2, 3, 4):
        for n_modules in (8, 16):
            for module_dim in (16, 24, 32):
                d_model = n_modules * module_dim
                m = build_nt(vocab, d_model, n_layers, n_modules, module_dim)
                p = count_params(m)
                rel = abs(p - target_params) / target_params
                if best is None or rel < best["rel"]:
                    best = {"params": p, "rel": rel, "n_layers": n_layers,
                            "n_modules": n_modules, "module_dim": module_dim,
                            "d_model": d_model}
                if rel <= tolerance:
                    return best
    return best
