"""XRFM-NeuroTopo — tiny CPU training loop for Phase 16 smoke comparison."""

from __future__ import annotations

import time
from dataclasses import asdict

import torch
import torch.nn as nn

from xrfm.core import NeuroTopoModel
from xrfm.neurotopo.config import MemoryConfig, NeuroTopoConfig, TopologyConfig


def topo_diagnostics(model: NeuroTopoModel) -> dict:
    """Aggregate the last-step diagnostics from the most recent forward."""
    return {"note": "call record_diagnostics during training for per-step stats"}


def train_nt_tiny(
    vocab_size: int,
    token_ids: torch.Tensor,
    seq_len: int = 64,
    n_layers: int = 2,
    n_modules: int = 8,
    module_dim: int = 16,
    steps: int = 60,
    batch_size: int = 8,
    lr: float = 3e-3,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    cfg = NeuroTopoConfig(
        vocab_size=vocab_size, d_model=n_modules * module_dim, n_layers=n_layers,
        max_seq_len=seq_len, pad_id=0,
        topology=TopologyConfig(n_modules=n_modules, module_dim=module_dim,
                                  local_degree=2, longrange_topk=4, rank=8,
                                  connectivity_reg=False),
        memory=MemoryConfig(enabled=False),
    )
    model = NeuroTopoModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    chunks = token_ids.split(seq_len)
    chunks = [c for c in chunks if c.numel() == seq_len]
    n = len(chunks)
    assert n > batch_size, "not enough data for a batch"

    losses = []
    t0 = time.time()
    model.train()
    for s in range(steps):
        idx = torch.randint(0, n, (batch_size,))
        batch = torch.stack([chunks[i] for i in idx])
        labels = batch.clone()
        opt.zero_grad()
        out = model(batch, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(out.loss.item())
    dt = time.time() - t0
    tokens = steps * batch_size * seq_len
    return {
        "params": sum(p.numel() for p in model.parameters()),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "ppl_last": float(torch.exp(torch.tensor(losses[-1]))),
        "tokens_per_sec": tokens / dt,
        "steps": steps,
    }
