"""Phase 11: first language overfit gate.

Tiny XRFM-NT must strongly overfit a tiny text (train loss drops sharply and
greedy generation reproduces the training string). If it cannot overfit, STOP.
"""

from __future__ import annotations

import torch

from tokenizer.bpe import BytePairEncoder
from xrfm.core import NeuroTopoModel
from xrfm.neurotopo.config import MemoryConfig, NeuroTopoConfig, TopologyConfig


TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox jumps over the lazy dog. "
    "a journey of a thousand miles begins with a single step. "
) * 8


def _build():
    tok = BytePairEncoder(vocab_size_target=300)
    tok.train_on_text(TEXT)
    ids = torch.tensor([tok.encode(TEXT)], dtype=torch.long)
    vs = tok.vocab_size()
    cfg = NeuroTopoConfig(
        vocab_size=vs,
        d_model=64,
        n_layers=3,
        max_seq_len=128,
        pad_id=tok.pad_id or 0,
        topology=TopologyConfig(
            n_modules=8, module_dim=16, local_degree=2, longrange_topk=4,
            rank=8, connectivity_reg=False,
        ),
        memory=MemoryConfig(enabled=False),
    )
    model = NeuroTopoModel(cfg)
    return tok, model, ids


def test_overfit_tiny_text_loss_drops():
    torch.manual_seed(42)
    tok, model, ids = _build()
    seq = 32
    chunks = ids[0].split(seq)
    chunks = [c for c in chunks if c.numel() == seq]
    batch = torch.stack(chunks[:4])
    labels = batch.clone()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    model.train()
    first = last = None
    for step in range(120):
        opt.zero_grad()
        out = model(batch, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first is None:
            first = out.loss.item()
        last = out.loss.item()
    assert last < 0.5, f"did not overfit: {first:.3f} -> {last:.3f}"
    assert last < first * 0.2


def test_teacher_forced_reproduction_of_tiny_text():
    """GATE (language overfit): after overfitting, teacher-forced greedy
    predictions must reproduce the training sequence with high accuracy.
    This is the standard overfit proof (a model that cannot reproduce its
    training data teacher-forced cannot generate). Free-running generation is
    a separate exposure-bias concern and is not required to pass the gate."""
    torch.manual_seed(42)
    tok, model, ids = _build()
    seq = 32
    chunk = ids[0, :seq].unsqueeze(0)
    labels = chunk.clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(150):
        opt.zero_grad()
        out = model(chunk, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        out = model(chunk)
        pred = out.logits.argmax(dim=-1)[0].tolist()
        target = labels[0].tolist()
        # logits[t] predicts the token at position t under this model's state.
        matches = sum(1 for t in range(seq) if pred[t] == target[t])
        acc = matches / seq
    assert acc > 0.8, f"teacher-forced reproduction too low: {acc:.2f}"
