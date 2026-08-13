"""Phase 16: tiny real-corpus training smoke test."""

from __future__ import annotations

import torch

from tokenizer.bpe import BytePairEncoder
from xrfm.nt_training.train import train_nt_tiny


def test_nt_trains_on_real_corpus():
    """GATE: XRFM-NT must produce meaningful learning (loss drops, finite PPL,
    positive throughput) on the existing small corpus."""
    with open("data/datasets/corpus.txt") as f:
        text = f.read()
    tok = BytePairEncoder(vocab_size_target=1024)
    tok.train_on_text(text[:300000])
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    assert ids.numel() > 1000

    res = train_nt_tiny(
        tok.vocab_size(), ids, seq_len=48, steps=40, batch_size=8, lr=3e-3
    )
    assert res["loss_last"] < res["loss_first"] * 0.8, res
    assert torch.isfinite(torch.tensor(res["ppl_last"]))
    assert res["tokens_per_sec"] > 0
    # Loss should be well below random (ln(vocab) ~ 6.9) after 40 steps.
    assert res["loss_last"] < 6.0, res
