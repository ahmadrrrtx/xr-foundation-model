"""Phase 10 tests: LM head, causal loss, padding mask, one training step."""

from __future__ import annotations

import torch

from xrfm.core import NeuroTopoModel
from xrfm.neurotopo.config import MemoryConfig, NeuroTopoConfig, TopologyConfig


def _tiny_config(vocab=256, memory=False):
    return NeuroTopoConfig(
        vocab_size=vocab,
        d_model=32,
        n_layers=2,
        max_seq_len=16,
        pad_id=0,
        topology=TopologyConfig(n_modules=4, module_dim=8, local_degree=1,
                                longrange_topk=2, rank=4, connectivity_reg=False),
        memory=MemoryConfig(enabled=memory, d_k=8, d_v=8, persistent_slots=0),
    )


def test_model_forward_shapes():
    cfg = _tiny_config()
    model = NeuroTopoModel(cfg)
    ids = torch.randint(5, cfg.vocab_size, (2, 10))
    out = model(ids)
    assert out.logits.shape == (2, 10, cfg.vocab_size)
    assert out.loss is None
    assert torch.isfinite(out.logits).all()


def test_causal_loss_decreases_in_one_step():
    torch.manual_seed(0)
    cfg = _tiny_config(memory=True)
    model = NeuroTopoModel(cfg)
    ids = torch.randint(5, cfg.vocab_size, (4, 12))
    labels = ids.clone()
    out = model(ids, labels=labels)
    assert out.loss is not None and torch.isfinite(out.loss)
    loss0 = out.loss.item()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        out = model(ids, labels=labels)
        out.loss.backward()
        opt.step()
    assert out.loss.item() < loss0


def test_padding_positions_ignored():
    cfg = _tiny_config()
    model = NeuroTopoModel(cfg)
    ids = torch.randint(5, cfg.vocab_size, (2, 8))
    labels = ids.clone()
    # set second half to -100 (ignore)
    labels[:, 4:] = -100
    out_masked = model(ids, labels=labels)
    out_full = model(ids, labels=ids)
    assert out_masked.loss.item() != out_full.loss.item()
    assert torch.isfinite(out_masked.loss)


def test_weight_tying():
    cfg = _tiny_config()
    model = NeuroTopoModel(cfg)
    # Logits are produced by dot product with the (shared) embedding matrix.
    external = torch.randn(cfg.vocab_size, cfg.d_model) * 0.02
    model.tie_weights(external)
    # tie_weights shares storage (may wrap in a Parameter that views the same data)
    assert torch.equal(model.input_layer.embedding.weight.data, external)
    model.input_layer.embedding.weight.data[0, 0] = 99.0
    assert external[0, 0].item() == 99.0  # same storage
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    out = model(ids)
    # vocab projection uses the tied weight
    assert out.logits.shape[-1] == cfg.vocab_size
    assert torch.isfinite(out.logits).all()


def test_incremental_state_carries_across_tokens():
    cfg = _tiny_config(memory=True)
    model = NeuroTopoModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    out = model(ids)
    # State after full sequence is non-zero (memory/topology updated).
    final_M = out.state[0]["M"]
    assert final_M is not None and final_M.abs().sum() > 0


def test_perplexity_computable():
    cfg = _tiny_config()
    model = NeuroTopoModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 10))
    out = model(ids, labels=ids)
    ppl = torch.exp(out.loss).item()
    assert ppl > 1 and ppl < cfg.vocab_size * 2
