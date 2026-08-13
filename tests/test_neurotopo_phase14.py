"""Phase 14: complete XRFM-NT model end-to-end (embedding -> blocks -> LM +
uncertainty head)."""

from __future__ import annotations

import torch

from tokenizer.bpe import BytePairEncoder
from xrfm.core import NeuroTopoModel
from xrfm.neurotopo.builder import build_neurotopo_model
from xrfm.neurotopo.config import MemoryConfig, NeuroTopoConfig, TopologyConfig


def _config(vocab=512):
    return NeuroTopoConfig(
        vocab_size=vocab,
        d_model=64,
        n_layers=2,
        max_seq_len=32,
        topology=TopologyConfig(n_modules=8, module_dim=16, local_degree=2,
                                 longrange_topk=4, rank=8, connectivity_reg=False),
        memory=MemoryConfig(enabled=True, d_k=16, d_v=16, persistent_slots=8),
        uncertainty=__import__("xrfm.neurotopo.config", fromlist=["UncertaintyConfig"]).UncertaintyConfig(enabled=True),
    )


def test_builder_returns_model_and_head():
    model, head = build_neurotopo_model(_config())
    assert isinstance(model, NeuroTopoModel)
    assert head is not None
    assert hasattr(head, "evidence")


def test_full_forward_with_labels_and_evidence():
    torch.manual_seed(0)
    cfg = _config()
    model, head = build_neurotopo_model(cfg)
    ids = torch.randint(5, cfg.vocab_size, (2, 16))
    out = model(ids, labels=ids)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert out.loss is not None and torch.isfinite(out.loss)
    # Build evidence from final-layer diagnostics and run the head.
    last_diag = out.diagnostics[-1][-1]
    H = out.state[-1]["H"]
    ei = model.blocks[-1].topo.edge_index
    ew = out.state[-1]["A"].unsqueeze(0).expand(2, -1)
    ev = head.evidence(H, ew, ei, memory_info={"surprise": torch.zeros(2), "margin": torch.zeros(2)})
    logits, calib = head(ev)
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()


def test_backward_updates_parameters():
    torch.manual_seed(1)
    cfg = _config()
    model, head = build_neurotopo_model(cfg)
    ids = torch.randint(5, cfg.vocab_size, (2, 12))
    out = model(ids, labels=ids)
    out.loss.backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(grads) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_end_to_end_with_real_tokenizer():
    tok = BytePairEncoder(vocab_size_target=400)
    tok.train_on_text("the quick brown fox. " * 30)
    cfg = _config(vocab=tok.vocab_size())
    model, head = build_neurotopo_model(cfg)
    ids = torch.tensor([tok.encode("the quick brown fox")], dtype=torch.long)
    out = model(ids)
    assert out.logits.shape[0] == 1 and torch.isfinite(out.logits).all()
