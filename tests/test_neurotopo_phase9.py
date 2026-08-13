"""Phase 9 tests: language input path (tokenizer -> embedding -> modules)."""

from __future__ import annotations

import torch

from tokenizer.bpe import BytePairEncoder
from xrfm.neurotopo.input_layer import NeuroTopoInput


def _tokenizer():
    tok = BytePairEncoder(vocab_size_target=512)
    tok.train_on_text("the cat sat on the mat . " * 50 + "a dog ran in the park . " * 50)
    return tok


def test_input_shapes():
    input_layer = NeuroTopoInput(vocab_size=512, d_model=64, n_modules=8, module_dim=8, padding_idx=0)
    ids = torch.randint(5, 512, (2, 12))
    H0 = input_layer(ids)
    assert H0.shape == (2, 12, 8, 8)
    assert torch.isfinite(H0).all()


def test_1d_input_batched():
    input_layer = NeuroTopoInput(512, 64, 8, 8)
    ids = torch.randint(0, 512, (10,))
    H0 = input_layer(ids)
    assert H0.shape[0] == 1 and H0.shape[1] == 10


def test_causal_conv_does_not_leak_future():
    """The conv is causal: output at position t must not depend on t+1.."""
    torch.manual_seed(0)
    layer = NeuroTopoInput(64, 32, 4, 8, conv_kernel=3)
    ids = torch.randint(0, 64, (1, 8))
    H_full = layer(ids)
    # Changing a future token should not change outputs before it.
    ids2 = ids.clone()
    ids2[0, 6:] = (ids2[0, 6:] + 7) % 64
    H_part = layer(ids2)
    # positions 0..4 must be identical (kernel 3: position 5 sees up to 5)
    assert torch.allclose(H_full[0, :5], H_part[0, :5], atol=1e-6)


def test_gradient_flow_through_input():
    layer = NeuroTopoInput(512, 64, 8, 8)
    ids = torch.randint(0, 512, (2, 10))
    H0 = layer(ids)
    H0.sum().backward()
    assert layer.embedding.weight.grad is not None
    assert layer.conv.weight.grad is not None
    assert layer.proj.weight.grad is not None


def test_uses_existing_tokenizer_end_to_end():
    """GATE: a tiny text forward pass works with the REAL XRFM BPE tokenizer
    (no new tokenizer created)."""
    tok = _tokenizer()
    vs = tok.vocab_size()
    pad = tok.pad_id or 0
    layer = NeuroTopoInput(vs, d_model=64, n_modules=8, module_dim=8, padding_idx=pad)
    ids = torch.tensor([tok.encode("the cat sat")], dtype=torch.long)
    H0 = layer(ids)
    assert H0.shape[0] == 1
    assert H0.shape[2] == 8 and H0.shape[3] == 8
    assert H0.shape[1] == ids.shape[1]
    assert torch.isfinite(H0).all()
    # Padding positions produce zero embedding (and thus finite) output
    padded = torch.full((1, 5), pad, dtype=torch.long)
    Hp = layer(padded)
    assert torch.isfinite(Hp).all()


def test_weight_tying_accepts_external_embedding():
    embed_w = torch.randn(512, 64) * 0.02
    layer = NeuroTopoInput(512, 64, 8, 8, embedding_weight=embed_w)
    assert torch.equal(layer.embedding.weight.data, embed_w)
