# ruff: noqa: E702, N802, N803, F841  — research bundle tests (pre-Phase-0 style)
"""Phase 15: parameter-matched controls (CONTROL-1, NT, CONTROL-2)."""

from __future__ import annotations

import torch

from xrfm.research.neurotopo.training.param_match import (
    build_nt,
    build_transformer,
    count_params,
    match_nt_to_transformer,
)


def test_transformer_control_builds():
    m = build_transformer(512, d_model=64, n_layers=2, n_heads=4)
    assert count_params(m) > 0
    x = torch.randint(0, 512, (2, 16))
    logits, _ = m(x)
    assert logits.shape == (2, 16, 512)


def test_nt_builds_and_counts():
    m = build_nt(512, d_model=128, n_layers=2, n_modules=8, module_dim=16)
    assert count_params(m) > 0
    out = m(torch.randint(0, 512, (2, 12)))
    assert out.logits.shape[-1] == 512


def test_match_nt_to_transformer_within_tolerance():
    # Build a small Transformer control, then match NT to its param count.
    ctrl = build_transformer(400, d_model=64, n_layers=2, n_heads=4)
    target = count_params(ctrl)
    match = match_nt_to_transformer(400, target, tolerance=0.10)
    assert match is not None
    # Matched within 10% (or best available)
    assert match["rel"] < 0.20, match
    nt = build_nt(
        400,
        d_model=match["d_model"],
        n_layers=match["n_layers"],
        n_modules=match["n_modules"],
        module_dim=match["module_dim"],
    )
    assert abs(count_params(nt) - target) / target < 0.20


def test_all_three_controls_forward_on_same_input():
    """CONTROL-1 (Transformer), EXPERIMENT (NT), and a no-graph NT ablation
    all forward the same token ids with the same vocab."""
    vocab = 300
    ids = torch.randint(5, vocab, (2, 10))
    ctrl1 = build_transformer(vocab, d_model=64, n_layers=2, n_heads=4)
    exp = build_nt(vocab, d_model=128, n_layers=2, n_modules=8, module_dim=16)
    ctrl2 = build_nt(vocab, d_model=128, n_layers=2, n_modules=8, module_dim=16, memory=False)
    l1, _ = ctrl1(ids)
    o_exp = exp(ids)
    o_c2 = ctrl2(ids)
    assert l1.shape[-1] == o_exp.logits.shape[-1] == o_c2.logits.shape[-1] == vocab
    assert torch.isfinite(o_exp.logits).all()
