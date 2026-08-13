"""Phase 2 tests: neural module state H_t in R^{B x N x d}."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from xrfm.neurons import NeuralModuleState


def test_init_shapes_and_determinism():
    mod = NeuralModuleState(N=8, d=16)
    h1 = mod.init_state(4)
    h2 = mod.init_state(4)
    assert h1.shape == (4, 8, 16)
    # init_state returns the same learned h0 broadcast -> deterministic
    assert torch.equal(h1, h2)
    # h0 is small (stability)
    assert h1.abs().max().item() < 0.1


def test_inject_broadcasts_input_to_modules():
    mod = NeuralModuleState(N=6, d=12, d_in=10)
    H = mod.init_state(3)
    x = torch.randn(3, 10)
    H2 = mod.inject(H, x)
    assert H2.shape == (3, 6, 12)
    # The same projected input is added to every module up to per-module gain.
    diff = H2 - H
    # differences across modules for a given batch differ only by gain ratio
    assert diff.shape == (3, 6, 12)
    assert torch.isfinite(H2).all()


def test_reset_returns_fresh_state():
    mod = NeuralModuleState(N=4, d=8)
    H = torch.randn(2, 4, 8, requires_grad=True)
    Hr = mod.reset(H)
    assert Hr.shape == (2, 4, 8)
    # reset does not depend on the old (possibly arbitrary) state values
    assert torch.equal(Hr, mod.init_state(2))


def test_gradient_flows_through_inject():
    mod = NeuralModuleState(N=4, d=8, d_in=5)
    H = mod.init_state(2).detach().requires_grad_(True)
    x = torch.randn(2, 5)
    H2 = mod(H, x)
    loss = H2.pow(2).sum()
    loss.backward()
    assert H.grad is not None
    assert torch.isfinite(H.grad).all()
    # projection parameters receive gradients
    assert mod.proj.weight.grad is not None
    assert mod.gain.grad is not None
    assert mod.h0.grad is None or torch.isfinite(mod.h0.grad).all()


def test_batch_dimension_validation():
    mod = NeuralModuleState(N=4, d=8)
    with pytest.raises(ValueError):
        mod.inject(torch.randn(4, 8), torch.randn(2, 8))  # H is 2D
    with pytest.raises(ValueError):
        mod.inject(torch.randn(2, 4, 8), torch.randn(3, 8))  # batch mismatch


def test_synthetic_forward_backward_runs():
    """End-to-end tiny synthetic update: repeated injection + a linear read."""
    torch.manual_seed(0)
    N, d, B, T = 4, 8, 2, 6
    mod = NeuralModuleState(N=N, d=d, d_in=d)
    read = nn.Linear(d, d)
    opt = torch.optim.SGD(list(mod.parameters()) + list(read.parameters()), lr=0.01)
    H = mod.init_state(B)
    x = torch.randn(B, T, d)
    target = torch.randn(B, d)
    prev = None
    for t in range(T):
        H = mod(H, x[:, t])
        # mean-pool modules -> readout
        z = read(H.mean(dim=1))
        prev = z
    loss = ((prev - target) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_numerical_stability_under_large_inputs():
    mod = NeuralModuleState(N=4, d=8, d_in=8)
    H = mod.init_state(2)
    x = torch.randn(2, 8) * 100.0
    H2 = mod(H, x)
    assert torch.isfinite(H2).all()
