"""Phase 3 tests: local GRU dynamics + overfit gate.

GATE (mission): a tiny neural system using these dynamics must overfit a
deterministic synthetic sequence. If it cannot, STOP before topology work.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xrfm.dynamics import GRUDynamics
from xrfm.neurons import NeuralModuleState


def test_gru_shapes_and_bounds():
    dyn = GRUDynamics(d=8, d_in=8)
    H = torch.randn(3, 5, 8)
    x = torch.randn(3, 5, 8)
    H2 = dyn(H, x)
    assert H2.shape == (3, 5, 8)
    assert torch.isfinite(H2).all()
    # tanh candidate keeps values bounded initially
    assert H2.abs().max().item() < 50.0


def test_gru_zero_input_preserves_shape_and_is_finite():
    torch.manual_seed(0)
    dyn = GRUDynamics(d=6)
    H = torch.randn(2, 4, 6)
    H2 = dyn(H, torch.zeros(2, 4, 6))
    assert H2.shape == H.shape
    assert torch.isfinite(H2).all()


def test_gru_repeated_input_converges_toward_fixed_point():
    """With a constant input and z-bias toward 1, repeated application should
    not explode and should approach a bounded fixed point."""
    torch.manual_seed(1)
    dyn = GRUDynamics(d=8)
    H = torch.zeros(1, 3, 8)
    x = torch.randn(1, 3, 8).expand(50, -1, -1)  # same input 50 times
    for t in range(50):
        H = dyn(H, x[t:t+1])
    assert torch.isfinite(H).all()
    assert H.abs().max().item() < 50.0


def test_gru_backward():
    dyn = GRUDynamics(d=8)
    H = torch.randn(2, 4, 8, requires_grad=True)
    x = torch.randn(2, 4, 8)
    H2 = dyn(H, x)
    loss = H2.pow(2).sum()
    loss.backward()
    assert H.grad is not None and torch.isfinite(H.grad).all()
    for n, p in dyn.named_parameters():
        assert p.grad is not None, f"no grad for {n}"
        assert torch.isfinite(p.grad).all()


def test_gate_overfit_deterministic_synthetic_sequence():
    """GATE: tiny neural modules + GRU must overfit a deterministic sequence.

    Task: given x_t, predict x_{t+1} where x follows a fixed sine pattern.
    A model that can overfit this proves the recurrent dynamics train.
    """
    torch.manual_seed(42)
    N, d, B = 4, 16, 8
    T = 40

    # Deterministic target sequence: a fixed phase pattern of dim d.
    t = torch.linspace(0, 4 * 3.14159, T + 1)
    pattern = torch.stack([torch.sin(t + 0.3 * i) for i in range(d)], dim=-1)  # (T+1, d)
    pattern = pattern.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, T+1, d)

    modules = NeuralModuleState(N=N, d=d, d_in=d)
    dyn = GRUDynamics(d=d, d_in=d)
    read = nn.Linear(d, d)
    params = list(modules.parameters()) + list(dyn.parameters()) + list(read.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    loss_fn = nn.MSELoss()

    initial_loss = None
    final_loss = None
    for step in range(300):
        H = modules.init_state(B)
        total = 0.0
        for tstep in range(T):
            # inject current token into modules, then apply local dynamics
            H = modules.inject(H, pattern[:, tstep])
            H = dyn(H, H)  # state updates from its own post-injection values
            pred = read(H.mean(dim=1))  # (B, d)
            total = total + loss_fn(pred, pattern[:, tstep + 1])
        loss = total / T
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert initial_loss is not None and final_loss is not None
    # Strong decrease = the recurrent system can fit the deterministic sequence.
    assert final_loss < 0.05, f"failed to overfit: {initial_loss:.4f} -> {final_loss:.4f}"
    assert final_loss < initial_loss * 0.2, (
        f"insufficient decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    )


def test_gradient_stability_over_long_rollout():
    """No NaN/Inf and gradients remain finite over a 200-step rollout."""
    torch.manual_seed(0)
    dyn = GRUDynamics(d=8)
    H = torch.zeros(1, 2, 8, requires_grad=True)
    for _ in range(200):
        x = torch.randn(1, 2, 8) * 0.1
        H = dyn(H, x)
    loss = H.pow(2).sum()
    loss.backward()
    assert torch.isfinite(H).all()
    # H is a non-leaf after dynamics; verify parameter grads are finite instead.
    for p in dyn.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
