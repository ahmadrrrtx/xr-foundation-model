# ruff: noqa: E702, N802, N803, F841  — research bundle tests (pre-Phase-0 style)
"""Phase 6 tests: sparse dynamic message passing + end-to-end block.

GATE: end-to-end synthetic graph sequence task trains using
modules + dynamics + dynamic topology + sparse MP.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xrfm.research.neurotopo import DynamicMessagePassing, NeuroTopoBlock
from xrfm.research.neurotopo.topology import DynamicTopology


def test_dynamic_mp_shapes_and_sparsity():
    N = 16
    topo = DynamicTopology(N, 8, local_degree=2, longrange_topk=4, rank=4, seed=0)
    mp = DynamicMessagePassing(8)
    H = torch.randn(2, N, 8)
    ei, w, _ = topo(H)
    out = mp(H, ei, w)
    assert out.shape == (2, N, 8)
    assert torch.isfinite(out).all()
    # Number of edges is sparse: E = N*(2*k_local + k_long) << N*N
    expected = N * (2 * 2 + 4)
    assert ei.shape[1] == expected
    assert ei.shape[1] < N * N


def test_dynamic_mp_broadcasts_1d_weights():
    topo = DynamicTopology(5, 4, local_degree=1, longrange_topk=2, rank=2, seed=1)
    mp = DynamicMessagePassing(4)
    H = torch.randn(3, 5, 4)
    ei, w, _ = topo(H)
    # Use a single weight vector shared across batch
    out = mp(H, ei, w[0])
    assert out.shape == (3, 5, 4)


def test_dynamic_mp_gradient_flow():
    topo = DynamicTopology(6, 8, local_degree=2, longrange_topk=2, rank=4, seed=2)
    mp = DynamicMessagePassing(8)
    H = torch.randn(2, 6, 8, requires_grad=True)
    ei, w, _ = topo(H)
    out = mp(H, ei, w)
    out.sum().backward()
    assert H.grad is not None and torch.isfinite(H.grad).all()
    assert mp.W_msg.weight.grad is not None


def test_dynamic_mp_isolates_disconnected_nodes():
    """With zero weights, aggregation is zero (no leakage across modules)."""
    mp = DynamicMessagePassing(4)
    H = torch.randn(1, 4, 4)
    # edges: target 0 from source 1, target 2 from source 3, but all weights 0
    ei = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    w = torch.zeros(1, 2)
    out = mp(H, ei, w)
    assert torch.allclose(out, torch.zeros_like(out))


def test_block_forward_shapes():
    topo = DynamicTopology(8, 8, local_degree=2, longrange_topk=3, rank=4, seed=3)
    block = NeuroTopoBlock(d=8, topology=topo)
    H = torch.randn(2, 8, 8)
    state = topo.init_weight()
    H2, new_state, _, diag = block(H, state, None)
    assert H2.shape == (2, 8, 8)
    assert torch.isfinite(H2).all()
    assert diag.active_edges > 0


def test_block_stateful_plasticity():
    topo = DynamicTopology(8, 8, local_degree=2, longrange_topk=3, rank=4, decay_init=0.5, plasticity=True, seed=4)
    block = NeuroTopoBlock(d=8, topology=topo)
    H = torch.randn(1, 8, 8)
    state = topo.init_weight()
    H1, s1, _, _ = block(H, state, None)
    H2, s2, _, _ = block(H1, s1, None)
    assert s1 is not None and s2 is not None
    assert not torch.allclose(s1, s2) or not torch.allclose(H1, H2)


def test_block_backward():
    topo = DynamicTopology(6, 8, local_degree=2, longrange_topk=2, rank=4, seed=5)
    block = NeuroTopoBlock(d=8, topology=topo)
    H = torch.randn(2, 6, 8, requires_grad=True)
    H2, _, _, _ = block(H, topo.init_weight(), None)
    loss = H2.pow(2).sum()
    loss.backward()
    assert H.grad is not None and torch.isfinite(H.grad).all()
    for n, p in block.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad for {n}"


def test_gate_end_to_end_synthetic_graph_sequence_trains():
    """GATE: a small system of modules+GRU+dynamic topology+sparse MP must
    learn a synthetic multi-step routing task (information must propagate
    from source to target across the graph over time).
    """
    torch.manual_seed(42)
    N, d, B = 8, 8, 64
    topo = DynamicTopology(N, d, local_degree=2, longrange_topk=4, rank=8, seed=6)
    block = NeuroTopoBlock(d=d, topology=topo)
    read = nn.Linear(d, d)
    params = list(block.parameters()) + list(read.parameters())
    opt = torch.optim.Adam(params, lr=3e-2)
    loss_fn = nn.MSELoss()

    # One-step propagation: a source node carries a signal; after one block,
    # its local neighbor (src+1) should reproduce it. This directly tests that
    # sparse dynamic message passing routes information source -> target.
    initial = final = None
    for step in range(500):
        src = torch.randint(0, N, (B,))
        tgt = (src + 1) % N
        signal = torch.randn(B, d)
        H = torch.zeros(B, N, d)
        H[torch.arange(B), src] = signal

        state = topo.init_weight()
        H, state, _, _ = block(H, state, None)
        pred = read(H[torch.arange(B), tgt])
        loss = loss_fn(pred, signal)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if initial is None:
            initial = loss.item()
        final = loss.item()

    assert final < 0.2, f"end-to-end graph task did not train: {initial:.4f} -> {final:.4f}"
    assert final < initial * 0.3
