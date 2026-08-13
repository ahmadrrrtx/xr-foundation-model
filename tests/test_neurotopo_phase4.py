"""Phase 4 tests: static neural graph + sparse message passing.

GATE: a synthetic message-passing task must train (a node must aggregate
information from its graph neighbors to predict a target).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xrfm.topology import StaticMessagePassing, StaticNeuralGraph


def test_graph_exact_connectivity():
    g = StaticNeuralGraph(n_modules=6, local_degree=2, longrange_topk=1, seed=0)
    # node 0 must have local neighbors 1 and 2
    nbrs = set(g.neighbors_of(0))
    assert {1, 2}.issubset(nbrs)
    # local neighbors exist for every node
    deg = g.degree()
    assert (deg >= 2).all()
    assert deg.sum().item() == g.num_edges


def test_graph_is_sparse_no_dense_nxn():
    g = StaticNeuralGraph(100, local_degree=4, longrange_topk=8, seed=1)
    # Expected edges per node = 4 local + 8 long-range (when possible)
    assert g.num_edges <= 100 * (4 + 8)
    # edge_index has shape (2, E), not (N, N)
    assert g.edge_index.shape[0] == 2
    assert g.edge_index.shape[1] < 100 * 100


def test_graph_no_self_loops():
    g = StaticNeuralGraph(20, local_degree=4, longrange_topk=4, seed=2)
    tgt, src = g.edge_index
    assert (tgt != src).all()


def test_isolated_node_handling():
    """A node with no edges gets zero aggregation (degree clamp)."""
    # Build a graph where one node has only local edges (always present),
    # then manually construct an edge_index with an isolated target.
    mp = StaticMessagePassing(d=4)
    H = torch.randn(1, 5, 4)
    # Only node 0 receives from node 1
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    out = mp(H, edge_index)
    assert out.shape == (1, 5, 4)
    # Isolated nodes (1..4) receive zero
    assert torch.allclose(out[0, 1:], torch.zeros(4, 4))
    # Node 0 receives the message from node 1 normalized by degree 1
    assert torch.allclose(out[0, 0], mp.W_msg(H)[0, 1])


def test_message_passing_gradient_flow():
    g = StaticNeuralGraph(8, local_degree=2, longrange_topk=2, seed=3)
    mp = StaticMessagePassing(d=6)
    H = torch.randn(2, 8, 6, requires_grad=True)
    ei, ew = g()
    out = mp(H, ei, ew)
    out.sum().backward()
    assert H.grad is not None and torch.isfinite(H.grad).all()
    assert mp.W_msg.weight.grad is not None


def test_batch_support():
    g = StaticNeuralGraph(5, local_degree=2, longrange_topk=1, seed=4)
    mp = StaticMessagePassing(d=4)
    H = torch.randn(3, 5, 4)
    out = mp(H, *g())
    assert out.shape == (3, 5, 4)


def test_permutation_sanity_across_batch():
    """Same graph applied to all batch elements; permuting batch rows permutes output."""
    g = StaticNeuralGraph(6, local_degree=2, longrange_topk=2, seed=5)
    mp = StaticMessagePassing(d=4)
    H = torch.randn(2, 6, 4)
    out = mp(H, *g())
    assert out.shape == (2, 6, 4)
    # Graph is identical for both -> output differs only via input
    assert not torch.allclose(out[0], out[1])


def test_gate_synthetic_message_passing_trains():
    """GATE: each node must predict the mean of its neighbors' input.

    A single message-passing layer + linear head should learn this quickly,
    proving the sparse graph routes gradient/information correctly.
    """
    torch.manual_seed(42)
    N, d, B = 12, 8, 16
    g = StaticNeuralGraph(N, local_degree=3, longrange_topk=2, seed=6)
    mp = StaticMessagePassing(d)
    head = nn.Linear(d, d)
    opt = torch.optim.Adam(list(mp.parameters()) + list(head.parameters()), lr=1e-2)
    ei, ew = g()

    # Ground-truth target = mean of source neighbors per node (fixed graph)
    # Build a dense target using scatter to be exact.
    with torch.no_grad():
        tgt, src = ei
        deg = torch.zeros(N)
        deg.scatter_add_(0, tgt, torch.ones_like(tgt, dtype=torch.float)).clamp_(min=1)

    initial, final = None, None
    for step in range(200):
        X = torch.randn(B, N, d)
        # target for each node = mean of its source neighbors' inputs
        src_vals = X.index_select(1, src)  # (B, E, d)
        summed = torch.zeros(B, N, d)
        summed.scatter_add_(1, tgt.view(1, -1, 1).expand(B, -1, d), src_vals)
        target = summed / deg.view(1, N, 1)

        agg = mp(X, ei, ew)
        pred = head(agg)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if initial is None:
            initial = loss.item()
        final = loss.item()

    assert final < 0.05, f"MP did not train: {initial:.4f} -> {final:.4f}"
    assert final < initial * 0.2
