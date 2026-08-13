"""Phase 5 tests: dynamic topology generator.

GATE: demonstrate empirically that dynamic topology != static topology on a
controlled synthetic task (the weights must change with context and provide a
training signal a fixed graph cannot).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xrfm.topology import DynamicTopology


def _make_topo(**kw):
    defaults = dict(
        n_modules=8, module_dim=8, local_degree=2, longrange_topk=3,
        rank=4, decay_init=0.9, plasticity=True, memory_dim=0, seed=0,
    )
    defaults.update(kw)
    return DynamicTopology(**defaults)


def test_topology_changes_with_context():
    torch.manual_seed(0)
    topo = _make_topo()
    H1 = torch.randn(2, 8, 8)
    H2 = torch.randn(2, 8, 8)
    _, w1, _ = topo(H1)
    _, w2, _ = topo(H2)
    assert w1.shape == (2, topo.edge_index.shape[1])
    # Different inputs produce different long-range weights
    assert not torch.allclose(w1, w2)
    # Local edges are fixed at 1.0
    assert torch.allclose(w1[:, topo.local_mask], torch.ones_like(w1[:, topo.local_mask]))


def test_topology_stable_when_input_unchanged():
    topo = _make_topo(plasticity=False)
    H = torch.randn(1, 8, 8)
    _, w1, d1 = topo(H)
    _, w2, d2 = topo(H)
    assert torch.allclose(w1, w2, atol=1e-6)
    assert d1.active_edges == d2.active_edges


def test_topk_sparsity_constraint():
    topo = _make_topo()
    H = torch.randn(3, 8, 8)
    ei, w, diag = topo(H)
    # Each node has 2*local_degree (bidirectional) + longrange_topk edges
    deg = torch.zeros(8)
    deg.scatter_add_(0, ei[0], torch.ones(ei.shape[1]))
    assert (deg == 2 * 2 + 3).all()
    # Number of edges is sparse (<< N*N)
    assert ei.shape[1] <= 8 * (2 * 2 + 3)


def test_plasticity_blends_previous_weights():
    torch.manual_seed(1)
    topo = _make_topo(plasticity=True, decay_init=0.5)
    H = torch.randn(1, 8, 8)
    prev = topo.init_weight()
    _, w_no_prev, _ = topo(H, prev_weight=None)
    _, w_prev, _ = topo(H, prev_weight=prev)
    # With previous weights present, result differs from no-prev
    assert not torch.allclose(w_no_prev, w_prev)
    alpha = topo.decay().item()
    expected = alpha * prev + (1 - alpha) * w_no_prev
    assert torch.allclose(w_prev, expected, atol=1e-6)


def test_gradients_flow_through_scores():
    # Use plasticity so the decay parameter also receives a gradient path.
    topo = _make_topo(plasticity=True)
    H = torch.randn(2, 8, 8, requires_grad=True)
    prev = topo.init_weight().detach().requires_grad_(True)
    _, w, _ = topo(H, prev_weight=prev)
    loss = w.sum()
    loss.backward()
    assert H.grad is not None and torch.isfinite(H.grad).all()
    for name, p in topo.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_no_nans_under_extreme_inputs():
    topo = _make_topo()
    H = torch.randn(2, 8, 8) * 100.0
    _, w, d = topo(H)
    assert torch.isfinite(w).all()
    assert 0.0 <= d.hub_concentration <= 1.0


def test_reproducibility_with_seed():
    t1 = _make_topo(seed=123)
    t2 = _make_topo(seed=123)
    assert torch.equal(t1.edge_index, t2.edge_index)
    H = torch.randn(1, 8, 8)
    _, w1, _ = t1(H)
    _, w2, _ = t2(H)
    assert torch.allclose(w1, w2)


def test_memory_term_influences_scores():
    topo = _make_topo(memory_dim=8)
    H = torch.randn(1, 8, 8)
    M = torch.randn(1, 8, 8)
    _, w_with_M, _ = topo(H, M=M)
    _, w_no_M, _ = topo(H, M=None)
    assert not torch.allclose(w_with_M, w_no_M)


def test_diagnostics_reasonable():
    topo = _make_topo()
    H = torch.randn(2, 8, 8)
    _, w, d = topo(H)
    assert d.active_edges > 0
    assert d.average_degree > 0
    assert d.max_degree >= d.min_degree
    assert d.routing_entropy >= 0
    assert 0.0 <= d.hub_concentration <= 1.0


def test_gate_dynamic_outperforms_static_on_context_dependent_task():
    """GATE: dynamic topology must provide a signal a static/frozen graph cannot.

    Task: given two active source nodes, a context bit tells which source each
    target should attend to. We read out a per-target label from the *incoming
    edge weights*. A DYNAMIC graph can set weights based on the context bit; a
    STATIC graph produces identical weights regardless of context and thus
    cannot solve it. We train both and require dynamic to reach low loss while
    a frozen-edge readout stays near chance.
    """
    torch.manual_seed(42)
    N, d, B = 8, 8, 64
    E_LOCAL = 2

    topo = _make_topo(n_modules=N, module_dim=d, local_degree=E_LOCAL, longrange_topk=5, rank=8)
    E = topo.edge_index.shape[1]
    # Per-target readout over its incoming edge weights.
    head = nn.Linear(E, N)
    opt = torch.optim.Adam(list(topo.parameters()) + list(head.parameters()), lr=3e-2)

    def batch():
        # context = 0 -> source A=1 is relevant; context = 1 -> source B=5
        ctx = torch.randint(0, 2, (B,))
        H = torch.randn(B, N, d) * 0.1
        H[torch.arange(B), 1] += 1.0
        H[torch.arange(B), 5] += 1.0
        # Encode context as a global direction in module 0's state.
        H[:, 0, :] += torch.where(ctx.view(B, 1) == 0, -1.0, 1.0)
        label = torch.where(ctx == 0, 1, 5)
        return H, label

    initial = final = None
    for _ in range(300):
        H, label = batch()
        _, w, _ = topo(H)
        logits = head(w)
        loss = nn.functional.cross_entropy(logits, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if initial is None:
            initial = loss.item()
        final = loss.item()

    # Dynamic must clearly beat chance (ln 2 ~= 0.69).
    assert final < 0.25, f"dynamic topology did not learn: {initial:.4f} -> {final:.4f}"

    # STATIC ablation: freeze edge weights at init (a fixed graph) and train only
    # the readout. Since weights do not depend on H/context, the readout input is
    # constant and the task is unsolvable.
    with torch.no_grad():
        static_w = None
        H0, _ = batch()
        H1, _ = batch()
        _, w0, _ = topo(H0)
        _, w1, _ = topo(H1)
        # Before training this isn't the static control; build a fixed weight.
        fixed_w = torch.sigmoid(torch.zeros(E))
    static_head = nn.Linear(E, N)
    opt2 = torch.optim.Adam(static_head.parameters(), lr=3e-2)
    s_init = s_fin = None
    for _ in range(200):
        _, label = batch()
        logits = static_head(fixed_w.unsqueeze(0).expand(B, -1))
        loss = nn.functional.cross_entropy(logits, label)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
        if s_init is None:
            s_init = loss.item()
        s_fin = loss.item()
    # A constant input cannot separate the two classes -> stays near chance.
    assert s_fin > 0.55, (
        f"static control unexpectedly solved task: {s_init:.4f} -> {s_fin:.4f}"
    )

    # And explicitly: learned dynamic weights separate the two contexts.
    H0 = torch.randn(1, N, d) * 0.1; H0[0, 1] += 1; H0[0, 5] += 1; H0[0, 0] += -1.0
    H1 = torch.randn(1, N, d) * 0.1; H1[0, 1] += 1; H1[0, 5] += 1; H1[0, 0] += 1.0
    with torch.no_grad():
        _, w0, _ = topo(H0)
        _, w1, _ = topo(H1)
    separation = (w0 - w1).abs().mean().item()
    assert separation > 0.05, (
        f"dynamic topology does not separate contexts: mean |w0-w1|={separation:.4f}"
    )
