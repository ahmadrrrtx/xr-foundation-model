# ruff: noqa: E702, N802, N803, F841  — research bundle tests (pre-Phase-0 style)
"""Phase 8 tests: integrate topology + memory (state H,A,M,C) and ablate."""

from __future__ import annotations

import torch
import torch.nn as nn

from xrfm.research.neurotopo import NeuroTopoBlock, NTState
from xrfm.research.neurotopo.dynamics import GRUDynamics
from xrfm.research.neurotopo.memory import GatedDeltaMemory
from xrfm.research.neurotopo.topology import DynamicTopology, StaticMessagePassing, StaticNeuralGraph


def _dynamic_block(n_modules=8, d=8, with_memory=True, **kw):
    topo = DynamicTopology(n_modules, d, local_degree=2, longrange_topk=4, rank=8, seed=kw.pop("seed", 0), **kw)
    mem = GatedDeltaMemory(d, d, d_hidden=d) if with_memory else None
    return NeuroTopoBlock(d=d, topology=topo, memory=mem), topo, mem


def test_block_with_memory_forward_shapes():
    block, topo, mem = _dynamic_block()
    H = torch.randn(2, 8, 8)
    A = topo.init_weight()
    M = mem.init_state(2)
    H2, A2, M2, diag = block(H, A, M)
    assert H2.shape == (2, 8, 8)
    assert A2 is not None and A2.shape == A.shape
    assert M2 is not None and M2.shape == (2, 8, 8)
    assert torch.isfinite(H2).all() and torch.isfinite(M2).all()


def test_block_without_memory_runs():
    block, topo, _ = _dynamic_block(with_memory=False)
    H = torch.randn(2, 8, 8)
    H2, A2, M2, _ = block(H, topo.init_weight(), None)
    assert M2 is None
    assert torch.isfinite(H2).all()


def test_state_bundle_detach():
    s = NTState(
        H=[torch.randn(1, 4, 4, requires_grad=True)],
        A=[torch.randn(3)],
        M=[torch.randn(1, 4, 4)],
        C=None,
    )
    sd = s.detach()
    assert not sd.H[0].requires_grad
    assert not sd.M[0].requires_grad


def test_memory_influences_topology_via_M_term():
    topo = DynamicTopology(8, 8, local_degree=2, longrange_topk=4, rank=8, memory_dim=8, seed=1)
    mem = GatedDeltaMemory(8, 8, d_hidden=8)
    block = NeuroTopoBlock(d=8, topology=topo, memory=mem)
    H = torch.randn(1, 8, 8)
    A = topo.init_weight()
    with torch.no_grad():
        _, w0, _ = topo(H, prev_weight=A, M=mem.init_state(1))
        _, w1, _ = topo(H, prev_weight=A, M=torch.randn(1, 8, 8))
    assert not torch.allclose(w0, w1)


def test_block_backward_with_memory():
    block, topo, mem = _dynamic_block()
    H = torch.randn(2, 8, 8, requires_grad=True)
    H2, _, M2, _ = block(H, topo.init_weight(), mem.init_state(2))
    loss = H2.pow(2).sum() + M2.pow(2).sum()
    loss.backward()
    assert H.grad is not None and torch.isfinite(H.grad).all()
    # All parameters on the active compute path get gradients. (mem_signal is
    # unused when the topology has no memory term, so it may not.)
    active = [n for n, p in block.named_parameters() if "mem_signal" not in n]
    for name in active:
        p = dict(block.named_parameters())[name]
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"bad grad for {name}"


def test_multistep_rollout_stable():
    block, topo, mem = _dynamic_block()
    H = torch.randn(2, 8, 8)
    A = topo.init_weight()
    M = mem.init_state(2)
    for _ in range(20):
        H, A, M, _ = block(H, A, M)
    assert torch.isfinite(H).all() and torch.isfinite(M).all()


class _StaticAblation(nn.Module):
    """Ablation A: static graph, no memory, no dynamic topology."""

    def __init__(self, N, d):
        super().__init__()
        self.graph = StaticNeuralGraph(N, local_degree=2, longrange_topk=4, seed=0)
        self.mp = StaticMessagePassing(d)
        self.proj = nn.Linear(d, d)
        self.dyn = GRUDynamics(d, d)

    def forward(self, H):
        ei, ew = self.graph()
        agg = self.mp(H, ei, ew)
        return self.dyn(H, self.proj(agg))


def test_gate_ablation_dynamic_plus_memory_outperforms():
    """GATE: delayed-copy task. A signal is written to memory, then distractor
    steps follow; the model must reproduce the signal by reading memory. The
    no-memory model has no persistent store and must fail. This honestly tests
    whether integrated memory adds value (A/B/C comparison: static, dynamic,
    dynamic+memory).
    """
    torch.manual_seed(42)
    N, d, B, T = 8, 8, 64, 4

    class WithMem(nn.Module):
        def __init__(self):
            super().__init__()
            self.block, self.topo, self.mem = _dynamic_block(seed=3)
            self.write_h = nn.Linear(d, d)
            self.read_out = nn.Linear(d, d)

        def forward(self, signal):
            H = torch.zeros(B, N, d)
            A = self.topo.init_weight()
            M = self.mem.init_state(B)
            # Write signal into memory.
            c, M, _ = self.mem(torch.tanh(self.write_h(signal)), M)
            # Distractor steps.
            for _ in range(T):
                dist = torch.randn(B, N, d)
                H, A, M, _ = self.block(H + dist, A, M)
            # Read memory using same key projection.
            c_final, M, _ = self.mem(torch.tanh(self.write_h(signal)), M)
            return self.read_out(c_final)

    class NoMem(nn.Module):
        def __init__(self):
            super().__init__()
            self.block, self.topo, _ = _dynamic_block(with_memory=False, seed=3)
            self.out = nn.Linear(d, d)

        def forward(self, signal):
            H = torch.zeros(B, N, d)
            H[:, 0] = signal
            A = self.topo.init_weight()
            for _ in range(T):
                dist = torch.randn(B, N, d) * 0.5
                dist[:, 0] = 0.0
                H, A, _, _ = self.block(H + dist, A, None)
            return self.out(H[:, N - 1])

    wm = WithMem()
    nm = NoMem()
    static = _StaticAblation(N, d)
    sread = nn.Linear(d, d)
    opt = torch.optim.Adam(
        list(wm.parameters()) + list(nm.parameters()) + list(static.parameters()) + list(sread.parameters()),
        lr=2e-2,
    )
    loss_fn = nn.MSELoss()
    w0 = n0 = s0 = wf = nf = sf = None
    for step in range(300):
        signal = torch.randn(B, d)
        lw = loss_fn(wm(signal), signal)
        ln = loss_fn(nm(signal), signal)
        Hs = torch.zeros(B, N, d)
        Hs[:, 0] = signal
        for _ in range(T):
            Hs = static(Hs + torch.randn(B, N, d) * 0.3)
        ls = loss_fn(sread(Hs[:, N - 1]), signal)
        opt.zero_grad()
        (lw + ln + ls).backward()
        opt.step()
        if w0 is None:
            w0, n0, s0 = lw.item(), ln.item(), ls.item()
        wf, nf, sf = lw.item(), ln.item(), ls.item()

    assert wf < 0.3, f"memory model failed delayed copy: {w0:.3f} -> {wf:.3f}"
    assert nf > 0.6, f"no-memory control unexpectedly solved it: {n0:.3f} -> {nf:.3f}"
    assert sf > 0.5, f"static control unexpectedly solved it: {s0:.3f} -> {sf:.3f}"
    assert wf < nf * 0.5, f"memory did not help: mem={wf:.3f} no-mem={nf:.3f}"
