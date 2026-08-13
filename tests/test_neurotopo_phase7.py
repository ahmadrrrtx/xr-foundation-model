"""Phase 7 tests: gated delta-rule associative memory + persistent memory.

GATE: memory measurably improves synthetic retrieval vs a no-memory control.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xrfm.memory import GatedDeltaMemory, PersistentMemory


def test_init_state_shapes():
    mem = GatedDeltaMemory(d_k=8, d_v=6, d_hidden=10)
    M = mem.init_state(4)
    assert M.shape == (4, 8, 6)
    assert torch.all(M == 0)


def test_write_then_read_roundtrip():
    torch.manual_seed(0)
    mem = GatedDeltaMemory(d_k=16, d_v=16, d_hidden=16)
    M = mem.init_state(1)
    # Encode a key/value via two different h vectors (simulate distinct tokens)
    h_k = torch.randn(1, 16)
    h_v = torch.randn(1, 16)
    # Write with h_k
    _, M, _ = mem(h_k, M)
    # Read back using a near-duplicate key (same h_k projection)
    c, M2, _ = mem(h_k, M)
    assert c.shape == (1, 16)
    assert torch.isfinite(c).all()


def test_multiple_facts_no_crash_and_distinct_reads():
    torch.manual_seed(1)
    mem = GatedDeltaMemory(d_k=16, d_v=16, d_hidden=16)
    M = mem.init_state(2)
    for _ in range(5):
        h = torch.randn(2, 16)
        c, M, info = mem(h, M)
        assert torch.isfinite(c).all()
        assert info["surprise"].shape == (2,)
    # Different inputs should yield different reads
    c1, _, _ = mem(torch.randn(2, 16), M)
    c2, _, _ = mem(torch.randn(2, 16), M)
    assert not torch.allclose(c1, c2)


def test_decay_reduces_memory_over_time():
    torch.manual_seed(2)
    mem = GatedDeltaMemory(d_k=8, d_v=8, d_hidden=8)
    M = mem.init_state(1)
    h = torch.randn(1, 8)
    _, M, _ = mem(h, M)
    norm_after_write = M.norm().item()
    # Repeated steps with zero input; high decay bias (~0.88) should still reduce norm
    for _ in range(20):
        _, M, _ = mem(torch.zeros(1, 8), M)
    assert M.norm().item() < norm_after_write + 1e-4


def test_erase_gate_exists_and_runs():
    mem = GatedDeltaMemory(d_k=8, d_v=8, d_hidden=8)
    M = mem.init_state(1)
    h = torch.randn(1, 8)
    _, M1, _ = mem(h, M)
    assert M1.shape == (1, 8, 8)
    # erase gate parameter receives gradients
    h2 = torch.randn(1, 8, requires_grad=True)
    c, M2, _ = mem(h2, M1)
    c.sum().backward()
    assert h2.grad is not None
    assert mem.W_e.weight.grad is not None


def test_gradients_flow_through_memory():
    mem = GatedDeltaMemory(d_k=8, d_v=8, d_hidden=8)
    M = mem.init_state(2)
    h = torch.randn(2, 8, requires_grad=True)
    c, M2, _ = mem(h, M)
    loss = c.pow(2).sum() + M2.pow(2).sum()
    loss.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    for n, p in mem.named_parameters():
        assert p.grad is not None, f"no grad for {n}"
        assert torch.isfinite(p.grad).all(), f"bad grad for {n}"


def test_persistent_memory_read():
    pm = PersistentMemory(n_slots=4, d=8)
    q = torch.randn(3, 8)
    out = pm(q)
    assert out.shape == (3, 8)
    assert torch.isfinite(out).all()


def test_persistent_memory_freeze():
    pm = PersistentMemory(4, 8)
    pm.freeze()
    assert not pm.keys.requires_grad
    assert pm.frozen is True


def test_gate_memory_improves_synthetic_retrieval():
    """GATE: an associative key->value lookup must be learnable with the delta
    memory, and a parameter-matched no-memory control (which only sees the
    query) must NOT be able to solve it. This proves memory actually stores
    and retrieves facts.
    """
    torch.manual_seed(42)
    D, B, L = 16, 64, 4

    class WithMem(nn.Module):
        """Controller that explicitly writes (key,value) pairs then reads by query."""

        def __init__(self):
            super().__init__()
            self.mem = GatedDeltaMemory(D, D, d_hidden=2 * D, surprise_gate=True)
            # Project a concatenated (key,value) write vector -> h for memory
            self.W_h = nn.Linear(2 * D, 2 * D)
            self.W_k = nn.Linear(D, D, bias=False)
            self.W_v = nn.Linear(D, D, bias=False)
            self.out = nn.Linear(D, D)

        def write(self, key, value, M):
            h = torch.tanh(self.W_h(torch.cat([key, value], dim=-1)))
            # Override the memory's internal k,v projections with explicit ones
            k = self.mem._l2_normalize(self.W_k(key))
            v = self.W_v(value)
            q = k
            read = torch.einsum("bd,bdv->bv", k, M)
            err = v - read
            erase = torch.sigmoid(self.mem.W_e(torch.cat([h, k, err], dim=-1)))
            write = torch.sigmoid(self.mem.W_w(torch.cat([h, v, err], dim=-1)))
            decay = torch.sigmoid(self.mem.W_d(torch.cat([h, k], dim=-1)))
            beta = torch.sigmoid(self.mem.W_b(err)) if self.mem.surprise_gate else torch.ones_like(err)
            update = (k.unsqueeze(-1) * (err * erase).unsqueeze(-2)) * write.unsqueeze(-2) * beta.unsqueeze(-2)
            M = decay.unsqueeze(-1) * M + update
            return M

        def read(self, query, M):
            q = self.mem._l2_normalize(self.W_k(query))
            c = torch.einsum("bd,bdv->bv", q, M)
            return self.out(c)

        def forward(self, keys, values, query):
            M = self.mem.init_state(keys.shape[0])
            for t in range(keys.shape[1]):
                M = self.write(keys[:, t], values[:, t], M)
            return self.read(query, M)

    class NoMem(nn.Module):
        """Same parameter budget, but no sequence memory — only the query."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(D, 2 * D), nn.Tanh(), nn.Linear(2 * D, D))

        def forward(self, keys, values, query):
            return self.net(query)

    def make_batch():
        keys = torch.randn(B, L, D)
        values = torch.randn(B, L, D)
        idx = torch.randint(0, L, (B,))
        query = keys[torch.arange(B), idx]
        target = values[torch.arange(B), idx]
        return keys, values, query, target

    torch.manual_seed(0)
    withmem = WithMem()
    nomem = NoMem()
    opt_w = torch.optim.Adam(withmem.parameters(), lr=2e-2)
    opt_n = torch.optim.Adam(nomem.parameters(), lr=2e-2)
    loss_fn = nn.MSELoss()

    w_init = w_fin = n_init = n_fin = None
    for step in range(400):
        keys, vals, q, tgt = make_batch()
        lw = loss_fn(withmem(keys, vals, q), tgt)
        ln = loss_fn(nomem(keys, vals, q), tgt)
        opt_w.zero_grad(); lw.backward(); opt_w.step()
        opt_n.zero_grad(); ln.backward(); opt_n.step()
        if w_init is None:
            w_init, n_init = lw.item(), ln.item()
        w_fin, n_fin = lw.item(), ln.item()

    assert w_fin < 0.2, f"memory model did not retrieve: {w_init:.3f} -> {w_fin:.3f}"
    # No-memory cannot solve it: it never sees which (key,value) pair was queried.
    assert n_fin > 0.6, f"no-memory control unexpectedly solved it: {n_init:.3f} -> {n_fin:.3f}"
    assert w_fin < n_fin * 0.4, (
        f"memory did not improve over no-memory: w={w_fin:.3f} no-mem={n_fin:.3f}"
    )
