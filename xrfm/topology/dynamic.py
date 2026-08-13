"""
XRFM-NeuroTopo — dynamic topology generator (Phase 5).

A_t = alpha * A_{t-1} + (1 - alpha) * sigma(S_t)

where long-range candidate scores are
    S_ij = (1/sqrt(r)) (U h_i)^T (V h_j)  +  gamma * (P m_i)^T (Q m_j)

Local ring edges are always present (structural prior). Long-range edges are
selected by top-k over candidate scores per target node. The adjacency is stored
sparsely as (edge_index, edge_weight); no dense N x N tensor is materialized for
message passing, though a temporary (N, N) score matrix is used to do top-k
(acceptable for N up to ~1024 on CPU; a custom kernel replaces it later).

The topology state carries the previous sparse weights for plasticity. This is
the research component that must empirically differ from a static graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TopoDiagnostics:
    active_edges: int
    average_degree: float
    max_degree: int
    min_degree: int
    routing_entropy: float
    hub_concentration: float


class DynamicTopology(nn.Module):
    """Context-dependent sparse topology over N modules.

    edge_index: (2, E) fixed union of local + long-range candidate slots.
        Local edges are fixed; long-range candidate identities are fixed
        (chosen at init) but their weights are context-dependent.
    edge_weight: (E,) gated weights in (0,1).
    """

    def __init__(
        self,
        n_modules: int,
        module_dim: int,
        local_degree: int = 4,
        longrange_topk: int = 8,
        rank: int = 16,
        decay_init: float = 0.9,
        plasticity: bool = True,
        memory_dim: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if n_modules <= 0:
            raise ValueError("n_modules must be positive")
        self.N = n_modules
        self.d = module_dim
        self.k_local = local_degree
        self.k_long = longrange_topk
        self.rank = min(rank, module_dim)
        self.plasticity = plasticity

        # Low-rank scorers for hidden-state similarity.
        self.U = nn.Linear(module_dim, self.rank, bias=False)
        self.V = nn.Linear(module_dim, self.rank, bias=False)
        # Optional memory term.
        self.has_memory = memory_dim > 0
        if self.has_memory:
            self.P = nn.Linear(memory_dim, self.rank, bias=False)
            self.Q = nn.Linear(memory_dim, self.rank, bias=False)
            self.gamma = nn.Parameter(torch.tensor(0.1))
        else:
            self.P = self.Q = None
            self.gamma = 0.0

        # Learned decay in (0,1) via sigmoid on a raw parameter.
        # decay_init near 0.9 -> edges persist.
        self.decay_raw = nn.Parameter(torch.tensor(_logit(decay_init)))

        # Build the fixed candidate edge set: local ring + long-range slots.
        self._build_candidate_edges(seed)
        self._init_weights(seed)

    # ---- construction ---------------------------------------------------
    def _build_candidate_edges(self, seed: int) -> None:
        targets: list[int] = []
        sources: list[int] = []
        is_local: list[bool] = []
        for i in range(self.N):
            for off in range(1, self.k_local + 1):
                # Symmetric local neighborhood: node i hears both its successors
                # and predecessors on the ring (bidirectional locality).
                for j in ((i + off) % self.N, (i - off) % self.N):
                    targets.append(i)
                    sources.append(j)
                    is_local.append(True)
        # Deterministic long-range candidate slots per target (k_long each).
        g = torch.Generator().manual_seed(seed)
        self.long_slot_start = len(targets)
        for i in range(self.N):
            local_set = {((i + off) % self.N) for off in range(1, self.k_local + 1)}
            local_set.add(i)
            candidates = [j for j in range(self.N) if j not in local_set]
            k = min(self.k_long, len(candidates))
            if k > 0:
                perm = torch.randperm(len(candidates), generator=g)[:k]
                for p in perm.tolist():
                    targets.append(i)
                    sources.append(candidates[p])
                    is_local.append(False)
        self.register_buffer(
            "edge_index", torch.tensor([targets, sources], dtype=torch.long), persistent=True
        )
        self.register_buffer("is_local", torch.tensor(is_local, dtype=torch.bool), persistent=True)
        # Edge-local vs long-range masks for scoring
        self.register_buffer(
            "long_mask", ~self.is_local, persistent=False
        )
        self.register_buffer(
            "local_mask", self.is_local, persistent=False
        )

    def _init_weights(self, seed: int) -> None:
        # Deterministic per-instance init so the same seed reproduces weights.
        g = torch.Generator().manual_seed(seed + 7919)
        for lin in (self.U, self.V):
            nn.init.xavier_uniform_(lin.weight, gain=0.5, generator=g)
        if self.has_memory and self.P is not None and self.Q is not None:
            nn.init.xavier_uniform_(self.P.weight, gain=0.5, generator=g)
            nn.init.xavier_uniform_(self.Q.weight, gain=0.5, generator=g)

    # ---- core -----------------------------------------------------------
    def decay(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_raw)

    def score_longrange(
        self, H: torch.Tensor, M: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return per-batch scores for the long-range edges, shape (B, E_long)."""
        tgt = self.edge_index[0][self.long_mask]  # (E_long,)
        src = self.edge_index[1][self.long_mask]
        Uh = self.U(H)  # (B, N, r)
        Vh = self.V(H)
        # (B, E, r) sum-dot over r -> (B, E)
        q = Uh.index_select(1, tgt)
        k = Vh.index_select(1, src)
        scores = (q * k).sum(dim=-1) / (self.rank ** 0.5)
        if self.has_memory and M is not None and self.P is not None and self.Q is not None:
            Pm = self.P(M)
            Qm = self.Q(M)
            qm = Pm.index_select(1, tgt)
            km = Qm.index_select(1, src)
            scores = scores + self.gamma * (qm * km).sum(dim=-1) / (self.rank ** 0.5)
        return scores

    def forward(
        self,
        H: torch.Tensor,
        prev_weight: torch.Tensor | None = None,
        M: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, TopoDiagnostics]:
        """Generate sparse edge weights.

        Args:
            H: (B, N, d) module states.
            prev_weight: (E,) previous gated weights for plasticity, or None.
            M: (B, N, memory_dim) optional memory states.
        Returns:
            edge_index: (2, E)
            edge_weight: (B, E) in (0,1)
            diagnostics
        """
        if H.dim() != 3 or H.shape[1:] != (self.N, self.d):
            raise ValueError(f"H must be (B, {self.N}, {self.d}), got {tuple(H.shape)}")
        B = H.shape[0]
        device = H.device
        E = self.edge_index.shape[1]

        weights = torch.empty(B, E, device=device, dtype=H.dtype)
        # Local edges always have a fixed high gate.
        weights[:, self.local_mask] = 1.0

        # Long-range edges get context-dependent scores.
        long_scores = self.score_longrange(H, M)  # (B, E_long)
        long_gates = torch.sigmoid(long_scores)
        weights[:, self.long_mask] = long_gates

        # Plasticity: blend with previous weights.
        if self.plasticity and prev_weight is not None:
            alpha = self.decay()
            weights = alpha * prev_weight.unsqueeze(0) + (1.0 - alpha) * weights

        diag = self.diagnostics(weights)
        return self.edge_index, weights, diag

    def init_weight(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Initial edge weights (local=1, long=0)."""
        w = torch.zeros(self.edge_index.shape[1], device=device)
        w[self.local_mask] = 1.0
        return w

    # ---- diagnostics ----------------------------------------------------
    def diagnostics(self, weights: torch.Tensor) -> TopoDiagnostics:
        # weights: (B, E)
        w = weights.detach()
        # Count an edge as "active" if gate > 0.5 (batch-averaged).
        active = (w > 0.5).float().mean(dim=0)  # (E,)
        n_active = int((active > 0.5).sum().item())
        # Per-node in-degree weighted by active edges.
        tgt = self.edge_index[0]
        deg = torch.zeros(self.N)
        deg.scatter_add_(0, tgt, active.cpu())
        avg = float(deg.mean().item())
        maxd = int(deg.max().item())
        mind = int(deg.min().item())
        # Routing entropy over edge usage distribution.
        p = active / (active.sum().clamp(min=1e-8))
        entropy = float(-(p * (p.clamp(min=1e-8)).log()).sum().item())
        # Hub concentration: fraction of edge mass on top 10% of nodes.
        sorted_deg, _ = deg.sort(descending=True)
        top = max(1, self.N // 10)
        hub = float(sorted_deg[:top].sum().item() / max(deg.sum().item(), 1e-8))
        return TopoDiagnostics(
            active_edges=n_active,
            average_degree=avg,
            max_degree=maxd,
            min_degree=mind,
            routing_entropy=entropy,
            hub_concentration=hub,
        )


def _logit(p: float) -> float:
    import math

    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))
