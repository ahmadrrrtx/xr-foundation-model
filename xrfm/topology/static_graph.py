"""
XRFM-NeuroTopo — static (context-independent) neural graph (Phase 4).

A static graph defines, for each of N modules, a fixed set of incoming neighbors:
  - local edges: nearest modules on a 1D ring lattice (dense locally)
  - long-range edges: deterministic pseudo-random top-k "small-world" links
The adjacency is sparse (each node has k_local + k_long incoming edges) and is
NOT materialized as an N x N matrix. We store index tensors (sources per node)
and fixed scalar weights, suitable for gather/scatter message passing.

This is the scientific ABLATION baseline ("static graph"). Phase 5 replaces the
fixed weights/long-range links with a context-dependent topology generator.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StaticNeuralGraph(nn.Module):
    """Fixed sparse graph over N modules with small-world connectivity.

    edge_index: (2, E) long tensor, rows = [target, source] (message flows
        source -> target).
    edge_weight: (E,) fixed scalar weight per edge.
    """

    def __init__(
        self,
        n_modules: int,
        local_degree: int = 4,
        longrange_topk: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if n_modules <= 0:
            raise ValueError("n_modules must be positive")
        if local_degree < 0 or longrange_topk < 0:
            raise ValueError("degrees must be non-negative")
        self.N = n_modules
        self.k_local = local_degree
        self.k_long = longrange_topk

        targets: list[int] = []
        sources: list[int] = []

        # Local edges: each node connects to k_local nearest neighbors on a ring.
        for i in range(n_modules):
            for off in range(1, local_degree + 1):
                j = (i + off) % n_modules
                targets.append(i)
                sources.append(j)

        # Long-range edges: deterministic per-node pseudo-random links that are
        # NOT local neighbors (small-world rewiring). Fixed seed for reproducibility.
        g = torch.Generator().manual_seed(seed)
        for i in range(n_modules):
            local_set = {((i + off) % n_modules) for off in range(1, local_degree + 1)}
            local_set.add(i)
            candidates = [j for j in range(n_modules) if j not in local_set]
            if not candidates or longrange_topk == 0:
                continue
            k = min(longrange_topk, len(candidates))
            perm = torch.randperm(len(candidates), generator=g)[:k]
            for p in perm.tolist():
                targets.append(i)
                sources.append(candidates[p])

        edge_index = torch.tensor([targets, sources], dtype=torch.long)  # (2, E)
        # Fixed uniform weight; softmax-normalized per target at forward time.
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
        self.register_buffer("edge_index", edge_index, persistent=True)
        self.register_buffer("edge_weight", edge_weight, persistent=True)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def degree(self) -> torch.Tensor:
        """In-degree per module (number of incoming edges)."""
        deg = torch.zeros(self.N, dtype=torch.long)
        deg.scatter_add_(0, self.edge_index[0], torch.ones_like(self.edge_index[0]))
        return deg

    def neighbors_of(self, node: int) -> list[int]:
        mask = self.edge_index[0] == node
        return self.edge_index[1][mask].tolist()

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (edge_index, edge_weight)."""
        return self.edge_index, self.edge_weight


class StaticMessagePassing(nn.Module):
    """Gather/scatter message passing over a static graph.

    m_{j->i} = W_msg h_j ;  h'_i = (1/deg_i) sum_j A_ij m_{j->i}
    Operates on (B, N, d). No dense N x N tensor is formed.
    """

    def __init__(self, d: int, edge_dim: int = 0) -> None:
        super().__init__()
        self.d = d
        self.edge_dim = edge_dim
        self.W_msg = nn.Linear(d, d, bias=False)
        if edge_dim > 0:
            self.W_edge = nn.Linear(edge_dim, d, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W_msg.weight, gain=0.5)
        if self.edge_dim > 0:
            nn.init.xavier_uniform_(self.W_edge.weight, gain=0.5)

    def forward(
        self,
        H: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        edge_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if H.dim() != 3 or H.shape[-1] != self.d:
            raise ValueError(f"H must be (B, N, {self.d}), got {tuple(H.shape)}")
        B = H.shape[0]
        tgt, src = edge_index[0], edge_index[1]
        E = tgt.shape[0]

        # Messages from every source module: (B, N, d) -> gather sources -> (B, E, d)
        msg = self.W_msg(H)
        src_msg = msg.index_select(1, src)  # (B, E, d)
        if edge_feat is not None and self.edge_dim > 0:
            src_msg = src_msg + self.W_edge(edge_feat)  # broadcast over B

        if edge_weight is not None:
            src_msg = src_msg * edge_weight.view(1, E, 1)

        # Scatter-add into targets: (B, N, d)
        out = torch.zeros_like(H)
        tgt_exp = tgt.view(1, E, 1).expand(B, E, self.d)
        out.scatter_add_(1, tgt_exp, src_msg)

        # Normalize by in-degree (number of incoming edges per target)
        deg = torch.zeros(H.shape[1], device=H.device, dtype=H.dtype)
        deg.scatter_add_(0, tgt, torch.ones(E, device=H.device, dtype=H.dtype)).clamp_(min=1.0)
        out = out / deg.view(1, H.shape[1], 1)
        return out
