"""
XRFM-NeuroTopo — dynamic sparse message passing + integrated block (Phase 8).

The block composes local dynamics, dynamic topology, sparse message passing, and
gated associative memory into one recurrent step. State is (H, A, M) per layer;
persistent memory C is model-level and optional.

    H -> RMSNorm -> DynamicTopology(A) -> sparse aggregation
                                                |
    summary of H ----> GatedDeltaMemory(M) -> read c
                                                |
    GRU dynamics(H, proj(agg) + c) -> H'
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xrfm.dynamics import GRUDynamics
from xrfm.memory import GatedDeltaMemory, PersistentMemory


class DynamicMessagePassing(nn.Module):
    """Sparse, batched message passing driven by dynamic edge weights."""

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
        edge_weight: torch.Tensor,
        edge_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if H.dim() != 3 or H.shape[-1] != self.d:
            raise ValueError(f"H must be (B, N, {self.d}), got {tuple(H.shape)}")
        B, N, _ = H.shape
        tgt, src = edge_index[0], edge_index[1]
        E = tgt.shape[0]

        msg = self.W_msg(H)
        src_msg = msg.index_select(1, src)
        if edge_feat is not None and self.edge_dim > 0:
            src_msg = src_msg + self.W_edge(edge_feat)
        if edge_weight.dim() == 1:
            edge_weight = edge_weight.unsqueeze(0).expand(B, E)
        src_msg = src_msg * edge_weight.unsqueeze(-1)

        out = torch.zeros_like(H)
        tgt_exp = tgt.view(1, E, 1).expand(B, E, self.d)
        out.scatter_add_(1, tgt_exp, src_msg)

        deg = torch.zeros(N, device=H.device, dtype=H.dtype)
        deg.scatter_add_(0, tgt, torch.ones(E, device=H.device, dtype=H.dtype)).clamp_(min=1.0)
        return out / deg.view(1, N, 1)


class NeuroTopoBlock(nn.Module):
    """One NeuroTopo layer: dynamics + dynamic topology + sparse MP + memory."""

    def __init__(
        self,
        d: int,
        topology: nn.Module,
        edge_dim: int = 0,
        memory: GatedDeltaMemory | None = None,
    ) -> None:
        super().__init__()
        self.d = d
        self.topo = topology
        self.mp = DynamicMessagePassing(d, edge_dim=edge_dim)
        self.dynamics = GRUDynamics(d, d_in=d)
        self.norm = nn.RMSNorm(d)
        self.proj = nn.Linear(d, d, bias=True)
        self.memory = memory
        if memory is not None:
            # Project memory read into the dynamics input space.
            self.proj_mem = nn.Linear(memory.d_v, d, bias=False)
            # Project flattened memory matrix into a per-module signal for topology.
            self.mem_signal = nn.Linear(memory.d_k * memory.d_v, d, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)
        if self.memory is not None:
            nn.init.xavier_uniform_(self.proj_mem.weight, gain=0.5)
            nn.init.xavier_uniform_(self.mem_signal.weight, gain=0.2)

    def forward(
        self,
        H: torch.Tensor,
        topo_state: torch.Tensor | None,
        M: torch.Tensor | None = None,
        C: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, object]:
        # Build a per-module memory signal for topology scoring if memory exists.
        M_signal = None
        if self.memory is not None and M is not None:
            flat = M.flatten(start_dim=1)  # (B, d_k*d_v)
            sig = self.mem_signal(flat)    # (B, d)
            M_signal = sig.unsqueeze(1).expand(-1, H.shape[1], -1)  # (B,N,d)
        Hn = self.norm(H)
        edge_index, edge_weight, diag = self.topo(Hn, prev_weight=topo_state, M=M_signal)
        agg = self.mp(Hn, edge_index, edge_weight)

        new_M = M
        if self.memory is not None and M is not None:
            h_summary = Hn.mean(dim=1)
            c, new_M, _info = self.memory(h_summary, M)
            agg = agg + self.proj_mem(c).unsqueeze(1)
            if C is not None and isinstance(C, PersistentMemory):
                p = C(h_summary)
                agg = agg + p.unsqueeze(1)

        H_out = self.dynamics(H, self.proj(agg))
        new_topo = edge_weight.detach().mean(dim=0) if self.topo.plasticity else None
        return H_out, new_topo, new_M, diag
