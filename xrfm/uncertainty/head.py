"""
XRFM-NeuroTopo — graph/memory evidence uncertainty head (Phase 12).

Produces probabilities over five output states:
    ANSWER, QUALIFY, ABSTAIN, REQUEST_CONTEXT, REQUEST_EVIDENCE
from evidence signals that are NOT just next-token probability:
  - graph support / conflict (weight of supporting vs contradicting edges)
  - memory retrieval margin (best - second-best association)
  - surprise (memory prediction error)
  - effective rank of module states (epistemic signal)
  - routing entropy (how concentrated the module routing is)

This is an architectural head, not a prompt-time trick.
"""

from __future__ import annotations

import torch
import torch.nn as nn

STATE_NAMES = (
    "ANSWER",
    "QUALIFY",
    "ABSTAIN",
    "REQUEST_CONTEXT",
    "REQUEST_EVIDENCE",
)


def effective_rank(H: torch.Tensor) -> torch.Tensor:
    """Effective rank via normalized singular-value entropy.

    H: (B, N, d). Returns (B,) in [1, rank].
    """
    s = torch.linalg.svdvals(H.float())
    s = s / (s.sum(dim=-1, keepdim=True) + 1e-8)
    ent = -(s * (s + 1e-8).log()).sum(dim=-1)
    return ent / torch.log(torch.tensor(H.shape[1], dtype=torch.float, device=H.device))


def routing_entropy(edge_weight: torch.Tensor, edge_index: torch.Tensor, N: int) -> torch.Tensor:
    """Per-batch entropy of edge usage across target nodes. Returns (B,)."""
    B, E = edge_weight.shape
    tgt = edge_index[0]
    deg = torch.zeros(B, N, device=edge_weight.device, dtype=edge_weight.dtype)
    deg.scatter_add_(1, tgt.unsqueeze(0).expand(B, E), edge_weight)
    p = deg / (deg.sum(dim=-1, keepdim=True) + 1e-8)
    ent = -(p * (p + 1e-8).log()).sum(dim=-1)
    return ent / torch.log(torch.tensor(N, dtype=torch.float, device=edge_weight.device))


class UncertaintyHead(nn.Module):
    """MLP over an evidence vector -> 5-state logits + scalar calibration."""

    STATE_NAMES = STATE_NAMES

    def __init__(self, d_evidence: int = 9, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_evidence, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, len(STATE_NAMES)),
        )
        self.calib = nn.Linear(d_evidence, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def evidence(
        self,
        H: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_index: torch.Tensor,
        memory_info: dict | None = None,
        next_token_entropy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the (B, d_evidence) vector."""
        B, N, d = H.shape
        eps = 1e-8
        mean_edge = edge_weight.mean(dim=1)
        min_edge = edge_weight.amin(dim=1)
        # "Conflict" proxy: variance of edge weights (disagreement).
        conflict = edge_weight.var(dim=1)
        # "Support" proxy: fraction of edges above 0.5.
        support = (edge_weight > 0.5).float().mean(dim=1)
        # Module activation concentration.
        mod_norm = H.norm(dim=-1)
        mod_conc = mod_norm.max(dim=-1).values / (mod_norm.mean(dim=-1) + eps)
        # Effective rank.
        effrank = effective_rank(H)
        # Routing entropy.
        rent = routing_entropy(edge_weight, edge_index, N)
        feats = [mean_edge, min_edge, conflict, support, mod_conc, effrank, rent]
        if memory_info is not None:
            surprise = memory_info.get("surprise")
            margin = memory_info.get("margin")
            if surprise is None:
                surprise = torch.zeros(B, device=H.device)
            if margin is None:
                margin = torch.zeros(B, device=H.device)
            feats += [surprise, margin]
        else:
            feats += [torch.zeros(B, device=H.device), torch.zeros(B, device=H.device)]
        if next_token_entropy is not None:
            feats.append(next_token_entropy)
        ev = torch.stack(feats[:9], dim=-1)
        return ev

    def forward(self, evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(evidence), self.calib(evidence).squeeze(-1)

    @torch.no_grad()
    def decide(self, logits: torch.Tensor, threshold: float = 0.5) -> list[str]:
        probs = torch.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        out = []
        for c, i in zip(conf.tolist(), idx.tolist()):
            out.append(STATE_NAMES[i] if c >= threshold else "QUALIFY")
        return out
