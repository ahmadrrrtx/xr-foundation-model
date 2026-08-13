"""XRFM-NeuroTopo — full recurrent state bundle (Phase 8): (H, A, M, C)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NTState:
    """Bundled state across all layers of a NeuroTopo model.

    H: list[Tensor]  per-layer module states (B, N, d)
    A: list[Tensor]  per-layer topology edge weights (E,) or None (plasticity off)
    M: list[Tensor]  per-layer associative memory (B, d_k, d_v) or None
    C: Tensor or None  persistent memory shared across layers (K, d)
    """

    H: list[torch.Tensor]
    A: list[torch.Tensor | None]
    M: list[torch.Tensor | None]
    C: torch.Tensor | None = None

    def detach(self) -> "NTState":
        return NTState(
            H=[h.detach() for h in self.H],
            A=[None if a is None else a.detach() for a in self.A],
            M=[None if m is None else m.detach() for m in self.M],
            C=None if self.C is None else self.C.detach(),
        )
