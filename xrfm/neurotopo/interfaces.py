"""
XRFM-NeuroTopo (XRFM-NT) — abstract component interfaces (Phase 1).

These define the contracts that every NeuroTopo component implements. They are
intentionally minimal: Phase 1 establishes module boundaries only; concrete
behavior arrives in later phases.

The existing Transformer (model/) is CONTROL-1 and is never imported by this
package; NeuroTopo is independently selectable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class NeuroTopoState:
    """Bundled recurrent state for a NeuroTopo model/layer.

    H: module states         (batch, N, d)
    A: topology state        (sparse edge representation; opaque at Phase 1)
    M: associative memory    (batch, d_k, d_v) or None
    C: persistent memory     (K, d) shared across batch, or None
    """

    H: torch.Tensor
    A: object | None = None
    M: torch.Tensor | None = None
    C: torch.Tensor | None = None

    def detach(self) -> "NeuroTopoState":
        return NeuroTopoState(
            H=self.H.detach(),
            A=self.A,
            M=None if self.M is None else self.M.detach(),
            C=None if self.C is None else self.C.detach(),
        )


class NeuralModule(ABC):
    """A set of N neural modules with recurrent state H_t in R^{N x d}."""

    N: int
    d: int

    @abstractmethod
    def init_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return initial H_0 of shape (batch, N, d)."""

    @abstractmethod
    def inject(self, H: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Inject token input x (batch, d_in) into module states H."""

    @abstractmethod
    def reset(self, H: torch.Tensor) -> torch.Tensor:
        """Reset module state (e.g. at sequence boundaries)."""


class Dynamics(ABC):
    """Per-module local recurrent dynamics: h_i(t+1) = f(h_i(t), input_i)."""

    @abstractmethod
    def forward(self, H: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """Apply dynamics to every module. Returns H' of shape (batch, N, d)."""


class TopologyGenerator(ABC):
    """Generates a (sparse) context-dependent topology A_t over modules."""

    @abstractmethod
    def forward(
        self,
        H: torch.Tensor,
        state: object | None = None,
        memory: torch.Tensor | None = None,
    ) -> tuple[object, object]:
        """Return (edge representation, new topology state). Phase 5 fills behavior."""


class MessageFunction(ABC):
    """Edge message: m_{j->i} = msg(h_j, edge_feature)."""

    @abstractmethod
    def forward(self, H_j: torch.Tensor, edge_feat: torch.Tensor | None) -> torch.Tensor:
        ...


class MessagePassingEngine(ABC):
    """Sparse aggregation: h_i' = Update(h_i, sum_{j in N(i)} A_ij msg(h_j))."""

    @abstractmethod
    def forward(
        self,
        H: torch.Tensor,
        edges: object,
        msg_fn: MessageFunction,
    ) -> torch.Tensor:
        ...


class MemorySystem(ABC):
    """Associative memory with write/read and fixed capacity (no external RAG)."""

    @abstractmethod
    def init_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        ...

    @abstractmethod
    def forward(
        self,
        H: torch.Tensor,
        M: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read/write. Returns (read_context c, updated M)."""


class UncertaintyHead(ABC):
    """Maps hidden + graph/memory evidence to a calibrated abstention state."""

    STATE_NAMES: tuple[str, ...] = (
        "ANSWER",
        "QUALIFY",
        "ABSTAIN",
        "REQUEST_CONTEXT",
        "REQUEST_EVIDENCE",
    )

    @abstractmethod
    def forward(self, z: torch.Tensor, evidence: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return logits over the 5 states, shape (batch, 5)."""


class NeuroTopoBlock(ABC):
    """One NeuroTopo layer: dynamics + topology + message passing + memory + FFN."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        state: NeuroTopoState,
    ) -> tuple[torch.Tensor, NeuroTopoState]:
        ...


class NeuroTopoModel(ABC):
    """Full model: embedding -> blocks -> readout -> LM head (+ uncertainty)."""

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        state: NeuroTopoState | None = None,
    ) -> dict[str, torch.Tensor | NeuroTopoState]:
        ...
