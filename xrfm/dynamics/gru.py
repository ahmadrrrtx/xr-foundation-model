"""
XRFM-NeuroTopo — local GRU dynamics (Phase 3).

h_i(t+1) = Dynamics(h_i(t), input_i), applied identically to each of the N
modules (with N-independent parameters). This is the first recurrent update;
diagonal SSM (Mamba-2-style) will later slot behind the same `Dynamics`
interface. Stability: orthogonal-ish init for recurrent kernels, sigmoid
gates bounded in (0,1), small candidate init.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xrfm.neurotopo.interfaces import Dynamics


class GRUDynamics(nn.Module, Dynamics):
    """Minimal GRU applied over the module dimension d.

    Inputs/Outputs are (B, N, d); the same GRUCell-style parameters are
    applied to every module (modules differ via their state and input).
    """

    def __init__(self, d: int, d_in: int | None = None) -> None:
        nn.Module.__init__(self)
        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")
        d_in = d_in or d
        self.d = d
        # Gates: reset r, update z, candidate n
        self.W_ir = nn.Linear(d_in, d, bias=True)
        self.W_hr = nn.Linear(d, d, bias=False)
        self.W_iz = nn.Linear(d_in, d, bias=True)
        self.W_hz = nn.Linear(d, d, bias=False)
        self.W_in = nn.Linear(d_in, d, bias=True)
        self.W_hn = nn.Linear(d, d, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for linear in (
            self.W_ir, self.W_hr, self.W_iz, self.W_hz, self.W_in, self.W_hn
        ):
            nn.init.xavier_uniform_(linear.weight, gain=0.5)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        # Initialize update gate bias positive so memory persists initially.
        nn.init.constant_(self.W_iz.bias, 1.0)

    def forward(self, H: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        if H.dim() != 3 or H.shape[-1] != self.d:
            raise ValueError(f"H must be (B, N, {self.d}), got {tuple(H.shape)}")
        if input.shape[-1] != self.W_ir.in_features or input.shape[0] != H.shape[0]:
            raise ValueError("input shape incompatible with H")
        r = torch.sigmoid(self.W_ir(input) + self.W_hr(H))
        z = torch.sigmoid(self.W_iz(input) + self.W_hz(H))
        n = torch.tanh(self.W_in(input) + r * self.W_hn(H))
        return (1 - z) * n + z * H
