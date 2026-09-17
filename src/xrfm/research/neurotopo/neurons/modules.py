"""
XRFM-NeuroTopo — neural module state (Phase 2).

H_t in R^{batch x N x d}, where N = number of neural modules and d = module dim.
This module owns ONLY module state representation: initialization, token-input
injection, reset, and batching. Local dynamics arrive in Phase 3; topology in
Phase 4/5. No Transformer control code is imported.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from xrfm.research.neurotopo.interfaces import NeuralModule


class NeuralModuleState(nn.Module, NeuralModule):
    """A bank of N neural modules with recurrent state H.

    Phase 2 provides a stable, simple update path:
      - init_state: zeros (deterministic), shape (B, N, d).
      - inject: broadcast token input x (B, d_in) into modules via a learned
        projection + per-module gains, added to the current state.
      - reset: zero the state (sequence boundaries).

    The injection is intentionally simple and linear-stable so gradient flow
    can be verified before recurrent dynamics are added.
    """

    def __init__(self, N: int, d: int, d_in: int | None = None) -> None:
        nn.Module.__init__(self)
        if N <= 0 or d <= 0:
            raise ValueError(f"N and d must be positive, got N={N}, d={d}")
        self.N = N
        self.d = d
        d_in = d_in or d
        # Project token input to module space; per-module learned gain.
        self.proj = nn.Linear(d_in, d, bias=True)
        self.gain = nn.Parameter(torch.full((N, 1), 0.1))
        # Small learned initial state (close to zero for stability).
        self.h0 = nn.Parameter(torch.zeros(N, d))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight, gain=0.02 / math.sqrt(2))
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.h0, std=0.02)
        # gain starts small so injection is stable at the beginning of training.
        nn.init.constant_(self.gain, 0.1)

    def init_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        h0 = self.h0.to(device)
        return h0.unsqueeze(0).expand(batch_size, self.N, self.d).contiguous()

    def inject(self, H: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Add token input x (B, d_in) to every module, scaled by per-module gain."""
        if H.dim() != 3 or H.shape[1:] != (self.N, self.d):
            raise ValueError(f"H must be (B, {self.N}, {self.d}), got {tuple(H.shape)}")
        if x.dim() != 2 or x.shape[0] != H.shape[0]:
            raise ValueError(f"x must be (B, d_in), got {tuple(x.shape)}")
        u = self.proj(x)  # (B, d)
        return H + self.gain * u.unsqueeze(1)  # broadcast over N

    def reset(self, H: torch.Tensor) -> torch.Tensor:
        return self.init_state(H.shape[0], H.device)

    def forward(self, H: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Alias for inject (Phase 2 has no dynamics yet)."""
        return self.inject(H, x)
