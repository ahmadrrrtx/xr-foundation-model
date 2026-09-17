"""
XRFM-NeuroTopo — gated delta-rule associative memory (Phase 7).

A fixed-capacity key->value memory M in R^{d_k x d_v}:

    read   = M^T k                         (current association for key k)
    err    = v - read                      (prediction error / surprise)
    erase  = sigmoid(W_e [h; k; err])      (key-axis channel gate in [0,1]^{d_k})
    write  = sigmoid(W_w [h; v; err])      (value-axis channel gate in [0,1]^{d_v})
    decay  = sigmoid(W_d [h; k])           (retention in [0,1])
    beta   = sigmoid(W_b err)              (step size, surprise-modulated)
    M      = diag(decay) M + beta * write * (k outer (err * erase))

Read for a query q: c = M^T q.

This is the v1 episodic/working memory. It is differentiable, has bounded size,
and is part of the neural computation (no external RAG). A persistent (frozen-
after-pretraining) memory C is provided separately.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedDeltaMemory(nn.Module):
    """Batched gated delta-rule associative memory."""

    def __init__(self, d_k: int, d_v: int, d_hidden: int, surprise_gate: bool = True) -> None:
        super().__init__()
        if d_k <= 0 or d_v <= 0:
            raise ValueError("d_k, d_v must be positive")
        self.d_k = d_k
        self.d_v = d_v
        self.surprise_gate = surprise_gate
        # Key/query/value projections from module-state summary
        self.W_k = nn.Linear(d_hidden, d_k, bias=False)
        self.W_q = nn.Linear(d_hidden, d_k, bias=False)
        self.W_v = nn.Linear(d_hidden, d_v, bias=False)
        # Gates
        self.W_e = nn.Linear(d_hidden + d_k + d_v, d_k)  # erase (key axis)
        self.W_w = nn.Linear(d_hidden + d_v + d_v, d_v)  # write (value axis)
        self.W_d = nn.Linear(d_hidden + d_k, d_k)  # decay/retention
        if surprise_gate:
            self.W_b = nn.Linear(d_v, d_v, bias=True)  # beta from error
        else:
            self.register_parameter("W_b", None)
        self._init_weights()

    def _init_weights(self) -> None:
        for lin in (self.W_k, self.W_q, self.W_v):
            nn.init.xavier_uniform_(lin.weight, gain=0.5)
        for lin in (self.W_e, self.W_w, self.W_d):
            nn.init.xavier_uniform_(lin.weight, gain=0.2)
            nn.init.zeros_(lin.bias)
        if self.W_b is not None:
            nn.init.xavier_uniform_(self.W_b.weight, gain=0.2)
            nn.init.zeros_(self.W_b.bias)
        # Bias decay toward high retention and erase toward "keep" (0 = no erase).
        nn.init.constant_(self.W_d.bias, 2.0)  # sigmoid(2) ~ 0.88
        nn.init.constant_(self.W_e.bias, -2.0)  # sigmoid(-2) ~ 0.12 erase
        nn.init.constant_(self.W_w.bias, 1.0)  # mostly write

    def init_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(batch_size, self.d_k, self.d_v, device=device)

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def forward(self, h: torch.Tensor, M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Write then read.

        Args:
            h: (B, d_hidden) summary of module states for this step.
            M: (B, d_k, d_v) memory.
        Returns:
            c: (B, d_v) read context for this step's query.
            M_new: (B, d_k, d_v) updated memory.
            info: dict with surprise, margin, read.
        """
        k = self._l2_normalize(self.W_k(h))
        q = self._l2_normalize(self.W_q(h))
        v = self.W_v(h)

        # Current association for the key k: (B, d_v)
        read = torch.einsum("bd,bdv->bv", k, M)
        err = v - read

        erase = torch.sigmoid(self.W_e(torch.cat([h, k, err], dim=-1)))
        write = torch.sigmoid(self.W_w(torch.cat([h, v, err], dim=-1)))
        decay = torch.sigmoid(self.W_d(torch.cat([h, k], dim=-1)))  # (B, d_k)

        if self.surprise_gate and self.W_b is not None:
            beta = torch.sigmoid(self.W_b(err))  # (B, d_v)
        else:
            beta = torch.ones_like(err)

        # Outer product: (B, d_k, d_v)
        update = k.unsqueeze(-1) * (err * erase).unsqueeze(-2)
        # Channel-wise write gate on value axis, beta scales the whole write.
        update = update * write.unsqueeze(-2) * beta.unsqueeze(-2)

        M_new = decay.unsqueeze(-1) * M + update

        c = torch.einsum("bd,bdv->bv", q, M_new)
        surprise = err.detach().pow(2).mean(dim=-1)
        # Retrieval margin: norm of read vs competitors (proxy)
        with torch.no_grad():
            margin = read.norm(dim=-1)
        return c, M_new, {"surprise": surprise, "margin": margin, "read": read.detach()}


class PersistentMemory(nn.Module):
    """A frozen-after-pretraining key/value memory (Titans persistent branch).

    During pretraining the entries are learnable; call freeze() after pretraining.
    Readout is attention over K slots.
    """

    def __init__(self, n_slots: int, d: int) -> None:
        super().__init__()
        self.K = n_slots
        self.keys = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.values = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.frozen = False

    def freeze(self) -> None:
        self.keys.requires_grad_(False)
        self.values.requires_grad_(False)
        self.frozen = True

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q: (B, d) -> (B, d)
        scores = torch.matmul(q, self.keys.t()) / (q.shape[-1] ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, self.values)
