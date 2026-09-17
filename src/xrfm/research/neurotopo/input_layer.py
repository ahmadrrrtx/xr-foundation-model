"""
XRFM-NeuroTopo — language input path (Phase 9).

Reuses the existing XRFM embedding (weight-tied) and tokenizer; adds a small
causal 1D convolution (Mamba-style local context) and an input projection that
injects each token into the N neural modules. No new tokenizer is created, and
the control Transformer is untouched.

    token ids -> Embedding -> CausalConv1d -> Linear -> injected H_0 (B,N,d)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NeuroTopoInput(nn.Module):
    """Embedding + causal conv + token->modules injection."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_modules: int,
        module_dim: int,
        conv_kernel: int = 3,
        padding_idx: int = 0,
        embedding_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if module_dim != d_model // n_modules and d_model % n_modules == 0:
            # Allow d_model to differ but require explicit module_dim.
            pass
        self.d_model = d_model
        self.N = n_modules
        self.d = module_dim

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        if embedding_weight is not None:
            self.embedding.weight = nn.Parameter(embedding_weight)
        else:
            nn.init.normal_(self.embedding.weight, std=0.02)
            if padding_idx is not None:
                with torch.no_grad():
                    self.embedding.weight[padding_idx].zero_()

        # Causal conv over the sequence for local token context.
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=1,
        )
        nn.init.normal_(self.conv.weight, std=0.02)
        nn.init.zeros_(self.conv.bias)
        self.conv_kernel = conv_kernel

        # Project d_model -> N*module_dim, then reshape into modules.
        self.proj = nn.Linear(d_model, n_modules * module_dim, bias=True)
        nn.init.normal_(self.proj.weight, std=0.02 / (2**0.5))
        nn.init.zeros_(self.proj.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return token-level module inputs of shape (B, T, N, d)."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, T = input_ids.shape
        x = self.embedding(input_ids)  # (B, T, d_model)
        # Causal conv: (B, d, T) then trim the right padding.
        xc = x.transpose(1, 2)
        yc = self.conv(xc)[:, :, :T]
        y = yc.transpose(1, 2)  # (B, T, d_model)
        y = torch.nn.functional.silu(y)
        h = self.proj(y)  # (B, T, N*d)
        return h.view(B, T, self.N, self.d)
