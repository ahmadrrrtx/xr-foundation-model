"""
XRFM-NeuroTopo — full causal language model (Phase 10/14).

Wires the input layer, stacked NeuroTopoBlocks, module readout, and a weight-tied
LM head into a next-token model. Per-token recurrent state is carried across the
sequence (no growing KV cache; state is fixed-size). Supports causal cross-entropy
with padding masking and perplexity evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from xrfm.research.neurotopo.message_passing import NeuroTopoBlock
from xrfm.research.neurotopo.memory import GatedDeltaMemory, PersistentMemory
from xrfm.research.neurotopo.config import NeuroTopoConfig
from xrfm.research.neurotopo.input_layer import NeuroTopoInput
from xrfm.research.neurotopo.topology import DynamicTopology


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    state: list  # final per-layer (H, A, M) for incremental decode
    diagnostics: list


class NeuroTopoModel(nn.Module):
    """Causal LM over sparse dynamic neural modules with associative memory."""

    def __init__(self, config: NeuroTopoConfig, embedding_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.config = config
        tcfg = config.topology
        self.N = tcfg.n_modules
        self.d = tcfg.module_dim

        self.input_layer = NeuroTopoInput(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_modules=self.N,
            module_dim=self.d,
            conv_kernel=3,
            padding_idx=config.pad_id,
            embedding_weight=embedding_weight,
        )

        self.blocks = nn.ModuleList()
        for _ in range(config.n_layers):
            topo = DynamicTopology(
                n_modules=self.N,
                module_dim=self.d,
                local_degree=tcfg.local_degree,
                longrange_topk=tcfg.longrange_topk,
                rank=tcfg.rank,
                decay_init=tcfg.decay_init,
                plasticity=tcfg.plasticity,
                memory_dim=self.d if config.memory.enabled else 0,
            )
            mem = None
            if config.memory.enabled:
                mem = GatedDeltaMemory(
                    d_k=config.memory.d_k,
                    d_v=config.memory.d_v,
                    d_hidden=self.d,
                    surprise_gate=config.memory.surprise_gate,
                )
            self.blocks.append(NeuroTopoBlock(d=self.d, topology=topo, memory=mem))

        self.norm_final = nn.RMSNorm(self.d)
        self.readout = nn.Linear(self.d, self.d)
        # Learned query to attend over modules (better than mean-pool for LM readout).
        self.readout_query = nn.Parameter(torch.randn(self.d) * 0.02)
        # Weight-tied LM head: project d -> d_model then dot with embedding.
        if self.d != config.d_model:
            self.head_proj = nn.Linear(self.d, config.d_model, bias=False)
        else:
            self.head_proj = nn.Identity()
        self.lm_head = None  # set when embedding_weight is shared

        if config.memory.persistent_slots > 0:
            self.persistent = PersistentMemory(config.memory.persistent_slots, self.d)
        else:
            self.persistent = None

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def tie_weights(self, embedding_weight: torch.Tensor) -> None:
        """Share an external embedding matrix (e.g., the control's)."""
        self.input_layer.embedding.weight = nn.Parameter(embedding_weight)

    def _logits_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, d) -> project to d_model -> (B,T,vocab)
        z = self.head_proj(h)
        return torch.matmul(z, self.input_layer.embedding.weight.t())

    def init_state(self, batch_size: int, device: torch.device | str = "cpu") -> list:
        states = []
        for block in self.blocks:
            A = block.topo.init_weight(device)
            M = block.memory.init_state(batch_size, device) if block.memory is not None else None
            H = torch.zeros(batch_size, self.N, self.d, device=device)
            states.append({"H": H, "A": A, "M": M})
        return states

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        state: list | None = None,
    ) -> ModelOutput:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, T = input_ids.shape
        device = input_ids.device

        if state is None:
            state = self.init_state(B, device)

        H_tokens = self.input_layer(input_ids)  # (B,T,N,d)
        diagnostics = []
        logits_list = []

        # Step through tokens, carrying per-layer state (H, A, M).
        for t in range(T):
            x = H_tokens[:, t]  # (B,N,d)
            new_state = []
            layer_diag = []
            for li, block in enumerate(self.blocks):
                st = state[li]
                # For incremental (single-token) decode, resume carried H.
                H_in = st["H"] if (T == 1 and st.get("H") is not None) else x
                H_out, A_out, M_out, diag = block(H_in, st["A"], st["M"], C=self.persistent)
                # Inject the token input alongside the carried recurrent state.
                if T == 1:
                    H_out = H_out + x
                new_state.append({"H": H_out.detach(), "A": A_out, "M": M_out})
                x = H_out
                layer_diag.append(diag)
            state = new_state
            diagnostics.append(layer_diag)
            # Readout: attention over modules with a learned query.
            xn = self.norm_final(x)
            # Learned attention over modules: (B,N,d) -> (B,d)
            scores = torch.einsum("bnd,d->bn", xn, self.readout_query) / (self.d**0.5)
            alpha = torch.softmax(scores, dim=1)
            pooled = torch.einsum("bn,bnd->bd", alpha, xn)
            h = self.readout(pooled)
            logits_list.append(self._logits_from_hidden(h))

        logits = torch.stack(logits_list, dim=1)  # (B,T,vocab)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=self.config.pad_id if False else -100,
            )
        return ModelOutput(logits=logits, loss=loss, state=state, diagnostics=diagnostics)
