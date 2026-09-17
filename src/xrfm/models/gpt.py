"""
XRFM decoder-only transformer (the core "GPT" model).

Combines embedding, stacked transformer blocks, final norm, and lm_head.
Supports KV cache for efficient autoregressive inference.

Public name: :class:`XRFMModel` (``GPTModel`` remains as a backward-compat
alias; they are the same class).

Forward contract
----------------
* **Input**: ``input_ids`` of shape ``(batch, seq)`` or ``(seq,)`` with
  values in ``[0, vocab_size)``.
* **Output**: ``(logits, present_key_values)`` where ``logits`` has shape
  ``(batch, seq, vocab_size)`` (squeezed to ``(seq, vocab)`` for 1-D input)
  and ``present_key_values`` is a per-layer list of ``(K, V)`` caches when
  ``use_cache=True``, else ``None``.
* **Attention mask**: optional ``mask`` forwarded to every block
  (``None`` = causal masking applied internally by each block).
* **Modes**: plain ``nn.Module`` semantics — ``model.train()`` enables
  dropout, ``model.eval()`` disables it; generation code must call
  ``eval()`` (``xrfm.inference`` does).
* **Device**: the model lives wherever the caller moved it
  (``model.to(device)``); the framework never relocates it implicitly.
  ``xrfm.training.Trainer`` moves *batches* to the model's device.

Conceptual references (not copied):
- Vaswani et al. (2017) — Attention Is All You Need
- Meta AI (2024) — Llama 3 architecture
- DeepSeek-AI (2024) — DeepSeek-V3 architecture

Implementation is original.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from xrfm.config.loader import load_config
from xrfm.config.schema import ModelConfig, XRFMConfig
from xrfm.models.embedding import XRFMEmbedding
from xrfm.models.layers.rmsnorm import RMSNorm
from xrfm.models.layers.transformer_block import TransformerBlock

__all__ = ["XRFMModel", "GPTModel"]


def _resolve_model_config(config: XRFMConfig | ModelConfig | str | Path | None) -> ModelConfig:
    """Accept a typed config, a YAML path (legacy), or None (packaged default)."""
    if config is None:
        return load_config(None).model
    if isinstance(config, ModelConfig):
        return config
    if isinstance(config, XRFMConfig):
        return config.model
    if isinstance(config, (str, Path)):
        # Legacy spelling: GPTModel(config_path="config/tiny.yaml").
        return load_config(config).model
    raise TypeError(
        f"config must be XRFMConfig, ModelConfig, str/Path (YAML path), or None; got {type(config).__name__}"
    )


class XRFMModel(nn.Module):
    """Decoder-only transformer with KV cache support.

    Architecture:
        Embedding -> [TransformerBlock x n_layers] -> Final Norm -> LM Head

    Forward with use_cache=True returns present_key_values for each layer,
    enabling incremental generation without recomputation.

    Example:
        >>> from xrfm import XRFM, XRFMConfig, ModelConfig
        >>> cfg = XRFMConfig(model=ModelConfig(vocab_size=512, d_model=64,
        ...     n_layers=2, n_heads=4, d_ff=128, max_seq_len=64, dropout=0.0))
        >>> model = XRFM(cfg)                    # doctest: +SKIP
        >>> logits, kv = model(input_ids, use_cache=True)  # doctest: +SKIP
    """

    def __init__(
        self,
        config: XRFMConfig | ModelConfig | str | Path | None = None,
        weight_tied: bool = True,
        vocab_size: int | None = None,
        bias: bool | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        super().__init__()

        # Legacy keyword: GPTModel(config_path="config/tiny.yaml").
        if config is None and config_path is not None:
            config = config_path
        model_cfg = _resolve_model_config(config)
        # Forensic-audit fix (F-13), kept: allow the caller to override the
        # config's vocab_size with the tokenizer's ACTUAL vocabulary size,
        # keeping the embedding/LM-head and the tokenizer coherent.
        if vocab_size is not None:
            if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
                raise ValueError(f"vocab_size override must be a positive int, got {vocab_size}")
            model_cfg = ModelConfig.from_dict(model_cfg.to_dict())  # validated copy
            model_cfg.vocab_size = vocab_size
        # Validation already ran in ModelConfig.__post_init__; re-validate in
        # case of the override above (defense in depth, zero cost).
        model_cfg.validate()

        self.config = model_cfg
        self.weight_tied = weight_tied
        self.max_seq_len = model_cfg.max_seq_len
        self.dropout_p = model_cfg.dropout
        # bias=None -> read from config (model.use_bias); modern LLMs use False.
        if bias is None:
            bias = bool(model_cfg.use_bias)
        self.use_bias = bias

        # Embedding — padding row comes from the config's pad_token_id
        # (tokenizer contract), not a hard-coded 0 (Phase 0 fix).
        self.embedding = XRFMEmbedding(
            vocab_size=model_cfg.vocab_size,
            d_model=model_cfg.d_model,
            weight_tied=weight_tied,
            padding_idx=model_cfg.pad_token_id,
        )

        # Stacked transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=model_cfg.d_model,
                    n_heads=model_cfg.n_heads,
                    d_ff=model_cfg.d_ff,
                    dropout=model_cfg.dropout,
                    use_rmsnorm=model_cfg.use_rmsnorm,
                    use_rope=model_cfg.use_rope,
                    bias=self.use_bias,
                )
                for _ in range(model_cfg.n_layers)
            ]
        )

        # Final norm
        self.norm_final = RMSNorm(model_cfg.d_model) if model_cfg.use_rmsnorm else nn.LayerNorm(model_cfg.d_model)

        # LM head (weight-tied with embedding by default)
        self.lm_head = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)
        if weight_tied:
            self.lm_head.weight = self.embedding.embedding.weight
        else:
            nn.init.xavier_uniform_(self.lm_head.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]] | None,
    ]:
        """Forward pass with optional KV cache (see module docstring for the contract)."""
        # Validate input
        if input_ids.dim() not in (1, 2):
            raise ValueError(f"input_ids must be 1D or 2D, got {input_ids.dim()}D")

        original_dim = input_ids.dim()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # Check vocabulary bounds
        if (input_ids >= self.embedding.vocab_size).any():
            max_id = int(input_ids.max().item())
            raise IndexError(f"Token ID {max_id} exceeds vocab size ({self.embedding.vocab_size})")

        batch_size, seq_len = input_ids.shape

        # Embedding
        x = self.embedding(input_ids)

        # Pass through transformer blocks with KV cache
        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if (past_key_values is not None and i < len(past_key_values)) else None
            x, present_kv = block(x, mask=mask, past_kv=past_kv, use_cache=use_cache)
            if use_cache and present_kv is not None:
                present_key_values.append(present_kv)

        # Final norm + LM head
        x = self.norm_final(x)
        logits = self.lm_head(x)

        # Restore original dimensions for single-sequence input
        if batch_size == 1 and original_dim == 1:
            logits = logits.squeeze(0)

        return (
            logits,
            present_key_values if (use_cache and present_key_values) else None,
        )

    def parameter_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters())


# Historical public name — the same class. New code should use XRFMModel.
GPTModel = XRFMModel
