"""
GPT Decoder-Only Transformer for XRFM (v0.6.0).

Combines embedding, stacked transformer blocks, final norm, and lm_head.
Supports KV cache for efficient autoregressive inference.

Conceptual references (not copied):
- Vaswani et al. (2017) — Attention Is All You Need
- Meta AI (2024) — Llama 3 architecture
- DeepSeek-AI (2024) — DeepSeek-V3 architecture

Implementation is original.
"""

import torch
import torch.nn as nn

from model.embedding import XRFMEmbedding
from model.layers.rmsnorm import RMSNorm
from model.layers.transformer_block import TransformerBlock
from xrfm.config.loader import ConfigLoader


class GPTModel(nn.Module):
    """Decoder-only transformer with KV cache support.

    Architecture:
        Embedding -> [TransformerBlock x n_layers] -> Final Norm -> LM Head

    Forward with use_cache=True returns present_key_values for each layer,
    enabling incremental generation without recomputation.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        weight_tied: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(config_path, str):
            raise TypeError(f"config_path must be str, got {type(config_path).__name__}")

        try:
            loader = ConfigLoader(config_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Config not found: '{config_path}'") from exc

        model_cfg = loader.model_config()
        self.config_path = config_path
        self.weight_tied = weight_tied
        self.max_seq_len = model_cfg.max_seq_len
        self.dropout_p = model_cfg.dropout

        # Validate architecture parameters
        if model_cfg.d_model <= 0:
            raise ValueError(f"d_model must be positive: {model_cfg.d_model}")
        if model_cfg.n_layers <= 0:
            raise ValueError(f"n_layers must be positive: {model_cfg.n_layers}")
        if model_cfg.n_heads <= 0:
            raise ValueError(f"n_heads must be positive: {model_cfg.n_heads}")
        if model_cfg.d_model % model_cfg.n_heads != 0:
            raise ValueError(
                f"d_model ({model_cfg.d_model}) not divisible by n_heads ({model_cfg.n_heads})"
            )
        if model_cfg.d_ff <= 0:
            raise ValueError(f"d_ff must be positive: {model_cfg.d_ff}")
        if not (0.0 <= model_cfg.dropout < 1.0):
            raise ValueError(f"dropout out of range: {model_cfg.dropout}")
        if model_cfg.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive: {model_cfg.vocab_size}")

        # Embedding
        self.embedding = XRFMEmbedding(
            vocab_size=model_cfg.vocab_size,
            d_model=model_cfg.d_model,
            weight_tied=weight_tied,
            padding_idx=0,
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
                )
                for _ in range(model_cfg.n_layers)
            ]
        )

        # Final norm
        self.norm_final = (
            RMSNorm(model_cfg.d_model) if model_cfg.use_rmsnorm else nn.LayerNorm(model_cfg.d_model)
        )

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
        """Forward pass with optional KV cache.

        Args:
            input_ids: Token IDs (batch, seq) or (seq,).
            mask: Optional attention mask.
            past_key_values: List of (K_cache, V_cache) per layer from
                previous steps, or None for full forward pass.
            use_cache: Whether to return updated key values for caching.

        Returns:
            (logits, present_key_values) where logits is (batch, seq, vocab_size)
            and present_key_values is list of (K, V) per layer for caching,
            or None if use_cache=False.
        """
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
            past_kv = (
                past_key_values[i]
                if (past_key_values is not None and i < len(past_key_values))
                else None
            )
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
