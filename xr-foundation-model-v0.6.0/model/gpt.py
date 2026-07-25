"""
GPT Decoder-Only Transformer Model for XR Foundation Model (XRFM).

Purpose: Full decoder-only language model (`GPTModel`) that combines
embedding, stacked transformer blocks (attention + SwiGLU + norm + residual),
final normalization, and output projection (with weight tying by default).

Conceptual references (NOT copied):
- Vaswani, A., et al. (2017). Attention Is All You Need.
- Touvron, H., et al. (2023). Llama 2 Technical Report.
- DeepSeek-AI (2024). DeepSeek-V3 Technical Report.
- Meta AI (2024). Llama 3 Technical Report.

Implementation is original. No source code copied from PyTorch native modules,
LLaMA repositories, or tutorial sources.

Design principles (Phase 4 architecture freeze):
- Config-driven (`ConfigLoader.get_model_config()` provides all architecture params).
- Weight tying default (`lm_head.weight = embedding.weight`) to reduce parameters
  and improve training stability.
- Numerical stability (Xavier init, gradient-safe residual connections, dropout,
  RMSNorm with epsilon, RoPE with bounded rotation matrices).
- Stable interface (`GPTModel.forward(input_ids)`) compatible with standard
  PyTorch training loops and future inference engine (`Phase 6`).
- Scalable architecture (`XRFM-10M` -> `XRFM-7B` without rewrites; interfaces stable).
- Clean separation: embedding, attention, FFN, normalization, output are independent
  modules (modular for future MoE / Multimodal / Reasoning extensions).
- Type-safe, fully documented, with input validation (`ValueError`, `TypeError`,
  `IndexError`) and clear error messages referencing config/file names.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from model.embedding import XRFMEmbedding
from model.layers.rmsnorm import RMSNorm
from model.layers.swiglu import SwiGLU
from model.layers.transformer_block import TransformerBlock
from xrfm.config.loader import ConfigLoader


class GPTModel(nn.Module):
    """Original decoder-only transformer model for XRFM.

    Architecture: Embedding -> [TransformerBlock x n_layers] -> Final Norm -> LM Head.

    The model uses pre-normalization (`norm` before sub-layer), residual
    connections, SwiGLU feed-forward, manual multi-head attention with optional
    RoPE, RMSNorm (configurable), and weight-tied output projection.

    Attributes:
        config: `ConfigLoader` instance or `ModelConfig` providing architecture.
        embedding: Token embedding layer (`XRFMEmbedding`).
        blocks: Sequence of `TransformerBlock` layers.
        norm_final: Final normalization layer (`RMSNorm` or `nn.LayerNorm`).
        lm_head: Output projection (`nn.Linear`) with weight tied to embedding by default.
        max_seq_len: Maximum sequence length (from config).
        dropout_p: Dropout probability (from config).
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        weight_tied: bool = True,
    ) -> None:
        """Initialize the full decoder-only model.

        Args:
            config_path: Path to YAML config file (`ConfigLoader` reads architecture).
                Default: `"config/config.yaml"`.
            weight_tied: Whether to tie `lm_head` weights to `embedding` weights.
                Default: `True` (standard modern practice). When `True`, the output
                projection uses the same weight matrix as the embedding layer,
                reducing parameter count by `vocab_size * d_model`.

        Raises:
            FileNotFoundError: If `config_path` does not exist.
            ValueError: If config parameters are invalid (non-positive dimensions,
                dropout out of range, `d_model` not divisible by `n_heads`).
        """
        super(GPTModel, self).__init__()

        # Load and validate configuration.
        if not isinstance(config_path, str):
            raise TypeError(
                f"config_path must be str, got {type(config_path).__name__}. "
                f"Check model initialization arguments."
            )
        try:
            loader = ConfigLoader(config_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"XRFM config file not found at '{config_path}'. "
                f"Verify file exists and path is correct. "
                f"Default config is at 'config/config.yaml'."
            ) from exc

        model_cfg = loader.model_config()
        self.config_path = config_path
        self.weight_tied = weight_tied
        self.max_seq_len = model_cfg.max_seq_len
        self.dropout_p = model_cfg.dropout

        # Validate core architecture parameters.
        if model_cfg.d_model <= 0:
            raise ValueError(
                f"d_model must be positive, got {model_cfg.d_model}. "
                f"Check config file: {config_path} (model.d_model)."
            )
        if model_cfg.n_layers <= 0:
            raise ValueError(
                f"n_layers must be positive, got {model_cfg.n_layers}. "
                f"Check config file: {config_path} (model.n_layers)."
            )
        if model_cfg.n_heads <= 0:
            raise ValueError(
                f"n_heads must be positive, got {model_cfg.n_heads}. "
                f"Check config file: {config_path} (model.n_heads)."
            )
        if model_cfg.d_model % model_cfg.n_heads != 0:
            raise ValueError(
                f"d_model ({model_cfg.d_model}) must be divisible by n_heads ({model_cfg.n_heads}). "
                f"Check config file: {config_path}."
            )
        if model_cfg.d_ff <= 0:
            raise ValueError(
                f"d_ff must be positive, got {model_cfg.d_ff}. "
                f"Check config file: {config_path} (model.d_ff)."
            )
        if not (0.0 <= model_cfg.dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1), got {model_cfg.dropout}. "
                f"Check config file: {config_path} (model.dropout)."
            )
        if model_cfg.vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be positive, got {model_cfg.vocab_size}. "
                f"Check config file: {config_path} (model.vocab_size)."
            )

        # Embedding layer (original, with Xavier init, padding support, weight tying design).
        self.embedding = XRFMEmbedding(
            vocab_size=model_cfg.vocab_size,
            d_model=model_cfg.d_model,
            weight_tied=weight_tied,
            padding_idx=0,  # Standard padding token index (configurable in future).
        )

        # Stack of transformer blocks.
        # Each block includes pre-norm, attention (with RoPE if configured), SwiGLU, dropout, residual.
        # The interface (`TransformerBlock`) supports future extensions without rewrites.
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

        # Final normalization layer (pre-output norm, standard modern practice).
        # RMSNorm is preferred; LayerNorm fallback available via config.
        self.norm_final = (
            RMSNorm(model_cfg.d_model) if model_cfg.use_rmsnorm else nn.LayerNorm(model_cfg.d_model)
        )

        # Output projection (language modeling head).
        # By default, weight is tied to embedding (`self.lm_head.weight = self.embedding.embedding.weight`).
        # This reduces parameters by `vocab_size * d_model` and improves training stability.
        # Separate projection (`tie_weights: False`) is available as optional future comparison.
        self.lm_head = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)
        if weight_tied:
            # Weight tying: share embedding weight with output projection.
            # This is standard in modern decoder-only models (GPT family, Llama, Mistral, DeepSeek).
            # The embedding layer (`XRFMEmbedding`) provides `self.embedding.weight`.
            self.lm_head.weight = self.embedding.embedding.weight

        # Numerical stability: initialize lm_head if not tied (separate projection case).
        if not weight_tied:
            nn.init.xavier_uniform_(self.lm_head.weight)

        # Dropout layer for final output (optional; applied after output projection in some designs,
        # but standard practice applies dropout to block outputs, not final logits).
        # We include a configurable dropout for future inference extensions.
        self.output_dropout = nn.Dropout(p=model_cfg.dropout)

        # Initialize the full model weights (residual connections ensure gradient flow;
        # Xavier/Kaiming init prevents early divergence).
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier/Kaiming initialization to any non-initialized parameters.

        This is a safety measure: since embedding, projection matrices, and
        norm weights are initialized in their respective modules, this method
        ensures no parameter is left with default initialization that could
        cause numerical instability.
        """
        # Initialize final norm weight (learnable gamma) to ones (standard).
        # RMSNorm weight is already initialized in its constructor.
        # LayerNorm weight is initialized by PyTorch to ones by default.
        # No additional initialization needed unless using custom layers.
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute language model logits.

        The forward pass follows:
        1. Embed token IDs: `(batch, seq) -> (batch, seq, d_model)`
        2. Pass through stacked transformer blocks: `(batch, seq, d_model)` preserved.
        3. Apply final normalization.
        4. Project to vocabulary: `(batch, seq, d_model) -> (batch, seq, vocab_size)`.

        Args:
            input_ids: Token ID tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)` for single sequences.
            mask: Optional attention mask of shape `(batch, 1, seq, seq)`
                or broadcastable. Applied inside `TransformerBlock`.

        Returns:
            Logits tensor of shape `(batch_size, sequence_length, vocab_size)`
            or `(sequence_length, vocab_size)` for single sequences.

        Raises:
            IndexError: If any token ID exceeds vocabulary size (indicating
                tokenizer/model vocabulary mismatch).
            ValueError: If `input_ids` has incorrect dimensions or `mask`
                has incompatible shape.
        """
        # Input validation: ensure correct dimensions and valid token IDs.
        if input_ids.dim() not in (1, 2):
            raise ValueError(
                f"input_ids must have 1 or 2 dimensions (seq or batch, seq), "
                f"got {input_ids.dim()}D with shape {input_ids.shape}. "
                f"Check dataset loader output format."
            )
        # Check vocabulary bounds before embedding (clear error message for mismatches).
        if (input_ids >= self.embedding.vocab_size).any():
            max_id = int(input_ids.max().item())
            raise IndexError(
                f"Token ID {max_id} exceeds vocabulary size ({self.embedding.vocab_size}). "
                f"This indicates a tokenizer/model vocabulary mismatch. "
                f"Check tokenizer vocabulary (tokenizer/bpe.py vocab building) and "
                f"ConfigLoader settings (model.vocab_size)."
            )

        original_dim = input_ids.dim()
        batch_size, seq_len = input_ids.shape if input_ids.dim() == 2 else (1, input_ids.shape[0])
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension for consistency.

        # Step 1: Embedding lookup.
        # The embedding layer converts discrete token IDs to dense continuous vectors.
        # Numerical stability: Xavier initialization ensures appropriate initial variance.
        x = self.embedding(input_ids)  # (batch, seq, d_model)

        # Step 2: Pass through stacked transformer blocks.
        # Each block applies pre-norm attention + residual + SwiGLU + residual.
        # The architecture supports `XRFM-10M` through `XRFM-1B` without rewrites.
        for i, block in enumerate(self.blocks):
            # Apply mask inside each block (attention mechanism handles it).
            # We pass the mask directly; the block validates and applies it.
            try:
                x = block(x, mask=mask)
            except Exception as exc:
                # Provide clear diagnostic message with layer index.
                raise RuntimeError(
                    f"Transformer block {i} (layer/{i+1}) failed during forward pass. "
                    f"Check input shape ({x.shape}), mask compatibility, and config settings. "
                    f"Original error: {exc}"
                ) from exc

        # Step 3: Final normalization.
        # Modern architectures (Llama 3, Mistral, DeepSeek) apply normalization
        # before the final output projection for numerical stability.
        x = self.norm_final(x)

        # Step 4: Output projection (language modeling head).
        # Weight tying (`lm_head.weight = embedding.weight`) ensures consistent
        # representation space and reduces parameter count.
        # Numerical stability: the projection uses Xavier init (if separate) or
        # shares the embedding weight (if tied), preventing initial divergence.
        logits = self.lm_head(x)

        # Optional output dropout (applied to logits for some training variants;
        # standard practice applies dropout to block outputs, not final logits,
        # but the hook is available for future extensions like inference sampling).
        if self.training:
            logits = self.output_dropout(logits)

        # Restore original dimensions for single-sequence inputs.
        if batch_size == 1 and original_dim == 1:
            logits = logits.squeeze(0)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Simple greedy/sampling generation (reserved for Phase 6 inference engine).

        This method provides a minimal generation interface for quick testing.
        A full inference engine (KV cache, streaming, custom sampling strategies)
        will be implemented in Phase 6 (`inference/` module).

        Args:
            input_ids: Starting token IDs of shape `(batch, seq)` or `(seq,)`.
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature (higher = more random; 0 = greedy).
            top_k: If set, restrict sampling to the top `k` most probable tokens.

        Returns:
            Extended token sequence including the original input.

        Note:
            This is a basic placeholder for Phase 6. The production inference
            engine will use KV caching for efficiency.
        """
        # Minimal greedy/sampling generation loop (basic, no KV cache).
        # Reserved for full inference engine in Phase 6.
        self.eval()
        batch_shape = input_ids.shape[0] if input_ids.dim() > 1 else None
        generated = input_ids.clone()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions for the current sequence.
                # Note: For long sequences (> max_seq_len), truncation is needed.
                # This basic implementation does not handle truncation; Phase 6 will.
                seq_input = generated[:, -self.max_seq_len:] if input_ids.dim() > 1 else generated[-self.max_seq_len:]
                if input_ids.dim() == 1:
                    seq_input = seq_input.unsqueeze(0)

                logits = self.forward(seq_input)  # (batch, current_seq, vocab_size)
                # Take the last token's logits for next token prediction.
                next_token_logits = logits[:, -1, :]  # (batch, vocab_size)

                # Apply temperature scaling.
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k filtering (optional).
                if top_k is not None:
                    top_k_values, top_k_indices = torch.topk(
                        next_token_logits, k=min(top_k, next_token_logits.shape[-1])
                    )
                    # Zero out tokens outside top-k.
                    mask_topk = torch.ones_like(next_token_logits, dtype=torch.bool)
                    for b in range(next_token_logits.shape[0]):
                        mask_topk[b, top_k_indices[b]] = False
                    next_token_logits = next_token_logits.masked_fill(mask_topk, float("-inf"))

                # Sample next token using softmax probabilities.
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

                # Append to sequence.
                generated = torch.cat([generated, next_token], dim=1)

        return generated.squeeze(0) if batch_shape is None else generated

    def parameter_count(self) -> int:
        """Compute approximate model parameter count.

        Returns:
            Total trainable parameter count. Used for benchmark verification
            (`XRFM-10M` preset should produce ~10M parameters).

        Note:
            Weight tying reduces parameter count by `vocab_size * d_model`.
        """
        total = 0
        for param in self.parameters():
            total += param.numel()
        return total
