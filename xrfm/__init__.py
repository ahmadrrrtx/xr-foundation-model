"""
xrfm — XR Foundation Model Python package.

Original open-source foundation model platform.
Built from scratch in pure PyTorch.
Designed for scalability: 10M → 1B → 7B → MoE.

Conceptual references (not copied):
- Vaswani et al. (2017) — Attention mechanism
- Karpathy (2023) — nanoGPT (conceptual training loop reference)
- Raschka (2024) — LLMs-from-Scratch (conceptual tokenizer/architecture)
- Meta AI — Llama 3 architecture (RoPE, RMSNorm, SwiGLU, GQA concepts)
- DeepSeek-AI (2024) — Sparse attention and MoE architecture concepts

All implementations are original.
"""

from xrfm.config.loader import ConfigLoader, ConfigPresets

__version__ = "0.1.0"
__package_name__ = "xrfm"
