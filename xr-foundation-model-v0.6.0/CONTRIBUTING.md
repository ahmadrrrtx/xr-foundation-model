# Contributing to XR Foundation Model

We welcome contributions from researchers, engineers, and the open-source community.

## Philosophy

This is not a tutorial. This is an open-source foundation model platform designed for long-term scalability. Every contribution must uphold production-quality standards.

## How to Contribute

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Write or update tests (`tests/`).
4. Ensure type hints and docstrings are present.
5. Update `CHANGELOG.md` with a brief description.
6. Submit a Pull Request with a clear title and reference to related issues or design docs.

## Code Standards

- Type hints on all public functions.
- Docstrings for all modules, classes, and functions.
- No circular dependencies.
- Config-driven: no hard-coded hyperparameters.
- Original implementation: do not copy code line-for-line from existing tutorials. If inspired by a concept (e.g., from "LLMs from Scratch" by Sebastian Raschka or "nanoGPT" by Andrej Karpathy), document it explicitly in comments with citation.

## Testing

Run before submitting:

```bash
python -m pytest tests/
```

Every module must include at least basic unit tests.

## References and Attribution

This project is inspired by core research concepts from:
- Vaswani et al. (Attention Is All You Need)
- Karpathy (nanoGPT — conceptual reference for small-scale training loops)
- Raschka (LLMs-from-Scratch — conceptual reference for tokenizer and architecture design)
- DeepSeek-AI (sparse attention and MoE architecture concepts)
- Meta AI (Llama architecture: RoPE, RMSNorm, SwiGLU, GQA)

No source code is copied directly. All implementations are original.
