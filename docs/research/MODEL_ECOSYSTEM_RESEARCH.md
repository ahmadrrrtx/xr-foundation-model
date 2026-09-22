# Open-weight model ecosystem research for XRFM

**Date:** 2026-09-22. This document is a design input, not a claim that XRFM trained any of these models.

## Executive decision

XRFM should not block the runtime on a new large pretraining run. The runtime should expose a model-neutral `ModelBackend` and initially support:

1. **ScriptedBackend** for deterministic tests and a no-model local demo.
2. **NativeXRFMBackend** for the existing PyTorch model, clearly marked research/limited.
3. **OpenAICompatibleBackend** for llama.cpp, Ollama, vLLM, or hosted BYOK endpoints.

The default reference model for serious local agent experiments is a **Qwen-family instruct model** (Qwen3 1.7B/4B where available; Qwen2.5-Instruct is a compatibility fallback). Qwen is attractive because current releases have native chat/tool templates and permissive Apache 2.0 licensing for the relevant families. Model weights are not bundled in XRFM; users download them separately and retain the upstream license and notices.

## Comparison

| Family | Typical useful local sizes | Architecture / context | Tool calling | Local runtimes | License posture | XRFM decision |
|---|---:|---|---|---|---|---|
| Qwen3 | 0.6B, 1.7B, 4B, 8B+ | Dense and MoE variants; long context in current releases | Native chat templates; quality depends strongly on size | Transformers, llama.cpp GGUF, Ollama, vLLM | Apache 2.0 for the relevant Qwen3 releases; verify exact model card | **Primary adapter target** |
| Qwen2.5 Instruct | 0.5B–72B | Dense decoder Transformer; long-context variants | Mature tool-use templates and broad serving support | Transformers, llama.cpp, Ollama, vLLM | Qwen license; review exact repository terms before redistribution | Compatibility fallback; used in the local smoke demo |
| Llama 3.x | 1B/3B/8B/70B+ | Dense decoder, GQA at larger sizes, long context | Strong ecosystem; templates vary by checkpoint | Transformers, llama.cpp, Ollama, vLLM | Llama Community License, attribution/naming and scale conditions | Supported through generic adapter; not bundled |
| Mistral / Ministral | 3B/7B/Small+ | Dense, GQA/SWA in family variants | Native function calling in instruct releases | Transformers, llama.cpp, Ollama, vLLM | Apache 2.0 for many base releases; some code models have different terms | Good permissive alternative |
| Gemma 2/3 | 270M–27B | Dense, RMSNorm/GQA variants; multimodal in some releases | Structured output/tool support varies by checkpoint | Transformers, llama.cpp, Ollama, vLLM | Google Gemma Terms of Use, not plain Apache/MIT | Use only after license review |
| Phi-4 / Phi-4-mini | ~3.8B/14B | Dense decoder; strong small-model reasoning | Can follow structured prompts; native tool behavior varies | Transformers, llama.cpp, Ollama | MIT for relevant releases; verify model card | CPU-friendly option |
| DeepSeek distilled models | 1.5B–70B | Distilled reasoning models based on other families | Strong reasoning; tool formatting is checkpoint-dependent | Transformers, llama.cpp, Ollama, vLLM | MIT for many distilled releases; verify base/model terms | Useful reasoning backend, not first default |
| OLMo 2 | 1B–32B | Open decoder Transformer; fully open research artifacts | Less standardized native tool use | Transformers, llama.cpp where converted | Apache 2.0 | Best reproducibility reference, not first tool model |
| Pythia | 70M–12B | GPT-NeoX-style decoder | Base models; no native tool specialization | Transformers, some GGUF | Apache 2.0 | Research/control only |

## Engineering implications

* Tool reliability is a property of **model + chat template + decoder + runtime parser**. A model name alone is not a guarantee.
* Native tool calling must be distinguished from XR runtime-enforced JSON. A generic model can emit a valid XR protocol object because XR validates/retries it; that does not mean the model was trained for function calling.
* Constrained decoding is preferred where available: llama.cpp GBNF, vLLM/SGLang guided decoding, or an equivalent provider strict-schema mode. The first XR implementation uses post-generation parsing and validation so it remains dependency-light; a grammar hook is a documented next step.
* At Q4, a 0.5B model is feasible on CPU but weak for multi-step planning; 1.7B–4B is a better minimum for agent experiments. A 7B/8B class model is the practical quality/latency balance on a small GPU.
* `llama.cpp` is the first local systems target because GGUF quantization, CPU execution, and an OpenAI-compatible HTTP surface reduce coupling. `transformers` and vLLM are future optional adapters.

## License and redistribution policy

XRFM's MIT code license does not grant rights to upstream weights. The repository must not commit or bundle model weights. A distribution may reference a model identifier and download instructions, but must preserve the upstream license, model card, acceptable-use policy, attribution, and any derivative restrictions. A BYOK endpoint is the safest deployment option when redistribution terms are unclear.

## Sources

* Hugging Face Transformers tool-use guide: https://huggingface.co/docs/transformers/chat_extras
* Qwen/Hugging Face model cards and templates: https://huggingface.co/Qwen
* Llama model cards: https://huggingface.co/meta-llama
* Model Context Protocol architecture: https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture
* Local model comparison research was used as a lead, but exact license text and model cards remain authoritative.

## Non-claims

No benchmark number in this document is presented as an XRFM measurement. Published third-party scores are not substituted for the XR benchmark suite implemented in this project.
