"""DEPRECATED (Phase 0): import from ``xrfm.inference`` instead.

The library moved into the ``xrfm`` package (``src/xrfm``). This shim keeps
the historical import path working inside a repository checkout. It is NOT
installed with the package (the wheel ships only ``xrfm``) and will be
removed no earlier than v2.0 — see docs/adr/0001-xrfm-core-architecture.md.
"""

import warnings

warnings.warn(
    "The 'inference/__init__' import path is deprecated; import from 'xrfm.inference' instead",
    DeprecationWarning,
    stacklevel=2,
)

from xrfm.inference import GenerationEngine, GenerationResult, KVCache, generate, sample_greedy, sample_temperature, sample_token, sample_top_k, sample_top_p
