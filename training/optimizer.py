"""DEPRECATED (Phase 0): import from ``xrfm.training`` instead.

The library moved into the ``xrfm`` package (``src/xrfm``). This shim keeps
the historical import path working inside a repository checkout. It is NOT
installed with the package (the wheel ships only ``xrfm``) and will be
removed no earlier than v2.0 — see docs/adr/0001-xrfm-core-architecture.md.
"""

import warnings

warnings.warn(
    "The 'training/optimizer' import path is deprecated; import from 'xrfm.training' instead",
    DeprecationWarning,
    stacklevel=2,
)

from xrfm.training.optimizer import OptimizerLoader
