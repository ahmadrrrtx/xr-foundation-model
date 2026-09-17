# ruff: noqa: F401, F403  — compat shim: intentional re-exports
"""Deprecated path: ``neurotopo`` → :mod:`xrfm.research.neurotopo`.

NeuroTopo research code moved to ``xrfm/research/neurotopo`` in Phase 0.
This shim keeps old imports working and will be removed no earlier than
v2.0. Import from ``xrfm.research.neurotopo`` instead.
"""

import warnings

warnings.warn(
    "neurotopo is deprecated; import from xrfm.research.neurotopo instead",
    DeprecationWarning,
    stacklevel=2,
)

from xrfm.research.neurotopo.input_layer import NeuroTopoInput
