"""XRFM model subsystem.

Public surface: :class:`XRFMModel` (alias ``GPTModel`` for backward
compatibility). Internal layers live in ``xrfm.models.attention`` /
``xrfm.models.layers`` — import them only if you extend the architecture.
"""

from xrfm.models.embedding import XRFMEmbedding
from xrfm.models.gpt import GPTModel, XRFMModel

__all__ = ["GPTModel", "XRFMEmbedding", "XRFMModel"]
