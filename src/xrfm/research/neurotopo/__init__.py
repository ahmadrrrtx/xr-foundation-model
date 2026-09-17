"""XRFM-NeuroTopo — experimental research model (NOT the core XRFM LM).

This package isolates the NeuroTopo research architecture (sparse dynamic
neural-graph language model with associative memory) from the stable core
(``xrfm.models``, ``xrfm.training``, ...). It is **experimental**: interfaces
here may change without notice, nothing in core XRFM depends on it, and it
is not covered by the public-API compatibility policy.

Status and scope: see ``docs/architecture/DEAD_OR_EXPERIMENTAL.md`` and
``docs/architecture.md`` ("out of scope" section).

Core entry points::

    from xrfm.research.neurotopo import NeuroTopoModel, NeuroTopoConfig
    model = NeuroTopoModel(NeuroTopoConfig(vocab_size=..., d_model=..., ...))
"""

from xrfm.research.neurotopo.config import (
    MemoryConfig,
    NeuroTopoConfig,
    TopologyConfig,
    UncertaintyConfig,
)
from xrfm.research.neurotopo.interfaces import (
    Dynamics,
    MemorySystem,
    MessageFunction,
    MessagePassingEngine,
    NeuralModule,
    NeuroTopoBlock,
    NeuroTopoModel,
    NeuroTopoState,
    TopologyGenerator,
    UncertaintyHead,
)
from xrfm.research.neurotopo.state import NTState
from xrfm.research.neurotopo.input_layer import NeuroTopoInput
from xrfm.research.neurotopo.message_passing import DynamicMessagePassing, NeuroTopoBlock
from xrfm.research.neurotopo.model import ModelOutput, NeuroTopoModel

__all__ = [
    "Dynamics",
    "DynamicMessagePassing",
    "NTState",
    "MemoryConfig",
    "MemorySystem",
    "MessageFunction",
    "MessagePassingEngine",
    "ModelOutput",
    "NeuralModule",
    "NeuroTopoBlock",
    "NeuroTopoConfig",
    "NeuroTopoInput",
    "NeuroTopoModel",
    "NeuroTopoState",
    "TopologyConfig",
    "TopologyGenerator",
    "UncertaintyConfig",
    "UncertaintyHead",
]
