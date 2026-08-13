"""XRFM-NeuroTopo: sparse dynamic neural-graph language model.

Public interface:
    NeuroTopoConfig, NeuroTopoState, and the component abstract interfaces.
"""
from xrfm.neurotopo.config import (
    MemoryConfig,
    NeuroTopoConfig,
    TopologyConfig,
    UncertaintyConfig,
)
from xrfm.neurotopo.interfaces import (
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

__all__ = [
    "NeuroTopoConfig",
    "TopologyConfig",
    "MemoryConfig",
    "UncertaintyConfig",
    "NeuroTopoState",
    "NeuralModule",
    "Dynamics",
    "TopologyGenerator",
    "MessageFunction",
    "MessagePassingEngine",
    "MemorySystem",
    "UncertaintyHead",
    "NeuroTopoBlock",
    "NeuroTopoModel",
]

from xrfm.neurotopo.input_layer import NeuroTopoInput

__all__.append("NeuroTopoInput")
