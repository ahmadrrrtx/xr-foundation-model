"""XRFM-NeuroTopo topology: static graph (Phase 4) and dynamic generator (Phase 5)."""

from xrfm.research.neurotopo.topology.dynamic import DynamicTopology, TopoDiagnostics
from xrfm.research.neurotopo.topology.static_graph import StaticMessagePassing, StaticNeuralGraph

__all__ = [
    "StaticNeuralGraph",
    "StaticMessagePassing",
    "DynamicTopology",
    "TopoDiagnostics",
]
