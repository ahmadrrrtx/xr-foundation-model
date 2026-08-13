"""XRFM-NeuroTopo configuration dataclasses (Phase 1, minimal)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryConfig:
    enabled: bool = False
    d_k: int = 128
    d_v: int = 128
    surprise_gate: bool = True
    persistent_slots: int = 0


@dataclass
class UncertaintyConfig:
    enabled: bool = False
    n_states: int = 5


@dataclass
class TopologyConfig:
    n_modules: int = 64
    module_dim: int = 64
    local_degree: int = 8
    longrange_topk: int = 16
    edge_dim: int = 16
    rank: int = 16
    decay_init: float = 0.9
    plasticity: bool = True
    connectivity_reg: bool = True


@dataclass
class NeuroTopoConfig:
    """Top-level config for an XRFM-NT model.

    v1 intentionally excludes concept graph / hyperedges / TDA / MoE (mission Rule 5).
    """

    vocab_size: int = 2048
    d_model: int = 128
    n_layers: int = 4
    max_seq_len: int = 256
    dropout: float = 0.1
    pad_id: int = 0
    # Sub-configs
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    # Staged extensions — all OFF in v1
    concept_graph: bool = False
    moe_routing: bool = False
    tda_reg: bool = False
    hybrid_attention_every: int = 0
