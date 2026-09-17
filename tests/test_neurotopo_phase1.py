# ruff: noqa: E702, N802, N803, F841  — research bundle tests (pre-Phase-0 style)
"""Phase 1 tests: NeuroTopo package architecture and interfaces exist and import.

These verify only module boundaries and contracts (no complex behavior yet).
The existing Transformer control must remain importable and unaffected.
"""

from __future__ import annotations

import importlib

import torch

import xrfm.research.neurotopo as nt
from xrfm.research.neurotopo import NeuroTopoConfig, NeuroTopoState


def test_all_component_packages_import():
    packages = [
        "xrfm.core",
        "xrfm.topology",
        "xrfm.neurons",
        "xrfm.dynamics",
        "xrfm.memory",
        "xrfm.routing",
        "xrfm.uncertainty",
        "xrfm.knowledge",
        "xrfm.neurotopo",
        "xrfm.nt_training",
        "xrfm.nt_evaluation",
    ]
    for name in packages:
        mod = importlib.import_module(name)
        assert mod is not None, f"package failed to import: {name}"


def test_public_interface_symbols_present():
    expected = [
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
        "NeuroTopoConfig",
        "TopologyConfig",
        "MemoryConfig",
        "UncertaintyConfig",
    ]
    for sym in expected:
        assert hasattr(nt, sym), f"missing public symbol: {sym}"


def test_config_defaults_are_v1_minimal():
    cfg = NeuroTopoConfig()
    # v1 staged extensions must default OFF (mission Rule 5)
    assert cfg.concept_graph is False
    assert cfg.moe_routing is False
    assert cfg.tda_reg is False
    assert cfg.hybrid_attention_every == 0
    assert cfg.topology.n_modules > 0
    assert cfg.topology.longrange_topk < cfg.topology.n_modules  # sparse


def test_state_dataclass_holds_components():
    H = torch.zeros(2, 4, 8)
    M = torch.zeros(2, 8, 8)
    s = NeuroTopoState(H=H, A=None, M=M, C=None)
    assert s.H.shape == (2, 4, 8)
    assert s.M.shape == (2, 8, 8)
    sd = s.detach()
    assert sd.H.shape == s.H.shape


def test_control_transformer_still_imports_and_builds():
    """Phase 1 must not disturb CONTROL-1."""
    from xrfm.models.gpt import GPTModel

    model = GPTModel(config_path="config/tiny.yaml", vocab_size=1024)
    x = torch.randint(0, 1024, (1, 16))
    logits, _ = model(x)
    assert logits.shape == (1, 16, 1024)
