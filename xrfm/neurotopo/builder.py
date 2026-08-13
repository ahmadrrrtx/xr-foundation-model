"""XRFM-NeuroTopo config-based model builder (Phase 14)."""

from __future__ import annotations

import torch

from xrfm.core import NeuroTopoModel
from xrfm.neurotopo.config import NeuroTopoConfig
from xrfm.uncertainty import UncertaintyHead


def build_neurotopo_model(
    config: NeuroTopoConfig,
    embedding_weight: torch.Tensor | None = None,
    with_uncertainty: bool = True,
) -> tuple[NeuroTopoModel, UncertaintyHead | None]:
    """Construct an XRFM-NT model (and optional uncertainty head) from config.

    The uncertainty head is kept separate from the LM forward pass so it can be
    trained on labeled abstention data without affecting next-token training.
    """
    model = NeuroTopoModel(config, embedding_weight=embedding_weight)
    head = UncertaintyHead() if (with_uncertainty and config.uncertainty.enabled) else None
    return model, head
