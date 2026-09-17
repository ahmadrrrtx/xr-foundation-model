"""Phase 0 backward-compatibility shim tests.

The pre-Phase-0 import paths must keep working inside a repository
checkout. These shims are NOT part of the wheel; removal is planned no
earlier than v2.0 (see docs/adr/0001-xrfm-core-architecture.md).
"""

import importlib

import pytest

OLD_PATHS = [
    # v1.x core stack (repo-root shims)
    "model",
    "model.gpt",
    "model.embedding",
    "model.attention.multi_head",
    "model.attention.rope",
    "model.layers.rmsnorm",
    "model.layers.swiglu",
    "model.layers.transformer_block",
    "tokenizer",
    "tokenizer.bpe",
    "tokenizer.interface",
    "tokenizer.encode",
    "tokenizer.decode",
    "training",
    "training.loop",
    "training.optimizer",
    "training.scheduler",
    "training.checkpoint",
    "training.distributed",
    "training.mixed_precision",
    "training.metrics",
    "inference",
    "inference.engine",
    "inference.kv_cache",
    "inference.sampling",
    "evaluation",
    "evaluation.perplexity",
    "evaluation.benchmarks",
    "optimization",
    "optimization.flash_attention",
    "optimization.quantization",
    "optimization.speculative_decoding",
    # NeuroTopo research stack (in-package shims)
    "xrfm.core",
    "xrfm.neurotopo",
    "xrfm.topology",
    "xrfm.memory",
    "xrfm.dynamics",
    "xrfm.neurons",
    "xrfm.uncertainty",
    "xrfm.nt_training",
    "xrfm.nt_evaluation",
]


@pytest.mark.parametrize("modname", OLD_PATHS)
def test_old_path_imports(modname):
    with pytest.warns(DeprecationWarning):
        mod = importlib.import_module(modname)
    assert mod is not None


def test_shim_reexports_are_the_canonical_objects():
    from model.gpt import GPTModel as ShimGPT
    from tokenizer.bpe import BytePairEncoder as ShimBPE
    from xrfm.models import GPTModel as CanonGPT
    from xrfm.tokenization import BytePairEncoder as CanonBPE

    assert ShimGPT is CanonGPT
    assert ShimBPE is CanonBPE


def test_neurotopo_shim_reexports():
    from xrfm.core import NeuroTopoModel as ShimNT
    from xrfm.research.neurotopo import NeuroTopoModel as CanonNT

    assert ShimNT is CanonNT
