"""Phase 0 backward-compatibility shim tests.

The pre-Phase-0 import paths must keep working inside a repository
checkout. Root-level shims (``model``, ``tokenizer``, ``training``,
``inference``, ``evaluation``, ``optimization``) were removed at v2.0
(see docs/adr/0001-xrfm-core-architecture.md Section D7). In-package
shims (``xrfm.core``, ``xrfm.neurotopo``, etc.) remain as cheap re-exports
and are still tested here.
"""

import importlib

import pytest

OLD_PATHS = [
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


def test_neurotopo_shim_reexports():
    from xrfm.core import NeuroTopoModel as ShimNT
    from xrfm.research.neurotopo import NeuroTopoModel as CanonNT

    assert ShimNT is CanonNT
