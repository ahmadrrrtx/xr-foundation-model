"""Phase 0 package tests: public API surface, version source, CWD independence."""

import importlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path

import xrfm

REPO_ROOT = Path(__file__).parent.parent


class TestPublicAPI:
    """The documented `from xrfm import ...` surface must work."""

    def test_version_exists(self):
        assert isinstance(xrfm.__version__, str)
        assert xrfm.__version__.count(".") == 2

    def test_version_matches_installed_metadata(self):
        assert importlib.metadata.version("xrfm") == xrfm.__version__

    def test_canonical_imports(self):
        from xrfm import (  # noqa: F401
            XRFM,
            BPETokenizer,
            BytePairEncoder,
            ConfigError,
            ConfigLoader,
            Dataset,
            DatasetConfig,
            GenerationEngine,
            GenerationResult,
            GPTModel,
            ModelConfig,
            TextDataset,
            Tokenizer,
            TokenizerInterface,
            Trainer,
            TrainingConfig,
            TrainingLoop,
            TrainResult,
            XRFMConfig,
            XRFMEmbedding,
            XRFMModel,
            XRFMTextDataset,
            compute_perplexity,
            decode_text,
            default_config,
            encode_text,
            evaluate,
            generate,
            load_config,
        )

    def test_model_aliases_are_the_same_class(self):
        assert xrfm.XRFM is xrfm.XRFMModel is xrfm.GPTModel

    def test_tokenizer_aliases_are_the_same_class(self):
        assert xrfm.BPETokenizer is xrfm.BytePairEncoder
        assert xrfm.Tokenizer is xrfm.TokenizerInterface

    def test_dataset_aliases_are_the_same_class(self):
        assert xrfm.Dataset is xrfm.TextDataset is xrfm.XRFMTextDataset

    def test_all_exports_resolve(self):
        for name in xrfm.__all__:
            assert hasattr(xrfm, name), f"xrfm.__all__ entry {name!r} missing"

    def test_no_fastapi_needed_for_core_import(self):
        """Core must not import API/server dependencies."""
        code = (
            "import sys; import xrfm;"
            "assert 'fastapi' not in sys.modules, 'core imports fastapi';"
            "assert 'uvicorn' not in sys.modules, 'core imports uvicorn'"
        )
        subprocess.run([sys.executable, "-c", code], check=True, cwd="/")


class TestCWDIndependence:
    """Phase 0 #6: nothing may require CWD == repository root."""

    def test_import_and_config_outside_repo(self):
        code = (
            "import xrfm;"
            "cfg = xrfm.load_config();"
            "c = xrfm.ConfigLoader();"
            "assert c.get('model.d_model') == cfg.model.d_model;"
            "print('ok', cfg.model.d_model)"
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd="/", capture_output=True, text=True, check=True)
        assert "ok" in proc.stdout

    def test_default_config_is_packaged_resource(self):
        from xrfm.config import packaged_default_config_path

        path = Path(packaged_default_config_path())
        assert path.is_file(), path
        assert "site-packages" in str(path) or "src" in str(path) or "src" in str(path.resolve())

    def test_packaged_tokenizer_loads_outside_repo(self):
        code = (
            "from xrfm.tokenization import BPETokenizer;"
            "t = BPETokenizer.pretrained();"
            "ids = t.encode('hello');"
            "assert t.decode(ids) == 'hello';"
            "assert t.pad_token_id is not None"
        )
        subprocess.run([sys.executable, "-c", code], cwd="/", check=True)


class TestNoCircularImports:
    """Importing any subsystem alone must not explode (no import cycles)."""

    def test_subsystem_imports_isolated(self):
        for mod in (
            "xrfm.config",
            "xrfm.tokenization",
            "xrfm.data",
            "xrfm.models",
            "xrfm.training",
            "xrfm.inference",
            "xrfm.evaluation",
            "xrfm.optimization",
            "xrfm.search",
            "xrfm.research.neurotopo",
        ):
            importlib.import_module(mod)
