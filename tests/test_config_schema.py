"""Phase 0 configuration tests: validation, cross-field checks, serialization."""

from pathlib import Path

import pytest
import yaml

from xrfm.config import ConfigError, DatasetConfig, ModelConfig, TrainingConfig, XRFMConfig, load_config

REPO_ROOT = Path(__file__).parent.parent


def valid_model(**over):
    base = dict(
        vocab_size=2048,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        max_seq_len=256,
        dropout=0.1,
        use_rope=True,
        use_rmsnorm=True,
        use_swiglu=True,
        use_bias=True,
    )
    base.update(over)
    return ModelConfig(**base)


class TestModelConfigValidation:
    def test_valid_config_constructs(self):
        cfg = valid_model()
        assert cfg.d_model % cfg.n_heads == 0

    @pytest.mark.parametrize("field", ["vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "max_seq_len"])
    def test_positive_int_fields(self, field):
        with pytest.raises(ConfigError, match=field):
            valid_model(**{field: 0})
        with pytest.raises(ConfigError, match=field):
            valid_model(**{field: -4})

    def test_dropout_range(self):
        with pytest.raises(ConfigError, match="dropout"):
            valid_model(dropout=1.0)
        with pytest.raises(ConfigError, match="dropout"):
            valid_model(dropout=-0.1)

    def test_cross_field_divisibility(self):
        with pytest.raises(ConfigError, match="divisible"):
            valid_model(d_model=100, n_heads=6)

    def test_error_message_names_fields(self):
        with pytest.raises(ConfigError) as ei:
            valid_model(d_model=100, n_heads=6)
        assert "model.d_model" in str(ei.value)
        assert "model.n_heads" in str(ei.value)
        assert "100" in str(ei.value)

    def test_pad_token_id_bounds(self):
        cfg = valid_model(pad_token_id=2047)
        assert cfg.pad_token_id == 2047
        with pytest.raises(ConfigError, match="pad_token_id"):
            valid_model(pad_token_id=2048)
        with pytest.raises(ConfigError, match="pad_token_id"):
            valid_model(pad_token_id=-1)


class TestTrainingConfigValidation:
    def test_valid_default(self):
        cfg = TrainingConfig()
        assert cfg.learning_rate > 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"learning_rate": 0},
            {"learning_rate": -0.1},
            {"weight_decay": -1},
            {"batch_size": 0},
            {"max_steps": -5},
            {"warmup_steps": -1},
            {"grad_accum_steps": 0},
        ],
    )
    def test_invalid_values_raise(self, kwargs):
        with pytest.raises(ConfigError):
            TrainingConfig(**kwargs)

    def test_cross_field_warmup_lt_max_steps(self):
        with pytest.raises(ConfigError, match="warmup"):
            TrainingConfig(warmup_steps=100, max_steps=50)

    def test_ddp_fsdp_mutually_exclusive(self):
        with pytest.raises(ConfigError, match="mutually exclusive"):
            TrainingConfig(use_ddp=True, use_fsdp=True)

    def test_prefetch_auto_when_workers(self):
        cfg = TrainingConfig(num_workers=4, prefetch_factor=None)
        assert cfg.prefetch_factor == 2


class TestDatasetConfigValidation:
    def test_ratio_sum(self):
        with pytest.raises(ConfigError, match="sum"):
            DatasetConfig(train_ratio=0.9, val_ratio=0.9, test_ratio=0.9)

    def test_ratio_bounds(self):
        with pytest.raises(ConfigError):
            DatasetConfig(train_ratio=1.5, val_ratio=0.0, test_ratio=-0.5)

    def test_max_seq_len(self):
        with pytest.raises(ConfigError, match="max_seq_len"):
            DatasetConfig(max_seq_len=0)


class TestXRFMConfig:
    def test_full_config_roundtrip(self):
        cfg = XRFMConfig(model=valid_model())
        d = cfg.to_dict()
        cfg2 = XRFMConfig.from_dict(d)
        assert cfg.to_dict() == cfg2.to_dict()

    def test_cross_subconfig_seq_len_match(self):
        with pytest.raises(ConfigError, match="max_seq_len"):
            XRFMConfig(model=valid_model(max_seq_len=256), data=DatasetConfig(max_seq_len=128))

    def test_unknown_top_level_section_rejected(self):
        with pytest.raises(ConfigError, match="unknown.*section|Unknown"):
            load_config_from_dict({"model": valid_model().to_dict(), "bogus": {}})

    def test_unknown_model_field_rejected(self):
        bad = valid_model().to_dict()
        bad["d_modell"] = 5  # typo
        with pytest.raises(ConfigError, match="d_modell"):
            ModelConfig.from_dict(bad)

    def test_replace_nested(self):
        cfg = XRFMConfig(model=valid_model())
        cfg2 = cfg.replace(**{"model.d_model": 256, "training.max_steps": 10, "training.warmup_steps": 2})
        assert cfg2.model.d_model == 256
        assert cfg2.training.max_steps == 10
        assert cfg.model.d_model == 128  # original untouched
        with pytest.raises(ConfigError):
            cfg.replace(**{"model.nope": 1})


def load_config_from_dict(d):
    import tempfile

    from xrfm.config import load_config

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(d, f)
        return load_config(f.name)


class TestYAMLLoading:
    def test_repo_presets_all_load(self):
        for name in ("config.yaml", "tiny.yaml", "medium.yaml", "v1.1-medium.yaml"):
            cfg = load_config(REPO_ROOT / "config" / name)
            assert cfg.model.vocab_size > 0
            assert cfg.training.learning_rate > 0

    def test_repo_presets_have_no_version_field(self):
        """Version lives only in xrfm.__version__ (Phase 0 D8)."""
        for name in ("config.yaml", "tiny.yaml", "medium.yaml", "v1.1-medium.yaml"):
            with open(REPO_ROOT / "config" / name) as f:
                raw = yaml.safe_load(f)
            assert "version" not in raw.get("project", {}), name

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_invalid_yaml(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("model: [unclosed\n  - broken")
            path = f.name
        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_config(path)

    def test_empty_yaml(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("")
            path = f.name
        with pytest.raises(ConfigError, match="empty"):
            load_config(path)

    def test_invalid_value_fails_with_field_name(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump({"model": valid_model().to_dict() | {"d_model": 100, "n_heads": 6}}, f)
            path = f.name
        with pytest.raises(ConfigError, match="divisible"):
            load_config(path)
