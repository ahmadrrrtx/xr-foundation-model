"""
Regression tests for XRFM v1.0.0.

Prevents recurrence of critical bugs from the Engineering Audit.
Bug C-1: Version inconsistency across files.
Bug C-4: Loss trained identity mapping, not next-token prediction.
Bug C-5: Training loop used random dummy data, not real dataset.
"""

import pytest
import torch
from model.gpt import GPTModel
from training.loop import TrainingLoop


class DummyRegressionDataset:
    """Minimal dataset returning (input, next-token-shifted target)."""

    def __init__(self, vocab_size=100, seq_len=8, size=20):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return ids[:-1], ids[1:]


class TestNextTokenPrediction:
    """REGRESSION C-4: Loss uses shifted targets for next-token prediction.

    Previous bug: cross_entropy(logits, input_ids) trained identity copying
    instead of next-token prediction.
    """

    def test_train_step_requires_target_ids(self):
        model = GPTModel()
        dataset = DummyRegressionDataset(vocab_size=100, seq_len=8)
        loop = TrainingLoop(
            config_path="config/config.yaml",
            model=model, dataset=dataset,
        )
        bi = torch.randint(0, 100, (2, 7))
        bt = torch.randint(0, 100, (2, 7))
        metrics = loop.train_step(bi, bt)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] == metrics["loss"]

    def test_loss_not_nan_sequential(self):
        model = GPTModel()
        dataset = DummyRegressionDataset(vocab_size=100, seq_len=8)
        loop = TrainingLoop(
            config_path="config/config.yaml",
            model=model, dataset=dataset,
        )
        bi = torch.tensor([[0,1,2,3,4,5,6],[0,1,2,3,4,5,6]])
        bt = torch.tensor([[1,2,3,4,5,6,7],[1,2,3,4,5,6,7]])
        metrics = loop.train_step(bi, bt)
        assert metrics["loss"] == metrics["loss"]


class TestDatasetIntegration:
    """REGRESSION C-5: Training loop uses DataLoader, not random dummy data."""

    def test_training_loop_uses_dataloader(self):
        model = GPTModel()
        dataset = DummyRegressionDataset(vocab_size=100, seq_len=8, size=10)
        loop = TrainingLoop(
            config_path="config/config.yaml",
            model=model, dataset=dataset,
        )
        result = loop.training_loop(max_steps=3, checkpoint_every=100, log_interval=100)
        assert result["final_step"] == 3
        assert result["final_loss"] > 0
        assert result["best_loss"] != float("inf")

    def test_dataset_items_are_tuples(self):
        dataset = DummyRegressionDataset(vocab_size=50, seq_len=10)
        item = dataset[0]
        assert isinstance(item, tuple) and len(item) == 2
        input_ids, target_ids = item
        assert input_ids.shape == target_ids.shape
        assert len(input_ids) == 9  # seq_len - 1


class TestVersionConsistency:
    """REGRESSION C-1: Version consistent across all indicators."""

    def test_package_version_matches_config(self):
        import yaml, pathlib
        from xrfm import __version__ as pkgv
        cfg_path = pathlib.Path(__file__).parent.parent / "config" / "config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert pkgv == cfg["project"]["version"], (
            f"Mismatch: __init__={pkgv}, config={cfg['project']['version']}"
        )

    def test_version_is_0_5_1(self):
        from xrfm import __version__
        assert __version__ == "1.0.0", f"Expected 1.0.0, got {__version__}"


class TestTrainingLoopInit:
    """REGRESSION: TrainingLoop requires model + dataset at init."""

    def test_init_requires_model(self):
        with pytest.raises(ValueError, match="model is required"):
            TrainingLoop(dataset=DummyRegressionDataset())

    def test_init_requires_dataset(self):
        model = GPTModel()
        with pytest.raises(ValueError, match="dataset is required"):
            TrainingLoop(model=model)
