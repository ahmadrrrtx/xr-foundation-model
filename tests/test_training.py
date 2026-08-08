"""
Tests for training engine modules (v0.5.1).

Covers optimizer, scheduler, checkpoint, mixed precision, and training loop
with proper next-token prediction targets.
"""

import os
import tempfile

import pytest
import torch

from training.checkpoint import CheckpointLoader
from training.loop import TrainingLoop
from training.mixed_precision import MixedPrecisionLoader, NoOpScaler
from training.optimizer import OptimizerLoader
from training.scheduler import SchedulerLoader

# --- Shared fixtures ---


class MiniDataset:
    """Dataset returning (input_ids, target_ids) tuples for training tests."""

    def __init__(self, vocab_size=100, seq_len=16, size=50):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return ids[:-1], ids[1:]  # input, shifted target


# --- Optimizer ---


class TestOptimizerLoader:
    def test_adamw_init(self):
        model = torch.nn.Linear(10, 2)
        loader = OptimizerLoader(model.parameters(), learning_rate=0.001, weight_decay=0.01)
        assert loader.optimizer is not None
        assert loader.learning_rate == 0.001

    def test_adamw_step(self):
        model = torch.nn.Linear(5, 2)
        loader = OptimizerLoader(model.parameters(), learning_rate=0.01)
        loader.zero_grad()
        x = torch.randn(1, 5)
        loss = model(x).sum()
        loss.backward()
        loader.step()
        for param in model.parameters():
            assert torch.isfinite(param).all()

    def test_invalid_lr(self):
        model = torch.nn.Linear(4, 1)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            OptimizerLoader(model.parameters(), learning_rate=-0.1)

    def test_invalid_weight_decay(self):
        model = torch.nn.Linear(4, 1)
        with pytest.raises(ValueError, match="weight_decay must be non-negative"):
            OptimizerLoader(model.parameters(), weight_decay=-0.01)


# --- Scheduler ---


class TestSchedulerLoader:
    def test_cosine_warmup(self):
        opt = torch.optim.AdamW([torch.randn(2, 2, requires_grad=True)], lr=0.1)
        sched = SchedulerLoader(opt, base_lr=0.1, warmup_steps=5, max_steps=50)
        assert sched.get_lr() == 0.0
        for _ in range(3):
            sched.step()
        assert abs(sched.get_lr() - 0.06) < 1e-6

    def test_invalid_warmup(self):
        opt = torch.optim.AdamW([torch.randn(2, 2, requires_grad=True)], lr=0.1)
        with pytest.raises(ValueError, match="warmup_steps must be positive"):
            SchedulerLoader(opt, base_lr=0.1, warmup_steps=0)

    def test_invalid_max_steps(self):
        opt = torch.optim.AdamW([torch.randn(2, 2, requires_grad=True)], lr=0.1)
        with pytest.raises(ValueError, match="max_steps must be positive"):
            SchedulerLoader(opt, base_lr=0.1, max_steps=0)

    def test_warmup_ge_max_steps(self):
        opt = torch.optim.AdamW([torch.randn(2, 2, requires_grad=True)], lr=0.1)
        with pytest.raises(ValueError, match="warmup_steps.*less than max_steps"):
            SchedulerLoader(opt, base_lr=0.1, warmup_steps=50, max_steps=30)


# --- Checkpoint ---


class TestCheckpointLoader:
    def test_save_load_roundtrip(self):
        model = torch.nn.Linear(10, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
        loader = CheckpointLoader(checkpoint_dir=tempfile.mkdtemp())
        path = loader.save_checkpoint(model, opt, step=100, loss=2.5, best_loss=2.0)
        assert os.path.exists(path)
        meta = loader.load_checkpoint(path, model, opt)
        assert meta["step"] == 100
        assert meta["loss"] == 2.5

    def test_invalid_path(self):
        loader = CheckpointLoader()
        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            loader.load_checkpoint(
                "nonexistent.pt",
                torch.nn.Linear(4, 1),
                torch.optim.AdamW([torch.randn(2, 2)]),
            )

    def test_invalid_dir_type(self):
        with pytest.raises(ValueError, match="checkpoint_dir must be str"):
            CheckpointLoader(checkpoint_dir=123)


# --- Mixed Precision ---


class TestMixedPrecisionLoader:
    def test_noop_scaler_identity(self):
        loader = MixedPrecisionLoader(enabled=False)
        loss = torch.randn(2, 2, requires_grad=True).sum()
        scaled = loader.scale(loss)
        assert torch.equal(scaled, loss)

    def test_noop_scaler_step(self):
        scaler = NoOpScaler()
        opt = torch.optim.AdamW([torch.randn(2, 2, requires_grad=True)], lr=0.01)
        loss = torch.randn(1, requires_grad=True).sum()
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.unscale_(opt)
        assert scaler.step(opt) is True
        scaler.update()
        for param in opt.param_groups[0]["params"]:
            assert torch.isfinite(param).all()


# --- Training Loop (v0.5.1) ---


class TestTrainingLoop:
    def test_init_from_config(self):
        from model.gpt import GPTModel

        model = GPTModel()
        dataset = MiniDataset()
        loop = TrainingLoop(
            config_path="config/config.yaml",
            model=model,
            dataset=dataset,
        )
        assert loop.current_step == 0
        assert loop.best_loss == float("inf")
        assert loop.max_steps > 0  # from config
        assert loop.gradient_clip == 1.0
        assert loop.mixed_precision_loader.enabled is loop.config_loader.get("training.mixed_precision", False)

    def test_train_step_with_targets(self):
        from model.gpt import GPTModel

        model = GPTModel()
        dataset = MiniDataset(vocab_size=100, seq_len=16)
        loop = TrainingLoop(
            config_path="config/config.yaml",
            model=model,
            dataset=dataset,
        )
        bi = torch.randint(0, 100, (2, 15))
        bt = torch.randint(0, 100, (2, 15))
        metrics = loop.train_step(bi, bt)
        assert "loss" in metrics
        assert "learning_rate" in metrics
        assert metrics["step"] == 1

    def test_training_loop_runs(self):
        from model.gpt import GPTModel

        model = GPTModel()
        dataset = MiniDataset(vocab_size=100, seq_len=8, size=20)
        loop = TrainingLoop(
            config_path="config/config.yaml",
            model=model,
            dataset=dataset,
        )
        result = loop.training_loop(max_steps=2, checkpoint_every=100, log_interval=100)
        assert result["final_step"] == 2
        assert result["final_loss"] > 0

    def test_requires_model(self):
        with pytest.raises(ValueError, match="model is required"):
            TrainingLoop(dataset=MiniDataset())

    def test_requires_dataset(self):
        from model.gpt import GPTModel

        model = GPTModel()
        with pytest.raises(ValueError, match="dataset is required"):
            TrainingLoop(model=model)
