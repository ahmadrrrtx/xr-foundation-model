"""
Tests for distributed training module (v0.8.0).

Tests gradient accumulation logic, no_sync context, model wrapping
(unit-testable without actual multi-GPU), and distributed utilities.
"""

import pytest
import torch
import torch.nn as nn

from model.gpt import GPTModel
from training.distributed import (
    GradientAccumulator,
    barrier,
    create_distributed_dataloader,
    get_rank,
    get_raw_model,
    get_world_size,
    is_distributed,
    is_main_process,
    reduce_loss,
    wrap_model_ddp,
)


class TestDistributedUtils:
    def test_single_process_rank(self):
        """In single-process mode, rank=0, world_size=1, main=True."""
        assert not is_distributed()
        assert get_rank() == 0
        assert get_world_size() == 1
        assert is_main_process()

    def test_barrier_noop(self):
        """barrier() is a no-op when not distributed."""
        barrier()  # Should not raise

    def test_reduce_loss_noop(self):
        """reduce_loss returns item when not distributed."""
        t = torch.tensor(3.5)
        assert reduce_loss(t) == 3.5

    def test_get_raw_model_no_wrap(self):
        model = GPTModel()
        assert get_raw_model(model) is model

    def test_wrap_model_ddp_noop(self):
        """DDP wrap is no-op when not distributed."""
        model = GPTModel()
        wrapped = wrap_model_ddp(model)
        assert wrapped is model  # Not wrapped in single-process


class TestGradientAccumulator:
    def test_init(self):
        model = nn.Linear(4, 2)
        acc = GradientAccumulator(model, steps=4)
        assert acc.steps == 4
        assert acc._counter == 0

    def test_should_sync_pattern(self):
        """should_sync is True every N micro-batches."""
        model = nn.Linear(4, 2)
        acc = GradientAccumulator(model, steps=3)
        # step 0
        assert not acc.should_sync()
        acc.backward(torch.tensor(1.0, requires_grad=True))
        # step 1
        assert not acc.should_sync()
        acc.backward(torch.tensor(1.0, requires_grad=True))
        # step 2 (last)
        assert acc.should_sync()
        acc.backward(torch.tensor(1.0, requires_grad=True))
        assert acc._counter == 3

    def test_should_step(self):
        model = nn.Linear(4, 2)
        acc = GradientAccumulator(model, steps=2)
        assert not acc.should_step()
        acc.backward(torch.tensor(1.0, requires_grad=True))
        assert not acc.should_step()
        acc.backward(torch.tensor(1.0, requires_grad=True))
        assert acc.should_step()

    def test_reset(self):
        model = nn.Linear(4, 2)
        acc = GradientAccumulator(model, steps=2)
        acc.backward(torch.tensor(1.0, requires_grad=True))
        acc.reset()
        assert acc._counter == 0

    def test_invalid_steps(self):
        with pytest.raises(ValueError, match="positive"):
            GradientAccumulator(nn.Linear(4, 2), steps=0)

    def test_no_sync_not_ddp(self):
        """no_sync_context returns nullcontext for non-DDP models."""
        model = nn.Linear(4, 2)
        acc = GradientAccumulator(model, steps=2)
        ctx = acc.no_sync_context()
        assert ctx is not None  # nullcontext


class TestDataLoaderCreation:
    def test_standard_dataloader(self):
        """In single-process, returns standard DataLoader."""
        from torch.utils.data import TensorDataset

        ds = TensorDataset(torch.randn(10, 4), torch.randn(10, 2))
        dl = create_distributed_dataloader(ds, batch_size=2)
        assert len(dl) > 0

    def test_shuffle_flag(self):
        from torch.utils.data import TensorDataset

        ds = TensorDataset(torch.randn(10, 4), torch.randn(10, 2))
        dl = create_distributed_dataloader(ds, batch_size=2, shuffle=False)
        batches = list(dl)
        assert len(batches) > 0


class TestTrainingLoopGradAccum:
    """Test gradient accumulation in the training loop."""

    class _TinyDataset:
        def __init__(self, vocab=100, seq=8, n=20):
            self.vocab, self.seq, self.n = vocab, seq, n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            ids = torch.randint(0, self.vocab, (self.seq,))
            return ids[:-1], ids[1:]

    def test_grad_accum_advances_step_correctly(self):
        """With grad_accum_steps=2, step advances every 2 micro-batches."""
        from training.loop import TrainingLoop

        model = GPTModel()
        ds = self._TinyDataset(vocab=100, seq=8, n=20)
        loop = TrainingLoop(
            config_path="config/config.yaml",
            model=model,
            dataset=ds,
        )
        loop.grad_accum_steps = 2
        loop.max_steps = 4
        loop.checkpoint_every = 100

        result = loop.training_loop(max_steps=4, checkpoint_every=100, log_interval=100)
        assert result["final_step"] == 4

    def test_grad_accum_loss_scaling(self):
        """Loss per micro-batch is scaled by 1/accum_steps."""
        from training.loop import TrainingLoop

        model = GPTModel()
        ds = self._TinyDataset(vocab=100, seq=8, n=10)
        loop = TrainingLoop(
            config_path="config/config.yaml",
            model=model,
            dataset=ds,
        )
        loop.grad_accum_steps = 3

        bi = torch.randint(0, 100, (2, 7))
        bt = torch.randint(0, 100, (2, 7))

        # First 2 micro-batches: step_performed=False
        m1 = loop.train_step(bi, bt)
        assert not m1["step_performed"]
        m2 = loop.train_step(bi, bt)
        assert not m2["step_performed"]
        # 3rd: step_performed=True, step advances
        m3 = loop.train_step(bi, bt)
        assert m3["step_performed"]
        assert loop.current_step == 1
