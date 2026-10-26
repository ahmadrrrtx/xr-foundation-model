"""Regression tests for Phase 0 training-execution findings.

These tests intentionally exercise the public/current interfaces so they can
be run against the audited revision before the corresponding fixes land.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from xrfm.config.schema import ModelConfig, TrainingConfig
from xrfm.models.gpt import GPTModel
from xrfm.training.checkpoint import CheckpointLoader
from xrfm.training.distributed import xrfm_collate_fn
from xrfm.training.loop import TrainingLoop, _set_seed
from xrfm.training.mixed_precision import MixedPrecisionLoader
from xrfm.training.optimizer import OptimizerLoader
from xrfm.training.scheduler import SchedulerLoader


class FixedSequenceDataset(Dataset):
    """Deterministic examples with observable order and stable run identity."""

    data_manifest_hash = "sha256:test-data-v1"
    tokenizer_hash = "sha256:test-tokenizer-v1"
    pad_id = 7

    def __init__(self, size: int = 4, seq_len: int = 5) -> None:
        self.items = [
            (
                torch.tensor([(row + col) % 6 + 1 for col in range(seq_len)], dtype=torch.long),
                torch.tensor([(row + col + 1) % 6 + 1 for col in range(seq_len)], dtype=torch.long),
            )
            for row in range(size)
        ]
        self.seen: list[int] = []

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen.append(index)
        return self.items[index]


class RandomSequenceDataset(FixedSequenceDataset):
    """Same identity contract, but item access consumes global Torch RNG."""

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen.append(index)
        ids = torch.randint(1, 7, (6,), dtype=torch.long)
        return ids[:-1], ids[1:]


def tiny_model(seed: int = 123, dropout: float = 0.1) -> GPTModel:
    _set_seed(seed)
    return GPTModel(
        ModelConfig(
            vocab_size=32,
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
            max_seq_len=8,
            dropout=dropout,
            use_rope=False,
            pad_token_id=7,
        )
    )


def training_config(tmp_path: Path, resume_from: str | None = None) -> TrainingConfig:
    return TrainingConfig(
        batch_size=1,
        max_steps=8,
        learning_rate=1e-3,
        warmup_steps=1,
        grad_accum_steps=1,
        checkpoint_every=100,
        eval_every=100,
        num_workers=0,
        device="cpu",
        deterministic=True,
        mixed_precision=False,
        resume_from=resume_from,
        seed=123,
    )


def make_loop(
    tmp_path: Path,
    model: GPTModel,
    dataset: Dataset,
    resume_from: str | None = None,
) -> TrainingLoop:
    return TrainingLoop(
        config=training_config(tmp_path, resume_from),
        model=model,
        dataset=dataset,
        checkpoint_dir=str(tmp_path),
    )


def record_consumed_batches(loop: TrainingLoop) -> list[torch.Tensor]:
    original = loop.train_step
    consumed: list[torch.Tensor] = []

    def wrapped(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        consumed.append(inputs.detach().clone())
        return original(inputs, targets, mask)

    loop.train_step = wrapped  # type: ignore[method-assign]
    return consumed


def assert_nested_equal(actual: Any, expected: Any, path: str = "state") -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_nested_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, value in enumerate(expected):
            assert_nested_equal(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected), path
    else:
        assert actual == expected, path


def test_gradscaler_successful_step_and_scheduler_skip_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real optimizer step must count; a skipped step must not count anywhere."""

    class ScaleAwareScaler:
        def __init__(self, scale: float) -> None:
            self.scale = scale

        def step(self, optimizer: Any) -> None:
            # The old implementation incorrectly relies on this return value.
            optimizer.step()
            return None

        def update(self) -> None:
            pass

        def get_scale(self) -> float:
            return self.scale

        def unscale_(self, optimizer: Any) -> None:
            pass

        def scale(self, loss: torch.Tensor) -> torch.Tensor:
            return loss

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = MixedPrecisionLoader(enabled=False)
    loader.scaler = ScaleAwareScaler(1.0)
    assert loader.step(optimizer) is True

    loop = make_loop(tmp_path, tiny_model(), FixedSequenceDataset())
    monkeypatch.setattr(loop.mixed_precision_loader, "step", lambda _optimizer: False)
    monkeypatch.setattr(loop.mixed_precision_loader, "update", lambda: None)
    batch = loop.dataset[0]
    metrics = loop.train_step(batch[0].unsqueeze(0), batch[1].unsqueeze(0))
    assert metrics["step_performed"] is False
    assert loop.current_step == 0
    assert loop.scheduler.current_step == 0


def test_auto_precision_does_not_require_bf16_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeScaler:
        pass

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *args, **kwargs: FakeScaler())
    loader = MixedPrecisionLoader(enabled=True)
    assert loader._autocast_dtype == torch.float16


def test_oom_is_a_failed_attempt(tmp_path: Path) -> None:
    loop = make_loop(tmp_path, tiny_model(dropout=0.0), FixedSequenceDataset())

    def raise_oom(*args: Any, **kwargs: Any) -> Any:
        raise torch.cuda.OutOfMemoryError("deliberate test allocation failure")

    loop.model.forward = raise_oom  # type: ignore[method-assign]
    inputs, targets = loop.dataset[0]
    with pytest.raises(RuntimeError, match="abort|failed|OOM"):
        loop.train_step(inputs.unsqueeze(0), targets.unsqueeze(0))
    assert loop.current_step == 0


def test_sampler_epoch_is_set_for_each_epoch(tmp_path: Path) -> None:
    class TrackingSampler(SequentialSampler):
        def __init__(self, data_source: Dataset) -> None:
            super().__init__(data_source)
            self.epochs: list[int] = []

        def set_epoch(self, epoch: int) -> None:
            self.epochs.append(epoch)

    dataset = FixedSequenceDataset(size=1)
    sampler = TrackingSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)
    loop = make_loop(tmp_path, tiny_model(dropout=0.0), dataset)
    loop.training_loop(max_steps=2, dataloader=dataloader, checkpoint_every=100)
    assert sampler.epochs == [0, 1]


def test_collate_uses_explicit_pad_contract() -> None:
    batch = [
        (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
        (torch.tensor([4, 5]), torch.tensor([5, 6])),
    ]
    inputs, targets, mask = xrfm_collate_fn(batch, pad_id=11, return_attention_mask=True)
    assert inputs[1].tolist() == [4, 5, 11]
    assert targets[1].tolist() == [5, 6, -100]
    assert mask.tolist() == [[1, 1, 1], [1, 1, 0]]


def test_checkpoint_restores_rng_and_leaves_no_partial_file(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = OptimizerLoader(model.parameters(), learning_rate=1e-3)
    scheduler = SchedulerLoader(optimizer.optimizer, base_lr=1e-3, warmup_steps=1, max_steps=4)
    checkpoints = CheckpointLoader(str(tmp_path))

    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    path = checkpoints.save_checkpoint(model, optimizer, scheduler, step=1, loss=1.0)
    expected = (random.random(), float(np.random.random()), torch.rand(3))

    random.random()
    np.random.random()
    torch.rand(20)
    checkpoints.load_checkpoint(path, model, optimizer, scheduler)
    actual = (random.random(), float(np.random.random()), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_save_preserves_previous_checkpoint_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = OptimizerLoader(model.parameters(), learning_rate=1e-3)
    checkpoints = CheckpointLoader(str(tmp_path))
    path = checkpoints.save_checkpoint(model, optimizer, step=0, filename="stable.pt")
    original = Path(path).read_bytes()

    def fail_after_partial_write(*args: Any, **kwargs: Any) -> None:
        Path(args[1]).write_bytes(b"partial")
        raise RuntimeError("simulated save failure")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated save failure"):
        checkpoints.save_checkpoint(model, optimizer, step=1, filename="stable.pt")
    assert Path(path).read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_rejects_changed_data_identity(tmp_path: Path) -> None:
    interrupted = make_loop(tmp_path / "identity-source", tiny_model(), FixedSequenceDataset())
    checkpoint_path = interrupted.training_loop(max_steps=1, checkpoint_every=100)["checkpoint_path"]
    changed = FixedSequenceDataset()
    changed.data_manifest_hash = "sha256:different-data"
    with pytest.raises(RuntimeError, match="data_manifest_hash"):
        make_loop(tmp_path / "identity-target", tiny_model(), changed, resume_from=checkpoint_path)


def test_cpu_resume_is_bit_exact_with_data_position_and_rng(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    interrupted_dir = tmp_path / "interrupted"
    resumed_dir = tmp_path / "resumed"
    reference_dir.mkdir()
    interrupted_dir.mkdir()
    resumed_dir.mkdir()

    reference_data = FixedSequenceDataset()
    reference = make_loop(reference_dir, tiny_model(), reference_data)
    reference_batches = record_consumed_batches(reference)
    reference.training_loop(max_steps=3, checkpoint_every=100)
    reference_weights = {k: v.detach().clone() for k, v in reference.model.state_dict().items()}
    reference_metrics = copy.deepcopy(reference._metrics_records)
    reference_optimizer = copy.deepcopy(reference.optimizer.state_dict())
    reference_scheduler = copy.deepcopy(reference.scheduler.state_dict())
    reference_position = (reference.epoch, reference._batch_in_epoch)
    reference_rng = (random.random(), float(np.random.random()), torch.rand(3))

    interrupted_data = FixedSequenceDataset()
    interrupted = make_loop(interrupted_dir, tiny_model(), interrupted_data)
    interrupted_result = interrupted.training_loop(max_steps=1, checkpoint_every=100)
    checkpoint_path = interrupted_result["checkpoint_path"]
    resumed_data = FixedSequenceDataset()
    resumed = make_loop(resumed_dir, tiny_model(seed=999), resumed_data, resume_from=checkpoint_path)
    resumed_batches = record_consumed_batches(resumed)
    resumed.training_loop(max_steps=3, checkpoint_every=100)

    assert len(resumed_batches) == 2
    for actual, expected in zip(resumed_batches, reference_batches[1:]):
        assert torch.equal(actual, expected)
    assert resumed._metrics_records == reference_metrics[1:]
    assert_nested_equal(resumed.optimizer.state_dict(), reference_optimizer, "optimizer")
    assert resumed.scheduler.state_dict() == reference_scheduler
    assert (resumed.epoch, resumed._batch_in_epoch) == reference_position
    actual_rng = (random.random(), float(np.random.random()), torch.rand(3))
    assert actual_rng[0] == reference_rng[0]
    assert actual_rng[1] == reference_rng[1]
    assert torch.equal(actual_rng[2], reference_rng[2])
    for name, expected in reference_weights.items():
        actual = resumed.model.state_dict()[name]
        assert torch.equal(actual, expected), f"resume diverged at {name}"


def test_cpu_resume_restores_rng_when_positioning_reads_random_dataset(tmp_path: Path) -> None:
    reference = make_loop(tmp_path / "random-reference", tiny_model(), RandomSequenceDataset())
    reference_batches = record_consumed_batches(reference)
    reference.training_loop(max_steps=3, checkpoint_every=100)
    expected_weights = {k: v.detach().clone() for k, v in reference.model.state_dict().items()}
    expected_rng = torch.rand(4)

    interrupted = make_loop(tmp_path / "random-interrupted", tiny_model(), RandomSequenceDataset())
    checkpoint_path = interrupted.training_loop(max_steps=1, checkpoint_every=100)["checkpoint_path"]
    resumed = make_loop(
        tmp_path / "random-resumed",
        tiny_model(seed=999),
        RandomSequenceDataset(),
        resume_from=checkpoint_path,
    )
    resumed_batches = record_consumed_batches(resumed)
    resumed.training_loop(max_steps=3, checkpoint_every=100)

    for actual, expected in zip(resumed_batches, reference_batches[1:]):
        assert torch.equal(actual, expected)
    assert torch.equal(torch.rand(4), expected_rng)
    for name, expected in expected_weights.items():
        assert torch.equal(resumed.model.state_dict()[name], expected), name
