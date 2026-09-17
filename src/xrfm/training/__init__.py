"""XRFM training subsystem.

Public surface: :class:`Trainer` — the stable training boundary. The
internal loop, optimizer/scheduler builders, checkpointing, distributed
helpers, and metrics sink remain importable for advanced use:

    from xrfm.training import Trainer
    from xrfm.training.loop import TrainingLoop
    from xrfm.training.checkpoint import CheckpointLoader
"""

from xrfm.training.checkpoint import CheckpointLoader
from xrfm.training.distributed import xrfm_collate_fn
from xrfm.training.loop import TrainingLoop, _set_seed
from xrfm.training.metrics import MetricsWriter
from xrfm.training.optimizer import OptimizerLoader
from xrfm.training.scheduler import SchedulerLoader
from xrfm.training.trainer import Trainer, TrainResult

__all__ = [
    "CheckpointLoader",
    "MetricsWriter",
    "OptimizerLoader",
    "SchedulerLoader",
    "TrainResult",
    "Trainer",
    "TrainingLoop",
    "_set_seed",
    "xrfm_collate_fn",
]
