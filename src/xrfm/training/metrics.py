"""
Metrics logging for XRFM training (forensic-audit addition, F-27).

Writes training metrics as JSON Lines (one JSON object per optimizer step)
so loss curves, LR, and throughput are trackable and resumable-analysis friendly.
"""

import json
import os
from typing import Any


class MetricsWriter:
    """Append-only JSONL metrics sink.

    Attributes:
        path: Output file path (e.g., ``logs/training_metrics.jsonl``).
        header: Extra metadata written once (config hash, seed, ...).
    """

    def __init__(self, path: str, header: dict[str, Any] | None = None) -> None:
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")  # noqa: SIM115 - append mode
        if header:
            self._write({"event": "header", **header})

    def write_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        """Append one metrics record for a completed optimizer step."""
        record = {"event": "step", "step": int(step)}
        record.update({k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()})
        self._write(record)

    def close(self) -> None:
        """Close the underlying file handle."""
        if not self._file.closed:
            self._file.close()

    def _write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()
