"""Step-level training log (`logs/train_step_log.jsonl`).

One JSON line every `every` optimizer steps: epoch, step in epoch, global optimizer step,
micro-batch loss, learning rate used for the step, gradient norm before clipping, and whether
clipping fired. Only the main process writes. `every <= 0` disables logging. Reading the gradient
norm happens only on logged steps, so the other steps pay no extra device sync.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

STEP_LOG_NAME = "train_step_log.jsonl"
DEFAULT_EVERY = 10


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value.item() if hasattr(value, "item") else value)
    except (TypeError, ValueError, RuntimeError):
        return None


class StepLogWriter:
    def __init__(self, path: Path, *, every: int, max_grad_norm: float, enabled: bool = True) -> None:
        self.every = int(every)
        self.max_grad_norm = float(max_grad_norm)
        self.optimizer_step = 0
        self._handle = None
        if enabled and self.every > 0:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("w", encoding="utf-8")  # a rerun of the same fold replaces the log

    @property
    def active(self) -> bool:
        return self._handle is not None

    def count_optimizer_step(self) -> bool:
        """Call once per real optimizer step. True when this step should be logged."""
        self.optimizer_step += 1
        return self.active and self.optimizer_step % self.every == 0

    def write(self, *, epoch: int, step: int, loss: Any, lr: float | None, grad_norm: Any) -> None:
        if self._handle is None:
            return
        norm = _to_float(grad_norm)
        row = {
            "epoch": int(epoch),
            "step": int(step),
            "optimizer_step": self.optimizer_step,
            "loss": _to_float(loss),
            "lr": lr,
            "grad_norm": norm,
            "clipped": None if norm is None or not math.isfinite(norm) else bool(norm > self.max_grad_norm),
        }
        self._handle.write(json.dumps(row, allow_nan=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
