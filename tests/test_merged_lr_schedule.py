"""Merged trainer LR schedule: counted in real updates, independent of num_processes.

The pre-fix merged trainer passed its linear scheduler to ``accelerator.prepare``
and called ``scheduler.step()`` once per schedule block; Accelerate then stepped
it ``num_processes`` times, so the LR reached 0 after ``epochs / num_processes``
epochs (2.5 of 20 epochs at 8 processes).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from src.lr_schedule import (
    LR_SCHEDULE_UPDATE_COUNTED,
    UpdateTracker,
    build_update_counted_scheduler,
    plan_merged_block_schedule,
    simulate_legacy_merged_lrs,
)

EPOCHS = 20
WARMUP = 0.03
AUDIO_BLOCKS = 376  # merged audio / audio+text updates per epoch (j-20261114T132811Z-a278)
TEXT_BLOCKS = 13  # merged text updates per epoch
MERGED_TRAIN = Path(__file__).resolve().parents[1] / "src" / "merged" / "train.py"


@pytest.fixture(autouse=True)
def _reset_accelerate_state():
    yield
    from accelerate.state import AcceleratorState, PartialState

    AcceleratorState._reset_state(reset_partial_state=True)
    PartialState._reset_state()


def _first_zero_update(lrs: list[float]) -> int | None:
    for index, value in enumerate(lrs):
        if index > 0 and value <= 0.0:
            return index
    return None


def _run_plan(plan) -> list[float]:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    scheduler = build_update_counted_scheduler(optimizer, plan)
    tracker = UpdateTracker(plan)
    lrs: list[float] = []
    for epoch, blocks in enumerate(plan["blocks_per_epoch"], start=1):
        tracker.start_epoch(epoch)
        for _ in range(blocks):
            lrs.append(tracker.record_update(optimizer))
            optimizer.step()
            tracker.step_scheduler(scheduler, True)
        tracker.finish_epoch()
    assert scheduler.last_epoch == plan["total_updates"]
    return lrs


@pytest.mark.parametrize(
    ("blocks", "num_processes", "zero_epoch"),
    [(AUDIO_BLOCKS, 8, 2.5), (TEXT_BLOCKS, 4, 5.0), (AUDIO_BLOCKS, 4, 5.0)],
)
def test_legacy_merged_schedule_reproduces_the_bug(blocks, num_processes, zero_epoch):
    lrs = simulate_legacy_merged_lrs(
        blocks_per_epoch=[blocks] * EPOCHS, warmup_ratio=WARMUP, num_processes=num_processes
    )
    first_zero = _first_zero_update(lrs)
    assert first_zero is not None
    assert first_zero / blocks == pytest.approx(zero_epoch, abs=1.0 / blocks + 1e-9)
    assert all(value == 0.0 for value in lrs[first_zero:])


def test_legacy_simulation_matches_accelerate_prepared_scheduler(monkeypatch):
    """The legacy simulation equals what Accelerate's prepared scheduler does."""
    from accelerate import Accelerator
    from accelerate import scheduler as accelerate_scheduler
    from transformers import get_linear_schedule_with_warmup

    blocks = [7] * 6
    num_processes = 4
    accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    total = sum(blocks)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total * 0.1), num_training_steps=total
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    class _State:
        def __init__(self):
            self.num_processes = num_processes

    monkeypatch.setattr(accelerate_scheduler, "AcceleratorState", _State)
    seen = []
    for _ in range(total):
        seen.append(optimizer.param_groups[0]["lr"])
        model(torch.ones(1)).sum().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    expected = simulate_legacy_merged_lrs(blocks_per_epoch=blocks, warmup_ratio=0.1, num_processes=num_processes)
    assert seen == pytest.approx(expected)


@pytest.mark.parametrize("blocks", [AUDIO_BLOCKS, TEXT_BLOCKS, 1])
def test_update_counted_merged_plan_spans_every_epoch(blocks):
    plan = plan_merged_block_schedule([blocks] * EPOCHS, WARMUP)
    assert plan["lr_schedule_mode"] == LR_SCHEDULE_UPDATE_COUNTED
    assert plan["updates_per_epoch"] == [blocks] * EPOCHS
    assert plan["total_updates"] == blocks * EPOCHS
    lrs = _run_plan(plan)
    assert len(lrs) == blocks * EPOCHS
    assert all(value > 0.0 for value in lrs)  # no update at LR 0, including the first and last
    assert max(lrs) == pytest.approx(2e-4)
    # The last epoch still trains; LR reaches ~0 only at the final update.
    assert lrs[-1] == pytest.approx(2e-4 / (plan["total_updates"] - plan["warmup_updates"]))
    if blocks > 1:
        mid = lrs[(EPOCHS // 2) * blocks]
        assert 0.4 * 2e-4 < mid < 0.6 * 2e-4


def test_update_counted_plan_handles_uneven_epochs_and_tracker_guards():
    plan = plan_merged_block_schedule([5, 6, 4], 0.1)
    assert plan["updates_per_epoch"] == [5, 6, 4]
    _run_plan(plan)
    tracker = UpdateTracker(plan)
    tracker.start_epoch(1)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    tracker.record_update(optimizer)
    with pytest.raises(RuntimeError, match="expected 5"):
        tracker.finish_epoch()


def test_merged_trainer_does_not_prepare_the_update_counted_scheduler():
    """Static guard: the update-counted branch prepares only model and optimizer."""
    tree = ast.parse(MERGED_TRAIN.read_text(encoding="utf-8"))
    prepare_calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "prepare"
        ):
            prepare_calls.append([ast.unparse(arg) for arg in node.args])
    assert ["model", "optimizer"] in prepare_calls
    # The only prepare that receives the scheduler is the legacy_v1 branch.
    with_scheduler = [args for args in prepare_calls if "scheduler" in args]
    assert len(with_scheduler) == 1
    source = MERGED_TRAIN.read_text(encoding="utf-8")
    assert "plan_merged_block_schedule(" in source
    assert "update_tracker.step_scheduler(scheduler" in source
    direct_steps = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
        and ast.unparse(node.func.value) == "scheduler"
    ]
    assert not direct_steps  # the scheduler is stepped only through UpdateTracker
    assert "StepLogWriter(" in source
