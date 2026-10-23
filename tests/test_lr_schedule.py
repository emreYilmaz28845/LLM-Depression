"""Update-counted LR schedule: no update at LR 0, scheduler ends at the last real update."""

from __future__ import annotations

import math

import pytest
import torch

from src.lr_schedule import (
    LR_SCHEDULE_LEGACY_V1,
    UpdateTracker,
    assert_saved_adapter_trained,
    build_update_counted_scheduler,
    check_epoch_changed_parameters,
    plan_update_schedule,
    resolve_lr_schedule_mode,
    simulate_legacy_lrs,
    trainable_parameter_fingerprint,
    updates_for_micro_batches,
)

WORLD = 4
ACCUM = 32
EPOCHS = 20
WARMUP = 0.03


@pytest.fixture(autouse=True)
def _reset_accelerate_state():
    yield
    from accelerate.state import AcceleratorState, PartialState

    AcceleratorState._reset_state(reset_partial_state=True)
    PartialState._reset_state()


def _run_schedule(plan):
    """Apply the plan with a real optimizer; return the LR seen by every update."""
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    scheduler = build_update_counted_scheduler(optimizer, plan)
    lrs = []
    for _ in range(plan["total_updates"]):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs, scheduler, optimizer


def _plan_for(samples, epochs=EPOCHS):
    per_rank = math.ceil(samples / WORLD)
    return per_rank, plan_update_schedule([per_rank] * epochs, ACCUM, WARMUP)


def test_d3tec_text_first_update_has_lr_and_schedule_ends_at_last_update():
    per_rank, plan = _plan_for(39)
    assert per_rank == 10
    assert plan["updates_per_epoch"] == [1] * EPOCHS
    assert plan["total_updates"] == 20
    lrs, scheduler, optimizer = _run_schedule(plan)
    assert len(lrs) == 20
    assert lrs[0] > 0
    assert all(lr > 0 for lr in lrs)
    assert lrs == sorted(lrs, reverse=True)
    # The scheduler reaches its end exactly after the last real update.
    assert scheduler.last_epoch == plan["total_updates"]
    assert optimizer.param_groups[0]["lr"] == 0.0


def test_d3tec_legacy_schedule_reproduces_the_bug():
    lrs = simulate_legacy_lrs(
        unsharded_micro_batches_per_epoch=39,
        micro_batches_per_rank_per_epoch=10,
        gradient_accumulation_steps=ACCUM,
        epochs=EPOCHS,
        warmup_ratio=WARMUP,
        num_processes=WORLD,
    )
    assert lrs[0] == 0.0  # epoch 1's only update runs at LR 0
    assert lrs[1] > 0.9  # epoch 2 jumps to near peak
    assert lrs[10] == 0.0  # LR is already 0 from epoch 11 on


@pytest.mark.parametrize("samples", [62, 75, 107, 190])
def test_small_text_corpora_never_update_at_lr_zero(samples):
    _, plan = _plan_for(samples)
    lrs, scheduler, _ = _run_schedule(plan)
    assert all(lr > 0 for lr in lrs)
    assert scheduler.last_epoch == plan["total_updates"]


def test_large_audio_case_warmup_and_end():
    per_rank, plan = _plan_for(1600)
    assert per_rank == 400
    assert plan["updates_per_epoch"] == [13] * EPOCHS
    assert plan["total_updates"] == 260
    assert plan["warmup_updates"] == 7
    lrs, scheduler, _ = _run_schedule(plan)
    assert lrs[0] == pytest.approx(2e-4 / 8)
    assert max(lrs) == pytest.approx(2e-4)
    assert lrs.index(max(lrs)) == 7
    assert all(lr > 0 for lr in lrs)
    assert scheduler.last_epoch == 260


def test_partial_accumulation_window_counts_as_update():
    assert updates_for_micro_batches(33, 32) == 2
    assert updates_for_micro_batches(32, 32) == 1
    assert updates_for_micro_batches(10, 32) == 1
    plan = plan_update_schedule([10, 33, 64], 32, 0.0)
    assert plan["updates_per_epoch"] == [1, 2, 2]
    assert plan["total_updates"] == 5


def test_lr_schedule_mode_resolution():
    assert resolve_lr_schedule_mode({"training": {}}) == "update_counted"
    assert resolve_lr_schedule_mode({"training": {"lr_schedule_mode": "legacy_v1"}}) == LR_SCHEDULE_LEGACY_V1
    with pytest.raises(ValueError):
        resolve_lr_schedule_mode({"training": {"lr_schedule_mode": "cosine"}})


def test_accelerate_loop_matches_plan():
    """Single-process Accelerate run: real sync points equal the planned updates."""
    from accelerate import Accelerator

    accelerator = Accelerator(gradient_accumulation_steps=ACCUM, cpu=True)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loader = torch.utils.data.DataLoader(torch.randn(40, 2), batch_size=1)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    plan = plan_update_schedule([len(loader)] * 3, ACCUM, WARMUP)
    assert plan["updates_per_epoch"] == [2, 2, 2]
    scheduler = build_update_counted_scheduler(optimizer, plan)
    tracker = UpdateTracker(plan)
    for epoch in range(1, 4):
        tracker.start_epoch(epoch)
        for batch in loader:
            with accelerator.accumulate(model):
                accelerator.backward(model(batch).sum())
                if accelerator.sync_gradients:
                    tracker.record_update(optimizer)
                optimizer.step()
                tracker.step_scheduler(scheduler, accelerator.sync_gradients)
                optimizer.zero_grad()
        tracker.finish_epoch()
    assert tracker.total_updates == 6
    assert scheduler.last_epoch == 6


def test_tracker_refuses_lr_zero_and_plan_mismatch():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=0.0)
    tracker = UpdateTracker({"lr_schedule_mode": "update_counted", "updates_per_epoch": [2]})
    tracker.start_epoch(1)
    with pytest.raises(RuntimeError, match="LR 0"):
        tracker.record_update(optimizer)
    optimizer.param_groups[0]["lr"] = 1e-4
    tracker.record_update(optimizer)
    with pytest.raises(RuntimeError, match="expected 2"):
        tracker.finish_epoch()
    empty = UpdateTracker({"lr_schedule_mode": LR_SCHEDULE_LEGACY_V1})
    empty.start_epoch(1)
    with pytest.raises(RuntimeError, match="without any optimizer update"):
        empty.finish_epoch()


class _TinyLora(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.ones(2, 2))
        self.lora_B = torch.nn.Parameter(torch.zeros(2, 2))


def test_epoch_guard_fails_on_unchanged_parameters_and_zero_lora_b():
    model = _TinyLora()
    before = trainable_parameter_fingerprint(model, "cpu")
    with pytest.raises(RuntimeError, match="No trainable parameter changed"):
        check_epoch_changed_parameters(model, "cpu", before, epoch=1)
    with torch.no_grad():
        model.lora_A.mul_(0.5)
    with pytest.raises(RuntimeError, match="LoRA B elements are still zero"):
        check_epoch_changed_parameters(model, "cpu", before, epoch=1)
    with torch.no_grad():
        model.lora_B.add_(1e-3)
    after = check_epoch_changed_parameters(model, "cpu", before, epoch=1)
    assert after[2] > 0


def test_saved_adapter_check(tmp_path):
    from safetensors.torch import save_file

    zero = {"base_model.model.layers.0.q_proj.lora_B.weight": torch.zeros(4, 2),
            "base_model.model.layers.0.q_proj.lora_A.weight": torch.ones(2, 4)}
    save_file(zero, str(tmp_path / "adapter_model.safetensors"))
    with pytest.raises(RuntimeError, match="selected checkpoint is untrained"):
        assert_saved_adapter_trained(tmp_path)
    trained = dict(zero)
    trained["base_model.model.layers.0.q_proj.lora_B.weight"] = torch.full((4, 2), 1e-4)
    save_file(trained, str(tmp_path / "adapter_model.safetensors"))
    assert_saved_adapter_trained(tmp_path)
