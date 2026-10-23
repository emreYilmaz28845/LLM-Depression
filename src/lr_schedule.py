"""Learning-rate schedules counted in real optimizer updates.

The legacy schedule in ``src/train.py`` sized ``get_linear_schedule_with_warmup``
from the unsharded loader length before ``accelerator.prepare`` and then let
Accelerate step it ``num_processes`` times per optimizer update. With few
micro-batches per rank (standalone text: 10-27 per epoch against an
accumulation of 32) that gave one update per epoch, the first one at LR 0, so
the epoch-1 adapter kept every LoRA B matrix at zero.

The ``update_counted`` schedule fixes this:

* the plan counts optimizer updates from the per-rank (sharded) micro-batch
  count of every epoch, with a partial accumulation window at the end of an
  epoch counted as one update, as Accelerate performs it;
* the scheduler is stepped once per real update (it is not passed to
  ``accelerator.prepare``), so its length matches the updates it sees;
* warmup is shifted by one update, so the first update already has LR > 0, and
  the last planned update still has LR > 0.

``simulate_legacy_lrs`` reproduces the legacy LR per real update for audits and
tests; it is not used for training.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

LR_SCHEDULE_UPDATE_COUNTED = "update_counted"
LR_SCHEDULE_LEGACY_V1 = "legacy_v1"
LR_SCHEDULE_MODES = (LR_SCHEDULE_UPDATE_COUNTED, LR_SCHEDULE_LEGACY_V1)


def resolve_lr_schedule_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("training", {}).get("lr_schedule_mode", LR_SCHEDULE_UPDATE_COUNTED))
    if mode not in LR_SCHEDULE_MODES:
        raise ValueError(f"training.lr_schedule_mode must be one of {LR_SCHEDULE_MODES}, got {mode!r}.")
    return mode


def updates_for_micro_batches(micro_batches: int, gradient_accumulation_steps: int) -> int:
    """Optimizer updates Accelerate makes for one epoch on one rank.

    Accelerate syncs at every full accumulation window and again at the end of
    the dataloader, so a partial last window is still one update.
    """
    if micro_batches < 0:
        raise ValueError("micro_batches must be non-negative.")
    return math.ceil(int(micro_batches) / max(1, int(gradient_accumulation_steps)))


def plan_update_schedule(
    micro_batches_per_rank_per_epoch: Sequence[int],
    gradient_accumulation_steps: int,
    warmup_ratio: float,
) -> dict[str, Any]:
    updates_per_epoch = [
        updates_for_micro_batches(count, gradient_accumulation_steps)
        for count in micro_batches_per_rank_per_epoch
    ]
    total_updates = sum(updates_per_epoch)
    if total_updates < 1:
        raise ValueError("The training plan has no optimizer updates.")
    warmup_updates = int(total_updates * float(warmup_ratio))
    warmup_updates = min(warmup_updates, total_updates - 1)
    return {
        "lr_schedule_mode": LR_SCHEDULE_UPDATE_COUNTED,
        "micro_batches_per_rank_per_epoch": [int(c) for c in micro_batches_per_rank_per_epoch],
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "updates_per_epoch": updates_per_epoch,
        "total_updates": total_updates,
        "warmup_updates": warmup_updates,
        "warmup_ratio": float(warmup_ratio),
        "first_update_lr_factor": update_lr_factor(0, total_updates, warmup_updates),
        "last_update_lr_factor": update_lr_factor(total_updates - 1, total_updates, warmup_updates),
    }


def update_lr_factor(update_index: int, total_updates: int, warmup_updates: int) -> float:
    """LR multiplier for the 0-based ``update_index``-th optimizer update.

    Warmup rises (k + 1) / (W + 1), so update 0 is never at LR 0. Decay is
    linear and reaches zero only after the last planned update.
    """
    k = int(update_index)
    if k < warmup_updates:
        return float(k + 1) / float(warmup_updates + 1)
    return max(0.0, float(total_updates - k) / float(max(1, total_updates - warmup_updates)))


def build_update_counted_scheduler(optimizer, plan: dict[str, Any]):
    from torch.optim.lr_scheduler import LambdaLR  # noqa: PLC0415

    total = int(plan["total_updates"])
    warmup = int(plan["warmup_updates"])
    # LambdaLR rejects Accelerate's optimizer wrapper on some torch versions;
    # the wrapped torch optimizer holds the same param groups.
    base_optimizer = getattr(optimizer, "optimizer", optimizer)
    return LambdaLR(base_optimizer, lambda step: update_lr_factor(step, total, warmup))


def simulate_legacy_lrs(
    *,
    unsharded_micro_batches_per_epoch: int,
    micro_batches_per_rank_per_epoch: int,
    gradient_accumulation_steps: int,
    epochs: int,
    warmup_ratio: float,
    num_processes: int,
) -> list[float]:
    """LR factor of every real update under the legacy (pre-fix) schedule."""
    total_steps = max(
        1, math.ceil(unsharded_micro_batches_per_epoch / gradient_accumulation_steps) * epochs
    )
    warmup_steps = int(total_steps * warmup_ratio)

    def hf_linear(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    lrs: list[float] = []
    scheduler_step = 0
    per_epoch = updates_for_micro_batches(micro_batches_per_rank_per_epoch, gradient_accumulation_steps)
    for _ in range(epochs * per_epoch):
        lrs.append(hf_linear(scheduler_step))
        scheduler_step += num_processes
    return lrs


class UpdateTracker:
    """Counts real optimizer updates and steps the scheduler once per update.

    Under ``update_counted`` it refuses an update at LR <= 0 and an epoch whose
    update count differs from the plan. Under ``legacy_v1`` the scheduler was
    prepared by Accelerate, which gates and multiplies its steps itself.
    """

    def __init__(self, record: dict[str, Any], logger=None) -> None:
        self.mode = str(record["lr_schedule_mode"])
        self.planned_updates_per_epoch = record.get("updates_per_epoch")
        self.logger = logger
        self.epoch = 0
        self.epoch_updates = 0
        self.total_updates = 0

    def start_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.epoch_updates = 0

    def record_update(self, optimizer) -> float:
        lr = float(optimizer.param_groups[0]["lr"])
        if self.total_updates == 0 and self.logger is not None:
            self.logger.info("First optimizer update | epoch=%s lr=%.3e mode=%s", self.epoch, lr, self.mode)
        if self.mode == LR_SCHEDULE_UPDATE_COUNTED and lr <= 0.0:
            raise RuntimeError(
                f"Optimizer update {self.total_updates + 1} (epoch {self.epoch}) would run at LR {lr}; "
                "an update at LR 0 leaves the adapter untrained."
            )
        self.epoch_updates += 1
        self.total_updates += 1
        return lr

    def step_scheduler(self, scheduler, sync_gradients: bool) -> None:
        if self.mode == LR_SCHEDULE_LEGACY_V1:
            scheduler.step()
        elif sync_gradients:
            scheduler.step()

    def finish_epoch(self) -> None:
        if self.epoch_updates == 0:
            raise RuntimeError(f"Epoch {self.epoch} ended without any optimizer update.")
        if self.mode != LR_SCHEDULE_UPDATE_COUNTED or not self.planned_updates_per_epoch:
            return
        expected = int(self.planned_updates_per_epoch[self.epoch - 1])
        if self.epoch_updates != expected:
            raise RuntimeError(
                f"Epoch {self.epoch} made {self.epoch_updates} optimizer updates but the LR plan "
                f"expected {expected}; the scheduler length no longer matches the training loop."
            )


def trainable_parameter_fingerprint(model, device) -> list[float]:
    """[sum |p|, sum p^2, sum |LoRA B|, LoRA B element count] over all ranks.

    Under FSDP each rank sees only its shard of every parameter, so the local
    sums are all-reduced.
    """
    import torch  # noqa: PLC0415

    abs_sum = 0.0
    sq_sum = 0.0
    lora_b_abs = 0.0
    lora_b_count = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.numel() == 0:
                continue
            values = parameter.detach().double()
            abs_value = float(values.abs().sum().item())
            abs_sum += abs_value
            sq_sum += float((values * values).sum().item())
            if "lora_B" in name:
                lora_b_abs += abs_value
                lora_b_count += int(parameter.numel())
    stats = torch.tensor([abs_sum, sq_sum, lora_b_abs, float(lora_b_count)], dtype=torch.float64, device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats)
    return [float(value) for value in stats.tolist()]


def check_epoch_changed_parameters(model, device, previous: list[float], *, epoch: int) -> list[float]:
    current = trainable_parameter_fingerprint(model, device)
    if current[0] == previous[0] and current[1] == previous[1]:
        raise RuntimeError(
            f"No trainable parameter changed during epoch {epoch}; refusing to continue with an untrained model."
        )
    if current[3] > 0 and current[2] == 0.0:
        raise RuntimeError(
            f"All {int(current[3])} LoRA B elements are still zero after epoch {epoch}; "
            "the adapter is the untrained base model."
        )
    return current


def assert_saved_adapter_trained(adapter_dir) -> None:
    """Fail when a saved PEFT adapter has LoRA B tensors that are all zero."""
    from pathlib import Path  # noqa: PLC0415

    from safetensors import safe_open  # noqa: PLC0415

    path = Path(adapter_dir) / "adapter_model.safetensors"
    lora_b_tensors = 0
    if path.exists():
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                if "lora_B" not in key:
                    continue
                lora_b_tensors += 1
                if bool(handle.get_tensor(key).abs().max().item() > 0):
                    return
    else:
        import torch  # noqa: PLC0415

        path = Path(adapter_dir) / "adapter_model.bin"
        if not path.exists():
            raise RuntimeError(f"Saved checkpoint has no adapter weights in {adapter_dir}.")
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        for key, tensor in state.items():
            if "lora_B" not in key:
                continue
            lora_b_tensors += 1
            if bool(tensor.abs().max().item() > 0):
                return
    if lora_b_tensors:
        raise RuntimeError(
            f"All {lora_b_tensors} LoRA B tensors in {path} are zero; the selected checkpoint is untrained."
        )
