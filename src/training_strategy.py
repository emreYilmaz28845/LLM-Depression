"""Training strategy selection: DDP (default) or FSDP.

`training.strategy` is a config switch so every existing run keeps today's DDP
behavior. FSDP exists because the Qwen3.8-27B memory measurement showed that DDP
replication (a full model copy per rank) does not fit one 63 GiB H100 for the
training step, while the same 4-GPU lane shape does fit once the weights are
sharded.

Everything that is rank- or sharding-related lives here. Backends only answer
two questions: which modules FSDP should wrap, and how the model is loaded.
"""

from __future__ import annotations

from typing import Any, Callable

from src.utils import get_logger


LOGGER = get_logger(__name__)

TRAINING_STRATEGY_DDP = "ddp"
TRAINING_STRATEGY_FSDP = "fsdp"
SUPPORTED_TRAINING_STRATEGIES = (TRAINING_STRATEGY_DDP, TRAINING_STRATEGY_FSDP)

FSDP_SHARDING_STRATEGY_FULL_SHARD = "full_shard"


def resolve_training_strategy(config: dict[str, Any]) -> str:
    """Resolve ``training.strategy`` (default ``ddp``)."""
    raw_value = config.get("training", {}).get("strategy", TRAINING_STRATEGY_DDP)
    if raw_value in (None, ""):
        return TRAINING_STRATEGY_DDP
    normalized = str(raw_value).strip().lower()
    if normalized not in SUPPORTED_TRAINING_STRATEGIES:
        raise ValueError(
            f"Unsupported training.strategy={raw_value!r}. "
            f"Expected one of {', '.join(SUPPORTED_TRAINING_STRATEGIES)} (or unset)."
        )
    return normalized


def effective_global_batch_size(config: dict[str, Any], world_size: int) -> int:
    """Examples that make up one optimizer update across all ranks."""
    training_cfg = config.get("training", {})
    return (
        int(training_cfg.get("per_device_train_batch_size", 1))
        * int(training_cfg.get("gradient_accumulation_steps", 1))
        * int(world_size)
    )


def _resolve_wrap_policy(
    config: dict[str, Any], model, wrap_policy_names: Callable[[Any], list[str]] | None
):
    if wrap_policy_names is None:
        return None

    class_names = list(wrap_policy_names(model))
    if not class_names:
        raise ValueError(
            "The backend returned no FSDP transformer class names; refusing to fall "
            "back to an implicit wrap policy."
        )
    LOGGER.info("FSDP wrap policy | transformer classes=%s", class_names)
    return class_names


def build_fsdp_plugin(
    config: dict[str, Any],
    model,
    wrap_policy_names: Callable[[Any], list[str]] | None = None,
):
    """Build the Accelerate FSDP plugin for the resolved backend.

    ``use_orig_params`` keeps the original parameter tensors visible to PEFT and
    to the optimizer; without it PEFT's adapter handling and the checkpoint
    gather cannot be reasoned about. ``sync_module_states`` broadcasts rank 0's
    parameters, so no rank keeps a second full GPU copy during load.

    The sharding strategy is Accelerate's FSDP default (FULL_SHARD); the
    argument that used to set it is deprecated in current Accelerate versions, so
    it is not passed and the resolved value is logged instead.
    """
    from accelerate import FullyShardedDataParallelPlugin  # noqa: PLC0415

    plugin_kwargs: dict[str, Any] = {
        "use_orig_params": True,
        "sync_module_states": True,
    }
    transformer_cls_names = _resolve_wrap_policy(config, model, wrap_policy_names)
    if transformer_cls_names is not None:
        # Accelerate only builds the wrap policy from the class names when
        # ``auto_wrap_policy`` is the ``transformer_auto_wrap_policy`` function
        # itself; leaving it unset silently wraps the whole model as one unit,
        # which materialises the full model on every rank instead of sharding it.
        try:
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - version dependent
            raise RuntimeError(
                "torch.distributed.fsdp.wrap.transformer_auto_wrap_policy is required for the "
                "fsdp strategy; this torch build does not provide it."
            ) from exc

        plugin_kwargs["auto_wrap_policy"] = transformer_auto_wrap_policy
        plugin_kwargs["transformer_cls_names_to_wrap"] = transformer_cls_names
    plugin = FullyShardedDataParallelPlugin(**plugin_kwargs)
    LOGGER.info(
        "FSDP plugin | sharding_strategy=%s use_orig_params=%s sync_module_states=%s",
        getattr(plugin, "sharding_strategy", FSDP_SHARDING_STRATEGY_FULL_SHARD),
        getattr(plugin, "use_orig_params", None),
        getattr(plugin, "sync_module_states", None),
    )
    return plugin


def align_fsdp_model_dtypes(model, dtype=None) -> str:
    """Make every parameter share one dtype before FSDP shards the model.

    FSDP flattens the parameters of each wrapped unit into one flat parameter and
    refuses mixed dtypes ("Must flatten tensors with uniform dtype"). PEFT creates
    the adapter parameters in float32 while the base model is loaded in bfloat16,
    so a training model is routinely mixed. The alignment happens in place, which
    keeps an already created optimizer pointing at the same parameter objects.
    """
    import torch  # noqa: PLC0415

    parameters = list(model.parameters())
    if not parameters:
        raise ValueError("Cannot align dtypes of a model without parameters.")
    target = dtype if dtype is not None else parameters[0].dtype
    changed = 0
    for parameter in parameters:
        if parameter.dtype != target:
            parameter.data = parameter.data.to(target)
            changed += 1
    LOGGER.info(
        "FSDP dtype alignment | target=%s moved_parameters=%s total_parameters=%s",
        target,
        changed,
        len(parameters),
    )
    return str(target)


def broadcast_flag(accelerator, flag: bool) -> bool:
    """Broadcast a rank-0 decision so every rank agrees on collective work.

    A save that gathers sharded parameters is a collective: if rank 0 skips it
    while the other ranks enter it, the job hangs. Rank 0's decision therefore
    travels to every rank before the collective starts.
    """
    import torch  # noqa: PLC0415

    tensor = torch.tensor(1 if flag else 0, device=accelerator.device, dtype=torch.int32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(tensor, src=0)
    return bool(int(tensor.item()))


def save_training_checkpoint(accelerator, model, processor, output_dir, config: dict[str, Any]) -> None:
    """Save the training model's PEFT adapter, rank-safe under DDP and FSDP.

    Under FSDP every rank joins the state-dict gather and only the main process
    writes files; under DDP the main process writes through the backend hook.
    Either way the result is the plain adapter directory the single-GPU
    evaluation path loads, and the 27B base model is never written out.
    """
    if resolve_training_strategy(config) != TRAINING_STRATEGY_FSDP:
        if accelerator.is_main_process:
            from src.model.runtime import save_adapter_and_processor  # noqa: PLC0415

            save_adapter_and_processor(
                accelerator.unwrap_model(model), processor, output_dir, config=config
            )
        return

    state_dict = accelerator.get_state_dict(model)
    if not accelerator.is_main_process:
        return
    from pathlib import Path  # noqa: PLC0415

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(
        target, state_dict=state_dict, safe_serialization=True
    )
    processor.save_pretrained(target)
    LOGGER.info("Saved gathered LoRA adapter and tokenizer to %s", target)


def build_accelerator(
    config: dict[str, Any],
    model=None,
    wrap_policy_names: Callable[[Any], list[str]] | None = None,
):
    """Build the Accelerator for the resolved strategy.

    FSDP requires the model before the plugin can resolve the wrap policy, and it
    forbids the in-train rank-0 held-out evaluation path (the plan and the FSDP
    design both require a separate evaluation job there).
    """
    from accelerate import Accelerator, DistributedDataParallelKwargs  # noqa: PLC0415

    strategy = resolve_training_strategy(config)
    training_cfg = config.get("training", {})
    if strategy == TRAINING_STRATEGY_FSDP:
        if model is None:
            raise ValueError("The fsdp training strategy needs the model to resolve the wrap policy.")
        if bool(training_cfg.get("run_final_eval_in_train", False)):
            raise ValueError(
                "training.run_final_eval_in_train must be false under the fsdp strategy: "
                "the in-train held-out evaluation reloads the best model on rank 0 into a "
                "single GPU, which a sharded run cannot do. Use a separate evaluation job."
            )
        import torch  # noqa: PLC0415

        align_fsdp_model_dtypes(
            model,
            dtype=torch.bfloat16
            if bool(training_cfg.get("bf16", False)) and torch.cuda.is_available()
            else None,
        )
        # Accelerate takes the FSDP plugin as its own argument: the class is a
        # plugin, not a KwargsHandler, so it must not travel through
        # kwargs_handlers (that path asserts on the handler base class).
        accelerator = Accelerator(
            gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
            mixed_precision="bf16" if bool(training_cfg.get("bf16", False)) else "no",
            fsdp_plugin=build_fsdp_plugin(config, model, wrap_policy_names),
        )
        return accelerator

    accelerator = Accelerator(
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        mixed_precision="bf16" if bool(training_cfg.get("bf16", False)) else "no",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    return accelerator
