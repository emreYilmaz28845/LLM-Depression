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
        plugin_kwargs["transformer_cls_names_to_wrap"] = transformer_cls_names
    plugin = FullyShardedDataParallelPlugin(**plugin_kwargs)
    LOGGER.info(
        "FSDP plugin | sharding_strategy=%s use_orig_params=%s sync_module_states=%s",
        getattr(plugin, "sharding_strategy", FSDP_SHARDING_STRATEGY_FULL_SHARD),
        getattr(plugin, "use_orig_params", None),
        getattr(plugin, "sync_module_states", None),
    )
    return plugin


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
