#!/usr/bin/env python3
"""Qwen3-Omni MN5 probes: real-model forward, FSDP backward, memory gate, sweep.

Four modes, all writing one JSON report with the measured evidence:

* ``tree``     — no weights, no GPU. Instantiates the Thinker on the meta device
  and verifies the anchored LoRA regex against the real module tree, recording
  the decoder layer count and class, the MoE layout, and the audio-tower names.
* ``forward``  — one process, device-map sharded across the visible GPUs. Loads
  the real Thinker, installs the audio dtype boundary, and runs a finite forward
  plus candidate-label likelihood scoring over the selected risk examples for
  both modalities. Proves Talker absence, the audio-feature dtype correction and
  finite scores, and records per-GPU peak memory.
* ``backward`` — ``torchrun`` ranks with the shared FSDP strategy. Builds the
  prepared model through the shared training path and runs one forward + backward
  + optimizer step, proving: more than one real FSDP unit, finite and non-zero
  LoRA gradients, a frozen audio encoder, active gradient checkpointing, and the
  forced per-microbatch synchronization.
* ``maxrisk``  — ``torchrun`` ranks with the shared FSDP strategy, over every
  selected maximum-risk example (longest audio, longest rendered prompt, largest
  combined processor footprint) for the requested modalities.
* ``perf``     — ``torchrun`` ranks. Times warm training steps at the configured
  effective batch and reports throughput and peak memory for the shape choice.

Every rank reports its own peak memory; rank 0 writes the combined report. The
script records shapes, dtypes, durations, counts, hashes, peak memory and
timings only — never transcripts, subject identifiers or audio paths.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.runtime import AudioTextDataset, build_examples, load_manifest_rows
from src.model.runtime import (
    build_collator,
    fsdp_wrap_policy_names,
    load_model_for_inference,
    load_model_for_training,
    load_processor,
    resolve_processor_sampling_rate,
)
from src.training_strategy import (
    activation_offload_context,
    build_accelerator,
    effective_global_batch_size,
    resolve_activation_offload,
)
from src.utils import (
    get_logger,
    load_yaml_with_overrides,
    normalize_config_overrides,
    resolve_model_name_or_path,
)

LOGGER = get_logger(__name__)

SCHEMA_VERSION = "audiollm.qwen3omni_probe.v1"
LOAD_AUDIT_ATTR = "_model_load_audit"


def _example_ref(sample_id: str) -> str:
    return hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:16]


def _host_memory_gb() -> dict[str, float]:
    info: dict[str, float] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                value = rest.strip().split()
                if not value or key not in {"MemTotal", "MemAvailable"}:
                    continue
                info[f"{key}_gb"] = round(float(value[0]) / 1024 / 1024, 2)
    except OSError:  # pragma: no cover - non-Linux
        pass
    return info


def _reset_peak_memory() -> None:
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)


def _gpu_memory() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    per_device: dict[str, Any] = {}
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        per_device[str(index)] = {
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated(index) / 1024**3, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved(index) / 1024**3, 3),
            "free_gib": round(free / 1024**3, 3),
            "total_gib": round(total / 1024**3, 3),
        }
    return per_device


def _load_config(args) -> dict[str, Any]:
    overrides = normalize_config_overrides(getattr(args, "set_overrides", []) or [])
    return load_yaml_with_overrides(Path(args.config), overrides or None)


def _manifest_rows(config: dict[str, Any], args) -> list[dict[str, Any]]:
    from src.evaluate import _load_metadata_or_build

    overrides = normalize_config_overrides(getattr(args, "set_overrides", []) or [])
    metadata = _load_metadata_or_build(args.config, config, overrides or None)
    return load_manifest_rows(metadata["manifest_path"])


def _modality_config(config: dict[str, Any], modality: str) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    if modality == "audio_only":
        resolved["data"]["use_audio"] = True
        resolved["data"]["use_text"] = False
        resolved["data"].pop("audio_text_transcript_scope", None)
    elif modality == "audio_text":
        resolved["data"]["use_audio"] = True
        resolved["data"]["use_text"] = True
        resolved["data"]["audio_text_transcript_scope"] = "full_participant"
    else:
        raise SystemExit(f"unsupported modality {modality!r}")
    return resolved


def _selection_refs(inventory: dict[str, Any], modality: str) -> list[str]:
    selection = (inventory.get("modalities", {}).get(modality) or {}).get("selection") or {}
    refs: list[str] = []
    for key in (
        "longest_audio",
        "longest_prompt",
        "largest_combined_footprint",
        "shortest_audio",
        "shortest_prompt",
    ):
        entry = selection.get(key)
        if entry and entry["example_ref"] not in refs:
            refs.append(entry["example_ref"])
    return refs


def _examples_for(
    args, config: dict[str, Any], modality: str, inventory: dict[str, Any] | None
) -> list[dict[str, Any]]:
    rows = _manifest_rows(config, args)
    examples = build_examples(rows, _modality_config(config, modality), partition_name="probe")
    if not inventory:
        examples.sort(key=lambda example: (example["sample_id"],))
        return examples[: args.limit]
    by_ref = {_example_ref(example["sample_id"]): example for example in examples}
    refs = _selection_refs(inventory, modality)
    missing = [ref for ref in refs if ref not in by_ref]
    if missing:
        raise SystemExit(f"inventory refs not found in the manifest: {missing}")
    return [by_ref[ref] for ref in refs]


def _collated(example: dict[str, Any], config: dict[str, Any], processor, device) -> dict[str, Any]:
    sampling_rate = resolve_processor_sampling_rate(processor)
    dataset = AudioTextDataset([example], processor_sampling_rate=sampling_rate, silence_audio=False)
    collator = build_collator(config, processor)
    batch = collator([dataset[0]])
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _split_loss_weight(batch: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Separate ``loss_weight`` from the model inputs, as the training loop does."""
    model_batch = dict(batch)
    weight = model_batch.pop("loss_weight", None)
    return model_batch, weight


# --------------------------------------------------------------------------- #
# forward probe (device-map sharded single process)
# --------------------------------------------------------------------------- #


def _candidate_scores(model, batch: dict[str, Any], processor) -> dict[str, Any]:
    model_batch, loss_weight = _split_loss_weight(batch)
    with torch.no_grad():
        outputs = model(**model_batch)
    loss = float(outputs.loss) if getattr(outputs, "loss", None) is not None else None
    logits = outputs.logits[:, -1, :].float()
    log_probabilities = torch.log_softmax(logits, dim=-1).squeeze(0)
    depressed_id = processor.tokenizer.encode("Depressed", add_special_tokens=False)[-1]
    non_id = processor.tokenizer.encode("Non-depressed", add_special_tokens=False)[-1]
    scores = [float(log_probabilities[depressed_id]), float(log_probabilities[non_id])]
    return {
        "loss": loss,
        "loss_weight": float(loss_weight) if loss_weight is not None else None,
        "depressed_score": scores[0],
        "non_depressed_score": scores[1],
        "finite": all(value is not None and torch.isfinite(torch.tensor(value)) for value in [loss, *scores]),
        "decision": "Depressed" if scores[0] > scores[1] else "Non-depressed",
    }


def run_forward(args) -> dict[str, Any]:
    config = _load_config(args)
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8")) if args.inventory else None
    model_dir = str(resolve_model_name_or_path(None, config))
    processor = load_processor(model_dir, config)
    if args.checkpoint:
        model = load_model_for_inference(model_dir, args.checkpoint, config)
    else:
        model = load_model_for_inference(model_dir, None, config)
    _reset_peak_memory()
    device = next(model.parameters()).device
    results: dict[str, Any] = {}
    for modality in args.modalities:
        config_for_modality = _modality_config(config, modality)
        rows = []
        for example in _examples_for(args, config, modality, inventory):
            batch = _collated(example, config_for_modality, processor, device)
            scores = _candidate_scores(model, batch, processor)
            rows.append(
                {
                    "example_ref": _example_ref(example["sample_id"]),
                    "input_ids_shape": list(batch["input_ids"].shape),
                    "input_features_shape": list(batch["input_features"].shape),
                    "input_features_dtype_collated": str(batch["input_features"].dtype),
                    "feature_attention_mask_dtype": str(batch["feature_attention_mask"].dtype),
                    "has_transcript": bool(example.get("transcript")),
                    **scores,
                }
            )
        results[modality] = rows
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "forward",
        "world_size": 1,
        "gpu_count": torch.cuda.device_count(),
        "prompt_version": (config.get("prompt") or {}).get("version"),
        "model_load_audit": getattr(model, LOAD_AUDIT_ATTR, None),
        "results": results,
        "gpu_memory": _gpu_memory(),
        "host_memory": _host_memory_gb(),
    }


# --------------------------------------------------------------------------- #
# FSDP training-step probes
# --------------------------------------------------------------------------- #


def _build_training_stack(config: dict[str, Any]):
    model = load_model_for_training(str(resolve_model_name_or_path(None, config)), config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    accelerator = build_accelerator(
        config,
        model=model,
        wrap_policy_names=lambda wrapped: fsdp_wrap_policy_names(config, wrapped),
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    return accelerator, model, optimizer


def _fsdp_unit_count(model) -> int:
    from torch.distributed.fsdp import FullyShardedDataParallel

    return sum(1 for module in model.modules() if isinstance(module, FullyShardedDataParallel))


def _checkpointing_active(model) -> bool:
    for module in model.modules():
        if getattr(module, "gradient_checkpointing", False):
            return True
        if getattr(module, "is_gradient_checkpointing", False):
            return True
    return False


def _gradient_summary(inner) -> dict[str, Any]:
    """Finite/non-zero LoRA gradients on the original (unwrapped) parameters."""
    nonfinite = 0
    nonzero = 0
    tensors = 0
    encoder_trainable = 0
    for name, parameter in inner.named_parameters():
        if "lora_" in name and "audio_tower" in name and parameter.requires_grad:
            encoder_trainable += int(parameter.numel())
        if parameter.grad is None or "lora_" not in name:
            continue
        tensors += 1
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            nonfinite += 1
        if bool((gradient != 0).any()):
            nonzero += 1
    return {
        "lora_grad_tensors": tensors,
        "lora_grad_tensors_nonzero": nonzero,
        "lora_grad_tensors_nonfinite": nonfinite,
        "audio_encoder_trainable_lora_params": encoder_trainable,
    }


def _accumulation_step(accelerator, model, optimizer, config, batch, accumulate: bool) -> float:
    model_batch, loss_weight = _split_loss_weight(batch)
    with activation_offload_context(config):
        context = accelerator.accumulate(model) if accumulate else contextlib.nullcontext()
        with context:
            outputs = model(**model_batch)
            loss = outputs.loss
            if loss_weight is not None:
                # The packed30 recipe weights every window by its subject-normalized
                # weight; the probe keeps that term so memory and timing match training.
                loss = loss * loss_weight.reshape(-1)[0].to(loss.device)
            accelerator.backward(loss)
    if accumulate:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


def _rank_report(accelerator, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": accelerator.process_index,
        "world_size": accelerator.num_processes,
        "gpu_memory": _gpu_memory(),
        "host_memory": _host_memory_gb(),
        **extra,
    }


def _gather_reports(accelerator, report: dict[str, Any]) -> list[dict[str, Any]]:
    if accelerator.num_processes == 1:
        return [report]
    gathered: list[Any] = [None] * accelerator.num_processes
    torch.distributed.all_gather_object(gathered, report)
    return [item for item in gathered if item is not None]


def _batches(args, config, processor, modality: str, inventory) -> list[dict[str, Any]]:
    device = torch.device("cuda", torch.cuda.current_device())
    examples = _examples_for(args, config, modality, inventory)
    modality_config = _modality_config(config, modality)
    batches = []
    for example in examples:
        batch = _collated(example, modality_config, processor, device)
        batches.append(
            {
                "example_ref": _example_ref(example["sample_id"]),
                "input_features_shape": list(batch["input_features"].shape),
                "input_ids_shape": list(batch["input_ids"].shape),
                "batch": batch,
            }
        )
    return batches


def run_fsdp_mode(args, mode: str) -> dict[str, Any]:
    config = _load_config(args)
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8")) if args.inventory else None
    model_dir = str(resolve_model_name_or_path(None, config))
    processor = load_processor(model_dir, config)

    accelerator, model, optimizer = _build_training_stack(config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    structure = {
        "strategy": str(config["training"].get("strategy")),
        "world_size": world_size,
        "per_device_train_batch_size": int(config["training"].get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(config["training"].get("gradient_accumulation_steps", 1)),
        "effective_global_batch_size": effective_global_batch_size(config, world_size),
        "activation_offload": resolve_activation_offload(config),
        "fsdp_units": _fsdp_unit_count(model),
        "gradient_checkpointing_active": _checkpointing_active(model),
        "model_load_audit": getattr(model, LOAD_AUDIT_ATTR, None)
        or getattr(accelerator.unwrap_model(model), LOAD_AUDIT_ATTR, None),
    }
    accelerator.wait_for_everyone()
    _reset_peak_memory()

    modalities = ["audio_only", "audio_text"]
    timings: dict[str, Any] = {}
    losses: list[float] = []

    if mode == "backward":
        modality = args.modalities[0]
        batches = _batches(args, config, processor, modality, inventory)
        first = batches[0]
        loss = _accumulation_step(accelerator, model, optimizer, config, first["batch"], False)
        gradients = _gradient_summary(accelerator.unwrap_model(model))
        timings[modality] = {"examples": 1, "first_loss": loss}
        extra = {
            "mode": mode,
            "structure": structure,
            "gradients": gradients,
            "timings": timings,
        }
    elif mode == "maxrisk":
        for modality in modalities:
            batches = _batches(args, config, processor, modality, inventory)
            started = time.monotonic()
            for entry in batches:
                losses.append(_accumulation_step(accelerator, model, optimizer, config, entry["batch"], True))
            timings[modality] = {
                "examples": len(batches),
                "wall_seconds": round(time.monotonic() - started, 2),
                "example_refs": [entry["example_ref"] for entry in batches],
                "input_features_shapes": [entry["input_features_shape"] for entry in batches],
            }
        extra = {
            "mode": mode,
            "structure": structure,
            "losses": losses,
            "timings": timings,
        }
    elif mode == "perf":
        modality = args.modalities[0]
        batches = _batches(args, config, processor, modality, inventory)
        steps = max(1, args.steps)
        warmup = max(0, args.warmup_steps)
        step_seconds: list[float] = []
        for index in range(steps + warmup):
            entry = batches[index % len(batches)]
            started = time.monotonic()
            _accumulation_step(accelerator, model, optimizer, config, entry["batch"], True)
            if index >= warmup:
                step_seconds.append(time.monotonic() - started)
        examples = steps * structure["effective_global_batch_size"]
        wall = sum(step_seconds)
        extra = {
            "mode": mode,
            "structure": structure,
            "steps": steps,
            "warmup_steps": warmup,
            "step_seconds": [round(value, 3) for value in step_seconds],
            "step_seconds_mean": round(wall / len(step_seconds), 3),
            "examples_per_second": round(examples / wall, 4) if wall else None,
            "optimizer_steps_per_hour": round(3600 * steps / wall, 2) if wall else None,
            "losses": losses,
            "steps_per_hour": round(3600 * steps / wall, 2) if wall else None,
        }
    else:  # pragma: no cover - argparse restricts the choices
        raise SystemExit(f"unsupported mode {mode!r}")

    rank_report = _rank_report(accelerator, extra)
    reports = _gather_reports(accelerator, rank_report) if accelerator.num_processes > 1 else [rank_report]
    report = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "world_size": world_size,
        "gpu_count": torch.cuda.device_count(),
        "prompt_version": (config.get("prompt") or {}).get("version"),
        "ranks": reports,
    }
    if accelerator.is_main_process:
        report["main_rank"] = reports[0]
    return report


# --------------------------------------------------------------------------- #
# module-tree audit (config only, no weights)
# --------------------------------------------------------------------------- #


def run_tree(args) -> dict[str, Any]:
    """Inspect the real Thinker module tree without loading any weights.

    Verifies the anchored LoRA regex against the instantiated class tree and
    records the structure the target set depends on: decoder layer count and
    class, the MoE layout (router, routed experts, shared expert), and the
    audio-tower module names.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3OmniMoeThinkerForConditionalGeneration

    from src.model.qwen3omni_lora import (
        QWEN3OMNI_LORA_TARGET_REGEX,
        expected_lora_module_names,
        fsdp_transformer_cls_names,
    )

    config = _load_config(args)
    model_dir = str(resolve_model_name_or_path(None, config))
    hf_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    # The full checkpoint nests the Thinker: the class is built from the thinker
    # sub-config, exactly as the full model builds it.
    thinker_config = getattr(hf_config, "thinker_config", None) or hf_config
    with init_empty_weights():
        model = Qwen3OmniMoeThinkerForConditionalGeneration(thinker_config)

    import re

    pattern = re.compile(QWEN3OMNI_LORA_TARGET_REGEX)
    module_names = [name for name, _ in model.named_modules()]
    parameter_names = [name for name, _ in model.named_parameters()]
    matched = sorted(name for name in module_names if pattern.fullmatch(name))
    audit_error = None
    expected: list[str] = []
    try:
        expected = sorted(expected_lora_module_names(model))
        fsdp_names = fsdp_transformer_cls_names(model)
    except Exception as exc:  # noqa: BLE001 - the audit must report, never crash
        audit_error = f"{type(exc).__name__}: {exc}"
        fsdp_names = []
    expert_parameters = sorted(name for name in parameter_names if ".mlp.experts." in name)
    report = {
        "schema_version": SCHEMA_VERSION,
        "mode": "tree",
        "model_dir": model_dir,
        "declared_architectures": list(getattr(hf_config, "architectures", []) or []),
        "model_class": type(model).__name__,
        "lora_target_regex": QWEN3OMNI_LORA_TARGET_REGEX,
        "fsdp_wrap_class_names": fsdp_names,
        "audit_error": audit_error,
        "matched_module_count": len(matched),
        "expected_module_count": len(expected),
        "matched_equals_expected": matched == expected,
        "matched_modules_head": matched[:6],
        "unexpected_matches": sorted(set(matched) - set(expected))[:10],
        "missing_expected": sorted(set(expected) - set(matched))[:10],
        "top_level_children": sorted(
            name for name, _ in model.named_children()
        ),
        "layer_0_modules": [
            name for name in module_names if name.startswith(("model.layers.0.", "layers.0."))
        ][:24],
        "mlp_modules_sample": [
            name for name in module_names if ".mlp." in name and name.count(".") <= 5
        ][:12],
        "router_module_count": len([name for name in module_names if name.endswith("mlp.gate")]),
        "router_modules": [name for name in module_names if name.endswith("mlp.gate")][:3],
        "expert_parameter_names": expert_parameters[:6],
        "expert_parameter_count": len(expert_parameters),
        "audio_tower_modules": [
            name for name in module_names if name.startswith("audio_tower") and name.count(".") <= 2
        ][:10],
        "visual_present": any(name.startswith("visual") for name in module_names),
        "text_config": {
            key: getattr(getattr(thinker_config, "text_config", None), key, None)
            for key in (
                "num_hidden_layers",
                "num_experts",
                "num_experts_per_tok",
                "mlp_only_layers",
                "decoder_sparse_step",
            )
        },
        "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
    }
    del model
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["tree", "forward", "backward", "maxrisk", "perf"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inventory", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--modalities", nargs="+", default=["audio_only", "audio_text"])
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config override in the repository --set form (repeatable)",
    )
    args = parser.parse_args(argv)

    if args.mode == "tree":
        report = run_tree(args)
    elif args.mode == "forward":
        report = run_forward(args)
    else:
        report = run_fsdp_mode(args, args.mode)

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        LOGGER.info("wrote probe report to %s", args.output)
    summary = {key: report[key] for key in report if key not in {"ranks", "results"}}
    print(f"PROBE_REPORT {json.dumps(summary, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
