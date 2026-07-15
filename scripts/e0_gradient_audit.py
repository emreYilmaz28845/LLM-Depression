#!/usr/bin/env python3
"""Run one controlled full-label backward pass on the pinned DAIC checkpoint.

This is a diagnostic, not training: it constructs an optimizer only to audit
membership, performs exactly one forward and one backward, and never calls
``optimizer.step``.  The baked deterministic K=4 example is used verbatim.  In
particular, a CUDA OOM is recorded and returned as a failure; the script never
reduces K, audio duration, transcript length, model precision, or model size.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# This must be set before PyTorch creates a CUDA/cuBLAS handle.  A caller may
# explicitly select the other supported deterministic workspace size.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.optim import AdamW

from src.data.runtime import AudioTextDataset
from src.e0_perturbations import (
    LEGACY_VIEW_FAMILY,
    _load_checkpoint_config,
    _resolve_input_examples,
    _resolve_model_path,
)
from src.model.collator import Qwen2AudioSFTCollator
from src.model.runtime import load_processor, resolve_processor_sampling_rate, restore_model_for_training
from src.utils import set_seed, sha256_file, sha256_text


SCHEMA_VERSION = 1
PROTOCOL = "e0_single_real_k4_full_label_forward_backward_no_step_v1"
EXPECTED_ADAPTER_SHA256 = "06b6f7592dfdfd9a0864acd7be59e80661a474324e43c3f14080e4e6e7ce5ed2"
EXPECTED_CHECKPOINT_INVENTORY_SHA256 = (
    "3dc0940ad9751321195e0ee359541d203261eecc5dbd75b40303b39aba027ff3"
)
EXPECTED_CONFIG_SNAPSHOT_SHA256 = (
    "e0533b255580bdca1afe5ee8598e20ae840836ddf8228caa64584e50cc435251"
)
EXPECTED_MANIFEST_METADATA_SHA256 = (
    "910e09d69361bd7872aab35fbde534d2204bd57759e2f72a0bd4a88aa14b4b04"
)
EXPECTED_MANIFEST_FILE_SHA256 = (
    "e31385760a0536a06f9ff38fe20e3eab9fa5dd6736c38de5bb8cd577438f61e3"
)
EXPECTED_SUBJECT_PARTITIONS_SHA256 = (
    "12b1a48cbfcc77771c6047ffa040616d16a66e99f8ad6d956687ffc5fb4d5fe4"
)
EXPECTED_SUBJECT_ID = "300"
EXPECTED_EXAMPLE_SHA256 = "327a298afcb39045b131199aff107315805db61be2da705024343ecf98d2f06b"
EXPECTED_AUDIO_SHA256 = {
    "300_random_segment_1.wav": "dbe7e644a7c5733c8663c294fc70b3da59d6a118fea1be473cbf8ef14d3b3cd7",
    "300_random_segment_3.wav": "efcc35662dbd949741895a0343be4e06d8f6b3084caa8b6a49a978efd024f88f",
    "300_random_segment_6.wav": "38bd11ddeb717cf90f5fd9f1c743f3fbaffc33c0526601a8dda15e1e4b4e7683",
    "300_random_segment_9.wav": "b1c9c44588f183ed8bfefd677f79d63f23522457f0a3e047e822a922a26a87fe",
}
EXPECTED_BASE_MODEL_FILES = {
    "config.json": "ab122e112e2450f10cec59216185bd519ccd7529c79d4c1c00255d43b97a037d",
    "model.safetensors.index.json": "b6cc05302d1bd25fbab6915e3a033603c524416b984f661d213f9a1f8e3b3895",
    "model-00001-of-00005.safetensors": "383de5b5b06f7e7f276850a065d9164641c93009f79a377352a7c949d17c1d7a",
    "model-00002-of-00005.safetensors": "610e59a23cdf1f78d7e3b42e69f2e1a08578f29726391a1772aecac95db3c59c",
    "model-00003-of-00005.safetensors": "b9ea76e97226524a12b6acc5e896bebf01a01018fbcad20a72eb13c7950f6916",
    "model-00004-of-00005.safetensors": "d68dee591619e26e3d0af23a07e181c70134fbc3aa33e24d872224607aae2c62",
    "model-00005-of-00005.safetensors": "a447dab3e72f9debd37a0c4ae0b9179a1bad2b06357092a77c066a0405bbc7d2",
}
DEFAULT_CHECKPOINT = Path(
    "output_model/audits/e0/checkpoints/"
    "daic_posf1_tf_daic_audio_text_selmacrof1_tf/fold_0/best_model"
)
DEFAULT_MANIFEST_METADATA = Path("outputs/splits/daic_manifest_metadata.json")
DECODER_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json_new(path: Path, payload: Any) -> None:
    """Atomically create an immutable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable audit artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Gradient-audit output directory must be new or empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def checkpoint_inventory(checkpoint_dir: Path) -> dict[str, Any]:
    """Hash every regular checkpoint file and a canonical inventory envelope."""
    entries = []
    for path in sorted(item for item in checkpoint_dir.rglob("*") if item.is_file()):
        entries.append(
            {
                "relative_path": path.relative_to(checkpoint_dir).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise FileNotFoundError(f"Checkpoint directory contains no files: {checkpoint_dir}")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_file_count": len(entries),
        "checkpoint_inventory_sha256": sha256_text(canonical),
        "files": entries,
    }


def _pinned_file(path: Path, expected_sha256: str, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Pinned {description} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"Pinned {description} hash mismatch: observed={observed} "
            f"expected={expected_sha256} path={path}"
        )
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": observed,
    }


def _pinned_base_model_inventory(model_path: Path) -> dict[str, Any]:
    files = [
        _pinned_file(model_path / name, digest, f"base-model file {name}")
        | {"relative_path": name}
        for name, digest in sorted(EXPECTED_BASE_MODEL_FILES.items())
    ]
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "model_path": str(model_path.resolve()),
        "verified_file_count": len(files),
        "verified_files_sha256": sha256_text(canonical),
        "files": files,
    }


def _canonical_example_sha256(example: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        example,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256_text(serialized)


def select_single_k4_example(
    examples: list[dict[str, Any]], subject_id: str | None = None
) -> dict[str, Any]:
    """Select exactly one stable, already-materialized K=4 example."""
    ordered = sorted(examples, key=lambda item: str(item["subject_id"]))
    if subject_id is None:
        selected = ordered[0] if ordered else None
    else:
        selected = next(
            (item for item in ordered if str(item["subject_id"]) == str(subject_id)),
            None,
        )
    if selected is None:
        raise ValueError(f"No deterministic example found for subject_id={subject_id!r}.")
    audio_paths = list(selected.get("audio_paths") or [])
    clip_seconds = list(selected.get("audio_clip_seconds") or [])
    if len(audio_paths) != 4 or len(clip_seconds) != 4:
        raise ValueError(
            "The controlled backward audit requires the unmodified baked K=4 bundle; "
            f"subject={selected.get('subject_id')} has paths={len(audio_paths)} "
            f"clip_limits={len(clip_seconds)}."
        )
    if int(selected.get("chunks_per_subject", 4)) != 4:
        raise ValueError("Selected example does not declare chunks_per_subject=4.")
    return selected


def _example_identity(example: Mapping[str, Any]) -> dict[str, Any]:
    audio_paths = [str(path) for path in example["audio_paths"]]
    return {
        "subject_id": str(example["subject_id"]),
        "sample_id": str(example["sample_id"]),
        "label": int(example["label"]),
        "internal_label_text": str(example["internal_label_text"]),
        "audio_ids": [Path(path).stem for path in audio_paths],
        "audio_paths": audio_paths,
        "audio_clip_seconds": [float(value) for value in example["audio_clip_seconds"]],
        "k": len(audio_paths),
        "prompt_sha256": sha256_text(str(example["prompt_text"])),
        "training_text_sha256": sha256_text(str(example["training_text"])),
        "prompt_character_count": len(str(example["prompt_text"])),
        "training_text_character_count": len(str(example["training_text"])),
        "selection": "lexicographically_first_subject_unless_explicit_subject_id",
        "view": "unmodified_baked_legacy_deterministic_k4",
    }


def _shape_metadata(batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
        }
        for key, value in sorted(batch.items())
    }


def target_metadata(
    batch: Mapping[str, torch.Tensor],
    processor: Any,
    expected_label_text: str,
    expected_target_text: str,
) -> dict[str, Any]:
    labels = batch["labels"]
    input_ids = batch["input_ids"]
    if labels.ndim != 2 or labels.shape[0] != 1 or input_ids.shape != labels.shape:
        raise ValueError(f"Expected one collated label sequence, received {tuple(labels.shape)}.")
    supervised = labels[0].ne(-100)
    target_ids = labels[0][supervised].detach().cpu().tolist()
    if not target_ids:
        raise ValueError("The full-label collator left no supervised target tokens.")
    first_target = int(torch.nonzero(supervised, as_tuple=False)[0].item())
    if bool(supervised[:first_target].any().item()) or not bool(
        supervised[first_target:].all().item()
    ):
        raise ValueError("Label mask is not one contiguous prompt prefix and target suffix.")
    if not torch.equal(labels[0, first_target:], input_ids[0, first_target:]):
        raise ValueError("Supervised labels do not equal input_ids on the target suffix.")
    tokenizer = getattr(processor, "tokenizer", processor)
    decoded_target = str(tokenizer.decode(target_ids, skip_special_tokens=False))
    if decoded_target != str(expected_target_text):
        raise ValueError(
            "Collated supervised target is not the exact full training suffix: "
            f"decoded={decoded_target!r} expected={expected_target_text!r}."
        )
    return {
        "expected_internal_label_text": str(expected_label_text),
        "expected_full_target_text": str(expected_target_text),
        "sequence_tokens": int(labels.shape[1]),
        "masked_prompt_tokens": first_target,
        "supervised_target_tokens": len(target_ids),
        "target_token_ids": [int(value) for value in target_ids],
        "target_token_strings": list(tokenizer.convert_ids_to_tokens(target_ids)),
        "decoded_supervised_target": decoded_target,
        "mask_is_contiguous_prompt_then_target": True,
        "supervised_labels_equal_input_ids": True,
        "decoded_target_equals_expected_full_suffix": True,
        "ignore_index": -100,
        "collation": "Qwen2AudioSFTCollator over real training_text",
    }


def _move_batch_for_backward(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    moved = {key: value.to(device) for key, value in batch.items()}
    if "input_features" not in moved:
        raise ValueError("Audio-input sensitivity requires collated input_features.")
    # Detach after the device copy so this is a leaf whose gradient is retained.
    moved["input_features"] = moved["input_features"].detach().requires_grad_(True)
    return moved


def one_forward_backward(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    use_bf16_autocast: bool,
) -> tuple[float, dict[str, torch.Tensor]]:
    """Perform the protocol's only forward/backward pair, with no optimizer step."""
    moved = _move_batch_for_backward(batch, device)
    model.zero_grad(set_to_none=True)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16_autocast
        else nullcontext()
    )
    with autocast_context:
        outputs = model(**moved, use_cache=False)
        loss = outputs.loss
    if loss is None or loss.numel() != 1 or not bool(torch.isfinite(loss.detach()).item()):
        raise RuntimeError(f"Controlled forward returned invalid scalar loss: {loss!r}")
    loss.backward()
    return float(loss.detach().item()), moved


def _decoder_layer_index(parameter_name: str) -> int | None:
    match = DECODER_LAYER_PATTERN.search(parameter_name)
    return int(match.group(1)) if match else None


def _primary_parameter_group(name: str) -> str:
    if "audio_adapter" in name:
        return "audio_adapter"
    if "audio_tower" in name:
        return "frozen_audio_tower"
    if "multi_modal_projector" in name:
        return "frozen_projector"
    if "lora_" in name:
        return "lora_overall"
    return "other"


def parameter_records(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> list[dict[str, Any]]:
    optimizer_ids = {
        id(parameter)
        for optimizer_group in optimizer.param_groups
        for parameter in optimizer_group["params"]
    }
    records = []
    for name, parameter in model.named_parameters():
        records.append(
            {
                "name": name,
                "parameter": parameter,
                "group": _primary_parameter_group(name),
                "decoder_layer": _decoder_layer_index(name),
                "optimizer_member": id(parameter) in optimizer_ids,
            }
        )
    return records


def _combined_l2(tensors: Iterable[torch.Tensor]) -> float:
    squared_norm = 0.0
    for tensor in tensors:
        # Accumulate in FP32 without materializing a full FP32 copy of large BF16
        # weights. CUDA peak memory is sampled before these diagnostics.
        norm = torch.linalg.vector_norm(tensor.detach(), ord=2, dtype=torch.float32)
        value = float(norm.item())
        squared_norm += value * value
    return float(math.sqrt(squared_norm))


def _status(count: int, total: int) -> str:
    if total == 0:
        return "empty"
    if count == 0:
        return "none"
    if count == total:
        return "all"
    return "mixed"


def summarize_parameter_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    parameters = [record["parameter"] for record in records]
    requires_grad = [parameter for parameter in parameters if parameter.requires_grad]
    optimizer_members = [
        record["parameter"] for record in records if record["optimizer_member"]
    ]
    with_grad = [parameter for parameter in parameters if parameter.grad is not None]
    finite_grads = [
        parameter
        for parameter in with_grad
        if bool(torch.isfinite(parameter.grad.detach()).all().item())
    ]
    return {
        "parameter_tensor_count": len(parameters),
        "parameter_element_count": int(sum(parameter.numel() for parameter in parameters)),
        "parameter_l2_norm": _combined_l2(parameters),
        "requires_grad_status": _status(len(requires_grad), len(parameters)),
        "requires_grad_tensor_count": len(requires_grad),
        "requires_grad_element_count": int(sum(parameter.numel() for parameter in requires_grad)),
        "optimizer_membership_status": _status(len(optimizer_members), len(parameters)),
        "optimizer_member_tensor_count": len(optimizer_members),
        "optimizer_member_element_count": int(
            sum(parameter.numel() for parameter in optimizer_members)
        ),
        "gradient_status": _status(len(with_grad), len(parameters)),
        "gradient_tensor_count": len(with_grad),
        "gradient_element_count": int(sum(parameter.grad.numel() for parameter in with_grad)),
        "gradient_l2_norm": _combined_l2(parameter.grad for parameter in with_grad),
        "finite_gradient_tensor_count": len(finite_grads),
        "nonfinite_gradient_tensor_count": len(with_grad) - len(finite_grads),
        "parameter_dtypes": sorted({str(parameter.dtype).removeprefix("torch.") for parameter in parameters}),
        "example_parameter_names": [record["name"] for record in records[:5]],
    }


def parameter_audit(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    decoder_layer_count: int,
) -> dict[str, Any]:
    records = parameter_records(model, optimizer)
    primary_names = (
        "frozen_audio_tower",
        "frozen_projector",
        "audio_adapter",
        "lora_overall",
        "other",
    )
    primary = {
        name: summarize_parameter_records(
            [record for record in records if record["group"] == name]
        )
        for name in primary_names
    }
    lora_records = [record for record in records if record["group"] == "lora_overall"]
    per_layer = {
        str(index): summarize_parameter_records(
            [record for record in lora_records if record["decoder_layer"] == index]
        )
        for index in range(decoder_layer_count)
    }
    unassigned_lora = [record for record in lora_records if record["decoder_layer"] is None]
    trainable_names = {
        record["name"] for record in records if record["parameter"].requires_grad
    }
    optimizer_names = {record["name"] for record in records if record["optimizer_member"]}
    return {
        "group_definition": {
            "primary_groups_are_disjoint": True,
            "audio_adapter": "name contains audio_adapter (reported separately from tower)",
            "frozen_audio_tower": "name contains audio_tower, excluding audio_adapter",
            "frozen_projector": "name contains multi_modal_projector",
            "lora_overall": "name contains lora_ outside audio/projector groups",
            "other": "all remaining model parameters",
            "per_decoder_layer": "LoRA-only subsets parsed from .layers.<index>.",
        },
        "primary_groups": primary,
        "lora_per_decoder_layer": per_layer,
        "unassigned_lora": summarize_parameter_records(unassigned_lora),
        "membership_invariants": {
            "trainable_parameter_names_equal_optimizer_parameter_names": trainable_names
            == optimizer_names,
            "trainable_not_in_optimizer": sorted(trainable_names - optimizer_names),
            "frozen_in_optimizer": sorted(optimizer_names - trainable_names),
            "trainable_tensor_count": len(trainable_names),
            "optimizer_tensor_count": len(optimizer_names),
        },
    }


def audio_input_sensitivity(input_features: torch.Tensor) -> dict[str, Any]:
    gradient = input_features.grad
    return {
        "input_features_shape": list(input_features.shape),
        "input_features_dtype": str(input_features.dtype).removeprefix("torch."),
        "input_features_requires_grad": bool(input_features.requires_grad),
        "gradient_present": gradient is not None,
        "gradient_element_count": int(gradient.numel()) if gradient is not None else 0,
        "gradient_l2_norm": _combined_l2([gradient]) if gradient is not None else 0.0,
        "gradient_is_finite": bool(torch.isfinite(gradient).all().item())
        if gradient is not None
        else None,
        "gradient_is_nonzero": bool(torch.count_nonzero(gradient).item())
        if gradient is not None
        else False,
        "interpretation": (
            "A nonzero loss gradient with respect to the real mel input proves local "
            "input sensitivity for this subject; it does not prove useful or clinical audio use."
        ),
    }


def _determinism_metadata() -> dict[str, Any]:
    return {
        "seed": None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _memory_metadata(device: torch.device) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "device_name": properties.name,
        "device_total_memory_bytes": int(properties.total_memory),
        "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _model_training_state(model: torch.nn.Module, config: Mapping[str, Any]) -> dict[str, Any]:
    base = model.base_model.model if hasattr(model, "base_model") else model
    base_config = getattr(base, "config", None)
    return {
        "model_training": bool(model.training),
        "configured_bf16": bool(config["training"].get("bf16", False)),
        "configured_gradient_checkpointing": bool(
            config["training"].get("gradient_checkpointing", False)
        ),
        "model_is_gradient_checkpointing": bool(
            getattr(base, "is_gradient_checkpointing", False)
        ),
        "model_use_cache": getattr(getattr(model, "config", None), "use_cache", None),
        "base_model_use_cache": getattr(base_config, "use_cache", None),
        "floating_parameter_dtypes": sorted(
            {
                str(parameter.dtype).removeprefix("torch.")
                for parameter in model.parameters()
                if parameter.is_floating_point()
            }
        ),
        "restore_function": "src.model.runtime.restore_model_for_training",
    }


def _validate_training_state(state: Mapping[str, Any]) -> None:
    if not state["model_training"]:
        raise RuntimeError("Restored checkpoint is not in training mode.")
    if not state["configured_bf16"]:
        raise RuntimeError("Resolved checkpoint config does not enable BF16.")
    if not state["configured_gradient_checkpointing"]:
        raise RuntimeError("Resolved checkpoint config does not enable gradient checkpointing.")
    if not state["model_is_gradient_checkpointing"]:
        raise RuntimeError("Training restoration did not enable gradient checkpointing.")
    if state["model_use_cache"] is not False or state["base_model_use_cache"] is not False:
        raise RuntimeError("Training restoration did not disable the model KV cache.")


def _model_placement_metadata(
    model: torch.nn.Module, expected_device: torch.device
) -> dict[str, Any]:
    grouped: dict[str, dict[str, int]] = {}
    wrong_device: list[str] = []
    wrong_bf16_base: list[str] = []
    for name, parameter in model.named_parameters():
        group = _primary_parameter_group(name)
        key = f"{str(parameter.device)}|{str(parameter.dtype).removeprefix('torch.')}"
        grouped.setdefault(group, {})[key] = grouped.setdefault(group, {}).get(key, 0) + int(
            parameter.numel()
        )
        if parameter.device != expected_device:
            wrong_device.append(name)
        if (
            parameter.is_floating_point()
            and group in {"frozen_audio_tower", "frozen_projector", "other"}
            and parameter.dtype != torch.bfloat16
        ):
            wrong_bf16_base.append(name)
    if wrong_device:
        raise RuntimeError(f"Model parameters outside {expected_device}: {wrong_device[:8]}")
    if wrong_bf16_base:
        raise RuntimeError(
            "Frozen/base checkpoint parameters were not loaded in BF16: "
            f"{wrong_bf16_base[:8]}"
        )
    return {
        "expected_device": str(expected_device),
        "all_parameters_on_expected_cuda_device": True,
        "frozen_base_parameters_bfloat16": True,
        "group_device_dtype_element_counts": grouped,
        "lora_fp32_is_expected_from_saved_adapter": True,
    }


def _validate_determinism(settings: Mapping[str, Any]) -> None:
    if settings["cublas_workspace_config"] not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 before the audit."
        )
    if not settings["deterministic_algorithms_enabled"]:
        raise RuntimeError("PyTorch deterministic algorithms are not enabled.")
    if not settings["cudnn_deterministic"] or settings["cudnn_benchmark"]:
        raise RuntimeError("cuDNN deterministic settings were not restored.")
    if settings["python_hash_seed"] != "0":
        raise RuntimeError("Launch the audit with PYTHONHASHSEED=0.")


def _load_trainable_checkpoint(
    model_path: Path,
    checkpoint_dir: Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    # Lazy imports keep CPU unit tests independent of heavyweight model loading.
    from peft import PeftModel
    from transformers import Qwen2AudioForConditionalGeneration

    from src.model.qwen2audio_lora import (
        configure_trainable_audio_modules,
        enforce_audio_encoder_freeze,
        load_additional_audio_modules,
        resolve_audio_adapter_config,
    )

    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    audio_load = load_additional_audio_modules(base_model, checkpoint_dir)
    model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint_dir),
        is_trainable=True,
    )
    configure_trainable_audio_modules(model, resolve_audio_adapter_config(config))
    freeze_guard = enforce_audio_encoder_freeze(model, config)
    model.to(device)
    model.train()
    restore_model_for_training(model, config)
    return model, {
        "peft_adapter_is_trainable_requested": True,
        "additional_audio_modules": audio_load,
        "audio_freeze_guard": freeze_guard,
    }


def _is_cuda_oom(exception: BaseException) -> bool:
    if isinstance(exception, torch.cuda.OutOfMemoryError):
        return True
    message = str(exception).lower()
    return isinstance(exception, RuntimeError) and (
        "cuda out of memory" in message or "cuda error: out of memory" in message
    )


def _safe_cuda_failure_memory() -> dict[str, Any]:
    try:
        device = torch.device("cuda:0")
        return {
            "device": str(device),
            "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    except Exception as exception:
        return {"unavailable": True, "reason": str(exception)}


def _failure_payload(
    *,
    exception: BaseException,
    stage: str,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "failed_cuda_oom",
        "created_at_utc": _utc_now(),
        "failed_stage": stage,
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "request": dict(request),
        "determinism": _determinism_metadata(),
        "cuda_memory_at_failure": _safe_cuda_failure_memory(),
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "script_sha256": sha256_file(Path(__file__)),
        },
        "fallback_policy": {
            "fallback_attempted": False,
            "k_reduced": False,
            "audio_reduced": False,
            "text_reduced": False,
            "precision_changed": False,
            "model_changed": False,
        },
        "exit_code": 2,
    }


def _decoder_layer_count(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    value = getattr(text_config, "num_hidden_layers", None)
    if value is None:
        value = getattr(config, "num_hidden_layers", None)
    if value is None:
        raise ValueError("Could not resolve decoder layer count for per-layer audit.")
    return int(value)


def run_gpu_audit(
    *,
    checkpoint_dir: Path,
    model_path: Path,
    config: dict[str, Any],
    processor: Any,
    example: dict[str, Any],
    request: dict[str, Any],
    stage: dict[str, str],
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    stage["name"] = "load_trainable_bf16_checkpoint"
    model, checkpoint_load = _load_trainable_checkpoint(
        model_path, checkpoint_dir, config, device
    )
    training_state_before = _model_training_state(model, config)
    _validate_training_state(training_state_before)
    placement = _model_placement_metadata(model, device)
    determinism = _determinism_metadata()
    determinism["seed"] = int(request["seed"])
    determinism["determinism_level"] = "best_effort_warn_only"
    _validate_determinism(determinism)

    stage["name"] = "load_and_collate_unmodified_k4_example"
    sampling_rate = resolve_processor_sampling_rate(processor)
    if sampling_rate is None:
        raise ValueError("The Qwen2-Audio processor has no sampling rate.")
    dataset = AudioTextDataset(
        [example],
        processor_sampling_rate=sampling_rate,
        silence_audio=False,
        chunk_sampling="deterministic",
        audio_augment=None,
    )
    collator = Qwen2AudioSFTCollator(processor=processor, debug=True)
    batch = collator([dataset[0]])
    prompt_text = str(example["prompt_text"])
    training_text = str(example["training_text"])
    if not training_text.startswith(prompt_text):
        raise ValueError("The selected training_text does not preserve its prompt prefix.")
    target = target_metadata(
        batch,
        processor,
        str(example["internal_label_text"]),
        training_text[len(prompt_text) :],
    )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("PEFT checkpoint was loaded without trainable parameters.")
    optimizer = AdamW(
        trainable_parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    stage["name"] = "single_forward_backward"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    loss, moved_batch = one_forward_backward(
        model,
        batch,
        device=device,
        use_bf16_autocast=True,
    )
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    # Capture the scientific run's peak before norm reductions allocate scratch.
    memory = _memory_metadata(device)

    stage["name"] = "summarize_gradients"
    sensitivity = audio_input_sensitivity(moved_batch["input_features"])
    parameters = parameter_audit(
        model,
        optimizer,
        decoder_layer_count=_decoder_layer_count(model),
    )
    training_state_after = _model_training_state(model, config)
    _validate_training_state(training_state_after)
    if not parameters["membership_invariants"][
        "trainable_parameter_names_equal_optimizer_parameter_names"
    ]:
        raise RuntimeError("Optimizer membership does not exactly equal requires_grad membership.")
    for frozen_group in ("frozen_audio_tower", "frozen_projector"):
        summary = parameters["primary_groups"][frozen_group]
        if summary["parameter_tensor_count"] == 0:
            raise RuntimeError(f"Expected {frozen_group} parameters, but the group was empty.")
        if summary["requires_grad_tensor_count"] or summary["optimizer_member_tensor_count"]:
            raise RuntimeError(f"Expected {frozen_group} to be frozen and outside the optimizer.")
    if parameters["primary_groups"]["lora_overall"]["requires_grad_tensor_count"] == 0:
        raise RuntimeError("No trainable LoRA tensors were found after checkpoint restoration.")
    if parameters["unassigned_lora"]["parameter_tensor_count"]:
        raise RuntimeError("Some LoRA parameters could not be assigned to a decoder layer.")
    empty_lora_layers = [
        index
        for index, summary in parameters["lora_per_decoder_layer"].items()
        if summary["parameter_tensor_count"] == 0
    ]
    if empty_lora_layers:
        raise RuntimeError(f"Decoder layers without expected LoRA parameters: {empty_lora_layers}")

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "completed",
        "created_at_utc": _utc_now(),
        "request": request,
        "checkpoint_load": checkpoint_load,
        "example": _example_identity(example),
        "collation": {
            "cpu_input_shapes": _shape_metadata(batch),
            "cuda_input_shapes": _shape_metadata(moved_batch),
            "target": target,
        },
        "training_state": {
            "before_forward_backward": training_state_before,
            "after_forward_backward": training_state_after,
        },
        "model_placement": placement,
        "execution": {
            "forward_count": 1,
            "backward_count": 1,
            "optimizer_step_count": 0,
            "optimizer_zero_grad_after_backward_count": 0,
            "loss": loss,
            "elapsed_seconds": float(elapsed),
            "autocast_dtype": "bfloat16",
        },
        "audio_input_sensitivity": sensitivity,
        "parameter_audit": parameters,
        "cuda_memory": memory,
        "determinism": determinism,
        "interpretation_guardrail": (
            "This single-example local gradient is a wiring/sensitivity diagnostic. "
            "It is not evidence that audio improves held-out predictions or encodes "
            "clinically valid depression cues."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest-metadata", type=Path, default=DEFAULT_MANIFEST_METADATA)
    parser.add_argument("--model-name-or-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    config_snapshot = checkpoint_dir / "standalone_eval" / "eval_config.yaml"
    manifest_metadata = (
        args.manifest_metadata
        if args.manifest_metadata.is_absolute()
        else PROJECT_ROOT / args.manifest_metadata
    ).resolve()
    output_dir = args.output_dir.expanduser().resolve()

    adapter_artifact = _pinned_file(
        checkpoint_dir / "adapter_model.safetensors",
        EXPECTED_ADAPTER_SHA256,
        "DAIC PEFT adapter",
    )
    config_artifact = _pinned_file(
        config_snapshot,
        EXPECTED_CONFIG_SNAPSHOT_SHA256,
        "checkpoint config snapshot",
    )
    metadata_artifact = _pinned_file(
        manifest_metadata,
        EXPECTED_MANIFEST_METADATA_SHA256,
        "local DAIC manifest metadata",
    )
    config, config_envelope = _load_checkpoint_config(config_snapshot)
    examples, view, manifest_info, manifest_path, partition_path = _resolve_input_examples(
        config,
        manifest_metadata,
        partition="test",
        expected_k=4,
        view_family=LEGACY_VIEW_FAMILY,
        view_index=0,
    )
    manifest_artifact = _pinned_file(
        manifest_path,
        EXPECTED_MANIFEST_FILE_SHA256,
        "local DAIC JSONL manifest",
    )
    partition_artifact = _pinned_file(
        partition_path,
        EXPECTED_SUBJECT_PARTITIONS_SHA256,
        "DAIC subject partitions",
    )
    example = select_single_k4_example(examples, EXPECTED_SUBJECT_ID)
    example_sha256 = _canonical_example_sha256(example)
    if example_sha256 != EXPECTED_EXAMPLE_SHA256:
        raise ValueError(
            "Pinned materialized subject example changed: "
            f"observed={example_sha256} expected={EXPECTED_EXAMPLE_SHA256}."
        )
    audio_names = [Path(path).name for path in example["audio_paths"]]
    if set(audio_names) != set(EXPECTED_AUDIO_SHA256) or len(audio_names) != 4:
        raise ValueError(f"Pinned subject audio IDs changed: {audio_names}")
    audio_artifacts = [
        _pinned_file(Path(path), EXPECTED_AUDIO_SHA256[Path(path).name], "K=4 audio input")
        | {"audio_id": Path(path).stem}
        for path in example["audio_paths"]
    ]
    model_path = _resolve_model_path(args.model_name_or_path, config, checkpoint_dir)
    inventory = checkpoint_inventory(checkpoint_dir)
    if inventory["checkpoint_inventory_sha256"] != EXPECTED_CHECKPOINT_INVENTORY_SHA256:
        raise ValueError(
            "Pinned checkpoint inventory changed: "
            f"observed={inventory['checkpoint_inventory_sha256']} "
            f"expected={EXPECTED_CHECKPOINT_INVENTORY_SHA256}."
        )
    base_model_inventory = _pinned_base_model_inventory(model_path)
    request = {
        "seed": int(args.seed),
        "checkpoint_dir": str(checkpoint_dir),
        "adapter_model_sha256": adapter_artifact["sha256"],
        "expected_adapter_model_sha256": EXPECTED_ADAPTER_SHA256,
        "checkpoint_inventory_sha256": inventory["checkpoint_inventory_sha256"],
        "checkpoint_file_count": inventory["checkpoint_file_count"],
        "checkpoint_files": inventory["files"],
        "config_snapshot": str(config_snapshot),
        "config_snapshot_sha256": config_artifact["sha256"],
        "config_snapshot_envelope": config_envelope,
        "manifest_metadata": str(manifest_metadata),
        "manifest_metadata_sha256": metadata_artifact["sha256"],
        "manifest": str(manifest_path),
        "manifest_hash": manifest_info.get("manifest_hash"),
        "manifest_file_sha256": manifest_artifact["sha256"],
        "subject_partitions": str(partition_path),
        "subject_partitions_sha256": partition_artifact["sha256"],
        "partition": "test",
        "model_name_or_path": str(model_path),
        "base_model_inventory": base_model_inventory,
        "selected_subject_id": str(example["subject_id"]),
        "selected_example_sha256": example_sha256,
        "selected_audio_files": audio_artifacts,
        "expected_k": 4,
        "view": view,
        "no_fallback_on_oom": True,
        "script_sha256": sha256_file(Path(__file__)),
    }
    _prepare_output_dir(output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("The pinned E0 gradient audit requires CUDA; CPU fallback is forbidden.")
    if not bool(config["training"].get("bf16", False)):
        raise ValueError("Pinned checkpoint config does not request BF16.")
    if not bool(config["training"].get("gradient_checkpointing", False)):
        raise ValueError("Pinned checkpoint config does not request gradient checkpointing.")

    set_seed(int(args.seed), deterministic=True)
    launch_determinism = _determinism_metadata()
    launch_determinism["seed"] = int(args.seed)
    _validate_determinism(launch_determinism)
    stage = {"name": "load_processor"}
    try:
        processor = load_processor(checkpoint_dir, config)
        result = run_gpu_audit(
            checkpoint_dir=checkpoint_dir,
            model_path=model_path,
            config=config,
            processor=processor,
            example=example,
            request=request,
            stage=stage,
        )
    except BaseException as exception:
        if not _is_cuda_oom(exception):
            raise
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        failure = _failure_payload(
            exception=exception,
            stage=stage["name"],
            request=request,
        )
        _atomic_write_json_new(output_dir / "gradient_audit_failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr, flush=True)
        return 2

    result["environment"] = {
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "script_sha256": sha256_file(Path(__file__)),
    }
    _atomic_write_json_new(output_dir / "gradient_audit.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
