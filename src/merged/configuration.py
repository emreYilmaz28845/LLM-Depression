from __future__ import annotations

import copy
from typing import Any


def _component_backend(record: dict[str, Any]) -> str:
    backend = str((record.get("config") or {}).get("model_backend") or "").strip().lower()
    return backend or "qwen"


def validate_shared_backend(merged_config: dict[str, Any], records: list[dict[str, Any]]) -> str:
    """Require one shared model backend across all merged components.

    A mixed-backend merged configuration is a hard stop: the merged model is
    one backbone, and every component must produce examples for that same
    backbone. Historical Qwen configs carry no ``model_backend`` (implicit
    Qwen), which is treated as the shared ``qwen`` backend.
    """
    if not records:
        raise ValueError("At least one component record is required to resolve a model backend.")
    declared = str(merged_config.get("model_backend") or "").strip().lower()
    backends = {_component_backend(record) for record in records}
    if len(backends) > 1:
        raise ValueError(
            f"Merged components must share one model backend; got {sorted(backends)}."
        )
    component_backend = next(iter(backends))
    if declared and component_backend and declared != component_backend:
        raise ValueError(
            f"Merged config model_backend {declared!r} does not match its "
            f"components' backend {component_backend!r}."
        )
    return declared or component_backend or "qwen"


def model_config(merged_config: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve the shared model config without importing torch.

    The backend is resolved from the merged config or the component records;
    every component must share one backend. For the Gemma backend the pinned
    revision and model path are carried into the resolved config so the
    runtime dispatch (loader, collator factory, example preparation) selects
    the Gemma path exactly like a standalone Gemma config.
    """
    if not records:
        raise ValueError("At least one component record is required to resolve a model config.")
    backend = validate_shared_backend(merged_config, records)
    config = copy.deepcopy(records[0]["config"])
    modality = str(merged_config["modality"]).lower()
    config["model_name_or_path"] = merged_config.get(
        "model_name_or_path", config.get("model_name_or_path")
    )
    if backend == "gemma4":
        config["model_backend"] = "gemma4"
        revision = merged_config.get("model_revision")
        if revision:
            config["model_revision"] = revision
    config.setdefault("data", {})["use_audio"] = modality in {"audio_text", "audio_only"}
    config.setdefault("data", {})["use_text"] = modality in {"audio_text", "text_only"}
    config["training"] = copy.deepcopy(merged_config.get("training", {}))
    config["training"]["selection_metric"] = "mean_dataset_macro_f1"
    config["training"]["selection_metric_mode"] = "max"
    # Marks the resolved config as the symmetric-merged context so the Gemma
    # validator applies only backend-level invariants (the per-dataset and
    # standalone selection-metric checks do not apply to a merged mix).
    config["merged"] = True
    config["evaluation"] = copy.deepcopy(records[0]["config"].get("evaluation", {}))
    config["evaluation"]["sample_prediction_mode"] = "original_teacher_forced"
    config["evaluation"]["headline_mode"] = "original_teacher_forced"
    config["output_dirs"] = copy.deepcopy(records[0]["config"].get("output_dirs", {}))
    return config
