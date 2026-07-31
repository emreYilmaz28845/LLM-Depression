from __future__ import annotations

import copy
from typing import Any


def model_config(merged_config: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve the shared Qwen model config without importing torch."""

    if not records:
        raise ValueError("At least one component record is required to resolve a model config.")
    config = copy.deepcopy(records[0]["config"])
    modality = str(merged_config["modality"]).lower()
    config["model_name_or_path"] = merged_config.get(
        "model_name_or_path", config.get("model_name_or_path")
    )
    config.setdefault("data", {})["use_audio"] = modality in {"audio_text", "audio_only"}
    config.setdefault("data", {})["use_text"] = modality in {"audio_text", "text_only"}
    config["training"] = copy.deepcopy(merged_config.get("training", {}))
    config["training"]["selection_metric"] = "mean_dataset_macro_f1"
    config["training"]["selection_metric_mode"] = "max"
    config["evaluation"] = copy.deepcopy(records[0]["config"].get("evaluation", {}))
    config["evaluation"]["sample_prediction_mode"] = "original_teacher_forced"
    config["evaluation"]["headline_mode"] = "original_teacher_forced"
    config["output_dirs"] = copy.deepcopy(records[0]["config"].get("output_dirs", {}))
    return config
