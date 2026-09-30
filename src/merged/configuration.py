from __future__ import annotations

import copy
from typing import Any


QWEN3_FAMILY_BACKENDS = ("qwen38", "qwen3omni")
EVALUATION_MODE_LIKELIHOOD = "likelihood"
EVALUATION_MODE_ORIGINAL_TEACHER_FORCED = "original_teacher_forced"
EVALUATION_CONTRACT_SCHEMA_VERSION = "symmetric_merged_evaluation_contract.v1"
EVALUATION_RESOURCE_SCHEMA_VERSION = "symmetric_merged_evaluation_resources.v1"


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


def is_qwen3_backend(backend: str | None) -> bool:
    """True when the resolved merged backend belongs to the Qwen3 family."""
    return str(backend or "").strip().lower() in QWEN3_FAMILY_BACKENDS


def head_support_ready(backend: str | None) -> bool:
    """Whether hidden-feature extraction and the head stage may run for a route.

    Qwen3 merged checkpoints are extracted through the same merged postprocess
    path as every other family: the extractor resolves the backend, the sharded
    device map and the route-specific hidden dimension from the checkpoint's own
    contract, and the extracted dimension is validated against the backend's
    recorded hidden size. Route-level *production* readiness is separate and
    lives in the submission guard's per-route ``head_ready`` table, which stays
    closed until that route's own bounded audit has run.
    """
    return True


def _evaluation_field(record: dict[str, Any], field: str) -> str:
    evaluation = (record.get("config") or {}).get("evaluation") or {}
    return str(evaluation.get(field) or "").strip()


def _record_dataset(record: dict[str, Any]) -> str:
    """Dataset identity of a component record.

    ``load_component_records`` reports it at the record level; lightweight
    callers (planner-side contract validation, tests) may only carry it inside
    the component config.
    """
    dataset = record.get("dataset")
    if dataset in (None, ""):
        dataset = (record.get("config") or {}).get("dataset")
    return str(dataset or "").strip().lower()


def validate_evaluation_contract(
    merged_config: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    backend: str | None = None,
) -> dict[str, Any]:
    """Validate the decision rule and the evidence view separately per component.

    ``evaluation.sample_prediction_mode`` (the decision rule) and
    ``evaluation.evaluation_view`` (the evidence qualifier recorded beside the
    metrics) are different fields and are validated separately. Qwen3 merged
    contracts must declare both in every component, select ``likelihood`` and
    agree on one view; a missing, partial or heterogeneous declaration fails
    closed before a job is submitted, and there is no silent teacher-forced
    fallback on the Qwen3 path. Legacy (non-Qwen3) contracts keep the historical
    merged decision rule and their declarations are not re-interpreted here.
    """
    resolved_backend = backend or validate_shared_backend(merged_config, records)
    component_modes = {
        _record_dataset(record): _evaluation_field(record, "sample_prediction_mode")
        for record in records
    }
    component_views = {
        _record_dataset(record): _evaluation_field(record, "evaluation_view")
        for record in records
    }
    component_headlines = {
        _record_dataset(record): _evaluation_field(record, "headline_mode")
        for record in records
    }
    contract: dict[str, Any] = {
        "schema_version": EVALUATION_CONTRACT_SCHEMA_VERSION,
        "backend": resolved_backend,
        "qwen3_family": is_qwen3_backend(resolved_backend),
        "declared": False,
        "sample_prediction_mode": "",
        "evaluation_view": "",
        "component_sample_prediction_modes": component_modes,
        "component_evaluation_views": component_views,
        "component_headline_modes": component_headlines,
    }
    if not contract["qwen3_family"]:
        return contract
    missing_modes = sorted(dataset for dataset, value in component_modes.items() if not value)
    if missing_modes:
        raise ValueError(
            "Qwen3 merged contracts must declare evaluation.sample_prediction_mode in every "
            f"component; missing for {missing_modes}."
        )
    wrong_modes = sorted(
        dataset
        for dataset, value in component_modes.items()
        if value.lower() != EVALUATION_MODE_LIKELIHOOD
    )
    if wrong_modes:
        raise ValueError(
            f"Qwen3 merged contracts must use evaluation.sample_prediction_mode="
            f"{EVALUATION_MODE_LIKELIHOOD!r}; refusing {sorted(set(component_modes.values()))} "
            f"for {wrong_modes}."
        )
    missing_headlines = sorted(dataset for dataset, value in component_headlines.items() if not value)
    if missing_headlines:
        raise ValueError(
            "Qwen3 merged contracts must declare evaluation.headline_mode in every component; "
            f"missing for {missing_headlines}."
        )
    headline_mismatch = sorted(
        dataset
        for dataset, value in component_headlines.items()
        if value.lower() != component_modes[dataset].lower()
    )
    if headline_mismatch:
        raise ValueError(
            "Qwen3 merged components must keep evaluation.headline_mode equal to "
            f"sample_prediction_mode; mismatched for {headline_mismatch}."
        )
    missing_views = sorted(dataset for dataset, value in component_views.items() if not value)
    if missing_views:
        raise ValueError(
            "Qwen3 merged contracts must declare evaluation.evaluation_view in every component; "
            f"missing for {missing_views}."
        )
    distinct_views = sorted(set(component_views.values()))
    if len(distinct_views) != 1:
        raise ValueError(
            "Qwen3 merged components must agree on one evaluation.evaluation_view; got "
            f"{component_views}."
        )
    contract.update(
        {
            "declared": True,
            "sample_prediction_mode": EVALUATION_MODE_LIKELIHOOD,
            "evaluation_view": distinct_views[0],
        }
    )
    return contract


def validate_merged_resources(
    merged_config: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    backend: str | None = None,
) -> dict[str, Any]:
    """Resolve the postprocess evaluation shape from the merged contract.

    ``execution.postprocess_gpus`` is the merged contract's declared evaluation
    shape; when it is declared it must agree with every component's
    ``resources.eval_gpus_per_node``, because the same number sizes the Slurm
    job and the model loader's device map. Without a declaration the historical
    first-component value is kept, so legacy merged configs are unchanged.
    """
    resolved_backend = backend or validate_shared_backend(merged_config, records)
    component_gpus = {
        _record_dataset(record): int(
            ((record["config"].get("resources") or {}).get("eval_gpus_per_node") or 1)
        )
        for record in records
    }
    # The FSDP recipe's evaluation shape is part of the model contract: the same
    # number sizes the Slurm job and the loader's device map, so a disagreement
    # with the components fails closed. Legacy DDP contracts keep their
    # historical resolution (the declared value or the first component's).
    fsdp_recipe = (
        str((merged_config.get("training") or {}).get("strategy") or "").strip().lower() == "fsdp"
    )
    declared = (merged_config.get("execution") or {}).get("postprocess_gpus")
    if declared in (None, ""):
        resolved_gpus = component_gpus[_record_dataset(records[0])]
    else:
        resolved_gpus = int(declared)
        disagree = sorted(
            dataset for dataset, value in component_gpus.items() if value != resolved_gpus
        )
        if fsdp_recipe and disagree:
            raise ValueError(
                f"merged execution.postprocess_gpus={resolved_gpus} disagrees with the component "
                f"resources.eval_gpus_per_node for {disagree}; the merged evaluation shape must "
                "match the model contract."
            )
    if resolved_gpus < 1 or resolved_gpus > 8:
        raise ValueError(f"merged postprocess GPUs must be between 1 and 8, got {resolved_gpus}.")
    return {
        "schema_version": EVALUATION_RESOURCE_SCHEMA_VERSION,
        "backend": resolved_backend,
        "eval_nodes": 1,
        "eval_gpus_per_node": resolved_gpus,
        "sharded": resolved_gpus > 1,
        "component_eval_gpus_per_node": component_gpus,
    }


def model_config(merged_config: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve the shared model config without importing torch.

    The backend is resolved from the merged config or the component records;
    every component must share one backend. For the Gemma backend the pinned
    revision and model path are carried into the resolved config so the
    runtime dispatch (loader, collator factory, example preparation) selects
    the Gemma path exactly like a standalone Gemma config. The evaluation
    contract (decision rule and evidence view) and the evaluation resource
    shape are validated here so the same checks gate planning and execution.
    """
    if not records:
        raise ValueError("At least one component record is required to resolve a model config.")
    backend = validate_shared_backend(merged_config, records)
    evaluation_contract = validate_evaluation_contract(merged_config, records, backend=backend)
    resources = validate_merged_resources(merged_config, records, backend=backend)
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
    for key in ("model_revision", "model_attn_implementation"):
        value = merged_config.get(key)
        if value not in (None, ""):
            config[key] = value
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
    # Preserve the component's declared decision backend. The canonical
    # likelihood components must not silently revert to teacher forcing here;
    # components without a declared mode keep the historical default. Qwen3
    # contracts declare both fields in every component and are validated above;
    # legacy contracts keep today's teacher-forced merged decision rule.
    config["evaluation"].setdefault("sample_prediction_mode", "original_teacher_forced")
    config["evaluation"].setdefault("headline_mode", "original_teacher_forced")
    if evaluation_contract["declared"]:
        config["evaluation"]["sample_prediction_mode"] = evaluation_contract["sample_prediction_mode"]
        config["evaluation"]["headline_mode"] = evaluation_contract["sample_prediction_mode"]
        if evaluation_contract["evaluation_view"]:
            config["evaluation"]["evaluation_view"] = evaluation_contract["evaluation_view"]
    config["resources"] = {
        "eval_nodes": resources["eval_nodes"],
        "eval_gpus_per_node": resources["eval_gpus_per_node"],
    }
    config["merged_evaluation_contract"] = evaluation_contract
    config["merged_evaluation_resources"] = resources
    config["output_dirs"] = copy.deepcopy(records[0]["config"].get("output_dirs", {}))
    return config
