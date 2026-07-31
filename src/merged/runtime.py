from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.merged.protocol import (
    DATASETS,
    build_final_partitions,
    load_component_records,
    namespace_example,
)
from src.utils import load_yaml_with_overrides, read_json, resolve_project_path


def load_merged_config(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_with_overrides(resolve_project_path(config_path), [])
    if str(config.get("protocol", "")) != "symmetric_merged":
        raise ValueError("Expected protocol=symmetric_merged config.")
    modality = str(config.get("modality", "")).strip().lower()
    if modality not in {"audio_text", "audio_only", "text_only"}:
        raise ValueError(f"Unsupported merged modality: {modality!r}")
    return config


def protocol_artifact_path(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["output_dirs"]["merged_root"]) / "merged_protocol.json"


def load_protocol_artifact(config: dict[str, Any]) -> dict[str, Any]:
    path = protocol_artifact_path(config)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing merged protocol artifact {path}. Run scripts/build_symmetric_merged_manifest.py first."
        )
    payload = read_json(path)
    if payload.get("schema_version") != "symmetric_merged_protocol.v1":
        raise ValueError(f"Unsupported merged protocol artifact schema at {path}.")
    stored_artifact_hash = payload.get("artifact_hash")
    if stored_artifact_hash:
        from src.merged.protocol import canonical_sha256

        unsigned = {key: value for key, value in payload.items() if key != "artifact_hash"}
        if canonical_sha256(unsigned) != stored_artifact_hash:
            raise ValueError(f"Merged protocol artifact hash mismatch: {path}")
    if payload.get("split_audit", {}).get("status") != "passed":
        raise ValueError(f"Merged split artifact did not pass its construction audit: {path}")
    return payload


def records_by_dataset(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(record["dataset"]).lower(): record for record in records}
    if set(result) != set(DATASETS):
        raise ValueError(f"Expected records for {DATASETS}, found {sorted(result)}")
    return result


def _source_subject_id(namespace_subject_id: str) -> tuple[str, str]:
    text = str(namespace_subject_id)
    if "::" not in text:
        raise ValueError(f"Expected namespaced subject identity, got {text!r}.")
    dataset, subject = text.split("::", 1)
    return dataset, subject


def build_namespaced_examples(
    records: list[dict[str, Any]],
    namespaced_subject_ids: Iterable[str],
    *,
    partition_name: str,
) -> dict[str, list[dict[str, Any]]]:
    """Build examples using each component's exact prompts and data policy."""

    requested = sorted(set(str(value) for value in namespaced_subject_ids))
    requested_by_dataset: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}
    for value in requested:
        dataset, subject = _source_subject_id(value)
        if dataset not in requested_by_dataset:
            raise ValueError(f"Unknown merged dataset in subject identity {value!r}.")
        requested_by_dataset[dataset].add(subject)

    result: dict[str, list[dict[str, Any]]] = {}
    by_dataset = records_by_dataset(records)
    seen_examples: set[str] = set()
    for dataset in DATASETS:
        record = by_dataset[dataset]
        source_subjects = requested_by_dataset[dataset]
        if not source_subjects:
            result[dataset] = []
            continue
        source_rows = [
            row for row in record["rows"] if str(row["subject_id"]) in source_subjects
        ]
        # Keep protocol/config planning importable without torch. The model
        # runtime is imported only when examples are actually materialized.
        from src.data.runtime import build_examples

        examples = build_examples(
            source_rows,
            record["config"],
            partition_name=partition_name,
            truncation_log_path=None,
        )
        wrapped: list[dict[str, Any]] = []
        for example in examples:
            merged_example = namespace_example(example, dataset)
            merged_example["component_config_path"] = str(record["config_path"])
            merged_example["component_manifest_hash"] = str(record["manifest_hash"])
            merged_example["partition"] = partition_name
            sample_id = str(merged_example["sample_id"])
            if sample_id in seen_examples:
                raise ValueError(f"Duplicate merged example identity: {sample_id}")
            seen_examples.add(sample_id)
            wrapped.append(merged_example)
        actual_subjects = {str(example["subject_id"]) for example in wrapped}
        expected_subjects = {f"{dataset}::{subject}" for subject in source_subjects}
        if actual_subjects != expected_subjects:
            raise ValueError(
                f"{dataset} {partition_name} examples do not match requested subjects: "
                f"missing={sorted(expected_subjects-actual_subjects)[:5]} "
                f"extra={sorted(actual_subjects-expected_subjects)[:5]}"
            )
        result[dataset] = sorted(wrapped, key=lambda item: str(item["sample_id"]))
    return result


def flatten_component_examples(grouped: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for dataset in DATASETS:
        result.extend(grouped.get(dataset, []))
    return result


def limit_grouped_subjects(
    grouped: dict[str, list[dict[str, Any]]], *, subjects_per_class: int
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Deterministically limit every component to a small smoke cohort."""

    if int(subjects_per_class) < 1:
        raise ValueError("subjects_per_class must be positive.")
    result: dict[str, list[dict[str, Any]]] = {}
    selected_all: list[str] = []
    for dataset in DATASETS:
        examples = list(grouped.get(dataset, []))
        labels: dict[str, int] = {}
        for example in examples:
            subject = str(example["subject_id"])
            label = int(example["label"])
            if subject in labels and labels[subject] != label:
                raise ValueError(f"Smoke subject {subject} has inconsistent labels.")
            labels[subject] = label
        selected: list[str] = []
        for label in (0, 1):
            selected.extend(
                sorted(subject for subject, value in labels.items() if value == label)[: int(subjects_per_class)]
            )
        selected_set = set(selected)
        result[dataset] = [
            example for example in examples if str(example["subject_id"]) in selected_set
        ]
        selected_all.extend(sorted(selected_set))
    return result, sorted(selected_all)


def fold_subject_ids(protocol: dict[str, Any], fold: int, partition: str) -> dict[str, list[str]]:
    protocol_payload = protocol.get("protocol", protocol)
    payload = protocol_payload.get("folds", {}).get(str(int(fold)))
    if payload is None:
        raise ValueError(f"Merged protocol has no outer fold {fold}.")
    if partition not in {"outer_train", "qwen_train", "inner_val", "outer_holdout"}:
        raise ValueError(f"Unsupported merged fold partition {partition!r}.")
    return {
        dataset: list(payload["components"][dataset][f"{partition}_subject_ids"])
        for dataset in DATASETS
    }


def final_subject_ids(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    final = build_final_partitions(records)
    return {
        dataset: list(final["by_dataset"][dataset]["train_subject_ids"])
        for dataset in DATASETS
    } | {"daic_official_test": list(final["daic_official_test_subject_ids"])}


def load_records_and_protocol(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = load_component_records(config, require_files=True)
    protocol = load_protocol_artifact(config)
    for record in records:
        dataset = str(record["dataset"]).lower()
        expected_hash = protocol["manifest"]["component_manifest_hashes"].get(dataset)
        if expected_hash and expected_hash != record["manifest_hash"]:
            raise ValueError(
                f"Component manifest hash changed since merged protocol creation for {dataset}."
            )
    return records, protocol


def make_fold_partitions(
    records: list[dict[str, Any]],
    protocol: dict[str, Any],
    fold: int,
) -> dict[str, Any]:
    grouped = {
        name: build_namespaced_examples(
            records,
            [subject for values in fold_subject_ids(protocol, fold, name).values() for subject in values],
            partition_name=name,
        )
        for name in ("outer_train", "qwen_train", "inner_val", "outer_holdout")
    }
    return {
        "subjects": {name: fold_subject_ids(protocol, fold, name) for name in ("outer_train", "qwen_train", "inner_val", "outer_holdout")},
        "examples": grouped,
        "flat_examples": {name: flatten_component_examples(values) for name, values in grouped.items()},
        "fold": int(fold),
    }


def make_final_partitions(records: list[dict[str, Any]]) -> dict[str, Any]:
    final = final_subject_ids(records)
    train_subjects = [subject for dataset in DATASETS for subject in final[dataset]]
    grouped_train = build_namespaced_examples(records, train_subjects, partition_name="final_train")
    grouped_test = build_namespaced_examples(records, final["daic_official_test"], partition_name="daic_official_test")
    return {
        "subjects": final,
        "examples": {"train": grouped_train, "daic_official_test": grouped_test},
        "flat_examples": {
            "train": flatten_component_examples(grouped_train),
            "daic_official_test": flatten_component_examples(grouped_test),
        },
    }


def serialized_identity(config: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_name": config.get("name"),
        "modality": config.get("modality"),
        "manifest_hash": protocol.get("manifest", {}).get("manifest_hash"),
        "split_hash": protocol.get("protocol", {}).get("split_hash"),
        "artifact_hash": protocol.get("artifact_hash"),
    }
