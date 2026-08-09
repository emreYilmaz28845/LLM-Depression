from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from src.data.split_utils import (
    assign_stratified_group_folds,
    deterministic_inner_split,
)
from src.utils import (
    load_yaml_with_overrides,
    read_json,
    read_jsonl,
    resolve_input_modality,
    resolve_project_path,
    sha256_file,
    sha256_jsonl_rows,
    write_jsonl,
)


DATASETS = ("daic", "cmdc", "turkish", "d3tec", "androids_interview")
MODALITIES = ("audio_text", "audio_only", "text_only")
METHODS = ("qwen", "logreg", "xgb_fixed", "xgb_optuna")
OUTER_FOLDS = 5
HEAD_INNER_FOLDS = 3
PROTOCOL_SCHEMA_VERSION = "symmetric_merged_protocol.v1"
MERGED_MANIFEST_SCHEMA_VERSION = "symmetric_merged_manifest.v1"
WEIGHT_SCHEMA_VERSION = "symmetric_hierarchical_weights.v1"
SCHEDULE_SCHEMA_VERSION = "symmetric_dataset_schedule.v1"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_head_inner_folds(config: dict[str, Any], stage: str) -> int:
    """Resolve grouped head folds, using a valid tiny smoke override.

    Production CV/final stages use the protocol's three grouped folds.  The
    smoke cohort is deliberately only two subjects per class, so three folds
    would create an empty validation fold for every dataset.  Smoke therefore
    uses the explicit two-fold execution override while retaining the
    production protocol setting.
    """

    stage_name = str(stage).strip().lower()
    if stage_name not in {"smoke", "cv", "final"}:
        raise ValueError(f"Unsupported merged stage for head folds: {stage!r}")
    protocol_settings = config.get("protocol_settings") or {}
    execution = config.get("execution") or {}
    if stage_name == "smoke":
        value = execution.get("smoke_head_inner_folds", 2)
    else:
        value = protocol_settings.get("head_inner_folds", HEAD_INNER_FOLDS)
    resolved = int(value)
    if resolved < 2 or resolved > HEAD_INNER_FOLDS:
        raise ValueError(
            f"Merged head tuning requires 2..{HEAD_INNER_FOLDS} grouped folds; "
            f"stage={stage_name} value={resolved}."
        )
    return resolved


def namespace_id(dataset: str, value: Any) -> str:
    """Return the protocol identity used in every merged artifact."""

    dataset_name = str(dataset).strip().lower()
    if not dataset_name:
        raise ValueError("Merged identities require a non-empty dataset name.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Merged {dataset_name} identity cannot be empty.")
    return f"{dataset_name}::{text}"


def _subject_key(row: dict[str, Any]) -> str:
    value = row.get("subject_id")
    if value in (None, ""):
        raise ValueError("Every merged row/example requires subject_id.")
    return str(value)


def _label_map(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for row in rows:
        subject_id = _subject_key(row)
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"Expected binary labels, got {label} for {subject_id}.")
        previous = labels.setdefault(subject_id, label)
        if previous != label:
            raise ValueError(f"Subject {subject_id} has inconsistent labels.")
    return labels


def namespace_row(row: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Copy a component manifest row into the collision-safe merged namespace."""

    dataset_name = str(dataset).strip().lower()
    original_subject = _subject_key(row)
    result = dict(row)
    result["component_dataset"] = dataset_name
    result["component_subject_id"] = original_subject
    result["component_sample_id"] = str(row.get("sample_id", ""))
    result["dataset"] = dataset_name
    result["subject_id"] = namespace_id(dataset_name, original_subject)
    if row.get("sample_id") not in (None, ""):
        result["sample_id"] = namespace_id(dataset_name, row["sample_id"])
    for field in ("response_id", "turn_key", "window_id", "bundle_id"):
        if row.get(field) not in (None, ""):
            result[f"component_{field}"] = str(row[field])
            result[field] = namespace_id(dataset_name, row[field])
    return result


def namespace_example(example: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Namespace an example after the component builder has applied its policy."""

    dataset_name = str(dataset).strip().lower()
    original_subject = _subject_key(example)
    result = dict(example)
    result["component_dataset"] = dataset_name
    result["component_subject_id"] = original_subject
    result["component_sample_id"] = str(example.get("sample_id", ""))
    result["dataset"] = dataset_name
    result["subject_id"] = namespace_id(dataset_name, original_subject)
    if example.get("sample_id") not in (None, ""):
        result["sample_id"] = namespace_id(dataset_name, example["sample_id"])
    for field in ("response_id", "turn_key", "window_id", "bundle_id"):
        if example.get(field) not in (None, ""):
            result[f"component_{field}"] = str(example[field])
            result[field] = namespace_id(dataset_name, example[field])
    return result


def _resolve_component_path(value: str | Path) -> Path:
    return resolve_project_path(value)


def _default_manifest_path(component_config: dict[str, Any]) -> Path:
    output_dir = _resolve_component_path(component_config["output_dirs"]["manifest_dir"])
    return output_dir / f"{str(component_config['dataset']).lower()}_manifest.jsonl"


def _default_metadata_path(component_config: dict[str, Any]) -> Path:
    output_dir = _resolve_component_path(component_config["output_dirs"]["split_dir"])
    return output_dir / f"{str(component_config['dataset']).lower()}_manifest_metadata.json"


def load_component_records(
    merged_config: dict[str, Any],
    *,
    require_files: bool = True,
) -> list[dict[str, Any]]:
    """Load component configs, manifests, and split metadata declared by a merged config."""

    records: list[dict[str, Any]] = []
    components = merged_config.get("components") or []
    if len(components) != len(DATASETS):
        raise ValueError(
            f"The symmetric protocol requires exactly {len(DATASETS)} components; "
            f"found {len(components)}."
        )
    seen: set[str] = set()
    for item in components:
        if not isinstance(item, dict):
            raise ValueError("Every merged component must be a mapping.")
        dataset = str(item.get("name") or item.get("dataset") or "").strip().lower()
        if dataset not in DATASETS:
            raise ValueError(f"Unsupported merged component dataset: {dataset!r}.")
        if dataset in seen:
            raise ValueError(f"Duplicate merged component: {dataset}.")
        seen.add(dataset)
        config_path = item.get("config")
        if not config_path:
            raise ValueError(f"Merged component {dataset} is missing config.")
        component_config_path = _resolve_component_path(config_path)
        component_config = load_yaml_with_overrides(component_config_path, [])
        if str(component_config.get("dataset", "")).lower() != dataset:
            raise ValueError(
                f"Component {dataset} config has dataset={component_config.get('dataset')!r}."
            )
        merged_modality = str(merged_config.get("modality", "")).strip().lower()
        if merged_modality and resolve_input_modality(component_config) != merged_modality:
            raise ValueError(
                f"Component {dataset} resolves to modality={resolve_input_modality(component_config)!r}, "
                f"but merged config requires {merged_modality!r}."
            )
        manifest_path = _resolve_component_path(
            item.get("manifest_path") or _default_manifest_path(component_config)
        )
        if item.get("metadata_path"):
            metadata_path = _resolve_component_path(item["metadata_path"])
        elif component_config.get("output_dirs", {}).get("split_dir"):
            metadata_path = _default_metadata_path(component_config)
        else:
            metadata_path = manifest_path.with_suffix(".metadata.json")
        if require_files and not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing {dataset} manifest {manifest_path}. Build the component manifests first."
            )
        rows = read_jsonl(manifest_path) if manifest_path.is_file() else []
        if not rows:
            raise ValueError(f"Merged component {dataset} manifest is empty: {manifest_path}")
        metadata = read_json(metadata_path) if metadata_path.is_file() else {}
        partition_rows: list[dict[str, Any]] = []
        if metadata.get("subject_partition_path"):
            partition_path = _resolve_component_path(metadata["subject_partition_path"])
            if partition_path.is_file():
                partition_rows = list(read_json(partition_path))
        folds: dict[str, Any] | dict[int, Any] = {}
        if metadata.get("folds_path"):
            folds_path = _resolve_component_path(metadata["folds_path"])
            if folds_path.is_file():
                folds = read_json(folds_path)
        labels = _label_map(rows)
        official_test = {
            str(subject_id)
            for subject_id in item.get("official_test_subject_ids", [])
        }
        official_test.update(
            str(row["subject_id"])
            for row in partition_rows
            if str(row.get("partition", "")).lower() == "test"
        )
        official_test.update(
            str(row["subject_id"])
            for row in rows
            if str(row.get("split_original", "")).lower() in {"test", "official_test"}
        )
        records.append(
            {
                "dataset": dataset,
                "config_path": str(component_config_path),
                "config": component_config,
                "manifest_path": str(manifest_path),
                "manifest_hash": sha256_jsonl_rows(rows),
                "manifest_file_sha256": sha256_file(manifest_path),
                "metadata_path": str(metadata_path),
                "metadata": metadata,
                "rows": rows,
                "labels": labels,
                "partition_rows": partition_rows,
                "folds": folds,
                "official_test_subject_ids": sorted(official_test),
            }
        )
    if set(seen) != set(DATASETS):
        raise ValueError(f"Merged components do not cover {DATASETS}: found={sorted(seen)}")
    return records


def build_merged_manifest(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Concatenate component manifests with namespaced identities and provenance."""

    if {str(record["dataset"]).lower() for record in records} != set(DATASETS):
        raise ValueError("Merged manifest requires all five protocol datasets.")
    rows: list[dict[str, Any]] = []
    seen_subjects: set[str] = set()
    seen_samples: set[str] = set()
    for record in sorted(records, key=lambda value: str(value["dataset"])):
        dataset = str(record["dataset"]).lower()
        for source_row in record["rows"]:
            row = namespace_row(source_row, dataset)
            row["component_config_path"] = str(record["config_path"])
            row["component_manifest_hash"] = str(record["manifest_hash"])
            row["official_test_subject"] = str(source_row.get("subject_id")) in set(
                record.get("official_test_subject_ids", [])
            )
            if row["subject_id"] in seen_subjects and row["sample_id"] not in seen_samples:
                # A subject appears on many response/window rows by design.
                pass
            seen_subjects.add(row["subject_id"])
            sample_id = str(row.get("sample_id", ""))
            if sample_id in seen_samples:
                raise ValueError(f"Duplicate merged sample identity: {sample_id}")
            seen_samples.add(sample_id)
            rows.append(row)
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["subject_id"]), str(row.get("sample_id", ""))))
    metadata = {
        "schema_version": MERGED_MANIFEST_SCHEMA_VERSION,
        "datasets": list(DATASETS),
        "row_count": len(rows),
        "subject_count": len({str(row["subject_id"]) for row in rows}),
        "sample_count": len(seen_samples),
        "dataset_row_counts": dict(sorted(Counter(str(row["dataset"]) for row in rows).items())),
        "dataset_subject_counts": {
            dataset: len({str(row["subject_id"]) for row in rows if str(row["dataset"]) == dataset})
            for dataset in DATASETS
        },
        "component_manifest_hashes": {
            str(record["dataset"]): str(record["manifest_hash"])
            for record in sorted(records, key=lambda value: str(value["dataset"]))
        },
        "manifest_hash": sha256_jsonl_rows(rows),
        "namespace": "dataset::subject_id and dataset::sample_id",
    }
    metadata["metadata_hash"] = canonical_sha256(metadata)
    return rows, metadata


def _read_fold_payload(folds: dict[Any, Any], fold: int) -> dict[str, Any] | None:
    if not folds:
        return None
    if str(fold) in folds:
        return folds[str(fold)]
    if fold in folds:
        return folds[fold]
    return None


def _validate_outer_coverage(
    dataset: str,
    folds: dict[int, dict[str, list[str]]],
    eligible_subject_ids: list[str],
    official_test_subject_ids: list[str],
) -> None:
    expected = set(eligible_subject_ids)
    observed: set[str] = set()
    official = set(official_test_subject_ids)
    if len(folds) != OUTER_FOLDS:
        raise ValueError(f"{dataset} requires five outer folds; found {sorted(folds)}")
    for fold, payload in sorted(folds.items()):
        train = set(payload["outer_train_subject_ids"])
        holdout = set(payload["final_eval_subject_ids"])
        if train & holdout:
            raise ValueError(f"{dataset} fold {fold} train/holdout overlap.")
        if holdout & official:
            raise ValueError(f"{dataset} official-test subject entered CV holdout: {sorted(holdout & official)[:5]}")
        if not holdout <= expected or not train <= expected:
            raise ValueError(f"{dataset} fold {fold} contains a subject outside its eligible development pool.")
        observed.update(holdout)
        if train != expected - holdout:
            raise ValueError(f"{dataset} fold {fold} does not contain exactly the other four folds.")
    if observed != expected:
        raise ValueError(
            f"{dataset} outer-fold coverage mismatch: missing={sorted(expected-observed)} extra={sorted(observed-expected)}"
        )


def build_component_outer_folds(
    record: dict[str, Any], *, outer_folds: int = OUTER_FOLDS, seed: int = 1337
) -> dict[int, dict[str, list[str]]]:
    """Resolve official component folds, falling back to deterministic stratified folds."""

    if int(outer_folds) != OUTER_FOLDS:
        raise ValueError("The symmetric protocol is fixed to five outer folds.")
    dataset = str(record["dataset"]).lower()
    labels = dict(record["labels"])
    official = set(str(value) for value in record.get("official_test_subject_ids", []))
    eligible = sorted(set(labels) - official)
    if not eligible:
        raise ValueError(f"{dataset} has no eligible development subjects.")
    resolved: dict[int, dict[str, list[str]]] = {}
    for fold in range(OUTER_FOLDS):
        payload = _read_fold_payload(record.get("folds") or {}, fold)
        if payload is None:
            continue
        train = sorted(str(value) for value in payload.get("outer_train_subject_ids", []))
        holdout = sorted(str(value) for value in payload.get("final_eval_subject_ids", []))
        resolved[fold] = {
            "outer_train_subject_ids": train,
            "final_eval_subject_ids": holdout,
        }
    if len(resolved) == OUTER_FOLDS:
        try:
            _validate_outer_coverage(dataset, resolved, eligible, sorted(official))
        except ValueError:
            split_mode = str((record.get("config") or {}).get("split", {}).get("mode", ""))
            if dataset != "daic" or split_mode != "fixed":
                raise
            # Standalone harmonized DAIC keeps the official train/val/test
            # split. Its stored fold file covers only the configured train
            # pool, while merged CV must cover the complete non-test
            # development pool (official train + validation). Generate that
            # CV here without ever admitting official-test subjects.
            resolved = {}
    if len(resolved) != OUTER_FOLDS:
        generated = assign_stratified_group_folds(
            {subject: labels[subject] for subject in eligible},
            n_splits=OUTER_FOLDS,
            seed=int(seed),
        )
        resolved = {
            int(fold): {
                "outer_train_subject_ids": sorted(payload["outer_train_subject_ids"]),
                "final_eval_subject_ids": sorted(payload["final_eval_subject_ids"]),
            }
            for fold, payload in generated.items()
        }
        source = (
            "deterministic_stratified_group_folds_from_fixed_development_pool"
            if dataset == "daic" and str((record.get("config") or {}).get("split", {}).get("mode", "")) == "fixed"
            else "deterministic_stratified_group_folds"
        )
    else:
        source = "component_official_folds"
    _validate_outer_coverage(dataset, resolved, eligible, sorted(official))
    for fold, payload in resolved.items():
        payload["fold"] = int(fold)
        payload["source"] = source
        payload["outer_train_subject_ids"] = sorted(payload["outer_train_subject_ids"])
        payload["final_eval_subject_ids"] = sorted(payload["final_eval_subject_ids"])
    return resolved


def build_protocol_splits(
    records: list[dict[str, Any]],
    *,
    seed: int = 1337,
    inner_val_ratio: float = 0.2,
    outer_folds: int = OUTER_FOLDS,
) -> dict[str, Any]:
    """Build namespaced five-fold outer splits and per-dataset inner selections."""

    if not 0 < float(inner_val_ratio) < 1:
        raise ValueError("inner_val_ratio must be between zero and one.")
    components: dict[str, Any] = {}
    for record in sorted(records, key=lambda value: str(value["dataset"])):
        dataset = str(record["dataset"]).lower()
        component_folds = build_component_outer_folds(record, outer_folds=outer_folds, seed=seed)
        labels = dict(record["labels"])
        namespaced_folds: dict[str, Any] = {}
        for fold, payload in sorted(component_folds.items()):
            outer_train = list(payload["outer_train_subject_ids"])
            inner = deterministic_inner_split(
                labels,
                outer_train,
                seed=int(seed) + int(fold),
                val_ratio=float(inner_val_ratio),
            )
            namespaced_folds[str(fold)] = {
                "fold": int(fold),
                "source": payload.get("source", "unknown"),
                "outer_train_subject_ids": [namespace_id(dataset, value) for value in outer_train],
                "outer_holdout_subject_ids": [
                    namespace_id(dataset, value) for value in payload["final_eval_subject_ids"]
                ],
                "qwen_train_subject_ids": [namespace_id(dataset, value) for value in inner["train_inner_subject_ids"]],
                "inner_val_subject_ids": [namespace_id(dataset, value) for value in inner["val_inner_subject_ids"]],
                "component_outer_train_subject_ids": outer_train,
                "component_outer_holdout_subject_ids": list(payload["final_eval_subject_ids"]),
                "component_qwen_train_subject_ids": list(inner["train_inner_subject_ids"]),
                "component_inner_val_subject_ids": list(inner["val_inner_subject_ids"]),
            }
        components[dataset] = {
            "dataset": dataset,
            "config_path": str(record["config_path"]),
            "manifest_hash": str(record["manifest_hash"]),
            "official_test_subject_ids": [namespace_id(dataset, value) for value in record.get("official_test_subject_ids", [])],
            "folds": namespaced_folds,
        }

    folds: dict[str, Any] = {}
    for fold in range(OUTER_FOLDS):
        fold_components = {dataset: components[dataset]["folds"][str(fold)] for dataset in DATASETS}
        folds[str(fold)] = {
            "fold": fold,
            "components": fold_components,
            "outer_train_subject_ids": sorted(
                subject
                for payload in fold_components.values()
                for subject in payload["outer_train_subject_ids"]
            ),
            "outer_holdout_subject_ids": sorted(
                subject
                for payload in fold_components.values()
                for subject in payload["outer_holdout_subject_ids"]
            ),
            "qwen_train_subject_ids": sorted(
                subject
                for payload in fold_components.values()
                for subject in payload["qwen_train_subject_ids"]
            ),
            "inner_val_subject_ids": sorted(
                subject
                for payload in fold_components.values()
                for subject in payload["inner_val_subject_ids"]
            ),
        }
        folds[str(fold)]["fold_hash"] = canonical_sha256(folds[str(fold)])
    protocol = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "datasets": list(DATASETS),
        "outer_folds": OUTER_FOLDS,
        "seed": int(seed),
        "inner_val_ratio": float(inner_val_ratio),
        "identity_namespace": "dataset::subject_id",
        "components": components,
        "folds": folds,
    }
    protocol["split_hash"] = canonical_sha256(protocol)
    return protocol


def build_final_partitions(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the final-training partition, keeping only DAIC official test untouched."""

    train: list[str] = []
    official_test: list[str] = []
    by_dataset: dict[str, Any] = {}
    for record in sorted(records, key=lambda value: str(value["dataset"])):
        dataset = str(record["dataset"]).lower()
        labels = dict(record["labels"])
        official = set(str(value) for value in record.get("official_test_subject_ids", []))
        if dataset != "daic" and official:
            raise ValueError(
                f"Only DAIC may declare an official-test partition in the final stage; {dataset} has {len(official)}."
            )
        eligible = sorted(set(labels) - official)
        namespaced_train = [namespace_id(dataset, value) for value in eligible]
        train.extend(namespaced_train)
        namespaced_test = [namespace_id(dataset, value) for value in sorted(official)]
        official_test.extend(namespaced_test)
        by_dataset[dataset] = {
            "train_subject_ids": namespaced_train,
            "official_test_subject_ids": namespaced_test,
            "train_count": len(namespaced_train),
            "official_test_count": len(namespaced_test),
        }
    overlap = sorted(set(train) & set(official_test))
    if overlap:
        raise ValueError(f"Final training/DAIC official-test overlap: {overlap[:10]}")
    return {
        "train_subject_ids": sorted(train),
        "daic_official_test_subject_ids": sorted(official_test),
        "by_dataset": by_dataset,
        "train_count": len(train),
        "daic_official_test_count": len(official_test),
    }


def limit_examples_by_dataset_subjects_per_class(
    examples: list[dict[str, Any]], *, subjects_per_class: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select a deterministic smoke cohort independently in every dataset."""

    limit = int(subjects_per_class)
    if limit < 1:
        raise ValueError("subjects_per_class must be positive.")
    labels_by_dataset: dict[str, dict[str, int]] = defaultdict(dict)
    for example in examples:
        dataset = str(example.get("dataset", "")).lower()
        if not dataset:
            raise ValueError("Every smoke example requires dataset.")
        subject = _subject_key(example)
        label = int(example["label"])
        if label not in (0, 1):
            raise ValueError(f"Expected binary labels, got {label} for {subject}.")
        prior = labels_by_dataset[dataset].setdefault(subject, label)
        if prior != label:
            raise ValueError(f"Subject {dataset}::{subject} has inconsistent labels.")

    selected_keys: set[tuple[str, str]] = set()
    selected_ids: list[str] = []
    for dataset in sorted(labels_by_dataset):
        labels = labels_by_dataset[dataset]
        for label in (0, 1):
            subjects = sorted(subject for subject, value in labels.items() if value == label)[:limit]
            selected_keys.update((dataset, subject) for subject in subjects)
            selected_ids.extend(
                namespace_id(dataset, subject) if "::" not in subject else subject
                for subject in subjects
            )
    selected = [
        example
        for example in examples
        if (str(example["dataset"]).lower(), _subject_key(example)) in selected_keys
    ]
    return selected, sorted(set(selected_ids))


def _response_key(row: dict[str, Any]) -> str:
    for field in ("response_id", "turn_key", "question_id", "sample_id"):
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    raise ValueError("Every eligible example requires a stable response/sample identity.")


def compute_hierarchical_example_weights(
    examples: list[dict[str, Any]],
    *,
    expected_datasets: Iterable[str] | None = DATASETS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply exhaustive dataset → subject → response → window weighting.

    Each present dataset receives raw total weight one.  Within it, each
    subject receives equal total weight, each response receives equal weight
    within that subject, and each response window receives equal weight.  A
    final global rescale makes the arithmetic mean example weight exactly one.
    No row is added or removed by this function.
    """

    if not examples:
        raise ValueError("Cannot weight an empty merged training pool.")
    datasets = sorted({str(example.get("dataset", "")).lower() for example in examples})
    if "" in datasets:
        raise ValueError("Every example must include dataset for merged weighting.")
    expected = sorted({str(value).lower() for value in expected_datasets}) if expected_datasets is not None else datasets
    missing = sorted(set(expected) - set(datasets))
    if missing:
        raise ValueError(f"Merged training pool is missing datasets: {missing}")

    subjects_by_dataset: dict[str, set[str]] = defaultdict(set)
    responses_by_subject: dict[tuple[str, str], set[str]] = defaultdict(set)
    rows_by_response: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        dataset = str(example["dataset"]).lower()
        subject = _subject_key(example)
        response = _response_key(example)
        subjects_by_dataset[dataset].add(subject)
        responses_by_subject[(dataset, subject)].add(response)
        rows_by_response[(dataset, subject, response)].append(index)

    raw_weights = [0.0] * len(examples)
    for (dataset, subject, response), indices in rows_by_response.items():
        subject_weight = 1.0 / len(subjects_by_dataset[dataset])
        response_weight = subject_weight / len(responses_by_subject[(dataset, subject)])
        window_weight = response_weight / len(indices)
        for index in indices:
            raw_weights[index] = window_weight
    raw_total = sum(raw_weights)
    if raw_total <= 0:
        raise ValueError("Hierarchical weighting produced no positive row weights.")
    scale = len(examples) / raw_total
    weighted: list[dict[str, Any]] = []
    for example, raw_weight in zip(examples, raw_weights):
        weighted.append(
            {
                **example,
                "raw_loss_weight": float(raw_weight),
                "loss_weight": float(raw_weight * scale),
            }
        )

    dataset_raw_totals: dict[str, float] = defaultdict(float)
    dataset_totals: dict[str, float] = defaultdict(float)
    subject_totals: dict[str, float] = defaultdict(float)
    response_totals: dict[str, float] = defaultdict(float)
    subject_totals_by_dataset: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    response_totals_by_subject: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    window_weights_by_response: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for example in weighted:
        dataset = str(example["dataset"]).lower()
        subject = _subject_key(example)
        response = _response_key(example)
        normalized_weight = float(example["loss_weight"])
        dataset_raw_totals[dataset] += float(example["raw_loss_weight"])
        dataset_totals[dataset] += normalized_weight
        subject_totals[subject] += normalized_weight
        response_totals[response] += normalized_weight
        subject_totals_by_dataset[dataset][subject] += normalized_weight
        response_totals_by_subject[dataset][subject][response] += normalized_weight
        window_weights_by_response[dataset][subject][response].append(normalized_weight)
    raw_reference = next(iter(dataset_raw_totals.values()))
    normalized_reference = next(iter(dataset_totals.values()))
    equal_dataset_raw = all(
        math.isclose(value, raw_reference, rel_tol=1e-12, abs_tol=1e-12)
        for value in dataset_raw_totals.values()
    )
    equal_dataset = all(
        math.isclose(value, normalized_reference, rel_tol=1e-12, abs_tol=1e-12)
        for value in dataset_totals.values()
    )
    equal_subject_within_dataset = all(
        all(
            math.isclose(value, next(iter(totals.values())), rel_tol=1e-12, abs_tol=1e-12)
            for value in totals.values()
        )
        for totals in subject_totals_by_dataset.values()
        if totals
    )
    equal_response_within_subject = all(
        all(
            math.isclose(value, next(iter(totals.values())), rel_tol=1e-12, abs_tol=1e-12)
            for value in totals.values()
        )
        for subject_totals_for_dataset in response_totals_by_subject.values()
        for totals in subject_totals_for_dataset.values()
        if totals
    )
    equal_windows_within_response = all(
        all(
            math.isclose(value, weights[0], rel_tol=1e-12, abs_tol=1e-12)
            for value in weights[1:]
        )
        for subjects in window_weights_by_response.values()
        for responses in subjects.values()
        for weights in responses.values()
        if weights
    )
    mean_loss_weight_one = (
        abs(sum(float(row["loss_weight"]) for row in weighted) / len(weighted) - 1.0)
        <= 1e-10
    )
    if not equal_dataset_raw or not equal_dataset:
        raise AssertionError("Merged weighting did not equalize dataset totals.")
    if not equal_subject_within_dataset:
        raise AssertionError("Merged weighting did not equalize subject totals within datasets.")
    if not equal_response_within_subject:
        raise AssertionError("Merged weighting did not equalize response totals within subjects.")
    if not equal_windows_within_response:
        raise AssertionError("Merged weighting did not equalize window totals within responses.")
    if not mean_loss_weight_one:
        raise AssertionError("Merged weighting was not normalized to global mean one.")
    audit = {
        "schema_version": WEIGHT_SCHEMA_VERSION,
        "policy": "equal_dataset_then_subject_then_response_then_window_natural_prevalence",
        "dataset_count": len(datasets),
        "datasets": datasets,
        "row_count": len(weighted),
        "subject_count": len(subject_totals),
        "response_count": len(response_totals),
        "raw_dataset_weight_totals": dict(sorted(dataset_raw_totals.items())),
        "dataset_weight_totals": dict(sorted(dataset_totals.items())),
        "subject_weight_totals": dict(sorted(subject_totals.items())),
        "response_weight_totals": dict(sorted(response_totals.items())),
        "hierarchical_invariants": {
            "equal_dataset_totals": equal_dataset,
            "equal_subject_totals_within_dataset": equal_subject_within_dataset,
            "equal_response_totals_within_subject": equal_response_within_subject,
            "equal_window_totals_within_response": equal_windows_within_response,
            "mean_loss_weight_one": mean_loss_weight_one,
        },
        "dataset_subject_counts": {dataset: len(values) for dataset, values in sorted(subjects_by_dataset.items())},
        "dataset_row_counts": dict(sorted(Counter(str(row["dataset"]).lower() for row in weighted).items())),
        "class_counts": {
            dataset: dict(sorted(Counter(int(row["label"]) for row in weighted if str(row["dataset"]).lower() == dataset).items()))
            for dataset in datasets
        },
        "raw_to_normalized_scale": float(scale),
        "mean_loss_weight": float(sum(float(row["loss_weight"]) for row in weighted) / len(weighted)),
        "equal_dataset_totals": equal_dataset,
        "natural_class_prevalence_preserved": True,
        "no_sampling": True,
        "no_duplication": True,
    }
    audit["audit_hash"] = canonical_sha256(audit)
    return weighted, audit


def build_dataset_aware_schedule(
    examples: list[dict[str, Any]],
    *,
    seed: int,
    epoch: int,
    accumulation_steps: int,
) -> dict[str, Any]:
    """Create a deterministic interleaved schedule with exact one-time coverage."""

    if int(accumulation_steps) < 1:
        raise ValueError("accumulation_steps must be positive.")
    if not examples:
        raise ValueError("Cannot schedule an empty training pool.")
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        dataset = str(example.get("dataset", "")).lower()
        if not dataset:
            raise ValueError("Every scheduled example requires dataset.")
        by_dataset[dataset].append(index)
    rng = random.Random(int(seed) + int(epoch) * 1_000_003)
    dataset_order = sorted(by_dataset)
    rng.shuffle(dataset_order)
    queues: dict[str, deque[int]] = {}
    for dataset in dataset_order:
        indices = list(by_dataset[dataset])
        indices.sort(key=lambda index: str(examples[index].get("sample_id", index)))
        rng.shuffle(indices)
        queues[dataset] = deque(indices)

    blocks: list[dict[str, Any]] = []
    flat_indices: list[int] = []
    rotation = 0
    while any(queues[dataset] for dataset in dataset_order):
        block: list[int] = []
        block_datasets: list[str] = []
        for _ in range(int(accumulation_steps)):
            available = [dataset for dataset in dataset_order if queues[dataset]]
            if not available:
                break
            # Round-robin over the shuffled queues is the balancing rule. It
            # gives every optimizer block all datasets while they remain and
            # drains short tails exactly once after their queues empty.
            chosen = available[rotation % len(available)]
            rotation += 1
            index = queues[chosen].popleft()
            block.append(index)
            block_datasets.append(chosen)
        if not block:
            raise AssertionError("Schedule made no progress.")
        flat_indices.extend(block)
        block_weight_by_dataset: dict[str, float] = defaultdict(float)
        for index in block:
            block_weight_by_dataset[str(examples[index]["dataset"]).lower()] += float(
                examples[index].get("loss_weight", 1.0)
            )
        blocks.append(
            {
                "block_index": len(blocks),
                "example_indices": block,
                "datasets": block_datasets,
                "dataset_counts": dict(sorted(Counter(block_datasets).items())),
                "dataset_weight_contributions": dict(sorted(block_weight_by_dataset.items())),
                "example_weight_total": float(sum(float(examples[index].get("loss_weight", 1.0)) for index in block)),
            }
        )
    expected = list(range(len(examples)))
    if sorted(flat_indices) != expected or len(flat_indices) != len(set(flat_indices)):
        raise AssertionError("Dataset-aware schedule omitted or duplicated examples.")
    contributions: dict[str, float] = defaultdict(float)
    for index in flat_indices:
        contributions[str(examples[index]["dataset"]).lower()] += float(
            examples[index].get("loss_weight", 1.0)
        )
    audit = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "seed": int(seed),
        "epoch": int(epoch),
        "accumulation_steps": int(accumulation_steps),
        "dataset_order": dataset_order,
        "global_shuffle": True,
        "queue_interleave": "round_robin_while_multiple_queues_remain_then_drain_tails",
        "example_count": len(flat_indices),
        "optimizer_block_count": len(blocks),
        "dataset_example_counts": dict(sorted(Counter(str(row["dataset"]).lower() for row in examples).items())),
        "realized_dataset_weight_contributions": dict(sorted(contributions.items())),
        "sample_occurrence_counts": dict(sorted(Counter(flat_indices).items())),
        "blocks": blocks,
    }
    audit["schedule_hash"] = canonical_sha256(audit)
    return {"indices": flat_indices, "blocks": blocks, "audit": audit}


def normalized_accumulated_loss(losses: list[float], weights: list[float]) -> float:
    if len(losses) != len(weights) or not losses:
        raise ValueError("Losses and weights must be non-empty and have equal length.")
    denominator = sum(float(weight) for weight in weights)
    if denominator <= 0:
        raise ValueError("Accumulated loss weights must have a positive sum.")
    return float(sum(float(loss) * float(weight) for loss, weight in zip(losses, weights)) / denominator)


def build_grouped_inner_folds(
    rows: list[dict[str, Any]], *, inner_folds: int = HEAD_INNER_FOLDS, seed: int = 1337
) -> dict[str, Any]:
    """Create subject-grouped inner folds independently inside each dataset."""

    if int(inner_folds) < 2 or int(inner_folds) > HEAD_INNER_FOLDS:
        raise ValueError(
            f"The symmetric head protocol supports 2..{HEAD_INNER_FOLDS} inner folds."
        )
    by_dataset: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for global_index, row in enumerate(rows):
        by_dataset[str(row["dataset"]).lower()].append((global_index, row))
    fold_assignments: dict[str, dict[int, dict[str, list[str]]]] = {}
    for dataset, indexed_rows in sorted(by_dataset.items()):
        labels = _label_map([row for _, row in indexed_rows])
        if len(labels) < int(inner_folds):
            raise ValueError(
                f"Head tuning needs at least {inner_folds} subjects in {dataset}; found {len(labels)}."
            )
        fold_assignments[dataset] = assign_stratified_group_folds(
            labels, n_splits=int(inner_folds), seed=int(seed)
        )
    assignments: list[dict[str, Any]] = []
    for fold in range(int(inner_folds)):
        train_indices: list[int] = []
        validation_indices: list[int] = []
        train_subjects: list[str] = []
        validation_subjects: list[str] = []
        for dataset in sorted(by_dataset):
            indexed_rows = by_dataset[dataset]
            dataset_rows = [row for _, row in indexed_rows]
            labels = _label_map(dataset_rows)
            subjects = sorted(labels)
            if len(subjects) < int(inner_folds):
                raise ValueError(
                    f"Head tuning needs at least {inner_folds} subjects in {dataset}; found {len(subjects)}."
                )
            payload = fold_assignments[dataset][fold]
            train_set = set(payload["outer_train_subject_ids"])
            validation_set = set(payload["final_eval_subject_ids"])
            for global_index, row in indexed_rows:
                if str(row["subject_id"]) in validation_set:
                    validation_indices.append(global_index)
                elif str(row["subject_id"]) in train_set:
                    train_indices.append(global_index)
                else:
                    raise AssertionError("Grouped fold lost a row.")
            train_subjects.extend(namespace_id(dataset, value) if "::" not in value else value for value in sorted(train_set))
            validation_subjects.extend(namespace_id(dataset, value) if "::" not in value else value for value in sorted(validation_set))
        assignments.append(
            {
                "fold": fold,
                "train_row_indices": sorted(train_indices),
                "validation_row_indices": sorted(validation_indices),
                "train_subject_ids": sorted(train_subjects),
                "validation_subject_ids": sorted(validation_subjects),
            }
        )
    validation_coverage = Counter(
        index
        for payload in assignments
        for index in payload["validation_row_indices"]
    )
    if set(validation_coverage) != set(range(len(rows))) or any(value != 1 for value in validation_coverage.values()):
        raise AssertionError("Head inner folds do not cover each row exactly once for validation.")
    result = {
        "schema_version": "symmetric_grouped_head_folds.v1",
        "inner_folds": int(inner_folds),
        "seed": int(seed),
        "folds": assignments,
    }
    result["assignments_hash"] = canonical_sha256(result)
    return result


def audit_protocol_splits(
    protocol: dict[str, Any], *, require_daic_official_test_count: bool = False
) -> dict[str, Any]:
    failures: list[str] = []
    for dataset in DATASETS:
        component = protocol.get("components", {}).get(dataset)
        if not component:
            failures.append(f"missing_component:{dataset}")
            continue
        official = {str(value) for value in component.get("official_test_subject_ids", [])}
        observed: Counter[str] = Counter()
        development_subjects: set[str] | None = None
        for fold in range(OUTER_FOLDS):
            payload = component.get("folds", {}).get(str(fold))
            if not payload:
                failures.append(f"missing_fold:{dataset}:{fold}")
                continue
            train = {str(value) for value in payload["outer_train_subject_ids"]}
            inner = {str(value) for value in payload["inner_val_subject_ids"]}
            qwen_train = {str(value) for value in payload["qwen_train_subject_ids"]}
            holdout = {str(value) for value in payload["outer_holdout_subject_ids"]}
            for identity in train | inner | qwen_train | holdout:
                if not identity.startswith(f"{dataset}::"):
                    failures.append(f"namespace_mismatch:{dataset}:{fold}:{identity}")
            if train & holdout or qwen_train & inner or qwen_train & holdout or inner & holdout:
                failures.append(f"overlap:{dataset}:{fold}")
            if holdout & official:
                failures.append(f"official_test_in_cv:{dataset}:{fold}")
            if qwen_train | inner != train or qwen_train & inner:
                failures.append(f"inner_partition_not_complete:{dataset}:{fold}")
            if (train | holdout) & official:
                failures.append(f"official_test_in_outer_training:{dataset}:{fold}")
            fold_development = train | holdout
            if development_subjects is None:
                development_subjects = fold_development
            elif fold_development != development_subjects:
                failures.append(f"development_pool_changed:{dataset}:{fold}")
            if fold_development & official:
                failures.append(f"official_test_in_development_pool:{dataset}:{fold}")
            observed.update(holdout)
        if any(count != 1 for count in observed.values()) or (
            development_subjects is not None and set(observed) != development_subjects
        ):
            failures.append(f"outer_holdout_coverage_mismatch:{dataset}")
        if require_daic_official_test_count and len(official) != 47 and dataset == "daic":
            failures.append(f"daic_official_test_count:{len(official)}")
        if dataset != "daic" and official:
            failures.append(f"non_daic_official_test_subjects:{dataset}")
    return {
        "schema_version": "symmetric_protocol_split_audit.v1",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "protocol_split_hash": protocol.get("split_hash"),
    }


def save_protocol_artifacts(
    merged_config: dict[str, Any],
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    seed: int,
    inner_val_ratio: float,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, manifest_metadata = build_merged_manifest(records)
    protocol = build_protocol_splits(records, seed=seed, inner_val_ratio=inner_val_ratio)
    final_partitions = build_final_partitions(records)
    manifest_path = output / "merged_manifest.jsonl"
    write_jsonl(rows, manifest_path)
    payload = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "config_identity": {
            "name": merged_config.get("name"),
            "modality": merged_config.get("modality"),
            "components": merged_config.get("components"),
        },
        "manifest": manifest_metadata,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "protocol": protocol,
        "final_partitions": final_partitions,
        "split_audit": audit_protocol_splits(
            protocol,
            require_daic_official_test_count=bool(
                (merged_config.get("protocol_settings") or {}).get("daic_official_test_only", False)
            ),
        ),
    }
    payload["artifact_hash"] = canonical_sha256(payload)
    (output / "merged_protocol.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
