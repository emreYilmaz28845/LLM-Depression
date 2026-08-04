from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.aggregate import aggregate_margin_predictions, aggregate_predictions
from src.daic_chunking import evenly_spaced_indices, matched_k_resamples
from src.evaluate import _write_csv
from src.utils import (
    AGGREGATION_LEVEL_SUBJECT,
    PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
    save_json,
    sha256_file,
    sha256_text,
    write_jsonl,
)


SECONDARY = ("median_score", "trimmed_mean_10", "majority_margin_tiebreak", "max_score")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def derive_fixed15_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[str(row["subject_id"])].append(row)
    output: list[dict[str, Any]] = []
    for subject_id in sorted(grouped):
        rows = sorted(grouped[subject_id], key=lambda row: int(row.get("bundle_id", 0)))
        if len(rows) not in {5, 15}:
            raise ValueError(f"fixed15 derivation expected 5 or 15 min-cover bundles for {subject_id}, got {len(rows)}")
        for bundle_id in range(15):
            row = copy.deepcopy(rows[bundle_id % len(rows)])
            row["sample_id"] = f"{subject_id}__derived_fixed15_bundle_{bundle_id:03d}"
            row["bundle_id"] = bundle_id
            row["derived_from_sample_id"] = rows[bundle_id % len(rows)]["sample_id"]
            row["derived_without_model_inference"] = True
            output.append(row)
    return output


def derive_matched10_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[str(row["subject_id"])].append(row)
    output: list[dict[str, Any]] = []
    for subject_id in sorted(grouped):
        rows = grouped[subject_id]
        if len(rows) < 10:
            raise ValueError(f"matched10 derivation requires at least 10 chunks for {subject_id}")
        for index in evenly_spaced_indices(len(rows), 10):
            row = copy.deepcopy(rows[index])
            row["derived_from_sample_id"] = rows[index]["sample_id"]
            row["derived_without_model_inference"] = True
            output.append(row)
    return output


def write_derived_evaluation(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    source_path: Path,
    view: str,
    checkpoint_name: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    headline_rows, headline_metrics, subject_rows, subject_metrics = aggregate_predictions(
        rows,
        mode=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
        aggregation_level=AGGREGATION_LEVEL_SUBJECT,
    )
    headline_metrics = {**headline_metrics, "checkpoint_name": checkpoint_name}
    subject_metrics = {**subject_metrics, "checkpoint_name": checkpoint_name}
    _write_csv(rows, output_dir / "predictions_sample_level.csv")
    _write_csv(headline_rows, output_dir / "predictions_headline_level.csv")
    _write_csv(subject_rows, output_dir / "predictions_subject_level.csv")
    write_jsonl(rows, output_dir / "predictions_sample_level.jsonl")
    write_jsonl(
        [row for row in rows if row.get("teacher_forced_prediction_text") == "INVALID"],
        output_dir / "predictions_invalid_sample_level.jsonl",
    )
    save_json(headline_metrics, output_dir / "metrics_original_teacher_forced.json")
    save_json(
        {
            "prediction_backend": PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
            "evaluation_protocol_name": "teacher_forced_label_span",
            "aggregation_level": AGGREGATION_LEVEL_SUBJECT,
            "binary_strict_confusion_matrix": headline_metrics["binary_strict_confusion_matrix"],
            "diagnostic_three_class_labels": headline_metrics["diagnostic_three_class_labels"],
            "diagnostic_three_class_confusion_matrix": headline_metrics["diagnostic_three_class_confusion_matrix"],
        },
        output_dir / "confusion_matrix.json",
    )
    secondary_results = {}
    for method in SECONDARY:
        secondary_rows, metrics = aggregate_margin_predictions(rows, method)
        metrics = {**metrics, "checkpoint_name": checkpoint_name, "aggregation_method": method}
        prediction_path = output_dir / f"predictions_subject_level_{method}.jsonl"
        write_jsonl(secondary_rows, prediction_path)
        save_json(metrics, output_dir / f"metrics_secondary_{method}.json")
        secondary_results[method] = {"metrics": metrics, "prediction_path": str(prediction_path)}
    save_json(
        {"requested": list(SECONDARY), "results": secondary_results},
        output_dir / "secondary_aggregations.json",
    )
    provenance = {
        "view": view,
        "derived_without_model_inference": True,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "logical_samples": len(rows),
        "actual_model_forwards": 0,
        "actual_audio_file_loads": 0,
    }
    save_json(provenance, output_dir / "derivation_metadata.json")
    return provenance


def materialize_cached_views(
    evaluation_root: Path,
    views: list[str],
    *,
    checkpoint_name: str,
    resample_seed: int = 1337,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if "fixed15" in views:
        source = evaluation_root / "mincover4" / "predictions_sample_level.jsonl"
        results.append(write_derived_evaluation(
            derive_fixed15_rows(_read_jsonl(source)), evaluation_root / "fixed15",
            source_path=source, view="fixed15", checkpoint_name=checkpoint_name,
        ))
    if "matched10_even" in views:
        source = evaluation_root / "all" / "predictions_sample_level.jsonl"
        results.append(write_derived_evaluation(
            derive_matched10_rows(_read_jsonl(source)), evaluation_root / "matched10_even",
            source_path=source, view="matched10_even", checkpoint_name=checkpoint_name,
        ))
    if "matched10_resampled" in views:
        source = evaluation_root / "all" / "predictions_sample_level.jsonl"
        rows = matched_k_resamples(_read_jsonl(source), k=10, iterations=1000, seed=resample_seed)
        target = evaluation_root / "matched10_resampled" / "predictions_subject_resamples.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(rows, target)
        metadata = {
            "view": "matched10_resampled", "derived_without_model_inference": True,
            "source_path": str(source), "source_sha256": sha256_file(source),
            "iterations": 1000, "seed": resample_seed, "actual_model_forwards": 0,
            "actual_audio_file_loads": 0,
        }
        save_json(metadata, target.parent / "derivation_metadata.json")
        results.append(metadata)
    return results


def _derive_hidden_partition(
    source_dir: Path, target_dir: Path, partition: str, view: str,
) -> int:
    source_rows = _read_jsonl(source_dir / f"{partition}_rows.jsonl")
    with np.load(source_dir / f"{partition}.npz") as payload:
        vectors = payload["vectors"]
    if len(source_rows) != len(vectors):
        raise ValueError(f"Hidden row/vector mismatch in {source_dir} {partition}")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(source_rows):
        grouped[str(row["subject_id"])].append(index)
    chosen: list[int] = []
    output_rows: list[dict[str, Any]] = []
    for subject_id in sorted(grouped):
        indices = grouped[subject_id]
        # Evaluation-view overrides are intentionally inactive for outer-train
        # examples in build_examples(). Preserve that source partition exactly;
        # only the held-out partition is transformed into the derived view.
        if partition == "outer_train":
            selected = indices
        elif view == "fixed15":
            indices = sorted(indices, key=lambda index: int(source_rows[index].get("bundle_id", 0)))
            if len(indices) not in {5, 15}:
                raise ValueError(f"fixed15 hidden derivation expected 5 or 15 rows for {subject_id}")
            selected = [indices[bundle_id % len(indices)] for bundle_id in range(15)]
        elif view == "matched10_even":
            selected = [indices[index] for index in evenly_spaced_indices(len(indices), 10)]
        else:
            raise ValueError(f"Unsupported derived hidden view {view}")
        for output_index, source_index in enumerate(selected):
            row = copy.deepcopy(source_rows[source_index])
            row["derived_from_sample_id"] = source_rows[source_index]["sample_id"]
            row["derived_without_model_inference"] = True
            if view == "fixed15":
                row["sample_id"] = f"{subject_id}__derived_fixed15_bundle_{output_index:03d}"
                row["bundle_id"] = output_index
            chosen.append(source_index)
            output_rows.append(row)
    matrix = vectors[np.asarray(chosen, dtype=np.int64)]
    np.savez_compressed(target_dir / f"{partition}.npz", vectors=matrix)
    write_jsonl(output_rows, target_dir / f"{partition}_rows.jsonl")
    return len(output_rows)


def materialize_cached_hidden_views(cache_root: Path, views: list[str]) -> list[dict[str, Any]]:
    results = []
    mapping = {"fixed15": "mincover4", "matched10_even": "all"}
    for view, source_view in mapping.items():
        if view not in views:
            continue
        source_dir, target_dir = cache_root / source_view, cache_root / view
        target_dir.mkdir(parents=True, exist_ok=True)
        source_metadata_path = source_dir / "extraction_metadata.json"
        metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        counts = {
            partition: _derive_hidden_partition(source_dir, target_dir, partition, view)
            for partition in ("outer_train", "final_eval")
        }
        metadata = copy.deepcopy(metadata)
        metadata["condition"] = str(metadata.get("condition", "" )).rsplit("_", 1)[0] + f"_{view}"
        metadata["derived_view"] = {
            "view": view, "derived_without_model_inference": True,
            "source_view": source_view, "source_metadata": str(source_metadata_path),
            "source_metadata_sha256": sha256_file(source_metadata_path),
            "actual_model_forwards": 0, "actual_audio_file_loads": 0,
        }
        metadata["evaluation_view"] = {**metadata.get("evaluation_view", {}), "derived_view": view}
        metadata["cache_config"] = {**metadata.get("cache_config", {}), "derived_view": metadata["derived_view"]}
        metadata["cache_config_sha256"] = sha256_text(json.dumps(metadata["cache_config"], sort_keys=True, separators=(",", ":")))
        for partition, count in counts.items():
            metadata.setdefault("partitions", {}).setdefault(partition, {})["rows"] = count
        save_json(metadata, target_dir / "extraction_metadata.json")
        results.append(metadata["derived_view"])
    return results
