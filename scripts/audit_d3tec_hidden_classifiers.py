#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import binary_auroc, classification_metrics
from src.utils import read_json, read_jsonl, save_json


STAGE1_ID = "xgb_optuna_raw_t150_d6_seed1337_inner1337"
ALL_POSITIVE_POS_F1 = 58 / 91


def _negative_f1(metrics: dict[str, Any]) -> float:
    tn, fp = metrics["confusion_matrix"][0]
    fn, _ = metrics["confusion_matrix"][1]
    precision = tn / (tn + fn) if tn + fn else 0.0
    recall = tn / (tn + fp) if tn + fp else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _pooled_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [int(row["label"]) for row in rows]
    y_pred = [int(row["prediction"]) for row in rows]
    metrics = classification_metrics(y_true, y_pred)
    metrics["negative_f1"] = _negative_f1(metrics)
    probabilities = [
        float(row.get("probability", row.get("dep_score", 0.0))) for row in rows
    ]
    metrics["auroc"] = binary_auroc(y_true, probabilities)
    return metrics


def _subject_metadata(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        subject_id = str(row["subject_id"])
        candidate = {
            "subject_id": subject_id,
            "label": int(row["label"]),
            "gender": str(row.get("gender", "")).strip(),
        }
        if subject_id in result and result[subject_id] != candidate:
            raise ValueError(f"Inconsistent canonical metadata for subject {subject_id}.")
        result[subject_id] = candidate
    return result


def _gender_analysis(
    rows: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    joined = []
    for row in rows:
        subject_id = str(row["subject_id"])
        if subject_id not in metadata:
            raise ValueError(f"Prediction subject missing canonical gender: {subject_id}.")
        joined.append({**row, "gender": metadata[subject_id]["gender"]})
    errors = Counter(
        row["gender"]
        for row in joined
        if int(row["prediction"]) != int(row["label"])
    )
    counts = Counter(row["gender"] for row in joined)
    return {
        "subject_counts": dict(sorted(counts.items())),
        "error_counts": dict(sorted(errors.items())),
        "error_rates": {
            gender: errors[gender] / count for gender, count in sorted(counts.items())
        },
    }


def _audit_head(
    result_dir: Path,
    *,
    condition: str,
    modality: str,
    fold: int,
    variant: str,
    optuna: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required = {
        "pipeline.joblib",
        "predictions_sample_level.jsonl",
        "predictions_subject_level.jsonl",
        "predictions_subject_level.csv",
        "predictions_sample_level.csv",
        "metrics.json",
        "classifier_metadata.json",
    }
    if not optuna:
        required.update(("result_config.json", "sampling_audit.json"))
        if variant == "majority_class":
            required.remove("pipeline.joblib")
    else:
        required.update(
            (
                "study_config.json",
                "study.sqlite3",
                "trials.csv",
                "best_params.json",
                "inner_subject_assignments.json",
                "inner_sampling_audits.json",
                "inner_weight_audits.json",
                "inner_fold_metrics.json",
                "inner_oof_metrics.json",
                "final_fit_weight_audit.json",
            )
        )
    if modality != "text_only":
        required.update(
            (
                "predictions_response_level.jsonl",
                "predictions_response_level.csv",
                "metrics_response_level.json",
            )
        )
    missing = sorted(name for name in required if not (result_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"{result_dir}: missing artifacts {missing}.")
    head_metadata = read_json(result_dir / "classifier_metadata.json")
    if (
        head_metadata["condition"] != condition
        or int(head_metadata["fold"]) != fold
        or head_metadata["classifier_variant"] != variant
    ):
        raise ValueError(f"Classifier identity mismatch: {result_dir}.")
    if optuna and int(head_metadata["completed_trials"]) != 150:
        raise ValueError(f"Optuna result does not contain exactly 150 trials: {result_dir}.")
    if set(head_metadata["training_subject_ids"]) & set(head_metadata["heldout_subject_ids"]):
        raise ValueError(f"Outer train/evaluation subject leakage: {result_dir}.")
    if not all(
        isinstance(value, str) and len(value) == 64
        for value in head_metadata["checkpoint_hashes"].values()
    ):
        raise ValueError(f"Missing checkpoint hashes: {result_dir}.")
    weight = (
        head_metadata["final_fit_weight_audit"]
        if optuna
        else head_metadata["fit_weight_audit"]
    )
    if modality == "text_only":
        if weight["policy"] != "one_vector_per_subject_unweighted":
            raise ValueError(f"Text-only result has an invalid weight policy: {result_dir}.")
    elif not (
        weight["equal_response_totals"]
        and weight["equal_subject_totals"]
        and set(weight["responses_per_subject"].values()) == {27}
    ):
        raise ValueError(f"Incomplete D3TEC response-weight audit: {result_dir}.")
    if optuna:
        assignments = head_metadata["inner_subject_assignments"]
        for inner in assignments["folds"]:
            if set(inner["train_subject_ids"]) & set(inner["validation_subject_ids"]):
                raise ValueError(f"Inner subject leakage: {result_dir}.")
        for inner in head_metadata["inner_sampling_audits"]:
            if not inner.get("subject_overlap_free", True):
                raise ValueError(f"Inner sampling leakage: {result_dir}.")
    rows = read_jsonl(result_dir / "predictions_subject_level.jsonl")
    if modality == "text_only":
        sample_rows = read_jsonl(result_dir / "predictions_sample_level.jsonl")
        if len(sample_rows) != len(rows):
            raise ValueError(f"Text-only must have one vector/prediction per subject: {result_dir}.")
    else:
        response_rows = read_jsonl(result_dir / "predictions_response_level.jsonl")
        counts = Counter(str(row["subject_id"]) for row in response_rows)
        if set(counts.values()) != {27}:
            raise ValueError(f"Expected exactly 27 evaluated responses per subject: {result_dir}.")
    return rows, {
        "fold": fold,
        "result_dir": str(result_dir),
        "subjects": len(rows),
        "metrics": read_json(result_dir / "metrics.json"),
        "cache_identity_sha256": head_metadata["cache_identity_sha256"],
        "checkpoint_hashes": head_metadata["checkpoint_hashes"],
        "split_hashes": head_metadata["split_hashes"],
    }


def audit(
    matrix_path: Path,
    results_root: Path,
    subject_metadata_path: Path,
    *,
    include_optuna: bool = True,
) -> dict[str, Any]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    expected_manifest = matrix["expected_manifest_sha256"]
    expected_split = matrix["expected_split_metadata_sha256"]
    canonical = _subject_metadata(subject_metadata_path)
    all_results = []
    heldout_by_condition: dict[str, set[str]] = defaultdict(set)
    labels_by_condition: dict[str, dict[str, int]] = defaultdict(dict)
    for experiment in matrix["experiments"]:
        condition = experiment["condition"]
        modality = experiment["modality"]
        run_name = Path(experiment["run_dir"]).name
        variants = list(experiment["variants"]) + ([STAGE1_ID] if include_optuna else [])
        for variant in variants:
            pooled = []
            folds = []
            seen = set()
            for fold in range(5):
                result_dir = (
                    results_root
                    / "d3tec"
                    / condition
                    / run_name
                    / f"fold_{fold}"
                    / variant
                )
                rows, fold_audit = _audit_head(
                    result_dir,
                    condition=condition,
                    modality=modality,
                    fold=fold,
                    variant=variant,
                    optuna=variant == STAGE1_ID,
                )
                split_hashes = fold_audit["split_hashes"]
                if (
                    split_hashes["manifest_sha256"] != expected_manifest
                    or split_hashes["split_metadata_sha256"] != expected_split
                ):
                    raise ValueError(f"Manifest/split hash mismatch: {result_dir}.")
                subjects = {str(row["subject_id"]) for row in rows}
                if seen & subjects:
                    raise ValueError(f"{condition}/{variant}: repeated held-out subjects.")
                seen.update(subjects)
                pooled.extend(rows)
                folds.append(fold_audit)
            labels = {str(row["subject_id"]): int(row["label"]) for row in pooled}
            if len(seen) != 62 or Counter(labels.values()) != Counter({0: 33, 1: 29}):
                raise ValueError(
                    f"{condition}/{variant}: expected 62 subjects (33/29), "
                    f"found {len(seen)} {dict(Counter(labels.values()))}."
                )
            heldout_by_condition[condition].update(seen)
            labels_by_condition[condition].update(labels)
            all_results.append(
                {
                    "condition": condition,
                    "modality": modality,
                    "variant": variant,
                    "folds": folds,
                    "pooled_metrics": _pooled_metrics(pooled),
                    "gender_analysis": _gender_analysis(pooled, canonical),
                }
            )
    condition_sets = list(heldout_by_condition.values())
    if any(subjects != condition_sets[0] for subjects in condition_sets[1:]):
        raise ValueError("Conditions do not evaluate the same 62 subjects.")
    subject_ids = sorted(condition_sets[0])
    sex_rule_rows = [
        {
            "subject_id": subject_id,
            "label": canonical[subject_id]["label"],
            "prediction": int(canonical[subject_id]["gender"].lower() == "female"),
            "probability": float(canonical[subject_id]["gender"].lower() == "female"),
        }
        for subject_id in subject_ids
    ]
    return {
        "schema_version": "d3tec_hidden_classifier_acceptance.v1",
        "status": "passed",
        "scope": "fixed_and_stage1_optuna" if include_optuna else "fixed_only",
        "matrix": str(matrix_path),
        "result_count": len(all_results),
        "conditions": sorted(heldout_by_condition),
        "unique_subjects": len(subject_ids),
        "label_counts": dict(sorted(Counter(canonical[s]["label"] for s in subject_ids).items())),
        "all_positive_baseline": {
            "positive_f1": ALL_POSITIVE_POS_F1,
            "fraction": "58/91",
        },
        "female_positive_sex_rule_baseline": _pooled_metrics(sex_rule_rows),
        "results": all_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Acceptance-audit D3TEC hidden classifiers.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/features/d3tec_hidden_matrix.yaml"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/hidden_classifiers"),
    )
    parser.add_argument(
        "--subject-metadata",
        type=Path,
        default=Path("outputs/manifests_d3tec/d3tec_manifest.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fixed-only",
        action="store_true",
        help="Audit the fixed-head stage before Stage-1 Optuna results exist.",
    )
    args = parser.parse_args()
    payload = audit(
        args.matrix,
        args.results_root,
        args.subject_metadata,
        include_optuna=not args.fixed_only,
    )
    save_json(payload, args.output)
    print(f"PASS: {args.output}")


if __name__ == "__main__":
    main()
