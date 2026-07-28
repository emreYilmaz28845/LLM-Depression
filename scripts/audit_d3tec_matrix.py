#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import binary_auroc, classification_metrics
from src.utils import read_json, save_json


EXPECTED_CONFIGS = {
    "audio_only_rotary",
    "audio_only_flat",
    "audio_only_normalized",
    "audio_text_rotary",
    "audio_text_flat",
    "audio_text_normalized",
    "text_only",
}


def _assert_resolved_config(config_id: str, run_config: dict[str, Any]) -> None:
    config = run_config["config"]
    data = config["data"]
    training = config["training"]
    evaluation = config["evaluation"]
    if config["dataset"] != "d3tec" or int(config["seed"]) != 1337:
        raise ValueError(f"{config_id}: invalid dataset/seed in resolved config.")
    if (
        float(data["segment_seconds"]) != 30.0
        or data["segment_partition"] != "equal_duration"
        or config["split"]["cv_protocol"] != "train_val_test"
        or int(config["split"]["outer_folds"]) != 5
        or float(config["split"]["inner_val_ratio"]) != 0.2
        or evaluation["sample_prediction_mode"] != "original_teacher_forced"
        or bool(evaluation.get("evaluate_last_checkpoint", True))
    ):
        raise ValueError(f"{config_id}: resolved protocol differs from the D3TEC plan.")
    if not config.get("full_transcript_path") or not config.get("segment_transcript_path"):
        raise ValueError(f"{config_id}: transcript sources are not explicit.")
    if config_id == "text_only":
        if data.get("use_audio") or not data.get("use_text"):
            raise ValueError("text_only: invalid modality flags.")
        if int(training["num_train_epochs"]) != 8:
            raise ValueError("text_only: expected eight natural subject epochs.")
        if evaluation["aggregation_level"] != "subject":
            raise ValueError("text_only: expected subject aggregation.")
        return
    modality, policy_name = config_id.rsplit("_", 1)
    expected_policy = {
        "rotary": "rotate_one_per_response",
        "flat": "all_segments_flat",
        "normalized": "all_segments_response_normalized",
    }[policy_name]
    expected_text = modality == "audio_text"
    if not data.get("use_audio") or bool(data.get("use_text")) != expected_text:
        raise ValueError(f"{config_id}: invalid modality flags.")
    if data["train_chunk_policy"] != expected_policy:
        raise ValueError(f"{config_id}: invalid chunk policy.")
    if (
        training.get("compute_budget_mode") != "rotary_reference"
        or int(training.get("reference_virtual_epochs", 0)) != 8
        or int(training.get("reference_examples_per_response", 0)) != 1
        or int(training["gradient_accumulation_steps"]) != 32
        or evaluation["aggregation_level"] != "response_subject"
    ):
        raise ValueError(f"{config_id}: invalid compute budget or aggregation protocol.")


def _strict_prediction(row: dict[str, Any]) -> int:
    prediction = int(row["prediction"])
    return prediction if prediction in (0, 1) else 1 - int(row["label"])


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [int(row["label"]) for row in rows]
    y_pred = [_strict_prediction(row) for row in rows]
    result = classification_metrics(y_true, y_pred)
    result["auroc"] = binary_auroc(
        y_true,
        [
            float(
                row.get(
                    "score_margin",
                    float(row.get("dep_score", 0.0)) - float(row.get("non_score", 0.0)),
                )
            )
            for row in rows
        ],
    )
    result["invalid_subjects"] = sum(int(row["prediction"]) not in (0, 1) for row in rows)
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_run(config_id: str, run_root: Path) -> dict[str, Any]:
    folds = sorted(run_root.glob("fold_*"))
    if [path.name for path in folds] != [f"fold_{index}" for index in range(5)]:
        raise ValueError(f"{config_id}: expected folds 0-4 under {run_root}.")
    manifest_hashes: set[str] = set()
    split_hashes: set[str] = set()
    pooled_subject_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    heldout_seen: set[str] = set()
    for fold_index, fold_dir in enumerate(folds):
        run_config_path = fold_dir / "run_config.yaml"
        split_path = fold_dir / "logs" / "split_used.json"
        selection_path = fold_dir / "logs" / "selected_checkpoint_selection_metrics.json"
        eval_dir = fold_dir / "eval" / "best_checkpoint"
        required = [
            run_config_path,
            split_path,
            selection_path,
            eval_dir / "predictions_sample_level.csv",
            eval_dir / "predictions_subject_level.csv",
            eval_dir / "metrics_original_teacher_forced.json",
        ]
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(f"{config_id} fold {fold_index} missing artifacts: {missing}")
        import yaml

        run_config = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
        _assert_resolved_config(config_id, run_config)
        split = read_json(split_path)
        selection = read_json(selection_path)
        fold_metrics_payload = read_json(
            eval_dir / "metrics_original_teacher_forced.json"
        )
        manifest_hashes.add(str(run_config["manifest_hash"]))
        split_hashes.add(str(run_config["split_metadata_hash"]))
        train = set(split["train_subject_ids"])
        validation = set(split["selection_subject_ids"])
        test = set(split["final_eval_subject_ids"])
        overlaps = {
            "train_validation": sorted(train & validation),
            "train_test": sorted(train & test),
            "validation_test": sorted(validation & test),
        }
        if any(overlaps.values()):
            raise ValueError(f"{config_id} fold {fold_index} subject leakage: {overlaps}")
        duplicate_holdout = heldout_seen & test
        if duplicate_holdout:
            raise ValueError(
                f"{config_id}: subjects held out more than once: {sorted(duplicate_holdout)}"
            )
        heldout_seen.update(test)
        subject_rows = _read_csv(eval_dir / "predictions_subject_level.csv")
        if {row["subject_id"] for row in subject_rows} != test:
            raise ValueError(f"{config_id} fold {fold_index}: subject prediction coverage mismatch.")
        sample_rows = _read_csv(eval_dir / "predictions_sample_level.csv")
        response_path = eval_dir / "predictions_response_level.csv"
        response_rows = _read_csv(response_path) if response_path.exists() else []
        if config_id != "text_only" and len(response_rows) != len(test) * 27:
            raise ValueError(
                f"{config_id} fold {fold_index}: expected 27 response predictions "
                f"per held-out subject."
            )
        pooled_subject_rows.extend(subject_rows)
        fold_rows.append(
            {
                "fold": fold_index,
                "train_subjects": len(train),
                "validation_subjects": len(validation),
                "test_subjects": len(test),
                "test_responses": len(response_rows),
                "test_segments": len(sample_rows),
                "selected_virtual_epoch": int(selection["selected_epoch"]),
                "invalid_segments": int(fold_metrics_payload.get("invalid_segments", 0)),
                "invalid_responses": int(fold_metrics_payload.get("invalid_responses", 0)),
                "invalid_subjects": int(fold_metrics_payload.get("invalid_subjects", 0)),
                **_metrics(subject_rows),
            }
        )
    if len(manifest_hashes) != 1 or len(split_hashes) != 1:
        raise ValueError(
            f"{config_id}: manifest/split hashes differ across folds: "
            f"{manifest_hashes} / {split_hashes}"
        )
    if len(heldout_seen) != 62 or len(pooled_subject_rows) != 62:
        raise ValueError(
            f"{config_id}: expected 62 unique OOF subjects; "
            f"unique={len(heldout_seen)} rows={len(pooled_subject_rows)}"
        )
    summary_metrics: dict[str, dict[str, float]] = {}
    for metric in ("accuracy", "precision", "recall", "positive_f1", "macro_f1", "weighted_f1", "auroc"):
        values = [float(row[metric]) for row in fold_rows]
        summary_metrics[metric] = {
            "mean": float(statistics.mean(values)),
            "std": float(statistics.stdev(values)),
        }
    pooled_metrics = _metrics(pooled_subject_rows)
    pooled_metrics["invalid_segments"] = sum(int(row["invalid_segments"]) for row in fold_rows)
    pooled_metrics["invalid_responses"] = sum(int(row["invalid_responses"]) for row in fold_rows)
    return {
        "config_id": config_id,
        "run_root": str(run_root),
        "manifest_hash": next(iter(manifest_hashes)),
        "split_hash": next(iter(split_hashes)),
        "folds": fold_rows,
        "fold_metric_summary": summary_metrics,
        "pooled_metrics": pooled_metrics,
        "subject_rows": sorted(pooled_subject_rows, key=lambda row: row["subject_id"]),
    }


def _bootstrap_difference(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    left_by_id = {row["subject_id"]: row for row in left}
    right_by_id = {row["subject_id"]: row for row in right}
    ids = sorted(left_by_id)
    if set(ids) != set(right_by_id):
        raise ValueError("Paired bootstrap configurations do not have identical subjects.")
    observed = _metrics(left)[metric] - _metrics(right)[metric]
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(samples):
        sampled_ids = [rng.choice(ids) for _ in ids]
        left_sample = [left_by_id[subject_id] for subject_id in sampled_ids]
        right_sample = [right_by_id[subject_id] for subject_id in sampled_ids]
        differences.append(_metrics(left_sample)[metric] - _metrics(right_sample)[metric])
    differences.sort()
    lower = differences[int(0.025 * samples)]
    upper = differences[min(samples - 1, int(0.975 * samples))]
    return {
        "left_minus_right": float(observed),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and summarize the complete D3TEC matrix.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="CONFIG_ID=RUN_ROOT; repeat exactly seven times.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_paths: dict[str, Path] = {}
    for value in args.run:
        config_id, separator, path = value.partition("=")
        if not separator or not config_id or not path:
            raise ValueError(f"Invalid --run value: {value!r}")
        run_paths[config_id] = Path(path)
    if set(run_paths) != EXPECTED_CONFIGS:
        raise ValueError(
            f"Expected config IDs {sorted(EXPECTED_CONFIGS)}; found {sorted(run_paths)}"
        )
    results = {key: audit_run(key, path) for key, path in sorted(run_paths.items())}
    manifest_hashes = {item["manifest_hash"] for item in results.values()}
    split_hashes = {item["split_hash"] for item in results.values()}
    if len(manifest_hashes) != 1 or len(split_hashes) != 1:
        raise ValueError("Manifest and fold hashes must be identical across all seven configs.")

    comparisons: dict[str, Any] = {}
    for modality in ("audio_only", "audio_text"):
        pairs = (
            ("rotary", "flat"),
            ("rotary", "normalized"),
            ("flat", "normalized"),
        )
        for left_policy, right_policy in pairs:
            left_id = f"{modality}_{left_policy}"
            right_id = f"{modality}_{right_policy}"
            key = f"{left_id}_vs_{right_id}"
            comparisons[key] = {
                metric: _bootstrap_difference(
                    results[left_id]["subject_rows"],
                    results[right_id]["subject_rows"],
                    metric=metric,
                    samples=args.bootstrap_samples,
                    seed=args.seed,
                )
                for metric in ("macro_f1", "positive_f1", "accuracy")
            }

    payload = {
        "schema_version": "d3tec_matrix_audit.v1",
        "manifest_hash": next(iter(manifest_hashes)),
        "split_hash": next(iter(split_hashes)),
        "bootstrap_samples": args.bootstrap_samples,
        "configs": {
            key: {field: value for field, value in item.items() if field != "subject_rows"}
            for key, item in results.items()
        },
        "paired_policy_comparisons": comparisons,
        "scientific_limitations": [
            "D3TEC contains only 62 subjects, so fold-level estimates have high variance.",
            "Segments inherit subject-level PHQ-9 labels and have no independent depression labels.",
            "Prompt meanings are unavailable in a machine-readable mapping.",
            "Audio+text uses machine-generated Spanish transcripts.",
            "The main matrix uses SM-27 audio only and does not establish cross-device robustness.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(payload, args.out / "d3tec_matrix_audit.json")
    with (args.out / "d3tec_matrix_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "config_id",
            "accuracy",
            "precision",
            "recall",
            "positive_f1",
            "macro_f1",
            "weighted_f1",
            "auroc",
            "invalid_subjects",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for config_id, item in sorted(results.items()):
            writer.writerow(
                {
                    key: value
                    for key, value in {"config_id": config_id, **item["pooled_metrics"]}.items()
                    if key in fieldnames
                }
            )
    print(json.dumps({"status": "passed", "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
