#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import classification_metrics
from src.utils import read_json, save_json


EXPECTED_CONFIGS = {
    "audio_only": ("audio_only", None, "response_subject"),
    "audio_text_segment_aligned": (
        "audio_text",
        "segment_aligned",
        "response_subject",
    ),
    "audio_text_full_turn": ("audio_text", "full_turn", "response_subject"),
    "text_only": ("text_only", None, "subject"),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _prediction(row: dict[str, str]) -> int:
    for key in ("prediction", "teacher_forced_prediction"):
        value = str(row.get(key, "")).strip()
        if value:
            prediction = int(float(value))
            if prediction in (0, 1):
                return prediction
            return 1 - int(float(row["label"]))
    raise ValueError(f"Missing prediction field in row for {row.get('subject_id')}")


def _audit_fold(
    fold_dir: Path,
    config_id: str,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    required = [
        fold_dir / "run_config.yaml",
        fold_dir / "logs" / "selected_checkpoint_selection_metrics.json",
        fold_dir / "eval" / "best_checkpoint" / "metrics_original_teacher_forced.json",
        fold_dir / "eval" / "best_checkpoint" / "predictions_sample_level.csv",
        fold_dir / "eval" / "best_checkpoint" / "predictions_subject_level.csv",
    ]
    if config_id != "text_only":
        required.append(
            fold_dir
            / "eval"
            / "best_checkpoint"
            / "predictions_response_level.csv"
        )
        required.append(fold_dir / "logs" / "androids_training_weight_audit.json")
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts for {fold_dir}: {missing}")
    invalid_rows_path = (
        fold_dir
        / "eval"
        / "best_checkpoint"
        / "predictions_invalid_sample_level.jsonl"
    )
    if not invalid_rows_path.is_file():
        raise FileNotFoundError(f"Missing invalid-sample audit artifact: {invalid_rows_path}")

    with (fold_dir / "run_config.yaml").open("r", encoding="utf-8") as handle:
        run_config = yaml.safe_load(handle)
    config = run_config["config"]
    expected_modality, expected_scope, expected_aggregation = EXPECTED_CONFIGS[config_id]
    modality = str(run_config.get("input_modality", ""))
    scope = config["data"].get("audio_text_transcript_scope")
    aggregation = config["evaluation"]["aggregation_level"]
    if (modality, scope, aggregation) != (
        expected_modality,
        expected_scope,
        expected_aggregation,
    ):
        raise ValueError(
            f"Resolved condition mismatch for {config_id}: "
            f"{(modality, scope, aggregation)}"
        )
    if (
        config_id != "text_only"
        and config["evaluation"].get("hierarchical_score_aggregation") != "mean"
    ):
        raise ValueError(f"Audio hierarchy is not equal-weight score averaging for {fold_dir}.")
    training = config["training"]
    early = training.get("early_stopping", {})
    if (
        int(training["num_train_epochs"]) > 20
        or training["selection_metric"] != "inner_val_macro_f1"
        or int(early.get("patience", -1)) != 3
        or not early.get("enabled")
        or not training.get("run_final_eval_in_train")
        or run_config.get("save_strategy") != "best_only"
    ):
        raise ValueError(f"Training protocol mismatch for {fold_dir}.")
    overlap = run_config.get("subject_overlap_proof", {})
    if not overlap or not overlap.get("passed", False):
        raise ValueError(f"Subject leakage proof failed for {fold_dir}.")
    selection = read_json(
        fold_dir / "logs" / "selected_checkpoint_selection_metrics.json"
    )
    truncation_count = 0
    for truncation_path in (
        fold_dir / "logs" / "train_truncation.jsonl",
        fold_dir / "logs" / "val_truncation.jsonl",
        fold_dir / "logs" / "final_eval_truncation.jsonl",
    ):
        if not truncation_path.is_file():
            raise FileNotFoundError(
                f"Missing transcript truncation audit: {truncation_path}"
            )
        truncation_count += sum(
            bool(line.strip())
            for line in truncation_path.read_text(encoding="utf-8").splitlines()
        )
    if truncation_count:
        raise ValueError(
            f"The current ANDROIDS corpus incurred {truncation_count} "
            f"transcript truncations in {fold_dir}."
        )
    headline_metrics = read_json(
        fold_dir
        / "eval"
        / "best_checkpoint"
        / "metrics_original_teacher_forced.json"
    )
    invalid_keys = {"invalid_subjects"}
    if config_id != "text_only":
        invalid_keys.update({"invalid_segments", "invalid_responses"})
    if not invalid_keys.issubset(headline_metrics):
        raise ValueError(
            f"Missing invalid-output counts in {fold_dir}: "
            f"{sorted(invalid_keys - set(headline_metrics))}"
        )
    subjects = _read_csv(
        fold_dir / "eval" / "best_checkpoint" / "predictions_subject_level.csv"
    )
    samples = _read_csv(
        fold_dir / "eval" / "best_checkpoint" / "predictions_sample_level.csv"
    )
    return {
        "fold": int(run_config["fold"]),
        "manifest_hash": run_config["manifest_hash"],
        "fold_hash": run_config["split_metadata_hash"],
        "subject_count": len(subjects),
        "sample_count": len(samples),
        "selected_epoch": selection.get("selected_epoch", selection.get("epoch")),
        "transcript_truncation_count": truncation_count,
    }, subjects, samples


def _paired_bootstrap(
    aligned: dict[str, dict[str, str]],
    full: dict[str, dict[str, str]],
    *,
    resamples: int = 10_000,
    seed: int = 1337,
) -> dict[str, Any]:
    subject_ids = sorted(aligned)
    if set(subject_ids) != set(full):
        raise ValueError("Paired bootstrap conditions have different subjects.")
    for subject_id in subject_ids:
        if int(aligned[subject_id]["label"]) != int(full[subject_id]["label"]):
            raise ValueError(f"Paired bootstrap label mismatch for {subject_id}.")
    rng = random.Random(seed)
    metrics = ("macro_f1", "positive_f1", "accuracy")
    deltas: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(resamples):
        sampled = [subject_ids[rng.randrange(len(subject_ids))] for _ in subject_ids]
        gold = [int(aligned[subject_id]["label"]) for subject_id in sampled]
        aligned_pred = [_prediction(aligned[subject_id]) for subject_id in sampled]
        full_pred = [_prediction(full[subject_id]) for subject_id in sampled]
        aligned_metrics = classification_metrics(gold, aligned_pred)
        full_metrics = classification_metrics(gold, full_pred)
        for metric in metrics:
            deltas[metric].append(
                float(aligned_metrics[metric]) - float(full_metrics[metric])
            )
    result: dict[str, Any] = {
        "comparison": "audio_text_segment_aligned_minus_audio_text_full_turn",
        "resamples": resamples,
        "seed": seed,
        "subject_count": len(subject_ids),
        "metrics": {},
    }
    for metric, values in deltas.items():
        ordered = sorted(values)
        result["metrics"][metric] = {
            "mean_delta": sum(values) / len(values),
            "ci95_low": ordered[int(0.025 * (len(ordered) - 1))],
            "ci95_high": ordered[int(0.975 * (len(ordered) - 1))],
            "probability_aligned_better": sum(value > 0 for value in values)
            / len(values),
        }
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    specs: dict[str, Path] = {}
    for raw in args.run:
        config_id, separator, path_text = raw.partition("=")
        if not separator or config_id in specs:
            raise ValueError(f"Invalid or duplicate --run spec: {raw!r}")
        specs[config_id] = Path(path_text)
    if set(specs) != set(EXPECTED_CONFIGS):
        raise ValueError(
            f"Expected run IDs {sorted(EXPECTED_CONFIGS)}, got {sorted(specs)}."
        )
    expected_folds = 1 if args.mode == "smoke" else 5
    report: dict[str, Any] = {
        "schema_version": "androids_interview_acceptance.v1",
        "mode": args.mode,
        "status": "passed",
        "configurations": {},
    }
    common_manifest_hashes: set[str] = set()
    common_fold_hashes: set[str] = set()
    pooled_by_config: dict[str, dict[str, dict[str, str]]] = {}
    samples_by_config_fold: dict[str, dict[int, list[dict[str, str]]]] = {}
    for config_id, run_root in sorted(specs.items()):
        fold_dirs = sorted(run_root.glob("fold_*"))
        if len(fold_dirs) != expected_folds:
            raise ValueError(
                f"{config_id} has {len(fold_dirs)} folds; expected {expected_folds}."
            )
        fold_reports: list[dict[str, Any]] = []
        pooled: dict[str, dict[str, str]] = {}
        samples_by_fold: dict[int, list[dict[str, str]]] = {}
        for fold_dir in fold_dirs:
            fold_report, subjects, samples = _audit_fold(fold_dir, config_id)
            fold_reports.append(fold_report)
            samples_by_fold[fold_report["fold"]] = samples
            common_manifest_hashes.add(fold_report["manifest_hash"])
            common_fold_hashes.add(fold_report["fold_hash"])
            for row in subjects:
                subject_id = row["subject_id"]
                if subject_id in pooled:
                    raise ValueError(
                        f"Duplicate out-of-fold subject for {config_id}: {subject_id}"
                    )
                pooled[subject_id] = row
        pooled_by_config[config_id] = pooled
        samples_by_config_fold[config_id] = samples_by_fold
        if args.mode == "matrix" and len(pooled) != 116:
            raise ValueError(f"{config_id} pooled subject count is {len(pooled)}, expected 116.")
        report["configurations"][config_id] = {
            "run_root": str(run_root),
            "folds": fold_reports,
            "pooled_subject_count": len(pooled),
            "pooled_label_counts": dict(
                Counter(int(row["label"]) for row in pooled.values())
            ),
        }
    if len(common_manifest_hashes) != 1 or len(common_fold_hashes) != 1:
        raise ValueError(
            "Configurations do not share one manifest and official-fold hash: "
            f"manifest={common_manifest_hashes} folds={common_fold_hashes}"
        )
    report["manifest_hash"] = next(iter(common_manifest_hashes))
    report["fold_hash"] = next(iter(common_fold_hashes))
    audio_config_ids = (
        "audio_only",
        "audio_text_segment_aligned",
        "audio_text_full_turn",
    )
    for fold in samples_by_config_fold["audio_only"]:
        inventories = []
        for config_id in audio_config_ids:
            inventories.append(
                sorted(
                    [
                        (
                            row["sample_id"],
                            row["response_id"],
                            row["start_time"],
                            row["end_time"],
                        )
                        for row in samples_by_config_fold[config_id][fold]
                    ]
                )
            )
        if not all(inventory == inventories[0] for inventory in inventories[1:]):
            raise ValueError(f"Audio window/interval inventories differ in fold {fold}.")
        if not any(
            int(float(row["num_segments"])) > 1
            for row in samples_by_config_fold["audio_only"][fold]
        ):
            raise ValueError(f"Smoke/production fold {fold} did not evaluate a multi-window turn.")
        full_turn_rows = samples_by_config_fold["audio_text_full_turn"][fold]
        hashes_by_turn: dict[str, set[str]] = {}
        for row in full_turn_rows:
            hashes_by_turn.setdefault(row["response_id"], set()).add(
                row["transcript_sha256"]
            )
        if any(len(values) != 1 for values in hashes_by_turn.values()):
            raise ValueError(f"Full-turn text is inconsistent within a parent turn in fold {fold}.")
        aligned_by_id = {
            row["sample_id"]: row
            for row in samples_by_config_fold["audio_text_segment_aligned"][fold]
        }
        full_by_id = {row["sample_id"]: row for row in full_turn_rows}
        if not any(
            aligned_by_id[sample_id]["prompt_sha256"]
            != full_by_id[sample_id]["prompt_sha256"]
            for sample_id in aligned_by_id
        ):
            raise ValueError(f"Audio+text prompt scopes did not differ in fold {fold}.")
        if any(
            aligned_by_id[sample_id]["prompt_context_sha256"]
            != full_by_id[sample_id]["prompt_context_sha256"]
            for sample_id in aligned_by_id
        ):
            raise ValueError(
                f"Audio+text prompts differ outside transcript scope in fold {fold}."
            )
    if args.mode == "matrix":
        report["primary_paired_bootstrap"] = _paired_bootstrap(
            pooled_by_config["audio_text_segment_aligned"],
            pooled_by_config["audio_text_full_turn"],
        )
    save_json(report, args.out)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "matrix"), required=True)
    parser.add_argument("--run", action="append", required=True, help="CONFIG_ID=RUN_ROOT")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = audit(parse_args())
    print(f"ANDROIDS Interview audit {result['status']}: {result['mode']}")
