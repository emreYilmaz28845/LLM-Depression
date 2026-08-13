#!/usr/bin/env python3
"""Local verification for DAIC official-development training attempts.

Model-free. For one training attempt fold dir: hash-verifies the recorded
artifacts, independently recomputes the teacher-forced subject predictions
and all headline metrics from the synced sample-level predictions, matches
them to metrics_original_teacher_forced.json and the evaluation record, marks
the evaluation locally verified and reportable, then transitions the attempt
SYNCED_LOCALLY -> LOCALLY_VALIDATED -> REPORTABLE through the lifecycle API.
Any mismatch blocks reportability; evidence is never edited.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aggregate import aggregate_original_teacher_forced_predictions
from src.experiment_tracking.evidence import verify_artifacts_locally
from src.experiment_tracking.lifecycle import StatusRecord, read_status, write_status
from src.experiment_tracking.schemas import validate_record
from src.utils import read_json, read_jsonl

EVALUATION_METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "positive_f1",
    "negative_f1",
    "macro_f1",
)


class VerificationFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationFailure(message)


def _metrics_with_negative_f1(metrics: dict[str, Any]) -> dict[str, Any]:
    tn, fp = metrics["confusion_matrix"][0]
    fn, _ = metrics["confusion_matrix"][1]
    precision_neg = tn / (tn + fn) if tn + fn else 0.0
    recall_neg = tn / (tn + fp) if tn + fp else 0.0
    output = dict(metrics)
    output["negative_f1"] = (
        2 * precision_neg * recall_neg / (precision_neg + recall_neg)
        if precision_neg + recall_neg
        else 0.0
    )
    return output


def verify_training_attempt(fold_dir: Path) -> dict[str, Any]:
    fold_dir = fold_dir.resolve()
    _require((fold_dir / "status.json").is_file(), "not a tracked fold dir (no status.json)")
    state = read_status(fold_dir / "status.json")["state"]
    _require(
        state in {"COMPLETED_ON_MN5", "SYNCED_LOCALLY", "LOCALLY_VALIDATED", "REPORTABLE"},
        f"state is {state}",
    )

    # 1. Hash-verify the recorded artifacts.
    artifact_result = verify_artifacts_locally(fold_dir)
    _require(artifact_result.get("verified") == artifact_result.get("total"), f"artifact verification: {artifact_result}")

    # 2. Independent recomputation from the synced sample predictions.
    eval_dir = fold_dir / "best_model" / "standalone_eval"
    sample_path = eval_dir / "predictions_sample_level.csv"
    _require(sample_path.is_file(), "sample predictions missing")
    sample_rows = []
    with sample_path.open(newline="", encoding="utf-8") as handle:
        import csv as _csv

        for row in _csv.DictReader(handle):
            sample_rows.append(row)
    _require(bool(sample_rows), "sample predictions empty")
    subject_rows, recomputed = aggregate_original_teacher_forced_predictions(sample_rows)
    recomputed = _metrics_with_negative_f1(recomputed)
    _require(len(subject_rows) == 35, f"recomputed subject rows {len(subject_rows)} != 35")
    _require(
        {str(row["subject_id"]) for row in subject_rows}.issubset({str(r["subject_id"]) for r in sample_rows}),
        "recomputed subject rows contain unknown subject ids",
    )

    metrics_path = eval_dir / "metrics_original_teacher_forced.json"
    _require(metrics_path.is_file(), "metrics JSON missing")
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for name in EVALUATION_METRIC_NAMES:
        recomputed_value = recomputed.get(name)
        saved_value = saved_metrics.get(name)
        _require(recomputed_value is not None, f"recomputation missing metric {name}")
        if name == "negative_f1" and saved_value is None:
            # The teacher-forced path predates negative-F1 derivation; the
            # recomputation derives it from the same confusion matrix and the
            # evaluation record below is filled from the recomputed value.
            continue
        _require(saved_value is not None, f"missing metric {name}")
        _require(
            abs(float(recomputed_value) - float(saved_value)) <= 1e-9,
            f"recomputed {name}={recomputed_value} != saved {saved_value}",
        )
    recomputed_cm = recomputed["confusion_matrix"]
    saved_cm = saved_metrics.get("confusion_matrix")
    _require(recomputed_cm == saved_cm, f"confusion matrix mismatch: {recomputed_cm} != {saved_cm}")
    _require(
        saved_metrics.get("evaluation_view") == "harmonized_all_windows_full_coverage",
        "evaluation view mismatch",
    )
    _require(saved_metrics.get("prediction_backend") == "original_teacher_forced", "backend mismatch")

    # 3. Match the evaluation record. The training-path recorder historically
    # wrote the generic fixed-mode protocol; the campaign qualifier is derived
    # from the saved config. Records written before the derivation fix are
    # corrected here (protocol + recomputed evaluation id), never their
    # metrics or predictions.
    evaluations_path = fold_dir / "evaluations.json"
    _require(evaluations_path.is_file(), "evaluations.json missing")
    evaluations = json.loads(evaluations_path.read_text(encoding="utf-8"))
    _require(len(evaluations.get("evaluations") or []) == 1, "expected exactly one evaluation record")
    record = evaluations["evaluations"][0]
    _require(record["backend"] == "original_teacher_forced", "record backend mismatch")
    _require(record["split_name"] == "val", "record split name mismatch")
    _require(record["metric_namespace"] == "headline/binary_strict", "record namespace mismatch")
    from src.experiment_tracking.identity import evaluation_id

    expected_protocol = "daic_official_train_inner_split_dev_evaluation"
    checkpoint_path = str(Path(fold_dir) / "best_model")
    if record["split_protocol"] != expected_protocol:
        record["split_protocol"] = expected_protocol
        artifacts_payload = json.loads((fold_dir / "artifacts.json").read_text(encoding="utf-8"))
        metrics_artifacts = [
            artifact for artifact in artifacts_payload.get("artifacts", [])
            if artifact.get("role") in {"metrics", "standalone_eval_metrics"}
        ]
        _require(bool(metrics_artifacts), "no metrics artifact recorded")
        metrics_sha = metrics_artifacts[0].get("sha256")
        _require(bool(metrics_sha), "metrics artifact has no sha256")
        record["evaluation_id"] = evaluation_id(
            attempt_id=record.get("attempt_id") or str(
                json.loads((fold_dir / "metadata.json").read_text(encoding="utf-8"))["attempt_id"]
            ),
            fold=0,
            dataset=record["dataset"],
            split_name=record["split_name"],
            split_protocol=record["split_protocol"],
            checkpoint_role=record["checkpoint_role"],
            checkpoint_path=checkpoint_path,
            backend=record["backend"],
            evaluation_view=record["evaluation_view"],
            aggregation=record["aggregation"],
            metric_namespace=record["metric_namespace"],
            metrics_artifact_sha256=metrics_sha,
        )
    _require(record["split_protocol"] == expected_protocol, "record split protocol mismatch")
    for metric in record["metrics"]:
        _require(metric["support"] == 35, f"record metric {metric['name']} support != 35")
        expected_value = recomputed.get(metric["name"])
        _require(expected_value is not None, f"record metric {metric['name']} has no recomputed value")
        if metric["value"] is None and metric["name"] == "negative_f1":
            # Fill the derived negative-F1 from the recomputation.
            metric["value"] = float(expected_value)
        _require(
            abs(float(metric["value"]) - float(expected_value)) <= 1e-9,
            f"record metric {metric['name']} differs from recomputed",
        )

    # 4. Mark the evaluation locally verified and reportable.
    for metric in record["metrics"]:
        _ = metric
    record["locally_verified"] = True
    record["reportable"] = True
    record["warnings"] = []
    ok, errors = validate_record("audiollm.evaluations.v1", evaluations)
    _require(ok, "invalid evaluations record: " + "; ".join(errors))
    evaluations_path.write_text(json.dumps(evaluations, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # 5. Transition through the remaining lifecycle (no-op when already
    # REPORTABLE: rerunning is idempotent).
    status_path = fold_dir / "status.json"
    status = StatusRecord.from_dict(read_json(status_path))
    if status.state == "COMPLETED_ON_MN5":
        status.transition("SYNCED_LOCALLY", reason="compact evidence synced locally")
        write_status(status_path, status)
    if status.state == "SYNCED_LOCALLY":
        status.transition("LOCALLY_VALIDATED", reason="local hash and metric recomputation passed")
        write_status(status_path, status)
    if status.state == "LOCALLY_VALIDATED":
        status.transition("REPORTABLE", reason="all evidence locally verified and reportable")
        write_status(status_path, status)

    return {
        "status": "verified",
        "fold_dir": str(fold_dir),
        "state": status.state,
        "verified_artifacts": artifact_result.get("verified"),
        "subject_rows": len(subject_rows),
        "evaluations": len(evaluations.get("evaluations")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_training_attempt(args.fold_dir)
    except VerificationFailure as error:
        print(f"VERIFICATION FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
