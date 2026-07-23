#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


REQUIRED_ARTIFACTS = (
    "study.sqlite3",
    "study_config.json",
    "trials.csv",
    "best_params.json",
    "inner_fold_metrics.json",
    "inner_oof_metrics.json",
    "inner_subject_assignments.json",
    "pipeline.joblib",
    "predictions_sample_level.jsonl",
    "predictions_sample_level.csv",
    "predictions_subject_level.jsonl",
    "predictions_subject_level.csv",
    "metrics.json",
    "classifier_metadata.json",
)


def _result_dir(root: Path, job: dict[str, Any]) -> Path:
    return (
        root
        / str(job["dataset"])
        / str(job.get("condition", job["modality"]))
        / Path(str(job["run_dir"])).name
        / f"fold_{int(job['fold'])}"
        / str(job["experiment_id"])
    )


def audit_manifest(matrix: Path, results_root: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(matrix.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs")
    if jobs is None:
        raise ValueError("Acceptance audit requires a resolved follow-up manifest with jobs.")
    failures: list[str] = []
    audited: list[dict[str, Any]] = []
    for job in jobs:
        result_dir = _result_dir(results_root, job)
        missing = [name for name in REQUIRED_ARTIFACTS if not (result_dir / name).is_file()]
        if missing:
            failures.append(f"{result_dir}: missing {missing}")
            continue
        config_payload = json.loads((result_dir / "study_config.json").read_text(encoding="utf-8"))
        config = config_payload["canonical_config"]
        metadata = json.loads((result_dir / "classifier_metadata.json").read_text(encoding="utf-8"))
        expected = {
            "experiment_id": str(job["experiment_id"]),
            "target_trials": int(job["target_trials"]),
            "inner_fold_count": int(job["inner_folds"]),
            "seed": int(job["seed"]),
            "inner_seed": int(job["inner_seed"]),
            "search_profile": str(job["search_profile"]),
            "objective": str(job["objective"]),
        }
        mismatches = {
            key: (config.get(key), value)
            for key, value in expected.items()
            if config.get(key) != value
        }
        if mismatches:
            failures.append(f"{result_dir}: config mismatches {mismatches}")
            continue
        if metadata.get("search_config_sha256") != config_payload.get("config_sha256"):
            failures.append(f"{result_dir}: metadata/config hash mismatch")
            continue
        if metadata.get("experiment_id") != expected["experiment_id"]:
            failures.append(f"{result_dir}: metadata experiment mismatch")
            continue
        if int(metadata.get("completed_trials", -1)) != expected["target_trials"]:
            failures.append(f"{result_dir}: metadata completed-trial mismatch")
            continue
        if int(metadata.get("outer_subject_overlap_count", -1)) != 0:
            failures.append(f"{result_dir}: outer subject overlap is nonzero")
            continue
        coverage = metadata.get("inner_subject_coverage", {})
        if not bool(coverage.get("validation_covers_each_subject_once")):
            failures.append(f"{result_dir}: incomplete inner validation coverage")
            continue
        train_subjects = [str(value) for value in metadata.get("training_subject_ids", [])]
        heldout_subjects = [str(value) for value in metadata.get("heldout_subject_ids", [])]
        if set(train_subjects) & set(heldout_subjects):
            failures.append(f"{result_dir}: train/held-out subject IDs overlap")
            continue
        with (result_dir / "trials.csv").open(encoding="utf-8", newline="") as handle:
            trial_rows = list(csv.DictReader(handle))
        states = Counter(row["state"] for row in trial_rows)
        if states != Counter({"COMPLETE": expected["target_trials"]}):
            failures.append(f"{result_dir}: unexpected trial states {dict(states)}")
            continue
        best = json.loads((result_dir / "best_params.json").read_text(encoding="utf-8"))
        if int(best.get("completed_trial_count", -1)) != expected["target_trials"]:
            failures.append(f"{result_dir}: best-parameter completed-trial mismatch")
            continue
        audited.append(
            {
                "dataset": job["dataset"],
                "condition": job.get("condition", job["modality"]),
                "fold": int(job["fold"]),
                "experiment_id": job["experiment_id"],
                "completed_trials": expected["target_trials"],
                "training_subjects": len(train_subjects),
                "heldout_subjects": len(heldout_subjects),
            }
        )
    expected_jobs = int(manifest["expected_jobs"])
    if len(jobs) != expected_jobs:
        failures.append(f"Manifest expected_jobs={expected_jobs}, jobs={len(jobs)}")
    payload = {
        "schema_version": "qwen_hidden_optuna_manifest_audit.v1",
        "matrix": str(matrix),
        "expected_jobs": expected_jobs,
        "audited_jobs": len(audited),
        "passed": not failures and len(audited) == expected_jobs,
        "failures": failures,
        "studies": audited,
    }
    if not payload["passed"]:
        raise ValueError(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a completed Optuna follow-up manifest.")
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/hidden_classifiers"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit_manifest(args.matrix, args.results_root)
    text = json.dumps(payload, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Audit passed for {payload['audited_jobs']} studies; wrote {args.output}")


if __name__ == "__main__":
    main()
