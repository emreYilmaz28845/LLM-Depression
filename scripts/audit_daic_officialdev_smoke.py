#!/usr/bin/env python3
"""Acceptance audit for DAIC official-development smoke extraction/head runs.

Model-free. Verifies that the smoke cache identity carries the
subject-selection hash, that fit and eval subjects come entirely from the
official training partition (both classes present, fit/eval disjoint), that no
official-development or official-test subject entered the smoke, that both
fixed-head variants completed with the correct backend identity, and that
subject predictions cover exactly the selected eval subjects. Writes a
task-owned audit JSON; exit 0 only when everything passes.
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

from src.experiment_tracking.canonical import read_json, read_jsonl


class SmokeAuditFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeAuditFailure(message)


def audit_smoke(
    *,
    attempt_dir: Path,
    parent_fold_dir: Path,
    backbone: str,
    modality: str,
) -> dict[str, Any]:
    features_dir = attempt_dir / "hidden_features"
    _require(features_dir.is_dir(), "hidden_features dir missing")
    metadata = read_json(features_dir / "extraction_metadata.json")
    cache_config = metadata.get("cache_config") or {}
    _require(
        bool(cache_config.get("subject_selection_sha256")),
        "smoke cache identity must carry subject_selection_sha256",
    )
    selection_sha = cache_config["subject_selection_sha256"]
    _require(len(str(selection_sha)) == 64, "subject_selection_sha256 is not a sha256 hex digest")

    run_config = read_json(parent_fold_dir / "run_config.yaml")
    split_payload = read_json(parent_fold_dir / "logs" / "split_used.json")
    partition_rows = read_json(run_config["split_metadata_path"])
    official_train = {
        str(row["subject_id"])
        for row in partition_rows
        if str(row["partition"]) == "train"
    }
    official_dev = {
        str(row["subject_id"])
        for row in partition_rows
        if str(row["partition"]) == "val"
    }
    official_test = {
        str(row["subject_id"])
        for row in partition_rows
        if str(row["partition"]) == "test"
    }
    saved_train = {str(s) for s in split_payload.get("train_subject_ids") or []}
    saved_selection = {str(s) for s in split_payload.get("selection_subject_ids") or []}

    partitions = metadata.get("partitions") or {}
    outer = partitions.get("outer_train") or {}
    final = partitions.get("final_eval") or {}
    fit_rows = read_jsonl(features_dir / "outer_train_rows.jsonl")
    eval_rows = read_jsonl(features_dir / "final_eval_rows.jsonl")
    fit_subjects = {str(row["subject_id"]) for row in fit_rows}
    eval_subjects = {str(row["subject_id"]) for row in eval_rows}
    _require(len(fit_rows) == outer.get("rows"), "fit row count mismatch")
    _require(len(eval_rows) == final.get("rows"), "eval row count mismatch")
    _require(fit_subjects.issubset(saved_train), "fit subjects not in the saved training partition")
    _require(
        fit_subjects.isdisjoint(official_dev) and eval_subjects.isdisjoint(official_dev),
        "official-development subjects entered the smoke",
    )
    _require(
        fit_subjects.isdisjoint(official_test) and eval_subjects.isdisjoint(official_test),
        "official-test subjects entered the smoke",
    )
    _require(eval_subjects.issubset(saved_train | saved_selection), "eval subjects not in the official training pool")
    _require(fit_subjects.isdisjoint(eval_subjects), "smoke fit/eval subjects overlap")
    fit_labels = {int(row["label"]) for row in fit_rows}
    eval_labels = {int(row["label"]) for row in eval_rows}
    _require(fit_labels == {0, 1}, f"smoke fit must contain both classes, got {sorted(fit_labels)}")
    _require(eval_labels == {0, 1}, f"smoke eval must contain both classes, got {sorted(eval_labels)}")

    expected_backends = {
        "logreg_raw": "gemma4_hidden_logreg_raw" if backbone == "gemma4" else "qwen_hidden_logreg_raw",
        "xgb_raw": "gemma4_hidden_xgb_raw" if backbone == "gemma4" else "qwen_hidden_xgb_raw",
    }
    variant_summary: dict[str, Any] = {}
    for variant in ("logreg_raw", "xgb_raw"):
        variant_dir = attempt_dir / "hidden_classifiers" / variant
        _require((variant_dir / "metrics.json").is_file(), f"{variant} metrics missing")
        classifier_metadata = read_json(variant_dir / "classifier_metadata.json")
        _require(
            classifier_metadata.get("prediction_backend") == expected_backends[variant],
            f"{variant} backend mismatch: {classifier_metadata.get('prediction_backend')}",
        )
        subject_rows = read_jsonl(variant_dir / "predictions_subject_level.jsonl")
        _require(
            {str(row["subject_id"]) for row in subject_rows} == eval_subjects,
            f"{variant} subject predictions do not cover the selected eval subjects",
        )
        metrics = read_json(variant_dir / "metrics.json")
        variant_summary[variant] = {
            "backend": expected_backends[variant],
            "subjects": len(subject_rows),
            "macro_f1": metrics.get("macro_f1"),
        }

    result = {
        "schema_version": "daic_officialdev_smoke_audit.v1",
        "status": "passed",
        "backbone": backbone,
        "modality": modality,
        "subject_selection_sha256": selection_sha,
        "fit_subjects": sorted(fit_subjects),
        "eval_subjects": sorted(eval_subjects),
        "fit_classes": sorted(fit_labels),
        "eval_classes": sorted(eval_labels),
        "official_train_only": True,
        "official_dev_absent": True,
        "official_test_absent": True,
        "variants": variant_summary,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--parent-fold-dir", required=True, type=Path)
    parser.add_argument("--backbone", required=True, choices=("qwen", "gemma4"))
    parser.add_argument("--modality", required=True, choices=("audio_only", "audio_text", "text_only"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        record = audit_smoke(
            attempt_dir=args.attempt_dir,
            parent_fold_dir=args.parent_fold_dir,
            backbone=args.backbone,
            modality=args.modality,
        )
    except SmokeAuditFailure as error:
        print(f"SMOKE AUDIT FAILED: {error}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"smoke audit passed: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
