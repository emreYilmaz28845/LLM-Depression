#!/usr/bin/env python3
"""Acceptance audit for the fold-0 DAIC independent-all modality matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


EXPECTED_MODALITIES = {"audio_text", "audio_only", "text_only"}
EXPECTED_VARIANTS = {"logreg_raw", "xgb_raw"}
EXPECTED_TEST_ROWS = {"audio_text": 540, "audio_only": 540, "text_only": 47}
EXPECTED_TRAIN_ROWS = {"audio_text": 1220, "audio_only": 1220, "text_only": 107}
REQUIRED_METRICS = {
    "accuracy",
    "positive_f1",
    "macro_f1",
    "precision",
    "recall",
    "confusion_matrix",
    "num_subjects",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _check_metrics(metrics: dict[str, Any], cell: str, failures: list[str]) -> None:
    missing = sorted(REQUIRED_METRICS - set(metrics))
    if missing:
        failures.append(f"{cell}: missing metrics {missing}")
        return
    if int(metrics["num_subjects"]) != 47:
        failures.append(f"{cell}: expected 47 subjects, got {metrics['num_subjects']}")
    if metrics.get("support_negative") != 33 or metrics.get("support_positive") != 14:
        failures.append(f"{cell}: expected held-out support 33/14")
    matrix = metrics["confusion_matrix"]
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix) or sum(map(sum, matrix)) != 47:
        failures.append(f"{cell}: invalid 47-subject confusion matrix")


def audit(
    matrix_path: Path,
    project_root: Path,
    sacct_path: Path | None = None,
) -> dict[str, Any]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    experiments = matrix.get("experiments") or []
    variants = set(matrix.get("variants") or [])
    failures: list[str] = []
    cells: list[dict[str, Any]] = []

    modalities = {str(item.get("modality")) for item in experiments}
    if modalities != EXPECTED_MODALITIES:
        failures.append(f"modalities={sorted(modalities)}, expected={sorted(EXPECTED_MODALITIES)}")
    if variants != EXPECTED_VARIANTS:
        failures.append(f"variants={sorted(variants)}, expected={sorted(EXPECTED_VARIANTS)}")
    if int(matrix.get("expected_jobs", -1)) != 3 or len(experiments) != 3:
        failures.append("matrix must declare exactly three jobs")

    reference_split: dict[str, Any] | None = None
    reference_manifest_hash: str | None = None
    for item in experiments:
        modality = str(item["modality"])
        condition = str(item.get("condition", modality))
        folds = item.get("folds")
        if folds != [0]:
            failures.append(f"{modality}: expected folds [0], got {folds}")
            continue
        run_dir = project_root / str(item["run_dir"]) / "fold_0"
        try:
            run_record = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
            split = _json(run_dir / "logs/split_used.json")
        except Exception as exc:
            failures.append(f"{modality}: missing run provenance: {exc}")
            continue
        config = run_record.get("config", {})
        data = config.get("data", {})
        if modality in {"audio_text", "audio_only"}:
            expected_flags = (True, modality == "audio_text")
            observed_flags = (data.get("use_audio"), data.get("use_text"))
            if observed_flags != expected_flags:
                failures.append(f"{modality}: modality flags are {observed_flags}, expected {expected_flags}")
            expected_construction = {
                "sample_mode": "subject_chunks",
                "train_chunk_policy": "all",
                "train_chunks_per_subject": "all",
                "eval_chunk_policy": "all",
                "eval_chunks_per_subject": "all",
                "equal_row_weight": True,
                "loss_weight_rescale": "none",
            }
            mismatches = {
                key: (data.get(key), value)
                for key, value in expected_construction.items()
                if data.get(key) != value
            }
            if mismatches:
                failures.append(f"{modality}: independent-all config mismatch {mismatches}")
        elif (
            data.get("use_audio") is not False
            or data.get("use_text") is not True
            or data.get("sample_mode") != "subject"
        ):
            failures.append("text_only: expected canonical one-example-per-subject construction")
        manifest_hash = str(run_record.get("manifest_hash", ""))
        if not manifest_hash:
            failures.append(f"{modality}: run record has no manifest hash")
        if reference_manifest_hash is None:
            reference_manifest_hash = manifest_hash
        elif manifest_hash != reference_manifest_hash:
            failures.append(f"{modality}: manifest hash differs from the matrix reference")
        if reference_split is None:
            reference_split = split
        elif split != reference_split:
            failures.append(f"{modality}: saved subject split differs from the matrix reference")
        train_ids = set(map(str, split.get("train_subject_ids", [])))
        val_ids = set(map(str, split.get("selection_subject_ids", [])))
        test_ids = set(map(str, split.get("final_eval_subject_ids", [])))
        if (len(train_ids), len(val_ids), len(test_ids)) != (107, 35, 47):
            failures.append(f"{modality}: expected subject counts 107/35/47")
        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            failures.append(f"{modality}: subject split overlap detected")

        qwen_dir = run_dir / "best_model/standalone_eval"
        try:
            qwen_metrics = _json(qwen_dir / "metrics_original_teacher_forced.json")
            qwen_samples = _jsonl(qwen_dir / "predictions_sample_level.jsonl")
            qwen_subjects = _csv(qwen_dir / "predictions_subject_level.csv")
            _check_metrics(qwen_metrics, f"{modality}/qwen", failures)
            if len(qwen_samples) != EXPECTED_TEST_ROWS[modality]:
                failures.append(f"{modality}/qwen: unexpected sample count {len(qwen_samples)}")
            if {str(row["subject_id"]) for row in qwen_subjects} != test_ids:
                failures.append(f"{modality}/qwen: held-out subject coverage mismatch")
            cells.append({"modality": modality, "head": "qwen", **qwen_metrics})
        except Exception as exc:
            failures.append(f"{modality}/qwen: {exc}")

        classifier_root = (
            project_root
            / "outputs/hidden_classifiers/daic"
            / condition
            / Path(str(item["run_dir"])).name
            / "fold_0"
        )
        existing_variants = {
            child.name
            for child in classifier_root.iterdir()
            if child.is_dir() and (child / "classifier_metadata.json").is_file()
        } if classifier_root.is_dir() else set()
        if existing_variants - variants:
            failures.append(f"{modality}: unexpected classifier heads {sorted(existing_variants - variants)}")
        for variant in sorted(variants):
            cell = f"{modality}/{variant}"
            head_dir = classifier_root / variant
            try:
                metrics = _json(head_dir / "metrics.json")
                metadata = _json(head_dir / "classifier_metadata.json")
                samples = _jsonl(head_dir / "predictions_sample_level.jsonl")
                subjects = _jsonl(head_dir / "predictions_subject_level.jsonl")
                _check_metrics(metrics, cell, failures)
                if metadata.get("classifier_variant") != variant:
                    failures.append(f"{cell}: metadata variant mismatch")
                if metadata.get("split_hashes", {}).get("manifest_sha256") != manifest_hash:
                    failures.append(f"{cell}: classifier manifest provenance mismatch")
                if set(map(str, metadata.get("training_subject_ids", []))) != train_ids | val_ids:
                    failures.append(f"{cell}: classifier training subjects must equal train+val")
                if set(map(str, metadata.get("heldout_subject_ids", []))) != test_ids:
                    failures.append(f"{cell}: classifier held-out subjects mismatch")
                weight_audit = metadata.get("fit_weight_audit", {})
                if weight_audit.get("policy") != "inverse_chunks_per_subject_rescaled_to_mean_one":
                    failures.append(f"{cell}: wrong DAIC classifier weight policy")
                if weight_audit.get("equal_subject_totals") is not True:
                    failures.append(f"{cell}: classifier fit weight is not equal per subject")
                if len(metadata.get("training_row_ids", [])) != EXPECTED_TRAIN_ROWS[modality] + (
                    410 if modality != "text_only" else 35
                ):
                    failures.append(f"{cell}: unexpected train+val classifier row count")
                if len(samples) != EXPECTED_TEST_ROWS[modality]:
                    failures.append(f"{cell}: unexpected held-out sample count {len(samples)}")
                if {str(row["subject_id"]) for row in subjects} != test_ids:
                    failures.append(f"{cell}: held-out subject coverage mismatch")
                cells.append({"modality": modality, "head": variant, **metrics})
            except Exception as exc:
                failures.append(f"{cell}: {exc}")

    slurm_jobs: list[dict[str, str]] = []
    if sacct_path is not None:
        for line in sacct_path.read_text(encoding="utf-8").splitlines():
            fields = line.split("|")
            if len(fields) < 4 or "." in fields[0]:
                continue
            job = {
                "job_id": fields[0],
                "state": fields[1],
                "exit_code": fields[2],
                "elapsed": fields[3],
            }
            slurm_jobs.append(job)
            if job["state"] != "COMPLETED" or job["exit_code"] != "0:0":
                failures.append(f"Slurm job did not complete successfully: {job}")
        if not slurm_jobs:
            failures.append("Slurm accounting file contains no top-level jobs")

    return {
        "schema_version": "daic_independent_modalities_audit.v1",
        "matrix": str(matrix_path),
        "passed": not failures and len(cells) == 9,
        "expected_cells": 9,
        "audited_cells": len(cells),
        "manifest_hash": reference_manifest_hash,
        "failures": failures,
        "cells": cells,
        "slurm_jobs": slurm_jobs,
        "slurm_accounting_path": str(sacct_path) if sacct_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--sacct", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit(
        args.matrix.resolve(),
        args.project_root.resolve(),
        args.sacct.resolve() if args.sacct else None,
    )
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
