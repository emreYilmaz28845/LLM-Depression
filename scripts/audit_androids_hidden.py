#!/usr/bin/env python3
"""Acceptance audit for the Androids hidden-state classifier experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.androids_hidden_policy import (
    ANDROID_CONTROL_COUNT,
    ANDROID_DATASET,
    ANDROID_HEADS,
    ANDROID_HIDDEN_FIXED_SCHEMA,
    ANDROID_HIDDEN_OPTUNA_SCHEMA,
    ANDROID_MANIFEST_HASH,
    ANDROID_PATIENT_COUNT,
    ANDROID_SPLIT_HASH,
    ANDROID_SUBJECT_COUNT,
    aggregate_androids_hidden_predictions,
    androids_training_weights,
    compact_cache_identity,
    file_sha256,
    load_androids_cache,
    metrics_close,
    read_json,
    read_jsonl,
    validate_androids_cache_metadata,
    validate_androids_row_inventory,
    write_csv,
)
from src.utils import save_json


MODALITIES = ("audio_only", "audio_text", "text_only")
FOLDS = tuple(range(5))
FIXED_HEADS = ("logreg_raw", "xgb_raw")
OPTUNA_HEAD = "xgb_optuna_150t_d6"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--classifier-root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--job-registry", type=Path)
    parser.add_argument("--smoke-extraction-dir", type=Path)
    parser.add_argument("--smoke-fixed-root", type=Path)
    parser.add_argument("--smoke-optuna-dir", type=Path)
    return parser.parse_args()


def _assert_equal(left: Any, right: Any, message: str) -> None:
    if left != right:
        raise ValueError(f"{message}: {left!r} != {right!r}")


def _portable_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "size_bytes": int(value["size_bytes"]),
            "sha256": str(value["sha256"]),
        }
        for name, value in identity.items()
    }


def _verify_checksum_manifest(root: Path, *, exclude_suffixes: tuple[str, ...] = ()) -> None:
    path = root / ("cache_sha256.tsv" if (root / "cache_sha256.tsv").is_file() else "artifact_sha256.tsv")
    if not path.is_file():
        raise FileNotFoundError(f"Missing checksum manifest: {path}")
    expected: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("\t", 1)
        expected[relative] = digest
    for relative, digest in expected.items():
        target = root / relative
        if not target.is_file():
            if any(target.name.endswith(suffix) for suffix in exclude_suffixes):
                continue
            raise ValueError(f"Checksum manifest mismatch: {target}")
        if file_sha256(target) != digest:
            raise ValueError(f"Checksum manifest mismatch: {target}")
    for target in root.rglob("*"):
        if not target.is_file() or target == path:
            continue
        if any(target.name.endswith(suffix) for suffix in exclude_suffixes):
            continue
        if target.relative_to(root).as_posix() not in expected:
            raise ValueError(f"Unrecorded file in audited artifact root: {target}")


def _subject_labels(rows: list[dict[str, Any]]) -> Counter[int]:
    labels: dict[str, int] = {}
    for row in rows:
        subject = str(row["subject_id"])
        label = int(row["label"])
        if subject in labels and labels[subject] != label:
            raise ValueError(f"Inconsistent label for subject {subject}.")
        labels[subject] = label
    return Counter(labels.values())


def _audit_cache(
    cache_dir: Path,
    *,
    modality: str,
    fold: int,
    source_commit: str,
    production: bool,
) -> dict[str, Any]:
    train_x, train_rows, eval_x, eval_rows, metadata = load_androids_cache(
        cache_dir,
        modality=modality,
        fold=fold,
        source_commit=source_commit,
        require_production=production,
        require_vectors=not production,
    )
    if production:
        validate_androids_row_inventory(train_rows, modality)
        validate_androids_row_inventory(eval_rows, modality)
        _verify_checksum_manifest(cache_dir, exclude_suffixes=(".npz",))
    if metadata.get("gold_label_protection", {}).get("labels_passed_to_model") is not False:
        raise ValueError(f"Gold-label protection proof failed for {cache_dir}.")
    return {
        "cache_dir": str(cache_dir),
        "fold": fold,
        "modality": modality,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_subjects": len({str(row["subject_id"]) for row in train_rows}),
        "eval_subjects": len({str(row["subject_id"]) for row in eval_rows}),
        "train_label_counts": dict(_subject_labels(train_rows)),
        "eval_label_counts": dict(_subject_labels(eval_rows)),
        "vector_dimension": int(train_x.shape[1]),
        "metadata": metadata,
    }


def _audit_inner_assignments(metadata: dict[str, Any], training_subjects: set[str]) -> None:
    assignments = metadata.get("inner_subject_assignments")
    if not isinstance(assignments, dict) or len(assignments.get("folds", [])) != 3:
        raise ValueError("Androids Optuna inner subject assignments are missing or not three-fold.")
    observed: list[str] = []
    for fold in assignments["folds"]:
        train = set(map(str, fold["train_subject_ids"]))
        validation = set(map(str, fold["validation_subject_ids"]))
        if train & validation or not train.issubset(training_subjects) or not validation.issubset(training_subjects):
            raise ValueError("Androids Optuna inner/outer subject leakage detected.")
        observed.extend(sorted(validation))
    assignment_subjects = [
        str(item.get("subject_id")) if isinstance(item, dict) else str(item)
        for item in assignments.get("subjects", [])
    ]
    if Counter(observed) != Counter(assignment_subjects):
        raise ValueError("Androids Optuna inner validation subjects are not covered exactly once.")


def _audit_head(
    *,
    output_dir: Path,
    cache_dir: Path,
    modality: str,
    fold: int,
    head: str,
    source_commit: str,
    production: bool,
    target_trials: int = 150,
) -> dict[str, Any]:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Missing Androids classifier output: {output_dir}")
    metadata = read_json(cache_dir / "extraction_metadata.json")
    result_metadata = read_json(output_dir / "classifier_metadata.json")
    expected_schema = ANDROID_HIDDEN_OPTUNA_SCHEMA if head == OPTUNA_HEAD else ANDROID_HIDDEN_FIXED_SCHEMA
    if result_metadata.get("schema_version") != expected_schema:
        raise ValueError(f"Wrong Androids result schema in {output_dir}")
    if result_metadata.get("head") != head or result_metadata.get("modality") != modality:
        raise ValueError(f"Androids result identity mismatch in {output_dir}")
    if int(result_metadata.get("fold", -1)) != fold:
        raise ValueError(f"Androids result fold mismatch in {output_dir}")
    if result_metadata.get("source_commit") != source_commit:
        raise ValueError(f"Androids result source commit mismatch in {output_dir}")
    if result_metadata.get("manifest_sha256") != ANDROID_MANIFEST_HASH or result_metadata.get("split_metadata_sha256") != ANDROID_SPLIT_HASH:
        raise ValueError(f"Androids result manifest/split provenance mismatch in {output_dir}")
    saved_cache_identity = result_metadata.get("cache_identity")
    if not isinstance(saved_cache_identity, dict):
        raise ValueError(f"Androids result cache identity is missing in {output_dir}")
    compact_identity = compact_cache_identity(cache_dir)
    if _portable_identity({name: saved_cache_identity.get(name) for name in compact_identity}) != _portable_identity(compact_identity):
        raise ValueError(f"Androids result compact cache identity mismatch in {output_dir}")
    sample_rows = read_jsonl(output_dir / "predictions_sample_level.jsonl")
    saved_subject_rows = read_jsonl(output_dir / "predictions_subject_level.jsonl")
    turn_rows, subject_rows, recomputed = aggregate_androids_hidden_predictions(sample_rows, modality)
    if len(subject_rows) != len(saved_subject_rows):
        raise ValueError(f"Androids subject prediction count mismatch in {output_dir}")
    saved_by_subject = {str(row["subject_id"]): row for row in saved_subject_rows}
    for row in subject_rows:
        other = saved_by_subject.get(str(row["subject_id"]))
        if other is None or int(other["prediction"]) != int(row["prediction"]) or abs(float(other["probability"]) - float(row["probability"])) > 1e-10:
            raise ValueError(f"Androids saved subject predictions do not match sample aggregation in {output_dir}")
    saved_metrics = read_json(output_dir / "metrics.json")
    if not metrics_close(saved_metrics, recomputed):
        raise ValueError(f"Androids saved metrics do not match recomputed predictions in {output_dir}")
    if production:
        train_x, train_rows, eval_x, eval_rows, _ = load_androids_cache(
            cache_dir,
            modality=modality,
            fold=fold,
            source_commit=source_commit,
            require_production=True,
            require_vectors=False,
        )
        if {str(row["subject_id"]) for row in subject_rows} != {str(row["subject_id"]) for row in eval_rows}:
            raise ValueError(f"Androids classifier held-out subject coverage mismatch in {output_dir}")
        weights, weight_audit = androids_training_weights(train_rows, modality)
        saved_weight_audit = read_json(output_dir / "fit_weight_audit.json" if head != OPTUNA_HEAD else output_dir / "final_fit_weight_audit.json")
        if saved_weight_audit.get("policy") != weight_audit.get("policy") or saved_weight_audit.get("equal_subject_totals") is not True:
            raise ValueError(f"Androids fit-weight audit mismatch in {output_dir}")
        if head == OPTUNA_HEAD:
            trial_metadata = result_metadata
            if int(trial_metadata.get("target_trials", -1)) != target_trials or int(trial_metadata.get("completed_trials", -1)) != target_trials or int(trial_metadata.get("failed_trials", -1)) != 0:
                raise ValueError(f"Androids Optuna trial count mismatch in {output_dir}")
            trial_rows = []
            with (output_dir / "trials.csv").open("r", encoding="utf-8", newline="") as handle:
                trial_rows = list(csv.DictReader(handle))
            if len(trial_rows) != target_trials or any(row.get("state") != "COMPLETE" for row in trial_rows):
                raise ValueError(f"Androids Optuna trials contain missing/failed states in {output_dir}")
            _audit_inner_assignments(trial_metadata, {str(row["subject_id"]) for row in train_rows})
        else:
            if result_metadata.get("no_pca") is not True or result_metadata.get("no_oversampling") is not True or result_metadata.get("no_controls") is not True:
                raise ValueError(f"Androids fixed-head protocol flags are invalid in {output_dir}")
        _verify_checksum_manifest(output_dir, exclude_suffixes=(".joblib", ".sqlite3"))
    return {
        "modality": modality,
        "fold": fold,
        "head": head,
        "output_dir": str(output_dir),
        "subject_count": len(subject_rows),
        "turn_count": len(turn_rows),
        "sample_count": len(sample_rows),
        "metrics": saved_metrics,
        "cache_dir": str(cache_dir),
        "source_commit": source_commit,
    }


def _read_registry(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _production_audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.cache_root is None or args.classifier_root is None:
        raise ValueError("Production audit requires --cache-root and --classifier-root.")
    fold_reports: list[dict[str, Any]] = []
    pooled_by_key: dict[str, list[dict[str, Any]]] = {}
    caches: dict[tuple[str, int], dict[str, Any]] = {}
    for modality in MODALITIES:
        for fold in FOLDS:
            cache_dir = args.cache_root / modality / f"fold_{fold}"
            caches[(modality, fold)] = _audit_cache(
                cache_dir,
                modality=modality,
                fold=fold,
                source_commit=args.source_commit,
                production=True,
            )
            for head in (*FIXED_HEADS, OPTUNA_HEAD):
                output_dir = args.classifier_root / modality / f"fold_{fold}" / head
                report = _audit_head(
                    output_dir=output_dir,
                    cache_dir=cache_dir,
                    modality=modality,
                    fold=fold,
                    head=head,
                    source_commit=args.source_commit,
                    production=True,
                )
                fold_reports.append(report)
                key = f"{modality}/{head}"
                pooled_by_key.setdefault(key, []).extend(read_jsonl(output_dir / "predictions_subject_level.jsonl"))
    pooled_reports: dict[str, Any] = {}
    for key, rows in sorted(pooled_by_key.items()):
        subjects = {str(row["subject_id"]) for row in rows}
        if len(rows) != ANDROID_SUBJECT_COUNT or len(subjects) != ANDROID_SUBJECT_COUNT:
            raise ValueError(f"{key} has {len(rows)} pooled subject predictions; expected {ANDROID_SUBJECT_COUNT}.")
        label_counts = _subject_labels(rows)
        if label_counts != Counter({0: ANDROID_CONTROL_COUNT, 1: ANDROID_PATIENT_COUNT}):
            raise ValueError(f"{key} pooled labels are {label_counts}, expected 52 controls/64 patients.")
        modality, head = key.split("/", 1)
        if modality == "text_only":
            _, pooled_subjects, metrics = aggregate_androids_hidden_predictions(
                [
                    {
                        **row,
                        "classifier_variant": head,
                        "modality": modality,
                    }
                    for row in rows
                ],
                "text_only",
            )
        else:
            # Audio subject rows are already reduced window -> turn -> subject;
            # pooled outer-fold metrics must be recomputed on those 116 rows,
            # never by flattening windows from different folds.
            from src.features.androids_hidden_policy import _metrics_for_subject_rows

            metrics = _metrics_for_subject_rows(rows)
            pooled_subjects = rows
        pooled_reports[key] = {
            "modality": modality,
            "head": head,
            "subject_count": len(pooled_subjects),
            "label_counts": dict(label_counts),
            "metrics": metrics,
        }
    result = {
        "schema_version": "androids_hidden_acceptance.v1",
        "status": "passed",
        "mode": "production",
        "run_id": args.run_id,
        "source_commit": args.source_commit,
        "manifest_sha256": ANDROID_MANIFEST_HASH,
        "split_metadata_sha256": ANDROID_SPLIT_HASH,
        "required_modalities": list(MODALITIES),
        "required_heads": list(ANDROID_HEADS),
        "counts": {
            "folds": 5,
            "fold_head_results": len(fold_reports),
            "pooled_results": len(pooled_reports),
            "pooled_subjects_per_result": ANDROID_SUBJECT_COUNT,
            "pooled_controls": ANDROID_CONTROL_COUNT,
            "pooled_patients": ANDROID_PATIENT_COUNT,
        },
        "fold_results": fold_reports,
        "pooled_results": pooled_reports,
        "cache_inventory": [caches[key] for key in sorted(caches)],
        "job_registry": _read_registry(args.job_registry),
    }
    return result


def _smoke_audit(args: argparse.Namespace) -> dict[str, Any]:
    required = (args.smoke_extraction_dir, args.smoke_fixed_root, args.smoke_optuna_dir)
    if any(path is None for path in required):
        raise ValueError("Smoke audit requires extraction, fixed, and Optuna paths.")
    extraction_metadata = read_json(args.smoke_extraction_dir / "extraction_metadata.json")
    validate_androids_cache_metadata(
        extraction_metadata,
        modality="audio_text",
        fold=0,
        source_commit=args.source_commit,
    )
    train_x, train_rows, eval_x, eval_rows, _ = load_androids_cache(
        args.smoke_extraction_dir,
        modality="audio_text",
        fold=0,
        source_commit=args.source_commit,
        require_production=False,
    )
    if not train_rows or not eval_rows or train_x.shape[1] != eval_x.shape[1]:
        raise ValueError("Real Androids extraction smoke did not produce usable partitions.")
    fixed_reports = []
    for head in FIXED_HEADS:
        fixed_reports.append(
            _audit_head(
                output_dir=args.smoke_fixed_root / head,
                cache_dir=args.smoke_fixed_root.parent / "synthetic_cache",
                modality="audio_only",
                fold=0,
                head=head,
                source_commit=args.source_commit,
                production=False,
            )
        )
    optuna_metadata = read_json(args.smoke_optuna_dir / "classifier_metadata.json")
    if int(optuna_metadata.get("completed_trials", -1)) != 2:
        raise ValueError("Androids Optuna smoke did not resume to exactly two completed trials.")
    optuna_report = _audit_head(
        output_dir=args.smoke_optuna_dir,
        cache_dir=args.smoke_optuna_dir.parent.parent / "synthetic_cache",
        modality="audio_only",
        fold=0,
        head=OPTUNA_HEAD,
        source_commit=args.source_commit,
        production=False,
        target_trials=2,
    )
    return {
        "schema_version": "androids_hidden_acceptance.v1",
        "status": "passed",
        "mode": "smoke",
        "run_id": args.run_id,
        "source_commit": args.source_commit,
        "real_extraction": {
            "path": str(args.smoke_extraction_dir),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "dimension": int(train_x.shape[1]),
        },
        "fixed_results": fixed_reports,
        "optuna_result": optuna_report,
    }


def main() -> None:
    args = _parse_args()
    result = _production_audit(args) if args.mode == "production" else _smoke_audit(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_json(result, args.out)
    print(json.dumps({"status": result["status"], "mode": result["mode"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
