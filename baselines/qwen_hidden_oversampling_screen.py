from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from baselines.qwen_hidden_classifier import _load_partition, _metrics_with_negative_f1
from baselines.qwen_hidden_xgb_optuna import build_inner_subject_assignments
from src.aggregate import aggregate_binary_classifier_predictions
from src.sampling import (
    SAMPLING_MODE_NONE,
    SAMPLING_MODE_SUBJECT_OVERSAMPLE,
    build_no_sampling_audit,
    build_subject_oversampling,
)
from src.utils import read_json, save_json_atomic, write_jsonl


HEADS = ("logreg_raw_unweighted", "xgb_raw_unweighted")
PROFILES = (
    (SAMPLING_MODE_NONE, None, 1337),
    (SAMPLING_MODE_SUBJECT_OVERSAMPLE, 0.75, 7),
    (SAMPLING_MODE_SUBJECT_OVERSAMPLE, 0.75, 1337),
    (SAMPLING_MODE_SUBJECT_OVERSAMPLE, 0.75, 2024),
    (SAMPLING_MODE_SUBJECT_OVERSAMPLE, 1.0, 7),
    (SAMPLING_MODE_SUBJECT_OVERSAMPLE, 1.0, 1337),
    (SAMPLING_MODE_SUBJECT_OVERSAMPLE, 1.0, 2024),
)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _profile_id(mode: str, ratio: float | None, seed: int) -> str:
    if mode == SAMPLING_MODE_NONE:
        return "none"
    ratio_token = f"{int(round(float(ratio) * 100)):03d}"
    return f"ros{ratio_token}_os{seed}"


def _classifier(head: str, seed: int):
    if head == "logreg_raw_unweighted":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight=None,
                        max_iter=5000,
                        random_state=seed,
                        solver="liblinear",
                    ),
                ),
            ]
        )
    if head == "xgb_raw_unweighted":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.25,
            reg_alpha=1.0,
            reg_lambda=10.0,
            scale_pos_weight=1.0,
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported screening head {head!r}.")


def _prediction_rows(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    *,
    metadata: dict[str, Any],
    head: str,
    profile_id: str,
    sampling_mode: str,
    ratio: float | None,
    sampling_seed: int,
    inner_fold: int,
) -> list[dict[str, Any]]:
    condition = str(metadata.get("condition") or metadata["input_modality"])
    return [
        {
            "dataset": metadata["dataset"],
            "modality": metadata["input_modality"],
            "condition": condition,
            "outer_fold": int(metadata["fold"]),
            "inner_fold": int(inner_fold),
            "sample_id": str(row["sample_id"]),
            "subject_id": str(row["subject_id"]),
            "label": int(row["label"]),
            "probability": float(probability),
            "predicted_class": int(probability >= 0.5),
            "head": head,
            "profile_id": profile_id,
            "sampling_mode": sampling_mode,
            "oversampling_ratio": ratio,
            "oversampling_seed": int(sampling_seed),
        }
        for row, probability in zip(rows, probabilities.tolist())
    ]


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_screen(
    *,
    cache_dir: Path,
    output_dir: Path,
    experiment_id: str,
    inner_folds: int = 3,
    inner_seed: int = 1337,
) -> list[dict[str, Any]]:
    if output_dir.name != experiment_id:
        raise ValueError("Output directory basename must match experiment_id.")
    train_x, train_rows = _load_partition(cache_dir, "outer_train")
    metadata = read_json(cache_dir / "extraction_metadata.json")
    assignments = build_inner_subject_assignments(
        train_rows, inner_folds=inner_folds, seed=inner_seed
    )
    identity = {
        "schema_version": "hidden_oversampling_screen.v1",
        "experiment_id": experiment_id,
        "dataset": metadata["dataset"],
        "condition": str(metadata.get("condition") or metadata["input_modality"]),
        "outer_fold": int(metadata["fold"]),
        "inner_folds": inner_folds,
        "inner_seed": inner_seed,
        "heads": list(HEADS),
        "profiles": [
            {
                "sampling_mode": mode,
                "oversampling_ratio": ratio,
                "oversampling_seed": seed,
                "profile_id": _profile_id(mode, ratio, seed),
            }
            for mode, ratio, seed in PROFILES
        ],
        "outer_train_rows_sha256": _canonical_hash(train_rows),
        "outer_train_shape": list(train_x.shape),
    }
    identity["configuration_sha256"] = _canonical_hash(identity)
    identity_path = output_dir / "screen_config.json"
    if identity_path.exists():
        existing = read_json(identity_path)
        if existing != identity:
            raise ValueError("Existing screen configuration is incompatible; refusing collision.")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json_atomic(identity, identity_path)
    save_json_atomic(assignments, output_dir / "inner_subject_assignments.json")

    summaries: list[dict[str, Any]] = []
    all_sample_predictions: list[dict[str, Any]] = []
    all_subject_predictions: list[dict[str, Any]] = []
    for sampling_mode, ratio, sampling_seed in PROFILES:
        profile_id = _profile_id(sampling_mode, ratio, sampling_seed)
        for head in HEADS:
            oof_sample_rows: list[dict[str, Any]] = []
            fold_metrics: list[dict[str, Any]] = []
            for fold in assignments["folds"]:
                source_train_indices = list(fold["train_row_indices"])
                validation_indices = list(fold["validation_row_indices"])
                source_train_rows = [train_rows[index] for index in source_train_indices]
                validation_rows = [train_rows[index] for index in validation_indices]
                if sampling_mode == SAMPLING_MODE_SUBJECT_OVERSAMPLE:
                    sampling = build_subject_oversampling(
                        source_train_rows,
                        ratio=ratio,
                        seed=sampling_seed,
                        expected_minority_label=0,
                        validation_rows=validation_rows,
                    )
                else:
                    sampling = build_no_sampling_audit(
                        source_train_rows,
                        seed=sampling_seed,
                        validation_rows=validation_rows,
                    )
                fit_indices = np.asarray(
                    [source_train_indices[index] for index in sampling.indices],
                    dtype=np.int64,
                )
                validation_idx = np.asarray(validation_indices, dtype=np.int64)
                model = _classifier(head, sampling_seed)
                model.fit(
                    train_x[fit_indices],
                    np.asarray(
                        [int(train_rows[index]["label"]) for index in fit_indices],
                        dtype=np.int64,
                    ),
                )
                probabilities = np.asarray(
                    model.predict_proba(train_x[validation_idx])[:, 1],
                    dtype=np.float64,
                )
                sample_predictions = _prediction_rows(
                    validation_rows,
                    probabilities,
                    metadata=metadata,
                    head=head,
                    profile_id=profile_id,
                    sampling_mode=sampling_mode,
                    ratio=ratio,
                    sampling_seed=sampling_seed,
                    inner_fold=int(fold["fold"]),
                )
                subject_predictions, metrics = aggregate_binary_classifier_predictions(
                    sample_predictions
                )
                metrics = _metrics_with_negative_f1(metrics)
                fold_metrics.append({"inner_fold": int(fold["fold"]), **metrics})
                oof_sample_rows.extend(sample_predictions)
                audit = {
                    **sampling.audit,
                    "head": head,
                    "profile_id": profile_id,
                    "outer_fold": int(metadata["fold"]),
                    "inner_fold": int(fold["fold"]),
                    "configuration_sha256": identity["configuration_sha256"],
                    "source_outer_train_row_indices": source_train_indices,
                    "validation_outer_train_row_indices": validation_indices,
                }
                save_json_atomic(
                    audit,
                    output_dir
                    / "sampling_audits"
                    / profile_id
                    / head
                    / f"inner_fold_{int(fold['fold'])}.json",
                )

            oof_subject_rows, pooled_metrics = aggregate_binary_classifier_predictions(
                oof_sample_rows
            )
            pooled_metrics = _metrics_with_negative_f1(pooled_metrics)
            expected_subjects = {str(row["subject_id"]) for row in train_rows}
            observed_subjects = [str(row["subject_id"]) for row in oof_subject_rows]
            if set(observed_subjects) != expected_subjects or len(observed_subjects) != len(
                expected_subjects
            ):
                raise ValueError("Incomplete or duplicate subject coverage in screening OOF.")
            for row in oof_subject_rows:
                row.update(
                    {
                        "dataset": metadata["dataset"],
                        "condition": str(
                            metadata.get("condition") or metadata["input_modality"]
                        ),
                        "outer_fold": int(metadata["fold"]),
                        "head": head,
                        "profile_id": profile_id,
                        "sampling_mode": sampling_mode,
                        "oversampling_ratio": ratio,
                        "oversampling_seed": int(sampling_seed),
                    }
                )
            summary = {
                "dataset": metadata["dataset"],
                "condition": str(metadata.get("condition") or metadata["input_modality"]),
                "outer_fold": int(metadata["fold"]),
                "head": head,
                "profile_id": profile_id,
                "sampling_mode": sampling_mode,
                "oversampling_ratio": ratio,
                "oversampling_seed": int(sampling_seed),
                "configuration_sha256": identity["configuration_sha256"],
                "inner_subject_count": len(expected_subjects),
                **{f"pooled_{key}": value for key, value in pooled_metrics.items()},
            }
            summaries.append(summary)
            all_sample_predictions.extend(oof_sample_rows)
            all_subject_predictions.extend(oof_subject_rows)
            save_json_atomic(
                fold_metrics,
                output_dir / "fold_metrics" / profile_id / f"{head}.json",
            )

    write_jsonl(all_sample_predictions, output_dir / "inner_oof_sample_predictions.jsonl")
    write_jsonl(all_subject_predictions, output_dir / "inner_oof_subject_predictions.jsonl")
    save_json_atomic(summaries, output_dir / "screen_summary.json")
    _write_csv(summaries, output_dir / "screen_summary.csv")
    save_json_atomic(
        {
            "status": "complete",
            "expected_profiles": len(PROFILES),
            "expected_heads": len(HEADS),
            "expected_inner_fits": len(PROFILES) * len(HEADS) * inner_folds,
            "observed_summary_rows": len(summaries),
            "observed_sampling_audits": len(
                list((output_dir / "sampling_audits").glob("*/*/*.json"))
            ),
            "final_eval_loaded": False,
            "configuration_sha256": identity["configuration_sha256"],
        },
        output_dir / "completion.json",
    )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen subject-oversampling profiles using outer-train inner OOF only."
    )
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment-id", default="hidden_os_screen")
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--inner-seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = run_screen(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        inner_folds=args.inner_folds,
        inner_seed=args.inner_seed,
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
