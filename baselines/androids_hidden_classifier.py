#!/usr/bin/env python3
"""Fit one deterministic Androids hidden-state classifier head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.features.androids_hidden_policy import (
    ANDROID_DATASET,
    ANDROID_HIDDEN_FIXED_SCHEMA,
    ANDROID_HEADS,
    ANDROID_THRESHOLD,
    aggregate_androids_hidden_predictions,
    androids_training_weights,
    cache_identity,
    canonical_sha256,
    load_androids_cache,
    read_json,
    write_csv,
    write_jsonl,
    write_sha256_manifest,
    _strict_probability_prediction,
)
from src.utils import save_json


FIXED_HEADS = ("logreg_raw", "xgb_raw")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=("audio_only", "audio_text", "text_only"))
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--heads", nargs="+", choices=FIXED_HEADS, default=list(FIXED_HEADS))
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def _pipeline(head: str, seed: int):
    # These are intentionally the repository's established logreg_raw and
    # xgb_raw defaults, without PCA, controls, or oversampling.
    from baselines.qwen_hidden_classifier import _variant_pipeline

    return _variant_pipeline(head, seed, unweighted=False)[0]


def _result_identity(
    *,
    cache_dir: Path,
    metadata: dict[str, Any],
    modality: str,
    fold: int,
    head: str,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "schema_version": ANDROID_HIDDEN_FIXED_SCHEMA,
        "dataset": ANDROID_DATASET,
        "modality": modality,
        "fold": int(fold),
        "head": head,
        "seed": int(seed),
        "threshold": ANDROID_THRESHOLD,
        "aggregation_policy": metadata["aggregation_policy"],
        "cache_identity": cache_identity(cache_dir),
        "source_commit": metadata["source_commit"],
        "checkpoint_hashes": {
            "adapter_config_sha256": metadata["adapter_config_sha256"],
            "adapter_sha256": metadata["adapter_sha256"],
        },
        "manifest_sha256": metadata["manifest_sha256"],
        "split_metadata_sha256": metadata["split_metadata_sha256"],
    }
    payload["result_config_sha256"] = canonical_sha256(payload)
    return payload


def _required_artifacts() -> tuple[str, ...]:
    return (
        "result_config.json",
        "pipeline.joblib",
        "predictions_sample_level.jsonl",
        "predictions_sample_level.csv",
        "predictions_turn_level.jsonl",
        "predictions_turn_level.csv",
        "predictions_subject_level.jsonl",
        "predictions_subject_level.csv",
        "metrics.json",
        "classifier_metadata.json",
        "fit_weight_audit.json",
    )


def _check_complete_or_collision(output_dir: Path, identity: dict[str, Any]) -> bool:
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return False
    identity_path = output_dir / "result_config.json"
    if not identity_path.is_file():
        raise ValueError(f"Refusing to overwrite incomplete Androids fixed output: {output_dir}")
    if read_json(identity_path) != identity:
        raise ValueError(f"Refusing incompatible Androids fixed output collision: {output_dir}")
    missing = [name for name in _required_artifacts() if not (output_dir / name).is_file()]
    if missing:
        raise ValueError(f"Androids fixed output is partial: {output_dir}; missing={missing}")
    return True


def _sample_rows(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    metadata: dict[str, Any],
    modality: str,
    fold: int,
    head: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row, probability in zip(rows, probabilities.tolist()):
        prediction = _strict_probability_prediction(float(probability))
        item = {
            "dataset": ANDROID_DATASET,
            "modality": modality,
            "fold": int(fold),
            "sample_id": str(row["sample_id"]),
            "subject_id": str(row["subject_id"]),
            "label": int(row["label"]),
            "probability": float(probability),
            "prediction": prediction,
            "predicted_class": prediction,
            "classifier_variant": head,
            "source_commit": metadata["source_commit"],
        }
        for key in (
            "recording_id",
            "turn_id",
            "turn_key",
            "response_id",
            "window_id",
            "window_index",
            "num_windows",
            "num_segments",
            "segment_index",
            "start_time",
            "end_time",
            "segment_duration",
            "turn_duration",
            "source_turn_count",
            "source_window_count",
            "source_turn_ids",
            "source_window_ids",
            "source_window_inventory_sha256",
        ):
            if key in row:
                item[key] = row[key]
        result.append(item)
    return result


def run_fixed_head(
    *,
    cache_dir: Path,
    output_root: Path,
    modality: str,
    fold: int,
    head: str,
    source_commit: str,
    seed: int = 1337,
) -> dict[str, Any]:
    if head not in FIXED_HEADS:
        raise ValueError(f"Androids fixed heads are exactly {FIXED_HEADS}; got {head!r}.")
    train_x, train_rows, eval_x, eval_rows, metadata = load_androids_cache(
        cache_dir,
        modality=modality,
        fold=fold,
        source_commit=source_commit,
        require_production=False,
    )
    identity = _result_identity(
        cache_dir=cache_dir,
        metadata=metadata,
        modality=modality,
        fold=fold,
        head=head,
        seed=seed,
    )
    output_dir = output_root / head
    if _check_complete_or_collision(output_dir, identity):
        return {"head": head, **read_json(output_dir / "metrics.json")}
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to use a non-empty Androids fixed output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Androids fixed-head training rows must contain both classes.")
    fit_weights, fit_weight_audit = androids_training_weights(train_rows, modality)
    model = _pipeline(head, seed)
    model.fit(train_x, labels, classifier__sample_weight=fit_weights)
    probabilities = np.asarray(model.predict_proba(eval_x)[:, 1], dtype=np.float64)
    sample_rows = _sample_rows(eval_rows, probabilities, metadata, modality, fold, head)
    turn_rows, subject_rows, metrics = aggregate_androids_hidden_predictions(sample_rows, modality)
    metrics.update(
        {
            "dataset": ANDROID_DATASET,
            "modality": modality,
            "fold": int(fold),
            "head": head,
            "schema_version": ANDROID_HIDDEN_FIXED_SCHEMA,
        }
    )
    import joblib

    joblib.dump(model, output_dir / "pipeline.joblib")
    write_jsonl(sample_rows, output_dir / "predictions_sample_level.jsonl")
    write_csv(sample_rows, output_dir / "predictions_sample_level.csv")
    write_jsonl(turn_rows, output_dir / "predictions_turn_level.jsonl")
    write_csv(turn_rows, output_dir / "predictions_turn_level.csv")
    write_jsonl(subject_rows, output_dir / "predictions_subject_level.jsonl")
    write_csv(subject_rows, output_dir / "predictions_subject_level.csv")
    save_json(metrics, output_dir / "metrics.json")
    save_json(fit_weight_audit, output_dir / "fit_weight_audit.json")
    classifier_metadata = {
        "schema_version": ANDROID_HIDDEN_FIXED_SCHEMA,
        "dataset": ANDROID_DATASET,
        "modality": modality,
        "fold": int(fold),
        "head": head,
        "seed": int(seed),
        "threshold": ANDROID_THRESHOLD,
        "aggregation_policy": metadata["aggregation_policy"],
        "source_commit": source_commit,
        "cache_dir": str(cache_dir),
        "cache_identity": identity["cache_identity"],
        "cache_identity_sha256": canonical_sha256(identity["cache_identity"]),
        "checkpoint_hashes": identity["checkpoint_hashes"],
        "manifest_sha256": identity["manifest_sha256"],
        "split_metadata_sha256": identity["split_metadata_sha256"],
        "training_subject_ids": sorted({str(row["subject_id"]) for row in train_rows}),
        "heldout_subject_ids": sorted({str(row["subject_id"]) for row in eval_rows}),
        "training_row_count": len(train_rows),
        "heldout_row_count": len(eval_rows),
        "input_dimension": int(train_x.shape[1]),
        "fit_weight_audit": fit_weight_audit,
        "no_pca": True,
        "no_oversampling": True,
        "no_controls": True,
        "result_config_sha256": identity["result_config_sha256"],
    }
    save_json(classifier_metadata, output_dir / "classifier_metadata.json")
    save_json(identity, output_dir / "result_config.json")
    write_sha256_manifest(
        output_dir,
        output_dir / "artifact_sha256.tsv",
        exclude_suffixes=(".joblib", ".sqlite3"),
    )
    return {"head": head, **metrics}


def main() -> None:
    args = _parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_fixed_head(
            cache_dir=args.cache_dir,
            output_root=args.output_root,
            modality=args.modality,
            fold=args.fold,
            head=head,
            source_commit=args.source_commit,
            seed=args.seed,
        )
        for head in args.heads
    ]
    save_json(summaries, args.output_root / "fixed_summary.json")
    write_csv(summaries, args.output_root / "fixed_summary.csv")
    write_sha256_manifest(
        args.output_root,
        args.output_root / "artifact_sha256.tsv",
        exclude_suffixes=(".joblib", ".sqlite3"),
    )
    print(json.dumps(summaries, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
