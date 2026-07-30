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

from baselines.qwen_hidden_classifier import _load_partition, _variant_pipeline
from src.aggregate import aggregate_mean_probability_predictions
from src.features.hidden_classifier_policy import (
    cache_identity,
    canonical_sha256,
    response_normalized_sample_weights,
)
from src.utils import read_json, save_json, write_jsonl


SCHEMA = "daic_chunking_classical_head.v1"


def _complete(path: Path) -> bool:
    return all(
        (path / name).is_file()
        for name in (
            "result_config.json",
            "predictions_sample_level.jsonl",
            "predictions_subject_level.jsonl",
            "metrics.json",
            "classifier_metadata.json",
        )
    )


def run(
    *,
    fit_cache: Path,
    eval_cache: Path,
    output_dir: Path,
    variant: str,
    seed: int,
    fitted_model_dir: Path | None = None,
) -> dict[str, Any]:
    if variant not in {"logreg_raw", "xgb_raw"}:
        raise ValueError("DAIC K experiment permits exactly logreg_raw and xgb_raw.")
    train_x, train_rows = _load_partition(fit_cache, "outer_train")
    eval_x, eval_rows = _load_partition(eval_cache, "final_eval")
    train_subjects = {str(row["subject_id"]) for row in train_rows}
    eval_subjects = {str(row["subject_id"]) for row in eval_rows}
    if train_subjects & eval_subjects:
        raise ValueError("Training/evaluation subject leakage.")
    fit_meta = read_json(fit_cache / "extraction_metadata.json")
    eval_meta = read_json(eval_cache / "extraction_metadata.json")
    identity = {
        "schema_version": SCHEMA,
        "variant": variant,
        "seed": int(seed),
        "fit_cache_identity": cache_identity(fit_cache),
        "eval_cache_identity": cache_identity(eval_cache),
        "aggregation_policy": "mean_depressed_probability_threshold_0.5",
        "fitted_model_source": str(fitted_model_dir) if fitted_model_dir else None,
    }
    identity["config_sha256"] = canonical_sha256(identity)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not _complete(output_dir) or read_json(output_dir / "result_config.json") != identity:
            raise ValueError(f"Partial, colliding, or incompatible output: {output_dir}")
        return read_json(output_dir / "metrics.json")
    output_dir.mkdir(parents=True, exist_ok=False)
    save_json(identity, output_dir / "result_config.json")

    import joblib

    fit_weights, weight_audit = response_normalized_sample_weights(train_rows, fit_meta)
    model_path = output_dir / "pipeline.joblib"
    if fitted_model_dir:
        source_config = read_json(fitted_model_dir / "result_config.json")
        if source_config["variant"] != variant:
            raise ValueError("Fitted-head variant mismatch.")
        if source_config["fit_cache_identity"] != identity["fit_cache_identity"]:
            raise ValueError("Fitted-head training-cache identity mismatch.")
        fitted = joblib.load(fitted_model_dir / "pipeline.joblib")
        model_path.write_bytes((fitted_model_dir / "pipeline.joblib").read_bytes())
        fit_action = "reused"
    else:
        fitted, _ = _variant_pipeline(variant, seed, unweighted=False)
        y = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
        fitted.fit(train_x, y, classifier__sample_weight=fit_weights)
        joblib.dump(fitted, model_path)
        fit_action = "fitted"

    probabilities = np.asarray(fitted.predict_proba(eval_x)[:, 1], dtype=np.float64)
    sample_rows = [
        {
            **row,
            "probability": float(probability),
            "predicted_class": int(probability >= 0.5),
            "classifier_variant": variant,
        }
        for row, probability in zip(eval_rows, probabilities.tolist())
    ]
    subject_rows, metrics = aggregate_mean_probability_predictions(sample_rows)
    metrics["aggregation_method"] = "mean_depressed_probability_threshold_0.5"
    metrics["predicted_positive_rate"] = (
        sum(int(row["prediction"]) for row in subject_rows) / len(subject_rows)
        if subject_rows
        else 0.0
    )
    write_jsonl(sample_rows, output_dir / "predictions_sample_level.jsonl")
    write_jsonl(subject_rows, output_dir / "predictions_subject_level.jsonl")
    save_json(metrics, output_dir / "metrics.json")
    save_json(
        {
            "schema_version": SCHEMA,
            "fit_action": fit_action,
            "variant": variant,
            "seed": seed,
            "fit_weight_audit": weight_audit,
            "fit_cache_metadata_sha256": canonical_sha256(fit_meta),
            "eval_cache_metadata_sha256": canonical_sha256(eval_meta),
            "train_subject_ids": sorted(train_subjects),
            "eval_subject_ids": sorted(eval_subjects),
            "threshold": 0.5,
        },
        output_dir / "classifier_metadata.json",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-cache", required=True, type=Path)
    parser.add_argument("--eval-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=("logreg_raw", "xgb_raw"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--fitted-model-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
