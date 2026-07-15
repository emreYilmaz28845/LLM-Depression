#!/usr/bin/env python3
"""Aggregate exactly eight deterministic E0 views without rerunning a model.

Each view root must contain ``<condition>/predictions_subject_level.jsonl`` for
all E0 conditions.  The script validates the paired subject/label universe and
canonical score schemas, reports each view separately, averages margins across
views before thresholding, and runs paired subject bootstraps on the averaged
predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import sklearn

try:
    from scripts import compare_e0_conditions as comparison
except ModuleNotFoundError:  # Support direct ``python scripts/...`` execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import compare_e0_conditions as comparison


EXPECTED_VIEW_COUNT = 8
BOOTSTRAP_REPETITIONS = 10_000
SCORE_MODES = (
    comparison.SCORE_MODE_FIRST_TOKEN,
    comparison.SCORE_MODE_CANDIDATE,
)
PAIR_SPECS = (
    ("real", "silence"),
    ("real", "audio_shuffle"),
    ("real", "audio_shuffle_same_class"),
    ("real", "transcript_shuffle"),
    ("audio_only_real", "audio_only_silence"),
    ("audio_only_real", "audio_only_shuffle"),
)
CONDITIONS = tuple(
    dict.fromkeys(condition for pair in PAIR_SPECS for condition in pair)
)
CANONICAL_SCORE_FIELDS = {
    comparison.SCORE_MODE_FIRST_TOKEN: (
        "first_token_margin",
        "first_token_prediction",
    ),
    comparison.SCORE_MODE_CANDIDATE: (
        "candidate_likelihood_margin",
        "candidate_likelihood_prediction",
    ),
}
COMPATIBILITY_SCORE_FIELDS = {"dep_score", "non_score", "likelihood_prediction"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _is_score_field(field: str) -> bool:
    return (
        "first_token" in field
        or field.startswith("candidate_")
        or field in COMPATIBILITY_SCORE_FIELDS
    )


def _score_schema(rows: Mapping[str, Mapping[str, Any]], path: Path) -> dict[str, str]:
    signatures: set[tuple[tuple[str, str], ...]] = set()
    for subject_id, row in rows.items():
        missing = [
            field
            for fields in CANONICAL_SCORE_FIELDS.values()
            for field in fields
            if field not in row
        ]
        if missing:
            raise ValueError(
                f"Canonical score schema is incomplete for subject_id={subject_id!r} "
                f"in {path}; missing={missing}"
            )
        signatures.add(
            tuple(
                sorted(
                    (field, _json_type(value))
                    for field, value in row.items()
                    if _is_score_field(field)
                )
            )
        )
    if len(signatures) != 1:
        raise ValueError(f"Score schema differs between rows in {path}")
    return dict(next(iter(signatures)))


def _read_stable_rows(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing E0 predictions file: {path}")
    before = _artifact(path)
    rows = comparison._read_rows(path)
    after_sha256 = _sha256_file(path)
    if after_sha256 != before["sha256"]:
        raise RuntimeError(f"Input changed while it was being read: {path}")
    return rows, before


def _parse_view_root(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        view_key, raw_path = spec.split("=", 1)
        view_key = view_key.strip()
        if not view_key:
            raise ValueError(f"Empty view key in --view-root {spec!r}")
        path = Path(raw_path).expanduser()
    else:
        path = Path(spec).expanduser()
        view_key = path.name
    if not view_key:
        raise ValueError(f"Could not infer a view key from --view-root {spec!r}")
    return view_key, path.resolve()


def load_view_mapping(
    *, view_root_specs: list[str] | None = None, view_map_path: Path | None = None
) -> dict[str, Path]:
    if bool(view_root_specs) == bool(view_map_path):
        raise ValueError("Provide either repeated --view-root values or one --view-map")
    pairs: list[tuple[str, Path]] = []
    if view_map_path is not None:
        map_path = view_map_path.expanduser().resolve()
        payload = json.loads(map_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload:
            raise ValueError("--view-map must be a non-empty JSON object of view_key to root")
        for key, value in payload.items():
            if not isinstance(key, str) or not key or not isinstance(value, str):
                raise ValueError("--view-map keys and values must be non-empty strings")
            root = Path(value).expanduser()
            if not root.is_absolute():
                root = map_path.parent / root
            pairs.append((key, root.resolve()))
    else:
        pairs = [_parse_view_root(spec) for spec in view_root_specs or []]

    mapping: dict[str, Path] = {}
    for key, root in pairs:
        if key in mapping:
            raise ValueError(f"Duplicate view key: {key!r}")
        mapping[key] = root
    if len(mapping) != EXPECTED_VIEW_COUNT:
        raise ValueError(
            f"Exactly {EXPECTED_VIEW_COUNT} views are required; received {len(mapping)}"
        )
    roots = [str(root) for root in mapping.values()]
    if len(set(roots)) != len(roots):
        raise ValueError("Each view key must resolve to a distinct root directory")
    return dict(sorted(mapping.items()))


def _validate_and_load(
    view_roots: Mapping[str, Path],
) -> tuple[
    dict[str, dict[str, dict[str, dict[str, Any]]]],
    list[str],
    np.ndarray,
    dict[str, Any],
    dict[str, str],
    dict[str, dict[str, str | int | None]],
]:
    if len(view_roots) != EXPECTED_VIEW_COUNT:
        raise ValueError(
            f"Exactly {EXPECTED_VIEW_COUNT} views are required; received {len(view_roots)}"
        )
    resolved_roots = [
        str(Path(root).expanduser().resolve()) for root in view_roots.values()
    ]
    if len(set(resolved_roots)) != EXPECTED_VIEW_COUNT:
        raise ValueError("Exactly eight distinct view root directories are required")
    loaded: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    inputs: dict[str, Any] = {}
    expected_subject_ids: list[str] | None = None
    expected_labels: np.ndarray | None = None
    expected_schema: dict[str, str] | None = None
    reported_view_ids: dict[str, str] = {}
    reported_view_indices: dict[str, int] = {}
    schema_versions: set[int] = set()

    for view_key, raw_root in sorted(view_roots.items()):
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"View root is not a directory: {root}")
        loaded[view_key] = {}
        input_conditions: dict[str, Any] = {}
        root_row_view_ids: set[str] = set()
        root_row_view_indices: set[int] = set()
        for condition in CONDITIONS:
            path = root / condition / "predictions_subject_level.jsonl"
            rows, artifact = _read_stable_rows(path)
            subject_ids = sorted(rows)
            schema = _score_schema(rows, path)
            for mode in SCORE_MODES:
                comparison._condition_arrays(rows, subject_ids, score_mode=mode)
            labels = np.asarray([int(rows[s]["label"]) for s in subject_ids], dtype=np.int64)

            if expected_subject_ids is None:
                expected_subject_ids = subject_ids
                expected_labels = labels
                expected_schema = schema
            elif subject_ids != expected_subject_ids:
                raise ValueError(
                    f"Subject IDs differ in view={view_key!r}, condition={condition!r}"
                )
            elif not np.array_equal(labels, expected_labels):
                mismatches = [
                    subject_id
                    for subject_id, actual, expected in zip(
                        subject_ids, labels, expected_labels, strict=True
                    )
                    if actual != expected
                ]
                raise ValueError(
                    f"Labels differ in view={view_key!r}, condition={condition!r}; "
                    f"subjects={mismatches[:10]}"
                )
            if schema != expected_schema:
                raise ValueError(
                    f"Score schema differs in view={view_key!r}, condition={condition!r}"
                )

            condition_values = {str(row.get("condition", condition)) for row in rows.values()}
            if condition_values != {condition}:
                raise ValueError(
                    f"Row condition metadata does not match directory {condition!r} in {path}"
                )
            if any("view_id" in row for row in rows.values()):
                if not all("view_id" in row for row in rows.values()):
                    raise ValueError(f"Partial view_id metadata in {path}")
                root_row_view_ids.update(str(row["view_id"]) for row in rows.values())
            if any("view_index" in row for row in rows.values()):
                if not all("view_index" in row for row in rows.values()):
                    raise ValueError(f"Partial view_index metadata in {path}")
                root_row_view_indices.update(int(row["view_index"]) for row in rows.values())
            schema_versions.update(
                int(row["schema_version"])
                for row in rows.values()
                if "schema_version" in row
            )
            loaded[view_key][condition] = rows
            input_conditions[condition] = artifact

        if len(root_row_view_ids) > 1:
            raise ValueError(
                f"Multiple row-level view_id values in view root {root}: "
                f"{sorted(root_row_view_ids)}"
            )
        if root_row_view_ids:
            reported_view_ids[view_key] = next(iter(root_row_view_ids))
        if len(root_row_view_indices) > 1:
            raise ValueError(
                f"Multiple row-level view_index values in view root {root}: "
                f"{sorted(root_row_view_indices)}"
            )
        if root_row_view_indices:
            reported_view_indices[view_key] = next(iter(root_row_view_indices))

        optional_root_artifacts: dict[str, Any] = {}
        for filename in ("run_provenance.json", "run_config.json"):
            candidate = root / filename
            if candidate.is_file():
                optional_root_artifacts[filename] = _artifact(candidate)
        inputs[view_key] = {
            "root": str(root),
            "conditions": input_conditions,
            "root_artifacts": optional_root_artifacts,
        }

    if len(reported_view_ids) not in (0, EXPECTED_VIEW_COUNT):
        raise ValueError("Row-level view_id metadata must be present in all views or none")
    if len(reported_view_indices) not in (0, EXPECTED_VIEW_COUNT):
        raise ValueError("Row-level view_index metadata must be present in all views or none")
    if reported_view_ids and reported_view_indices:
        identities = {
            (reported_view_ids[key], reported_view_indices[key])
            for key in sorted(view_roots)
        }
        if len(identities) != EXPECTED_VIEW_COUNT:
            raise ValueError(
                "The eight roots do not contain eight distinct row-level "
                "(view_id, view_index) identities"
            )
    elif reported_view_ids and len(set(reported_view_ids.values())) != EXPECTED_VIEW_COUNT:
        raise ValueError(
            "The eight roots do not contain eight distinct row-level view_id values"
        )
    elif reported_view_indices and len(set(reported_view_indices.values())) != EXPECTED_VIEW_COUNT:
        raise ValueError(
            "The eight roots do not contain eight distinct row-level view_index values"
        )
    if len(schema_versions) > 1:
        raise ValueError(f"Prediction schema_version values differ: {sorted(schema_versions)}")
    if expected_subject_ids is None or expected_labels is None or expected_schema is None:
        raise AssertionError("No E0 inputs were loaded")
    view_metadata = {
        view_key: {
            "reported_view_id": reported_view_ids.get(view_key),
            "reported_view_index": reported_view_indices.get(view_key),
        }
        for view_key in sorted(view_roots)
    }
    return loaded, expected_subject_ids, expected_labels, inputs, expected_schema, view_metadata


def _paired_report(
    reference_rows: dict[str, dict[str, Any]],
    control_rows: dict[str, dict[str, Any]],
    *,
    reference_condition: str,
    control_condition: str,
    score_mode: str,
    bootstrap_reps: int,
    seed: int,
    reference_path: Path,
    control_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    subject_ids = sorted(reference_rows)
    if set(subject_ids) != set(control_rows):
        raise ValueError("Mean-condition files do not contain identical subject IDs")
    labels, reference_scores, reference_predictions = comparison._condition_arrays(
        reference_rows, subject_ids, score_mode=score_mode
    )
    control_labels, control_scores, control_predictions = comparison._condition_arrays(
        control_rows, subject_ids, score_mode=score_mode
    )
    if not np.array_equal(labels, control_labels):
        raise ValueError("Mean-condition labels differ")

    reference_metrics = comparison._metrics(
        labels, reference_scores, reference_predictions
    )
    control_metrics = comparison._metrics(labels, control_scores, control_predictions)
    label_sign = labels * 2 - 1
    correct_class_margin_delta = label_sign * (reference_scores - control_scores)
    metric_replicates, margin_replicates = comparison._paired_bootstrap_fast(
        labels,
        reference_scores,
        control_scores,
        reference_predictions,
        control_predictions,
        correct_class_margin_delta,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )
    metric_intervals = {}
    for name, values in metric_replicates.items():
        interval = comparison._percentile_interval(values)
        metric_intervals[name] = {
            "point_estimate": float(reference_metrics[name] - control_metrics[name]),
            "ci_95_low": interval["ci_95_low"],
            "ci_95_high": interval["ci_95_high"],
            "valid_replicates": interval["valid_replicates"],
        }
    margin_interval = comparison._percentile_interval(margin_replicates)
    reference_correct = reference_predictions == labels
    control_correct = control_predictions == labels
    report = {
        "schema_version": 1,
        "comparison": f"{reference_condition}_minus_{control_condition}",
        "score_mode": score_mode,
        "reference_condition": reference_condition,
        "control_condition": control_condition,
        "reference_predictions_path": str(reference_path.resolve()),
        "control_predictions_path": str(control_path.resolve()),
        "n_subjects": len(subject_ids),
        "support_negative": int(np.sum(labels == 0)),
        "support_positive": int(np.sum(labels == 1)),
        "bootstrap": {
            "repetitions": bootstrap_reps,
            "seed": seed,
            "method": "paired subject resampling with replacement",
            "aggregation_before_bootstrap": "arithmetic mean of each subject's score across 8 views",
        },
        "conditions": {
            reference_condition: reference_metrics,
            control_condition: control_metrics,
        },
        "paired": {
            "prediction_disagreements": int(
                np.sum(reference_predictions != control_predictions)
            ),
            "reference_correct_control_wrong": int(
                np.sum(reference_correct & ~control_correct)
            ),
            "reference_wrong_control_correct": int(
                np.sum(~reference_correct & control_correct)
            ),
            "mean_absolute_raw_margin_change": float(
                np.mean(np.abs(reference_scores - control_scores))
            ),
            "correct_class_margin_delta": {
                "point_estimate": float(np.mean(correct_class_margin_delta)),
                "ci_95_low": margin_interval["ci_95_low"],
                "ci_95_high": margin_interval["ci_95_high"],
                "valid_replicates": margin_interval["valid_replicates"],
            },
            "metric_differences": metric_intervals,
        },
        "interpretation_guardrail": (
            "Across-view averaging reduces view-selection variance, but sensitivity to a "
            "control is not by itself evidence of clinically valid acoustic reasoning."
        ),
    }
    details = []
    for index, subject_id in enumerate(subject_ids):
        details.append(
            {
                "subject_id": subject_id,
                "label": int(labels[index]),
                "reference_score": float(reference_scores[index]),
                "control_score": float(control_scores[index]),
                "reference_prediction": int(reference_predictions[index]),
                "control_prediction": int(control_predictions[index]),
                "reference_correct": bool(reference_correct[index]),
                "control_correct": bool(control_correct[index]),
                "correct_class_margin_delta": float(correct_class_margin_delta[index]),
            }
        )
    return report, details


def _write_details_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_views(
    view_roots: Mapping[str, Path],
    output_dir: Path,
    *,
    seed: int = 1337,
    bootstrap_reps: int = BOOTSTRAP_REPETITIONS,
    invocation: list[str] | None = None,
) -> dict[str, Any]:
    if bootstrap_reps != BOOTSTRAP_REPETITIONS:
        raise ValueError(
            f"E0 across-view reports require exactly {BOOTSTRAP_REPETITIONS} bootstrap repetitions"
        )
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    (
        loaded,
        subject_ids,
        labels,
        inputs,
        score_schema,
        view_metadata,
    ) = _validate_and_load(view_roots)
    view_keys = sorted(loaded)

    per_view_metrics: dict[str, Any] = {
        "schema_version": 1,
        "n_views": len(view_keys),
        "n_subjects": len(subject_ids),
        "score_modes": list(SCORE_MODES),
        "views": {},
    }
    for view_key in view_keys:
        condition_metrics = {}
        for condition in CONDITIONS:
            condition_metrics[condition] = {}
            for mode in SCORE_MODES:
                mode_labels, scores, predictions = comparison._condition_arrays(
                    loaded[view_key][condition], subject_ids, score_mode=mode
                )
                if not np.array_equal(mode_labels, labels):
                    raise AssertionError("Labels changed after validation")
                condition_metrics[condition][mode] = comparison._metrics(
                    labels, scores, predictions
                )
        per_view_metrics["views"][view_key] = {
            **view_metadata[view_key],
            "conditions": condition_metrics,
        }

    mean_rows: dict[str, dict[str, dict[str, Any]]] = {}
    mean_metrics: dict[str, Any] = {
        "schema_version": 1,
        "aggregation": "arithmetic mean of subject margins across exactly 8 views; predict 1 iff mean > 0",
        "n_views": len(view_keys),
        "n_subjects": len(subject_ids),
        "conditions": {},
    }
    for condition in CONDITIONS:
        rows_by_subject: dict[str, dict[str, Any]] = {}
        for subject_index, subject_id in enumerate(subject_ids):
            first_scores = {
                view_key: float(
                    loaded[view_key][condition][subject_id]["first_token_margin"]
                )
                for view_key in view_keys
            }
            candidate_scores = {
                view_key: float(
                    loaded[view_key][condition][subject_id][
                        "candidate_likelihood_margin"
                    ]
                )
                for view_key in view_keys
            }
            first_mean = float(np.mean(list(first_scores.values()), dtype=np.float64))
            candidate_mean = float(
                np.mean(list(candidate_scores.values()), dtype=np.float64)
            )
            rows_by_subject[subject_id] = {
                "schema_version": 1,
                "aggregation": "mean_score_across_8_views",
                "condition": condition,
                "subject_id": subject_id,
                "label": int(labels[subject_index]),
                "view_keys": view_keys,
                "first_token_view_margins": first_scores,
                "first_token_margin": first_mean,
                "first_token_prediction": int(first_mean > 0.0),
                "candidate_likelihood_view_margins": candidate_scores,
                "candidate_likelihood_margin": candidate_mean,
                "candidate_likelihood_prediction": int(candidate_mean > 0.0),
            }
        mean_rows[condition] = rows_by_subject
        mean_metrics["conditions"][condition] = {}
        for mode in SCORE_MODES:
            mode_labels, scores, predictions = comparison._condition_arrays(
                rows_by_subject, subject_ids, score_mode=mode
            )
            mean_metrics["conditions"][condition][mode] = comparison._metrics(
                mode_labels, scores, predictions
            )

    final_mean_paths = {
        condition: output_dir
        / "mean_predictions"
        / condition
        / "predictions_subject_level.jsonl"
        for condition in CONDITIONS
    }
    paired_reports: dict[tuple[str, str, str], tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for reference, control in PAIR_SPECS:
        for mode in SCORE_MODES:
            paired_reports[(reference, control, mode)] = _paired_report(
                mean_rows[reference],
                mean_rows[control],
                reference_condition=reference,
                control_condition=control,
                score_mode=mode,
                bootstrap_reps=bootstrap_reps,
                seed=seed,
                reference_path=final_mean_paths[reference],
                control_path=final_mean_paths[control],
            )

    output_dir.mkdir(parents=True)
    per_view_path = output_dir / "per_view_metrics.json"
    mean_metrics_path = output_dir / "mean_condition_metrics.json"
    _write_json_new(per_view_path, per_view_metrics)
    _write_json_new(mean_metrics_path, mean_metrics)
    for condition in CONDITIONS:
        _write_jsonl_new(
            final_mean_paths[condition],
            [mean_rows[condition][subject_id] for subject_id in subject_ids],
        )

    comparison_paths: list[Path] = []
    for (reference, control, mode), (report, details) in paired_reports.items():
        stem = f"paired_{reference}_vs_{control}_{mode}"
        json_path = output_dir / "comparisons" / f"{stem}.json"
        csv_path = output_dir / "comparisons" / f"{stem}.csv"
        _write_json_new(json_path, report)
        _write_details_csv(csv_path, details)
        comparison_paths.extend((json_path, csv_path))

    output_artifacts = {
        "per_view_metrics": _artifact(per_view_path),
        "mean_condition_metrics": _artifact(mean_metrics_path),
        "mean_predictions": {
            condition: _artifact(path) for condition, path in final_mean_paths.items()
        },
        "paired_comparisons": [_artifact(path) for path in sorted(comparison_paths)],
    }
    subject_manifest = [
        {"subject_id": subject_id, "label": int(label)}
        for subject_id, label in zip(subject_ids, labels, strict=True)
    ]
    provenance = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": _artifact(Path(__file__).resolve()),
        "comparison_implementation": _artifact(Path(comparison.__file__).resolve()),
        "invocation": invocation if invocation is not None else list(sys.argv),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "protocol": {
            "required_views": EXPECTED_VIEW_COUNT,
            "view_keys": view_keys,
            "conditions": list(CONDITIONS),
            "pairs": [list(pair) for pair in PAIR_SPECS],
            "score_modes": list(SCORE_MODES),
            "bootstrap_repetitions": bootstrap_reps,
            "bootstrap_seed": seed,
            "mean_prediction_tie_break": "predict 0 when mean score equals zero",
        },
        "score_schema": score_schema,
        "subject_manifest_sha256": _sha256_json(subject_manifest),
        "input_manifest_sha256": _sha256_json(inputs),
        "inputs": inputs,
        "outputs": output_artifacts,
    }
    provenance_path = output_dir / "provenance.json"
    _write_json_new(provenance_path, provenance)
    return {
        "output_dir": str(output_dir),
        "n_views": len(view_keys),
        "n_subjects": len(subject_ids),
        "n_conditions": len(CONDITIONS),
        "n_paired_reports": len(paired_reports),
        "bootstrap_repetitions": bootstrap_reps,
        "provenance": str(provenance_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--view-root",
        action="append",
        metavar="[VIEW_KEY=]PATH",
        help="Repeat exactly eight times; a bare path uses its directory name as the key.",
    )
    source.add_argument(
        "--view-map",
        type=Path,
        help="JSON object mapping exactly eight view keys to root directories.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mapping = load_view_mapping(
        view_root_specs=args.view_root,
        view_map_path=args.view_map,
    )
    result = aggregate_views(
        mapping,
        args.output_dir,
        seed=int(args.seed),
        invocation=list(sys.argv if argv is None else [__file__, *argv]),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
