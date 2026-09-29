#!/usr/bin/env python3
"""Prespecified paired analysis for the Qwen3 DAIC label-vocabulary campaign.

The campaign plan freezes, before any production submission, the contrast family
(ten arm pairs in each of three model/modality cells), the metric set, the
correction rule, the iteration counts and the analysis seed. This tool reads that
frozen family (``experiments/definitions/significance_family_qwen3_daic_label_vocab_20260928.yaml``),
resolves the local REPORTABLE evidence for every run of the matrix, and writes the
deterministic tables the report and workbook consume.

Rules it enforces instead of assuming them:

* only locally validated ``REPORTABLE`` fold evidence is analysed;
* headline metrics are recomputed locally from subject-level predictions with
  INVALID counted as wrong (the same strict rule the validation gate uses);
* metrics are computed per training seed and then averaged (mean and ddof=1
  standard deviation); seed rows are never concatenated into one population;
* the paired permutation and the stratified bootstrap use the multi-seed
  subject-clustered functions of ``src.daic_statistics``, where all seed records
  of one subject move together;
* Holm correction runs separately inside each metric's own 30-member family, and
  the McNemar view is corrected as its own 90-test family;
* a missing run, a non-REPORTABLE run, a missing seed, a duplicate
  ``(subject_id, seed)`` key or a label mismatch is a hard analysis error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.daic_statistics import (  # noqa: E402
    exact_mcnemar,
    holm_adjust,
    paired_prediction_swap_permutation_many,
    stratified_paired_bootstrap_many,
)
from src.experiment_tracking.validate import recompute_strict_headline  # noqa: E402
from scripts.build_qwen3_daic_label_configs import ARMS  # noqa: E402
from tools.qwen3_daic_label_vocab_matrix import SOURCES, run_name  # noqa: E402

DEFAULT_FAMILY = ROOT / "experiments/definitions/significance_family_qwen3_daic_label_vocab_20260928.yaml"
EXPECTED_SCHEMA = "audiollm.qwen3_daic_label_vocab.significance_family.v1"
PREDICTIONS_RELATIVE = Path("best_model/standalone_eval/predictions_subject_level.csv")
PREDICTIONS_ROLE = "standalone_eval_predictions"
RUN_CONFIG_ROLE = "run_config"
METRIC_KEYS = {"macro_f1": "binary_strict_macro_f1", "positive_f1": "binary_strict_positive_f1", "macro_recall": "binary_strict_uar"}
EXPECTED_LABEL_KEYS = ("label_vocab_version", "internal_positive_label", "internal_negative_label", "external_positive_label", "external_negative_label")
EXPECTED_EVALUATION_VIEW = "harmonized_all_windows_full_coverage"
EXPECTED_SPLIT_SEED = 1337


class AnalysisError(RuntimeError):
    """Raised when the prespecified analysis cannot be computed."""


def load_family(path: Path) -> dict[str, Any]:
    family = yaml.safe_load(path.read_text(encoding="utf-8"))
    if family.get("schema_version") != EXPECTED_SCHEMA:
        raise AnalysisError(f"{path}: unexpected schema_version {family.get('schema_version')!r}")
    return family


def contrasts(family: dict[str, Any]) -> list[dict[str, Any]]:
    """All unordered arm pairs per cell, ordered later-minus-earlier."""
    order = list(family["arm_order"])
    members: list[dict[str, Any]] = []
    for cell in family["cells"]:
        for arm_a, arm_b in combinations(order, 2):  # combinations keeps order: a earlier than b
            members.append(
                {
                    "id": f"{cell['id']}|{arm_b}-minus-{arm_a}",
                    "cell": cell["id"],
                    "modality": cell["modality"],
                    "model": cell["model"],
                    "arm_earlier": arm_a,
                    "arm_later": arm_b,
                }
            )
    expected = family.get("contrast_count")
    if expected is not None and len(members) != int(expected):
        raise AnalysisError(f"derived {len(members)} contrasts, family declares {expected}")
    return members


def mcnemar_members(family: dict[str, Any]) -> list[dict[str, Any]]:
    members = []
    for contrast in contrasts(family):
        for seed in family["seeds"]:
            members.append({**contrast, "seed": int(seed)})
    return members


def validate_family(family: dict[str, Any]) -> None:
    for key in (
        "arm_order", "cells", "seeds", "metrics", "primary_metric", "alpha", "fold",
        "campaign", "smoke_campaign", "contrast_count", "tests", "multiple_comparison",
    ):
        if key not in family:
            raise AnalysisError(f"family is missing required key {key!r}")
    expected_metrics = {"macro_f1", "positive_f1", "macro_recall"}
    if set(family["metrics"]) != expected_metrics:
        raise AnalysisError(f"family metrics must be {sorted(expected_metrics)}")
    if len(family["arm_order"]) != 5 or len(set(family["arm_order"])) != 5:
        raise AnalysisError("family arm_order must list the five distinct arms")
    iteration_counts = family["tests"]["iteration_counts"]
    if int(iteration_counts["permutation"]) < 1_000_000:
        raise AnalysisError("the prespecified permutation count is 1,000,000; a smaller value is not prespecified")
    if int(iteration_counts["bootstrap"]) < 100_000:
        raise AnalysisError("the prespecified bootstrap count is 100,000; a smaller value is not prespecified")


def evidence_fold_dir(evidence_root: Path, family: dict[str, Any], *, cell: dict[str, Any], arm: str, seed: int) -> Path:
    name = run_name(smoke=False, modality=cell["modality"], tag=arm, seed=int(seed))
    return evidence_root / family["campaign"] / cell["modality"] / family["dataset"] / name / f"fold_{int(family['fold'])}"


def _read_status(fold_dir: Path) -> dict[str, Any]:
    status_path = fold_dir / "status.json"
    if not status_path.is_file():
        raise AnalysisError(f"missing status.json in {fold_dir}")
    return json.loads(status_path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recorded_artifacts(fold_dir: Path) -> dict[str, dict[str, Any]]:
    """Role -> recorded artifact record (path and sha256) from artifacts.json."""
    artifacts_path = fold_dir / "artifacts.json"
    if not artifacts_path.is_file():
        raise AnalysisError(f"{fold_dir}: missing artifacts.json; recorded evidence is required")
    payload = json.loads(artifacts_path.read_text(encoding="utf-8"))
    records = payload.get("artifacts")
    if not isinstance(records, list) or not records:
        raise AnalysisError(f"{artifacts_path}: no recorded artifacts")
    by_role: dict[str, dict[str, Any]] = {}
    for record in records:
        role = str(record.get("role") or "")
        if role:
            by_role.setdefault(role, record)
    return by_role


def _require_recorded_file(
    *, fold_dir: Path, role: str, path: Path, recorded: dict[str, dict[str, Any]]
) -> str:
    """Reject a file whose bytes do not match the recorded evidence hash."""
    record = recorded.get(role)
    if record is None:
        raise AnalysisError(f"{fold_dir}: no recorded artifact with role {role!r}")
    expected = str(record.get("sha256") or "")
    if not expected:
        raise AnalysisError(f"{fold_dir}: recorded artifact {role!r} has no sha256")
    actual = _sha256(path)
    if actual != expected:
        raise AnalysisError(
            f"{path}: sha256 {actual} does not match the recorded evidence hash {expected}; "
            "the analysis never rewrites run evidence"
        )
    return actual


def verify_run_record(
    *,
    fold_dir: Path,
    modality: str,
    arm: str,
    seed: int,
    campaign: str,
    dataset: str = "daic",
) -> dict[str, Any]:
    """Compare one run against the campaign contract and its recorded evidence.

    Checks the expected seed, model backend, model revision and label vocabulary
    against the run's own ``run_config.yaml``, the resolved run root and modality,
    and the recorded artifact hashes in ``artifacts.json``. A missing sidecar, a
    missing recorded artifact, a hash mismatch or a contradicting record is an
    error; nothing here writes to the run directory.
    """
    recorded = _recorded_artifacts(fold_dir)
    run_config_path = fold_dir / "run_config.yaml"
    if not run_config_path.is_file():
        raise AnalysisError(f"{fold_dir}: missing run_config.yaml")
    predictions_path = fold_dir / PREDICTIONS_RELATIVE
    if not predictions_path.is_file():
        raise AnalysisError(f"{fold_dir}: missing {PREDICTIONS_RELATIVE}")

    run_config_sha = _require_recorded_file(
        fold_dir=fold_dir, role=RUN_CONFIG_ROLE, path=run_config_path, recorded=recorded
    )
    predictions_sha = _require_recorded_file(
        fold_dir=fold_dir, role=PREDICTIONS_ROLE, path=predictions_path, recorded=recorded
    )

    document = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise AnalysisError(f"{run_config_path}: not a mapping")
    config = document.get("config") if isinstance(document.get("config"), dict) else document
    evaluation = config.get("evaluation") or {}
    labels = config.get("labels") or {}
    split = config.get("split") or {}
    tracking = document.get("tracking") or {}

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise AnalysisError(f"{fold_dir}: {message}")

    expected_backend = SOURCES[modality][2]
    expected_vocab, expected_positive, expected_negative = ARMS[arm]
    require(config.get("dataset") == dataset, f"run_config dataset is {config.get('dataset')!r}, expected {dataset!r}")
    require(int(config.get("seed", -1)) == int(seed), f"run_config seed is {config.get('seed')!r}, expected {seed}")
    require(
        int(split.get("seed", -1)) == EXPECTED_SPLIT_SEED,
        f"run_config split.seed is {split.get('seed')!r}, expected {EXPECTED_SPLIT_SEED}",
    )
    require(
        config.get("model_backend") == expected_backend,
        f"run_config backend is {config.get('model_backend')!r}, expected {expected_backend!r}",
    )
    require(
        str(config.get("input_modality") or document.get("input_modality") or "") == modality,
        f"run_config input_modality is {config.get('input_modality') or document.get('input_modality')!r}, "
        f"expected {modality!r}",
    )
    require(
        evaluation.get("sample_prediction_mode") == "likelihood",
        f"run_config backend is {evaluation.get('sample_prediction_mode')!r}, expected likelihood",
    )
    require(
        evaluation.get("evaluation_view") == EXPECTED_EVALUATION_VIEW,
        f"run_config evaluation view is {evaluation.get('evaluation_view')!r}, expected {EXPECTED_EVALUATION_VIEW!r}",
    )
    require(
        evaluation.get("aggregation_level") in {"subject", "subject_level"},
        f"run_config aggregation is {evaluation.get('aggregation_level')!r}, expected subject level",
    )
    require(
        set(labels) == set(EXPECTED_LABEL_KEYS),
        f"run_config labels block keys are {sorted(labels)}, expected {sorted(EXPECTED_LABEL_KEYS)}",
    )
    require(
        labels.get("label_vocab_version") == expected_vocab,
        f"run_config label vocabulary is {labels.get('label_vocab_version')!r}, expected {expected_vocab!r}",
    )
    require(
        labels.get("internal_positive_label") == expected_positive
        and labels.get("internal_negative_label") == expected_negative,
        f"run_config internal labels are {labels.get('internal_positive_label')!r}/"
        f"{labels.get('internal_negative_label')!r}, expected {expected_positive!r}/{expected_negative!r}",
    )
    expected_run_root_suffix = f"output_model/{campaign}/{modality}/{dataset}"
    run_root = str((config.get("output_dirs") or {}).get("run_root") or "")
    require(
        run_root.endswith(expected_run_root_suffix),
        f"run_config run_root is {run_root!r}, expected it to end with {expected_run_root_suffix!r}",
    )
    if expected_backend == "qwen38":
        require(bool(config.get("model_revision")), "run_config has no model_revision for the pinned Qwen3.8 snapshot")

    attempt_ids = {
        "status": _read_status(fold_dir).get("attempt_id"),
        "artifacts": json.loads((fold_dir / "artifacts.json").read_text(encoding="utf-8")).get("attempt_id"),
        "run_config": tracking.get("attempt_id"),
    }
    present = {key: value for key, value in attempt_ids.items() if value}
    require(len(set(present.values())) <= 1, f"attempt ids disagree between sidecars: {attempt_ids}")

    metadata_path = fold_dir / "metadata.json"
    source: dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source = metadata.get("source") or {}
        metadata_attempt = metadata.get("attempt_id")
        if metadata_attempt:
            require(
                not present or metadata_attempt in set(present.values()),
                f"metadata attempt id {metadata_attempt!r} disagrees with {sorted(set(present.values()))}",
            )

    job_ids: dict[str, list[str]] = {}
    jobs_path = fold_dir / "jobs.jsonl"
    if jobs_path.is_file():
        for line in jobs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            job_id = event.get("slurm_job_id")
            if not job_id:
                continue
            key = str(event.get("job_key"))
            recorded_ids = job_ids.setdefault(key, [])
            if str(job_id) not in recorded_ids:
                recorded_ids.append(str(job_id))

    return {
        "run_name": run_name(smoke=False, modality=modality, tag=arm, seed=int(seed)),
        "attempt_id": present.get("run_config") or present.get("status"),
        "state": _read_status(fold_dir).get("state"),
        "fold_dir": str(fold_dir),
        "run_config_path": str(run_config_path),
        "run_config_sha256": run_config_sha,
        "recorded_run_config_sha256": str((recorded.get(RUN_CONFIG_ROLE) or {}).get("sha256") or ""),
        "predictions_path": str(predictions_path),
        "predictions_sha256": predictions_sha,
        "recorded_predictions_sha256": str((recorded.get(PREDICTIONS_ROLE) or {}).get("sha256") or ""),
        "manifest_hash": document.get("manifest_hash"),
        "split_metadata_hash": document.get("split_metadata_hash"),
        "manifest_path": document.get("manifest_path"),
        "split_metadata_path": document.get("split_metadata_path"),
        "model_backend": config.get("model_backend"),
        "model_path": config.get("model_name_or_path") or document.get("resolved_model_name_or_path"),
        "model_revision": config.get("model_revision"),
        "label_vocab_version": labels.get("label_vocab_version"),
        "git_commit": source.get("git_commit"),
        "git_branch": source.get("git_branch"),
        "git_dirty": source.get("git_dirty"),
        "deployment_id": source.get("deployment_id"),
        "deployed_source_sha256": source.get("deployed_source_sha256"),
        "job_ids": job_ids,
    }


def load_seed_rows(
    fold_dir: Path,
    *,
    seed: int,
    modality: str | None = None,
    arm: str | None = None,
    campaign: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load strict subject rows for one run; INVALID predictions count as wrong."""
    status = _read_status(fold_dir)
    if status.get("state") != "REPORTABLE":
        raise AnalysisError(f"{fold_dir}: lifecycle state is {status.get('state')!r}, not REPORTABLE")
    predictions = fold_dir / PREDICTIONS_RELATIVE
    if not predictions.is_file():
        raise AnalysisError(f"{fold_dir}: missing {PREDICTIONS_RELATIVE}")
    verified: dict[str, Any] = {}
    if modality is not None and arm is not None and campaign is not None:
        verified = verify_run_record(
            fold_dir=fold_dir, modality=modality, arm=arm, seed=int(seed), campaign=campaign
        )
    strict = recompute_strict_headline(predictions)
    rows: list[dict[str, Any]] = []
    with predictions.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            label = int(record["label"])
            text = (record.get("prediction_text") or "").strip().lower()
            if text == "depressed":
                prediction = 1
            elif text == "non-depressed":
                prediction = 0
            else:
                prediction = 1 - label
            rows.append(
                {
                    "subject_id": str(record["subject_id"]),
                    "label": label,
                    "prediction": int(prediction),
                    "seed": int(seed),
                }
            )
    if not rows:
        raise AnalysisError(f"{predictions}: no subject rows")
    keys = {(row["subject_id"], row["seed"]) for row in rows}
    if len(keys) != len(rows):
        raise AnalysisError(f"{predictions}: duplicate (subject_id, seed) rows")
    provenance = {
        **verified,
        "fold_dir": str(fold_dir),
        "attempt_id": status.get("attempt_id"),
        "state": status.get("state"),
        "predictions_sha256": _sha256(predictions),
        "subjects": len(rows),
        "strict_headline": strict,
    }
    return rows, provenance


def seed_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    labels = [int(row["label"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    positive_f1 = _f1(labels, predictions, positive=True)
    negative_f1 = _f1(labels, predictions, positive=False)
    positive_recall = _recall(labels, predictions, positive=True)
    negative_recall = _recall(labels, predictions, positive=False)
    return {
        "macro_f1": (positive_f1 + negative_f1) / 2.0,
        "positive_f1": positive_f1,
        "macro_recall": (positive_recall + negative_recall) / 2.0,
    }


def _f1(labels: list[int], predictions: list[int], *, positive: bool) -> float:
    target = 1 if positive else 0
    tp = sum(1 for y, p in zip(labels, predictions) if y == target and p == target)
    fp = sum(1 for y, p in zip(labels, predictions) if y != target and p == target)
    fn = sum(1 for y, p in zip(labels, predictions) if y == target and p != target)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def _recall(labels: list[int], predictions: list[int], *, positive: bool) -> float:
    target = 1 if positive else 0
    tp = sum(1 for y, p in zip(labels, predictions) if y == target and p == target)
    fn = sum(1 for y, p in zip(labels, predictions) if y == target and p != target)
    return tp / (tp + fn) if (tp + fn) else 0.0


def load_all_rows(
    evidence_root: Path, family: dict[str, Any]
) -> tuple[dict[tuple[str, str, int], list[dict[str, Any]]], dict[str, Any]]:
    rows_by_key: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    provenance: dict[str, Any] = {}
    for cell in family["cells"]:
        for arm in family["arm_order"]:
            for seed in family["seeds"]:
                fold_dir = evidence_fold_dir(evidence_root, family, cell=cell, arm=str(arm), seed=int(seed))
                key = (cell["id"], str(arm), int(seed))
                rows, run_provenance = load_seed_rows(
                    fold_dir,
                    seed=int(seed),
                    modality=cell["modality"],
                    arm=str(arm),
                    campaign=family["campaign"],
                )
                rows_by_key[key] = rows
                provenance[f"{cell['id']}|{arm}|{seed}"] = {**run_provenance, "run_name": fold_dir.parent.name}
    return rows_by_key, provenance


def verify_all_records(evidence_root: Path, family: dict[str, Any]) -> dict[str, Any]:
    """Verify every expected run against the campaign contract and recorded evidence.

    Read-only: it reads the fold sidecars and hashes the files it verifies, and it
    never writes into a run directory. The returned payload is the provenance the
    workbook consumes.
    """
    records: dict[str, Any] = {}
    for cell in family["cells"]:
        for arm in family["arm_order"]:
            for seed in family["seeds"]:
                fold_dir = evidence_fold_dir(evidence_root, family, cell=cell, arm=str(arm), seed=int(seed))
                record = verify_run_record(
                    fold_dir=fold_dir,
                    modality=cell["modality"],
                    arm=str(arm),
                    seed=int(seed),
                    campaign=family["campaign"],
                )
                status = _read_status(fold_dir)
                if status.get("state") != "REPORTABLE":
                    raise AnalysisError(
                        f"{fold_dir}: lifecycle state is {status.get('state')!r}, not REPORTABLE"
                    )
                record["subjects"] = _subject_rows(fold_dir / PREDICTIONS_RELATIVE)
                record["strict_headline"] = recompute_strict_headline(fold_dir / PREDICTIONS_RELATIVE)
                record["cell"] = cell["id"]
                record["modality"] = cell["modality"]
                record["arm"] = str(arm)
                record["seed"] = int(seed)
                records[f"{cell['id']}|{arm}|{seed}"] = record
    return records


def _subject_rows(predictions: Path) -> int:
    with predictions.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        raise AnalysisError("cannot average an empty metric list")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, variance ** 0.5


def analyse(
    *,
    family: dict[str, Any],
    rows_by_key: dict[tuple[str, str, int], list[dict[str, Any]]],
    provenance: dict[str, Any],
    permutations: int,
    bootstrap_iterations: int,
    iterations_overridden: bool,
) -> dict[str, Any]:
    metrics = list(family["metrics"])
    seeds = [int(seed) for seed in family["seeds"]]

    seed_rows: list[dict[str, Any]] = []
    summary_long: list[dict[str, Any]] = []
    for cell in family["cells"]:
        for arm in family["arm_order"]:
            per_metric: dict[str, list[float]] = {metric: [] for metric in metrics}
            for seed in seeds:
                rows = rows_by_key[(cell["id"], str(arm), seed)]
                values = seed_metrics(rows)
                for metric in metrics:
                    per_metric[metric].append(float(values[metric]))
                seed_rows.append(
                    {
                        "cell": cell["id"],
                        "model": cell["model"],
                        "modality": cell["modality"],
                        "arm": arm,
                        "seed": seed,
                        **{metric: values[metric] for metric in metrics},
                    }
                )
            for metric in metrics:
                mean, std = _mean_std(per_metric[metric])
                summary_long.append(
                    {
                        "cell": cell["id"],
                        "model": cell["model"],
                        "modality": cell["modality"],
                        "arm": arm,
                        "metric": metric,
                        "mean": mean,
                        "std_ddof1": std,
                        "n_seeds": len(per_metric[metric]),
                    }
                )

    contrast_rows: list[dict[str, Any]] = []
    p_values_by_metric: dict[str, list[float]] = {metric: [] for metric in metrics}
    raw_rows: list[dict[str, Any]] = []
    for member in contrasts(family):
        earlier = _stack(rows_by_key, member["cell"], str(member["arm_earlier"]), seeds)
        later = _stack(rows_by_key, member["cell"], str(member["arm_later"]), seeds)
        try:
            permutation = paired_prediction_swap_permutation_many(
                earlier, later, metrics=tuple(metrics), iterations=int(permutations), seed=int(family["tests"]["analysis_seed"])
            )
            bootstrap = stratified_paired_bootstrap_many(
                earlier, later, metrics=tuple(metrics), iterations=int(bootstrap_iterations), seed=int(family["tests"]["analysis_seed"])
            )
        except (ValueError, KeyError) as error:
            raise AnalysisError(
                f"{member['id']}: paired rows are not analysable ({error}); "
                "a missing seed, a missing subject or a label mismatch is an analysis error"
            ) from error
        for metric in metrics:
            raw_rows.append(
                {
                    "cell": member["cell"],
                    "model": member["model"],
                    "modality": member["modality"],
                    "arm_earlier": member["arm_earlier"],
                    "arm_later": member["arm_later"],
                    "metric": metric,
                    "delta": float(permutation[metric]["observed_delta"]),
                    "p_value": float(permutation[metric]["p_value"]),
                    "ci_low": float(bootstrap[metric]["ci_low"]),
                    "ci_high": float(bootstrap[metric]["ci_high"]),
                    "subjects": int(permutation[metric]["subjects"]),
                    "keys": int(permutation[metric]["keys"]),
                }
            )
            p_values_by_metric[metric].append(float(permutation[metric]["p_value"]))

    for metric in metrics:
        adjusted = holm_adjust(p_values_by_metric[metric])
        for row, corrected in zip([item for item in raw_rows if item["metric"] == metric], adjusted):
            contrast_rows.append({**row, "p_value_holm": float(corrected), "alpha": float(family["alpha"])})

    mcnemar_rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for member in mcnemar_members(family):
        seed = int(member["seed"])
        earlier = rows_by_key[(member["cell"], str(member["arm_earlier"]), seed)]
        later = rows_by_key[(member["cell"], str(member["arm_later"]), seed)]
        result = exact_mcnemar(earlier, later)
        p_values.append(float(result["p_value"]))
        mcnemar_rows.append(
            {
                "cell": member["cell"],
                "arm_earlier": member["arm_earlier"],
                "arm_later": member["arm_later"],
                "seed": seed,
                "b": result["baseline_only_correct"],
                "c": result["comparison_only_correct"],
                "accuracy_delta": result["accuracy_delta"],
                "p_value": float(result["p_value"]),
            }
        )
    for row, corrected in zip(mcnemar_rows, holm_adjust(p_values)):
        row["p_value_holm"] = float(corrected)

    return {
        "seed_rows": seed_rows,
        "summary_long": summary_long,
        "contrasts": contrast_rows,
        "mcnemar": mcnemar_rows,
        "provenance": provenance,
        "iterations": {
            "permutation": int(permutations),
            "bootstrap": int(bootstrap_iterations),
            "iterations_overridden": bool(iterations_overridden),
            "analysis_seed": int(family["tests"]["analysis_seed"]),
        },
    }


def _stack(
    rows_by_key: dict[tuple[str, str, int], list[dict[str, Any]]], cell: str, arm: str, seeds: list[int]
) -> list[dict[str, Any]]:
    stacked: list[dict[str, Any]] = []
    for seed in seeds:
        stacked.extend(rows_by_key[(cell, arm, seed)])
    return stacked


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AnalysisError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(family: dict[str, Any], result: dict[str, Any]) -> str:
    metrics = list(family["metrics"])
    arms = list(family["arm_order"])
    lines = ["# Qwen3 DAIC label-vocabulary: fixed-test results", ""]
    lines.append(
        "DAIC fixed test partition (fold 0), likelihood backend, subject level, "
        "`harmonized_all_windows_full_coverage` view. Cells hold "
        "`Macro-F1 / Positive-F1 / UAR`. Each row is one model/modality cell and one training seed."
    )
    lines.append("")
    lines.append("| cell | seed | " + " | ".join(arms) + " |")
    lines.append("|---|---|" + "---|" * len(arms))
    summary = {(row["cell"], row["arm"], row["metric"]): row for row in result["summary_long"]}
    seed_index = {(row["cell"], row["arm"], row["seed"]): row for row in result["seed_rows"]}
    for cell in family["cells"]:
        for seed in family["seeds"]:
            cells = []
            for arm in arms:
                row = seed_index[(cell["id"], arm, int(seed))]
                cells.append(" / ".join(f"{row[metric]:.4f}" for metric in metrics))
            lines.append(f"| {cell['id']} | {seed} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Seed mean and standard deviation (ddof=1) per cell and arm:")
    lines.append("")
    lines.append("| cell | arm | " + " | ".join(f"{metric} mean ± sd" for metric in metrics) + " |")
    lines.append("|---|---|" + "---|" * len(metrics))
    for cell in family["cells"]:
        for arm in arms:
            cells = []
            for metric in metrics:
                row = summary[(cell["id"], arm, metric)]
                cells.append(f"{row['mean']:.4f} ± {row['std_ddof1']:.4f}")
            lines.append(f"| {cell['id']} | {arm} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Prespecified paired contrasts (later arm minus earlier arm).")
    lines.append("")
    lines.append("| cell | contrast | metric | delta | unadjusted 95% CI | p (permutation) | p (Holm, family of 30) |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in result["contrasts"]:
        lines.append(
            f"| {row['cell']} | {row['arm_earlier']} -> {row['arm_later']} | {row['metric']} | "
            f"{row['delta']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {row['p_value']:.4g} | "
            f"{row['p_value_holm']:.4g} |"
        )
    lines.append("")
    lines.append(
        "Bootstrap intervals are unadjusted 95% subject-clustered percentile intervals and are not a "
        "corrected decision. The exact McNemar view (90 tests, its own Holm family) is in `mcnemar.csv`."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default=str(DEFAULT_FAMILY))
    parser.add_argument("--evidence-root", default=str(ROOT / "output_model"))
    parser.add_argument("--output-dir", default=None, help="where the tables are written (required unless --check-family)")
    parser.add_argument("--permutations", type=int, default=None, help="development override; recorded in the report")
    parser.add_argument("--bootstrap-iterations", type=int, default=None, help="development override; recorded in the report")
    parser.add_argument("--check-family", action="store_true")
    parser.add_argument(
        "--verify-records",
        action="store_true",
        help="verify every run against the campaign contract and recorded evidence, then write the provenance payload; no statistics are computed",
    )
    parser.add_argument(
        "--provenance-output",
        default=None,
        help="where --verify-records writes its payload (default: <output-dir>/run_provenance.json)",
    )
    args = parser.parse_args(argv)

    family_path = Path(args.family)
    family = load_family(family_path)
    validate_family(family)
    if args.verify_records:
        records = verify_all_records(Path(args.evidence_root), family)
        payload = {
            "schema_version": "audiollm.qwen3_daic_label_vocab.run_provenance.v1",
            "campaign": family["campaign"],
            "dataset": family["dataset"],
            "fold": family["fold"],
            "family": {
                "path": str(family_path.relative_to(ROOT)) if str(family_path).startswith(str(ROOT)) else str(family_path),
                "sha256": hashlib.sha256(family_path.read_bytes()).hexdigest(),
            },
            "runs": records,
            "run_count": len(records),
            "passed": True,
        }
        target = Path(args.provenance_output) if args.provenance_output else (
            Path(args.output_dir) / "run_provenance.json" if args.output_dir else None
        )
        if target is None:
            raise AnalysisError("--output-dir or --provenance-output is required with --verify-records")
        if target.exists():
            raise AnalysisError(f"refusing to overwrite the existing provenance file {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"verified {len(records)} runs; provenance written to {target}")
        return 0

    if args.check_family:
        print(
            f"family ok: {len(contrasts(family))} contrasts, {len(mcnemar_members(family))} McNemar tests, "
            f"arms {family['arm_order']}, seeds {family['seeds']}"
        )
        return 0
    if not args.output_dir:
        raise AnalysisError("--output-dir is required unless --check-family is used")

    permutations = int(args.permutations or family["tests"]["iteration_counts"]["permutation"])
    bootstrap_iterations = int(
        args.bootstrap_iterations or family["tests"]["iteration_counts"]["bootstrap"]
    )
    iterations_overridden = (
        permutations != int(family["tests"]["iteration_counts"]["permutation"])
        or bootstrap_iterations != int(family["tests"]["iteration_counts"]["bootstrap"])
    )

    rows_by_key, provenance = load_all_rows(Path(args.evidence_root), family)
    result = analyse(
        family=family,
        rows_by_key=rows_by_key,
        provenance=provenance,
        permutations=permutations,
        bootstrap_iterations=bootstrap_iterations,
        iterations_overridden=iterations_overridden,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "seed_rows.csv", result["seed_rows"])
    _write_csv(output_dir / "summary_long.csv", result["summary_long"])
    _write_csv(output_dir / "contrasts.csv", result["contrasts"])
    _write_csv(output_dir / "mcnemar.csv", result["mcnemar"])
    payload = {
        "schema_version": "audiollm.qwen3_daic_label_vocab.analysis.v1",
        "family": {
            "path": str(family_path.relative_to(ROOT)) if str(family_path).startswith(str(ROOT)) else str(family_path),
            "sha256": hashlib.sha256(family_path.read_bytes()).hexdigest(),
            "arm_order": family["arm_order"],
            "seeds": family["seeds"],
            "alpha": family["alpha"],
            "metrics": family["metrics"],
            "analysis_seed": family["tests"]["analysis_seed"],
            "iterations": result["iterations"],
        },
        "seed_rows": result["seed_rows"],
        "summary_long": result["summary_long"],
        "contrasts": result["contrasts"],
        "mcnemar": result["mcnemar"],
        "provenance": result["provenance"],
    }
    (output_dir / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(_markdown(family, result), encoding="utf-8")
    if iterations_overridden:
        print("WARNING: iteration counts were overridden; this run is not the prespecified analysis")
    print(f"analysis written to {output_dir} ({len(result['contrasts'])} contrast rows, {len(result['mcnemar'])} McNemar rows)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
