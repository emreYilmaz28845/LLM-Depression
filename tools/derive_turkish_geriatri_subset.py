#!/usr/bin/env python3
"""Derive the matched Turkish original-cohort comparison from completed arms.

The four-source Turkish campaign evaluates every arm on its own five folds and
writes one subject-level prediction file per fold. The cross-arm question — does
adding the geriatri cohort to training change performance on the same original
participants? — is answered by restricting the treatment's already completed
full-population evaluation to the original participants, with identical
evaluation semantics, and comparing it with the baseline arm's own evaluation
of the same people on the same locked folds.

This command never retrains and never re-evaluates. It pools the per-fold
subject predictions, splits the declared populations by the subject-id cohort
namespace, verifies that subsetting does not change a single per-subject
prediction, recomputes the strict headline metrics, and checks the
full-population values against the recorded evaluation metrics.

Every fold directory is passed explicitly:

    python tools/derive_turkish_geriatri_subset.py \
      --cell qwen38_text_only --output-dir outputs/significance/<cell> \
      --baseline-fold 0=<dir> --baseline-fold 1=<dir> ... \
      --treatment-fold 0=<dir> --treatment-fold 1=<dir> ...
"""

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

from src.aggregate import _metrics_from_prediction_rows  # canonical strict metrics
from src.utils import AGGREGATION_LEVEL_SUBJECT, PREDICTION_MODE_LIKELIHOOD

GERIATRI_PREFIX = "geriatri:"
HEADLINE_FIELDS = ("binary_strict_macro_f1", "binary_strict_positive_f1", "binary_strict_uar")
HEADLINE_LABELS = {
    "binary_strict_macro_f1": "macro_f1",
    "binary_strict_positive_f1": "positive_f1",
    "binary_strict_uar": "uar",
}
EVAL_LOCATIONS = (
    "eval/best_validation",
    "best_model/standalone_eval",
    "eval/best_checkpoint",
    "last_model/standalone_eval",
)
METRICS_FILES = ("metrics_likelihood.json", "metrics.json")
POPULATIONS = ("full", "original", "geriatri")
TOLERANCE = 1e-12


class DerivationError(RuntimeError):
    """Raised when the derivation contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subject_set_sha256(subject_ids: list[str]) -> str:
    payload = json.dumps(sorted(subject_ids), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise DerivationError(f"empty prediction file: {path}")
    for row in rows:
        for field in ("subject_id", "label", "prediction"):
            if field not in row:
                raise DerivationError(f"{path} is missing the {field!r} column")
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def strict_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    metrics = _metrics_from_prediction_rows(
        [{"label": int(row["label"]), "prediction": int(row["prediction"])} for row in rows],
        backend_name=PREDICTION_MODE_LIKELIHOOD,
        aggregation_level=AGGREGATION_LEVEL_SUBJECT,
    )
    return {
        **{HEADLINE_LABELS[field]: metrics[field] for field in HEADLINE_FIELDS},
        "num_subjects": metrics["num_subjects"],
        "invalid_subjects": metrics["invalid_subjects"],
        "confusion_matrix": metrics["binary_strict_confusion_matrix"],
    }


def fold_mean(per_fold: dict[str, dict[str, Any]]) -> dict[str, float] | None:
    folds = sorted(per_fold)
    if not folds:
        return None
    return {
        HEADLINE_LABELS[field]: sum(per_fold[fold][HEADLINE_LABELS[field]] for fold in folds) / len(folds)
        for field in HEADLINE_FIELDS
    }


def locate_predictions(fold_dir: Path) -> tuple[Path, str]:
    for location in EVAL_LOCATIONS:
        candidate = fold_dir / location / "predictions_subject_level.csv"
        if candidate.is_file():
            return candidate, location
        jsonl = fold_dir / location / "predictions_subject_level.jsonl"
        if jsonl.is_file() and not candidate.is_file():
            raise DerivationError(
                f"{jsonl} is the only subject-level file; the derivation needs the CSV form"
            )
    raise DerivationError(f"no subject-level predictions under {fold_dir}")


def verify_recorded_metrics(fold_dir: Path, location: str, metrics: dict[str, Any]) -> dict[str, Any]:
    for name in METRICS_FILES:
        path = fold_dir / location / name
        if not path.is_file():
            continue
        recorded = json.loads(path.read_text(encoding="utf-8"))
        comparisons = {
            label: {
                "recorded": recorded.get(field),
                "recomputed": metrics[label],
                "equal": recorded.get(field) is not None
                and abs(float(recorded[field]) - float(metrics[label])) <= TOLERANCE,
            }
            for field, label in HEADLINE_LABELS.items()
        }
        return {"metrics_path": str(path), "metrics_sha256": sha256_file(path), "comparisons": comparisons}
    return {"metrics_path": None, "metrics_sha256": None, "comparisons": {}}


def population_of(subject_id: str) -> str:
    return "geriatri" if subject_id.startswith(GERIATRI_PREFIX) else "original"


def pool_arm(label: str, fold_dirs: dict[int, Path]) -> dict[str, Any]:
    pooled: dict[str, dict[str, str]] = {}
    per_fold_rows: dict[int, list[dict[str, str]]] = {}
    sources: dict[str, Any] = {}
    recorded_checks: dict[str, Any] = {}
    for fold in sorted(fold_dirs):
        fold_dir = fold_dirs[fold]
        if not fold_dir.is_dir():
            raise DerivationError(f"{label}: missing fold directory {fold_dir}")
        path, location = locate_predictions(fold_dir)
        rows = read_rows(path)
        per_fold_rows[fold] = rows
        sources[str(fold)] = {
            "fold_dir": str(fold_dir),
            "predictions_path": str(path),
            "predictions_sha256": sha256_file(path),
            "eval_location": location,
        }
        for row in rows:
            subject_id = str(row["subject_id"])
            if subject_id in pooled:
                raise DerivationError(
                    f"{label}: subject {subject_id!r} appears in more than one fold"
                )
            pooled[subject_id] = row
        recorded_checks[str(fold)] = verify_recorded_metrics(
            fold_dir, location, strict_metrics(rows)
        )
    populations = {
        population: [row for subject, row in pooled.items() if population_of(subject) == population]
        for population in ("original", "geriatri")
    }
    populations["full"] = list(pooled.values())
    per_fold_metrics: dict[str, dict[str, dict[str, Any]]] = {population: {} for population in POPULATIONS}
    for fold, rows in per_fold_rows.items():
        for population in POPULATIONS:
            if population == "full":
                subset = list(rows)
            else:
                subset = [
                    row for row in rows if population_of(str(row["subject_id"])) == population
                ]
            if subset:
                per_fold_metrics[population][str(fold)] = strict_metrics(subset)
    return {
        "label": label,
        "fold_dirs": {str(fold): str(path) for fold, path in fold_dirs.items()},
        "sources": sources,
        "pooled_rows": pooled,
        "populations": populations,
        "per_fold_metrics": per_fold_metrics,
        "recorded_checks": recorded_checks,
    }


def subset_is_unchanged(full: dict[str, dict[str, str]], subset: list[dict[str, str]]) -> bool:
    for row in subset:
        source = full[str(row["subject_id"])]
        for field in ("subject_id", "label", "prediction"):
            if str(source[field]) != str(row[field]):
                return False
    return True


def parse_fold_dirs(values: list[str], label: str) -> dict[int, Path]:
    fold_dirs: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise DerivationError(f"--{label}-fold expects <fold>=<path>, got {value!r}")
        raw_fold, raw_path = value.split("=", 1)
        fold = int(raw_fold)
        if fold in fold_dirs:
            raise DerivationError(f"--{label}-fold repeats fold {fold}")
        fold_dirs[fold] = Path(raw_path).resolve()
    return fold_dirs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cell", required=True, help="reader-facing cell label, e.g. qwen38_text_only")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--baseline-fold", action="append", default=[], metavar="FOLD=DIR",
        help="baseline arm fold directory (repeat per fold)",
    )
    parser.add_argument(
        "--treatment-fold", action="append", default=[], metavar="FOLD=DIR",
        help="treatment arm fold directory (repeat per fold)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    baseline_folds = parse_fold_dirs(args.baseline_fold, "baseline")
    treatment_folds = parse_fold_dirs(args.treatment_fold, "treatment")
    if not baseline_folds or not treatment_folds:
        raise DerivationError("both --baseline-fold and --treatment-fold are required")
    if sorted(baseline_folds) != sorted(treatment_folds):
        raise DerivationError(
            f"the arms disagree on the fold set: {sorted(baseline_folds)} vs {sorted(treatment_folds)}"
        )

    baseline = pool_arm("baseline", baseline_folds)
    treatment = pool_arm("treatment", treatment_folds)

    if baseline["populations"]["geriatri"]:
        raise DerivationError("the baseline arm must not contain geriatri subjects")
    baseline_ids = {str(row["subject_id"]) for row in baseline["populations"]["original"]}
    treatment_ids = {str(row["subject_id"]) for row in treatment["populations"]["original"]}
    if baseline_ids != treatment_ids:
        raise DerivationError(
            "the two arms do not evaluate the same original participants: "
            f"{len(baseline_ids)} vs {len(treatment_ids)}"
        )
    labels = {str(row["subject_id"]): str(row["label"]) for row in baseline["populations"]["original"]}
    for row in treatment["populations"]["original"]:
        if labels[str(row["subject_id"])] != str(row["label"]):
            raise DerivationError(f"label mismatch between arms for subject {row['subject_id']!r}")
    for arm in (baseline, treatment):
        for population in POPULATIONS:
            if not subset_is_unchanged(arm["pooled_rows"], arm["populations"][population]):
                raise DerivationError(
                    f"{arm['label']} {population} subset changed per-subject predictions"
                )
    bad = [
        (arm["label"], fold, label)
        for arm in (baseline, treatment)
        for fold, check in arm["recorded_checks"].items()
        for label, comparison in check["comparisons"].items()
        if comparison["recorded"] is not None and not comparison["equal"]
    ]
    if bad:
        raise DerivationError(f"recomputed metrics disagree with the recorded evaluation: {bad[:5]}")

    output_dir = Path(args.output_dir)
    written: dict[str, str] = {}
    for arm in (baseline, treatment):
        for population in POPULATIONS:
            rows = arm["populations"][population]
            if not rows:
                continue
            path = output_dir / f"{arm['label']}_{population}_predictions_subject_level.csv"
            write_rows(path, rows)
            written[str(path)] = sha256_file(path)

    audit = {
        "schema_version": "audiollm.turkish_subset_derivation.v1",
        "cell": args.cell,
        "populations": {
            population: {
                "baseline_subjects": len(baseline["populations"][population]),
                "treatment_subjects": len(treatment["populations"][population]),
            }
            for population in POPULATIONS
        },
        "matched_original_population": {
            "subject_count": len(baseline_ids),
            "subject_set_sha256": subject_set_sha256(sorted(baseline_ids)),
            "label_distribution": {
                label: sum(1 for value in labels.values() if value == label)
                for label in sorted(set(labels.values()))
            },
        },
        "arms": {
            arm["label"]: {
                "fold_dirs": arm["fold_dirs"],
                "runs": arm["sources"],
                "pooled_subject_count": len(arm["pooled_rows"]),
                "per_fold_metrics": arm["per_fold_metrics"],
                "fold_mean_metrics": {
                    population: fold_mean(arm["per_fold_metrics"][population])
                    for population in POPULATIONS
                },
                "recorded_metric_checks": arm["recorded_checks"],
            }
            for arm in (baseline, treatment)
        },
        "subset_operation_preserves_predictions": True,
        "outputs": written,
    }
    audit["audit_sha256"] = hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "subset_derivation_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "cell": args.cell,
        "original_subjects": len(baseline_ids),
        "baseline_original_fold_mean": audit["arms"]["baseline"]["fold_mean_metrics"]["original"],
        "treatment_original_fold_mean": audit["arms"]["treatment"]["fold_mean_metrics"]["original"],
        "treatment_full_fold_mean": audit["arms"]["treatment"]["fold_mean_metrics"]["full"],
        "treatment_geriatri_fold_mean": audit["arms"]["treatment"]["fold_mean_metrics"]["geriatri"],
        "outputs": sorted(written),
        "audit": str(audit_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DerivationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
