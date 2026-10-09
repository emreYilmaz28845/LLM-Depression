#!/usr/bin/env python3
"""Local strict-metric verification for the collected merged evidence.

Read-only over the lane worktree. For every collected group/stage/fold it
recomputes the Qwen strict subject metrics and the merged head metrics from the
saved subject predictions and compares them with the recorded metric files.
It also re-derives the frozen final epoch from the five CV selections.

Schema v2 records, per fold, the sha256 of every artifact the recomputation
read (qwen metrics/predictions per dataset, head metrics/predictions per
method, train-side identity/complete/config/provenance files, cv selected
checkpoint). ``build_coverage_audit.py`` re-hashes these files and refuses
verification entries whose inputs no longer match, so stale checks cannot
approve replaced artifacts.

Missing evidence is recorded with ``ok: false`` (fail-closed), never silently
skipped.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_tracking.validate import recompute_strict_headline  # noqa: E402
from src.metrics import classification_metrics  # noqa: E402

EVIDENCE = ROOT / "outputs/qwen3_androids_official_folds_20261008"
TOLERANCE = 1e-6
QWEN_KEYS = ("binary_strict_macro_f1", "binary_strict_positive_f1", "binary_strict_uar")
HEAD_KEYS = ("macro_f1", "positive_f1", "macro_recall")
TRAIN_INPUT_FILES = (
    "training_identity.json",
    "training_complete.json",
    "resolved_merged_config.json",
    "slurm_provenance.json",
)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare(recorded, recomputed) -> bool:
    if recorded is None or recomputed is None:
        return False
    try:
        return abs(float(recorded) - float(recomputed)) <= TOLERANCE
    except (TypeError, ValueError):
        return False


def verify_fold(root: Path, fold_dir: Path, train_fold: Path, stage: str) -> tuple[list[dict], dict]:
    results: list[dict] = []
    inputs: dict[str, str] = {}

    def record(path: Path) -> None:
        try:
            inputs[str(path.relative_to(root))] = _sha256(path)
        except OSError:
            pass

    qwen_dir = fold_dir / "qwen"
    if qwen_dir.is_dir():
        for ds_dir in sorted(path for path in qwen_dir.iterdir() if path.is_dir()):
            pred = ds_dir / "predictions_subject_level.csv"
            rec_path = ds_dir / "metrics_likelihood.json"
            if not pred.is_file() or not rec_path.is_file():
                results.append(
                    {"kind": "qwen", "dataset": ds_dir.name, "status": "missing_evidence", "ok": False}
                )
                continue
            record(pred)
            record(rec_path)
            recorded = json.loads(rec_path.read_text(encoding="utf-8"))
            recomputed = recompute_strict_headline(pred)
            for key in QWEN_KEYS:
                ok = _compare(recorded.get(key), recomputed.get(key))
                results.append(
                    {
                        "kind": "qwen",
                        "dataset": ds_dir.name,
                        "metric": key,
                        "recorded": recorded.get(key),
                        "recomputed": recomputed.get(key),
                        "ok": ok,
                    }
                )
    else:
        results.append({"kind": "qwen", "dataset": None, "status": "missing_evidence", "ok": False})

    for method in ("logreg", "xgb_fixed"):
        method_dir = fold_dir / "heads" / method
        pred = method_dir / "predictions_subject_level.csv"
        rec_path = method_dir / "metrics_by_dataset.json"
        if not pred.is_file() or not rec_path.is_file():
            results.append(
                {"kind": method, "dataset": None, "status": "missing_evidence", "ok": False}
            )
            continue
        record(pred)
        record(rec_path)
        recorded = json.loads(rec_path.read_text(encoding="utf-8"))
        rows = _read_csv(pred)
        by_dataset: dict[str, list[dict]] = {}
        for row in rows:
            by_dataset.setdefault(str(row["dataset"]), []).append(row)
        for dataset, dataset_rows in sorted(by_dataset.items()):
            labels = [int(row["label"]) for row in dataset_rows]
            predictions = [int(row["prediction"]) for row in dataset_rows]
            metrics = classification_metrics(labels, predictions)
            for key in HEAD_KEYS:
                ok = _compare((recorded.get(dataset) or {}).get(key), metrics.get(key))
                results.append(
                    {
                        "kind": method,
                        "dataset": dataset,
                        "metric": key,
                        "recorded": (recorded.get(dataset) or {}).get(key),
                        "recomputed": metrics.get(key),
                        "ok": ok,
                    }
                )

    for name in TRAIN_INPUT_FILES:
        path = train_fold / name
        if path.is_file():
            record(path)
    if stage == "cv":
        path = train_fold / "logs" / "selected_checkpoint.json"
        if path.is_file():
            record(path)
    return results, inputs


def frozen_epoch_check(run_root: Path, final_root: Path) -> dict:
    selections = []
    for fold in range(5):
        path = run_root / "cv" / f"fold_{fold}" / "logs" / "selected_checkpoint.json"
        if not path.is_file():
            return {"status": "missing_cv_selection", "fold": fold}
        selections.append(int(json.loads(path.read_text(encoding="utf-8"))["selected_epoch"]))
    expected = int(math.floor(float(statistics.median(selections)) + 0.5))
    identity_path = final_root / "fold_0" / "training_identity.json"
    if not identity_path.is_file():
        return {"status": "missing_final_identity", "selections": selections, "expected": expected}
    actual = int(json.loads(identity_path.read_text(encoding="utf-8")).get("epochs", -1))
    return {
        "status": "passed" if actual == expected else "mismatch",
        "selections": selections,
        "expected": expected,
        "actual": actual,
    }


def main() -> int:
    report: dict = {
        "schema_version": "audiollm.androids_fixed_merged.local_metric_verification.v2",
        "groups": [],
        "failures": [],
    }
    for run_root_dir in sorted(
        glob.glob(str(ROOT / "output_model/symmetric_merged/*/*/qmsm_*"))
    ):
        run_root = Path(run_root_dir)
        run_id = run_root.name
        for stage in ("cv", "final"):
            stage_train = run_root / stage
            merged_stage = None
            for candidate in glob.glob(
                str(ROOT / "outputs/symmetric_merged/*/*" / run_id / stage)
            ):
                merged_stage = Path(candidate)
                break
            if merged_stage is None or not stage_train.is_dir():
                continue
            fold_dirs = sorted(
                path for path in merged_stage.glob("fold_*") if path.is_dir()
            )
            if not fold_dirs:
                continue
            fold_results = []
            for fold_dir in fold_dirs:
                fold = int(fold_dir.name.split("_", 1)[1])
                checks, inputs = verify_fold(
                    ROOT, fold_dir, stage_train / f"fold_{fold}", stage
                )
                fold_results.append({"fold": fold, "checks": checks, "inputs": inputs})
                for check in checks:
                    if check.get("ok") is not True:
                        report["failures"].append(
                            f"{run_id}/{stage}/fold_{fold}: {check}"
                        )
            entry: dict = {"run_id": run_id, "stage": stage, "folds": fold_results}
            if stage == "final":
                entry["frozen_epoch"] = frozen_epoch_check(run_root, stage_train)
                if entry["frozen_epoch"].get("status") != "passed":
                    report["failures"].append(
                        f"{run_id}/final: frozen epoch {entry['frozen_epoch']}"
                    )
            report["groups"].append(entry)
    report["status"] = "passed" if not report["failures"] else "failed"
    (EVIDENCE / "local_metric_verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "failures": report["failures"][:10]}, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
