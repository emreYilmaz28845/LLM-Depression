from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


def _run_root(project_root: Path, config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return Path(
        str(config["output_dirs"]["run_root"]).replace(
            "${PROJECT_ROOT}", str(project_root.resolve())
        )
    )


def audit(matrix_path: Path, project_root: Path) -> dict[str, Any]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    findings = []
    failures = []
    pooled: dict[tuple[str, str], list[str]] = {}
    for job in matrix["jobs"]:
        fold_root = (
            _run_root(project_root, project_root / job["config"])
            / job["run_name"]
            / f"fold_{job['fold']}"
        )
        try:
            required = (
                "run_config.yaml",
                "logs/split_used.json",
                "logs/training_history.json",
                "logs/selected_checkpoint_selection_metrics.json",
                "eval/best_validation/metrics_original_teacher_forced.json",
                "eval/best_validation/predictions_subject_level.csv",
            )
            missing = [name for name in required if not (fold_root / name).is_file()]
            assert not missing, f"missing {missing}"
            resolved = yaml.safe_load(
                (fold_root / "run_config.yaml").read_text(encoding="utf-8")
            )
            training = resolved["config"]["training"]
            assert training["class_balance"] == job["sampling_mode"]
            assert training["selection_metric"] == "inner_val_macro_f1"
            assert int(resolved["config"]["threshold"]) == 17
            split = json.loads(
                (fold_root / "logs/split_used.json").read_text(encoding="utf-8")
            )
            train = set(split["train_subject_ids"])
            selection = set(split["selection_subject_ids"])
            final_eval = set(split["final_eval_subject_ids"])
            assert not train & selection
            assert not train & final_eval
            assert not selection & final_eval
            if job["profile"] == "oversampled":
                assert float(training["oversampling_ratio"]) == float(
                    job["oversampling_ratio"]
                )
                assert int(training["oversampling_seed"]) == int(
                    job["oversampling_seed"]
                )
                sampling = json.loads(
                    (fold_root / "sampling_audit.json").read_text(encoding="utf-8")
                )
                assert sampling["detected_minority_label"] == 0
                assert sampling["validation_indices_untouched"]
                assert sampling["evaluation_indices_untouched"]
            with (
                fold_root
                / "eval/best_validation/predictions_subject_level.csv"
            ).open(encoding="utf-8", newline="") as handle:
                predictions = list(csv.DictReader(handle))
            ids = [row["subject_id"] for row in predictions]
            assert len(ids) == len(set(ids)) == len(selection)
            key = (job["modality"], job["profile"])
            pooled.setdefault(key, []).extend(ids)
            findings.append(
                {
                    "run_name": job["run_name"],
                    "fold": int(job["fold"]),
                    "profile": job["profile"],
                    "subjects": len(ids),
                    "status": "pass",
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "run_name": job["run_name"],
                    "fold": int(job["fold"]),
                    "path": str(fold_root),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    for key, ids in pooled.items():
        if len(ids) != len(set(ids)):
            failures.append({"pooled_identity": key, "error": "duplicate outer subjects"})
        if matrix["stage"] == "full" and len(ids) != 120:
            failures.append(
                {"pooled_identity": key, "error": f"expected 120 subjects, found {len(ids)}"}
            )
    return {
        "schema_version": "turkish_oversampling_qwen_audit.v1",
        "stage": matrix["stage"],
        "expected_jobs": int(matrix["expected_jobs"]),
        "audited_jobs": len(findings),
        "status": (
            "pass"
            if not failures and len(findings) == int(matrix["expected_jobs"])
            else "fail"
        ),
        "findings": findings,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.matrix, args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
