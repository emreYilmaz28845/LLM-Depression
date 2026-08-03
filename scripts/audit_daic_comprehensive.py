from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.daic_comprehensive_audit import (
    audit_manifest_contract,
    audit_matrix,
    audit_schedule,
    audit_slurm,
    audit_test_gate,
)
from src.utils import read_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--slurm-accounting", type=Path)
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--partitions", type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    matrix = read_json(args.matrix)
    failures = audit_matrix(matrix)
    checks: dict[str, object] = {"matrix": {"passed": not failures, "task_count": len(matrix.get("tasks", []))}}
    manifest_path = args.manifest
    partitions_path = args.partitions
    folds_path = args.folds
    shared_splits = args.matrix.parent / "shared" / "splits"
    shared_manifests = args.matrix.parent / "shared" / "manifests"
    if manifest_path is None and (shared_manifests / "daic_manifest.jsonl").exists():
        manifest_path = shared_manifests / "daic_manifest.jsonl"
    if partitions_path is None and (shared_splits / "daic_subject_partitions.json").exists():
        partitions_path = shared_splits / "daic_subject_partitions.json"
    if folds_path is None and (shared_splits / "daic_folds.json").exists():
        folds_path = shared_splits / "daic_folds.json"
    if manifest_path:
        manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        partition_rows = json.loads(partitions_path.read_text(encoding="utf-8")) if partitions_path and partitions_path.exists() else None
        folds = json.loads(folds_path.read_text(encoding="utf-8")) if folds_path and folds_path.exists() else None
        manifest_failures = audit_manifest_contract(
            manifest_rows,
            partition_rows=partition_rows,
            folds=folds,
            expected_chunks_by_label={0: 10, 1: 15},
            expected_subjects_by_label={0: 133, 1: 56},
            split_seed=int(matrix.get("split_seed", 1337)),
        )
        failures.extend(manifest_failures)
        checks["manifest"] = {
            "path": str(manifest_path),
            "rows": len(manifest_rows),
            "failures": manifest_failures,
        }
    elif args.require_artifacts:
        failures.append("missing_manifest_for_acceptance_audit")
    if str(matrix.get("stage")) == "final" and not args.artifact_root:
        marker_value = (matrix.get("test_authorization") or {}).get("path")
        marker_path = Path(marker_value) if marker_value else args.matrix.parent / "FINAL_TEST_AUTHORIZED.json"
        if not marker_path.is_absolute():
            marker_path = args.matrix.parent / marker_path
        failures.extend(
            audit_test_gate(
                args.matrix.parent,
                "final",
                authorization_root=marker_path.parent,
                selection_hash=matrix.get("selection_hash"),
                implementation_commit=matrix.get("implementation_commit"),
                spec_hash=matrix.get("spec_hash"),
            )
        )
    if args.artifact_root:
        artifact_root = args.artifact_root
        marker_value = (matrix.get("test_authorization") or {}).get("path")
        marker_path = Path(marker_value) if marker_value else args.matrix.parent / "FINAL_TEST_AUTHORIZED.json"
        if not marker_path.is_absolute():
            marker_path = args.matrix.parent / marker_path
        failures.extend(
            audit_test_gate(
                artifact_root,
                str(matrix["stage"]),
                authorization_root=marker_path.parent,
                selection_hash=matrix.get("selection_hash"),
                implementation_commit=matrix.get("implementation_commit"),
                spec_hash=matrix.get("spec_hash"),
            )
        )
        missing = []
        schedule_failures = []
        views_by_cell = {
            task["cell_id"]: task.get("views", [])
            for task in matrix["tasks"] if task["kind"] == "evaluation"
        }
        for task in matrix["tasks"]:
            root = ROOT / task["output_root"]
            if artifact_root:
                run_stage_root = ROOT / "output_model" / "daic_chunking_comprehensive" / str(matrix["run_id"]) / str(matrix["stage"])
                try:
                    root = artifact_root / root.relative_to(run_stage_root)
                except ValueError:
                    # Preserve the historical repository-relative fallback for
                    # custom matrices whose output roots are outside the
                    # canonical comprehensive tree.
                    root = ROOT / task["output_root"]
            if task["kind"] == "train":
                schedule_path = root / "logs/daic_chunk_schedule_audit.json"
                objective = str(task.get("overrides", {}).get("training.objective", "token_ce"))
                mil_schedule_path = root / "logs/mil_training_audit.json"
                if schedule_path.exists():
                    schedule_failures.extend(f"{task['cell_id']}:{item}" for item in audit_schedule(read_json(schedule_path)))
                elif objective == "subject_mean_margin_mil" and mil_schedule_path.exists():
                    mil_payload = read_json(mil_schedule_path)
                    if int(mil_payload.get("complete_subject_updates", 0)) < 1:
                        schedule_failures.append(f"{task['cell_id']}:mil_no_complete_subject_update")
                    for epoch in mil_payload.get("epochs", []):
                        class_updates = epoch.get("class_updates") or {}
                        if int(class_updates.get("non_depressed", 0)) < 1 or int(class_updates.get("depressed", 0)) < 1:
                            schedule_failures.append(f"{task['cell_id']}:mil_missing_class_update:{epoch.get('epoch')}")
                elif args.require_artifacts:
                    missing.append(str(mil_schedule_path if objective == "subject_mean_margin_mil" else schedule_path))
                if args.require_artifacts:
                    for required in (root / "best_model", root / "run_config.yaml", root / "logs" / "split_used.json"):
                        if not required.exists():
                            missing.append(str(required))
            elif args.require_artifacts and task["kind"] == "evaluation":
                for view in task.get("views", []):
                    required = (
                        root / "evaluation" / view / "predictions_subject_resamples.jsonl"
                        if view == "matched10_resampled"
                        else root / "evaluation" / view / "metrics_original_teacher_forced.json"
                    )
                    if not required.exists():
                        missing.append(str(required))
                secondary_methods = task.get("overrides", {}).get("evaluation.secondary_aggregations") or []
                for view in task.get("views", []):
                    if view == "matched10_resampled":
                        continue
                    for method in secondary_methods:
                        for suffix in (
                            f"metrics_secondary_{method}.json",
                            f"predictions_subject_level_{method}.jsonl",
                        ):
                            required = root / "evaluation" / view / suffix
                            if not required.exists():
                                missing.append(str(required))
            elif args.require_artifacts and task["kind"] == "hidden":
                for view in views_by_cell.get(task["cell_id"], []):
                    if view == "matched10_resampled":
                        continue
                    required = root / "hidden" / view / "extraction_metadata.json"
                    if not required.exists():
                        missing.append(str(required))
            elif args.require_artifacts and task["kind"] == "classical":
                for view in views_by_cell.get(task["cell_id"], []):
                    if view == "matched10_resampled":
                        continue
                    for head in task.get("heads", []):
                        required = root / "classical" / view / head / "metrics.json"
                        if not required.exists():
                            missing.append(str(required))
        failures.extend(f"missing_artifact:{path}" for path in missing)
        failures.extend(schedule_failures)
        checks["artifacts"] = {"missing": missing, "schedule_failures": schedule_failures}
    if args.slurm_accounting:
        rows = [json.loads(line) for line in args.slurm_accounting.read_text().splitlines() if line.strip()]
        slurm_failures = audit_slurm(rows, {str(task["task_id"]) for task in matrix["tasks"]})
        failures.extend(slurm_failures)
        checks["slurm"] = {"rows": len(rows), "failures": slurm_failures}
    elif args.require_artifacts:
        failures.append("missing_authoritative_slurm_accounting")
    payload = {
        "schema_version": "daic_chunking_comprehensive_audit.v1",
        "run_id": matrix["run_id"], "stage": matrix["stage"],
        "passed": not failures, "failures": failures, "checks": checks,
        "matrix_spec_hash": matrix.get("spec_hash"),
        "implementation_commit": matrix.get("implementation_commit"),
    }
    output = args.output or args.matrix.parent / f"audit_{matrix['stage']}.json"
    save_json(payload, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
