from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.daic_comprehensive_audit import audit_matrix, audit_schedule, audit_slurm, audit_test_gate
from src.utils import read_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--slurm-accounting", type=Path)
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    matrix = read_json(args.matrix)
    failures = audit_matrix(matrix)
    checks: dict[str, object] = {"matrix": {"passed": not failures, "task_count": len(matrix.get("tasks", []))}}
    if args.artifact_root:
        artifact_root = args.artifact_root
        failures.extend(audit_test_gate(artifact_root, str(matrix["stage"])))
        missing = []
        schedule_failures = []
        views_by_cell = {
            task["cell_id"]: task.get("views", [])
            for task in matrix["tasks"] if task["kind"] == "evaluation"
        }
        for task in matrix["tasks"]:
            root = ROOT / task["output_root"]
            if task["kind"] == "train":
                schedule_path = root / "logs/daic_chunk_schedule_audit.json"
                if schedule_path.exists():
                    schedule_failures.extend(f"{task['cell_id']}:{item}" for item in audit_schedule(read_json(schedule_path)))
                elif args.require_artifacts:
                    missing.append(str(schedule_path))
                if args.require_artifacts and not (root / "best_model").exists():
                    missing.append(str(root / "best_model"))
            elif args.require_artifacts and task["kind"] == "evaluation":
                for view in task.get("views", []):
                    required = (
                        root / "evaluation" / view / "predictions_subject_resamples.jsonl"
                        if view == "matched10_resampled"
                        else root / "evaluation" / view / "metrics_original_teacher_forced.json"
                    )
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
