from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEED_RE = re.compile(r"^seed_(\d+)$")
FOLD_RE = re.compile(r"^fold_(\d+)$")


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.with_suffix(".jsonl").exists():
        return [json.loads(line) for line in path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.with_suffix(".csv").exists():
        with path.with_suffix(".csv").open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise FileNotFoundError(f"Missing subject prediction artifact beside {path}")


def _view_for_task(task: dict[str, Any], requested: str | None) -> str:
    if requested:
        if requested not in task.get("views", []):
            raise ValueError(f"Requested OOF view {requested!r} is not present in {task['task_id']}.")
        return requested
    default = "fixed15" if str(task.get("protocol_id", "")).startswith("j") else "all"
    if default not in task.get("views", []):
        raise ValueError(f"Task {task['task_id']} has no default OOF view {default!r}.")
    return default


def _elapsed_seconds(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text)
    fields = [int(item) for item in text.split(":")]
    if len(fields) == 3:
        hours, minutes, seconds = fields
    elif len(fields) == 2:
        hours, minutes, seconds = 0, fields[0], fields[1]
    else:
        return float(fields[0])
    return float(timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds).total_seconds())


def _gpu_count(value: Any) -> int:
    match = re.search(r"gres/gpu[^=]*=(\d+)", str(value or ""))
    return int(match.group(1)) if match else 0


def _gpu_hours(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    if row.get("gpu_hours") is not None:
        return float(row["gpu_hours"])
    gpu_text = str(row.get("allocated_tres", ""))
    if not re.search(r"gres/gpu[^=]*=(\d+)", gpu_text):
        return None
    return _elapsed_seconds(row.get("elapsed")) * _gpu_count(gpu_text) / 3600.0


def collect(
    matrix: dict[str, Any],
    artifact_root: Path,
    requested_view: str | None = None,
    accounting: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    canonical_root = ROOT / "output_model" / "daic_chunking_comprehensive" / str(matrix["run_id"]) / str(matrix["stage"])
    for task in matrix.get("tasks", []):
        if task.get("kind") != "evaluation":
            continue
        task_root = ROOT / str(task["output_root"])
        try:
            task_root = artifact_root / task_root.relative_to(canonical_root)
        except ValueError:
            pass
        view = _view_for_task(task, requested_view)
        prediction_path = task_root / "evaluation" / view / "predictions_subject_level"
        subject_rows = _rows(prediction_path)
        selected_epoch_path = task_root / "logs" / "selected_checkpoint_selection_metrics.json"
        selected_epoch = None
        if selected_epoch_path.exists():
            selected_epoch = json.loads(selected_epoch_path.read_text(encoding="utf-8")).get("selected_epoch")
        train_task_id = f"train__{task['cell_id']}"
        gpu_hours = _gpu_hours((accounting or {}).get(train_task_id))
        seen: set[str] = set()
        for row in subject_rows:
            subject_id = str(row.get("subject_id", ""))
            if not subject_id or subject_id in seen:
                raise ValueError(f"Duplicate or missing subject in {task['task_id']}: {subject_id!r}")
            seen.add(subject_id)
            if row.get("score_margin") is not None:
                margin = float(row["score_margin"])
            else:
                margin = float(row.get("dep_score", 0.0)) - float(row.get("non_score", 0.0))
            prediction = row.get("prediction")
            if prediction in ("", None):
                raise ValueError(f"Invalid subject prediction in {task['task_id']}: {subject_id}")
            rows.append(
                {
                    "protocol_id": task["protocol_id"],
                    "seed": int(task["seed"]),
                    "fold": int(task["fold"]),
                    "subject_id": subject_id,
                    "label": int(row["label"]),
                    "prediction": int(prediction),
                    "score_margin": margin,
                    "selected_epoch": selected_epoch,
                    "evaluation_view": view,
                    "task_id": task["task_id"],
                    "train_task_id": train_task_id,
                    "implementation_hash": task.get("implementation_hash", matrix.get("implementation_hash")),
                    "config_hash": task.get("config_hash"),
                    "gpu_hours": gpu_hours,
                }
            )
    return sorted(rows, key=lambda row: (str(row["protocol_id"]), int(row["seed"]), int(row["fold"]), str(row["subject_id"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view")
    parser.add_argument("--slurm-accounting", type=Path)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    accounting = None
    if args.slurm_accounting and args.slurm_accounting.exists():
        accounting = {
            str(row["task_id"]): row
            for row in (
                json.loads(line)
                for line in args.slurm_accounting.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
    rows = collect(matrix, args.artifact_root, args.view, accounting)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
