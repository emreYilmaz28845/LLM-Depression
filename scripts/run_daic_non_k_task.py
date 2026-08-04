from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))


def override_args(overrides: dict[str, Any]) -> str:
    def scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (list, dict)):
            return json.dumps(value, separators=(",", ":"))
        return str(value)
    return " ".join(f"--set {key}={scalar(value)}" for key, value in overrides.items())


def run(script: str, env: dict[str, str]) -> None:
    subprocess.run(["bash", str(ROOT / script)], cwd=ROOT, env={**os.environ, **env}, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--kind", choices=("train", "evaluation"), required=True)
    parser.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    tasks = [task for task in matrix["tasks"] if task["kind"] == args.kind]
    if not 0 <= args.index < len(tasks):
        raise SystemExit(f"Invalid {args.kind} task index {args.index}.")
    task = tasks[args.index]
    config = ROOT / task["base_config"]
    output_root = ROOT / task["output_root"]
    shared = ROOT / "outputs/daic_non_k" / matrix["run_id"] / "shared"
    overrides = {
        **task["overrides"],
        "output_dirs.run_root": str(output_root.parent),
        "output_dirs.manifest_dir": str(shared / "manifests"),
        "output_dirs.split_dir": str(shared / "splits"),
    }
    logs = ROOT / "logs/daic_non_k" / matrix["run_id"] / matrix["stage"] / args.kind / task["cell_id"]
    resume = os.environ.get("RESUME", "0") == "1"
    if args.kind == "train":
        if (output_root / "best_model").is_dir() and resume:
            print(f"RESUME skip {task['task_id']}")
            return
        if output_root.exists() and any(output_root.iterdir()) and not resume:
            raise SystemExit(f"Collision: {output_root}")
        run(
            "scripts/run_train_slurm.sh",
            {
                "PROJECT_ROOT": str(ROOT),
                "CONFIG": str(config),
                "FOLD": "0",
                "RUN_NAME": ".",
                "EXTRA_TRAIN_ARGS": override_args(overrides),
                "SKIP_MANIFEST_BUILD": "1",
                "LOG_ROOT": str(logs),
                "NPROC_PER_NODE": "4",
            },
        )
        return
    checkpoint = output_root / "best_model"
    metrics = output_root / "evaluation" / "metrics_original_teacher_forced.json"
    if metrics.is_file() and resume:
        print(f"RESUME skip {task['task_id']}")
        return
    if not checkpoint.is_dir():
        raise SystemExit(f"Missing best_model: {checkpoint}")
    run(
        "scripts/run_eval_slurm.sh",
        {
            "PROJECT_ROOT": str(ROOT),
            "CONFIG": str(config),
            "FOLD": "0",
            "CHECKPOINT_DIR": str(checkpoint),
            "OUTPUT_DIR": str(output_root / "evaluation"),
            "EXTRA_EVAL_ARGS": override_args(overrides),
            "SKIP_MANIFEST_BUILD": "1",
            "LOG_ROOT": str(logs),
        },
    )


if __name__ == "__main__":
    main()
