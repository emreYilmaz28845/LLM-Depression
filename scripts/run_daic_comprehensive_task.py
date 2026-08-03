from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def override_args(overrides: dict[str, Any]) -> str:
    def scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (list, dict)):
            return json.dumps(value, separators=(",", ":"))
        return str(value)
    return " ".join(f"--set {key}={scalar(value)}" for key, value in overrides.items())


def evaluation_override(view: str) -> dict[str, Any] | None:
    return {
        "fixed4": {"data.eval_chunk_policy": "fixed_k", "data.eval_chunks_per_subject": 4},
        "mincover4": {"data.eval_chunk_policy": "balanced_joint_cover", "data.eval_chunks_per_subject": 4},
        "fixed15": {"data.eval_chunk_policy": "fixed_count_balanced_joint_cover", "data.eval_chunks_per_subject": 4, "data.eval_bundles_per_subject": 15},
        "all": {"data.eval_chunk_policy": "all", "data.eval_chunks_per_subject": "all"},
        "matched10_even": {"data.eval_chunk_policy": "matched_k", "data.eval_chunks_per_subject": 10},
        "matched10_resampled": None,
    }[view]


def run(env: dict[str, str], script: str) -> None:
    subprocess.run(["bash", str(PROJECT_ROOT / script)], cwd=PROJECT_ROOT, env={**os.environ, **env}, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--kind", choices=("train", "evaluation", "hidden", "classical"), required=True)
    parser.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    tasks = [task for task in matrix["tasks"] if task["kind"] == args.kind]
    if not 0 <= args.index < len(tasks):
        raise SystemExit(f"Array index {args.index} is outside {args.kind} task count {len(tasks)}")
    task = tasks[args.index]
    output_root = PROJECT_ROOT / task["output_root"]
    resume = os.environ.get("RESUME", "0") == "1"
    config = PROJECT_ROOT / task["base_config"]
    common_overrides = {
        **task["overrides"],
        "output_dirs.run_root": str(output_root.parent),
        "output_dirs.manifest_dir": str(PROJECT_ROOT / "outputs/daic_chunking_comprehensive" / matrix["run_id"] / "shared/manifests"),
        "output_dirs.split_dir": str(PROJECT_ROOT / "outputs/daic_chunking_comprehensive" / matrix["run_id"] / "shared/splits"),
    }
    logs = PROJECT_ROOT / "logs/daic_chunking_comprehensive" / matrix["run_id"] / matrix["stage"] / task["kind"] / task["cell_id"]
    checkpoint = output_root / "best_model"
    if args.kind == "train":
        if (output_root / "best_model").exists() and resume:
            print(f"RESUME skip completed training task {task['task_id']}")
            return
        if output_root.exists() and any(output_root.iterdir()) and not resume:
            raise SystemExit(f"Collision: training output exists: {output_root}")
        run({
            "PROJECT_ROOT": str(PROJECT_ROOT), "CONFIG": str(config), "FOLD": str(task["fold"]),
            "RUN_NAME": ".", "EXTRA_TRAIN_ARGS": override_args(common_overrides),
            "SKIP_MANIFEST_BUILD": os.environ.get("SKIP_MANIFEST_BUILD", "1"),
            "LOG_ROOT": str(logs), "NPROC_PER_NODE": "1" if task["overrides"].get("training.objective") == "subject_mean_margin_mil" else "4",
        }, "scripts/run_train_slurm.sh")
        return
    if args.kind == "evaluation":
        for view in task["views"]:
            view_overrides = evaluation_override(view)
            if view_overrides is None:
                continue
            view_root = output_root / "evaluation" / view
            if (view_root / "metrics_original_teacher_forced.json").exists() and resume:
                print(f"RESUME skip completed evaluation {task['cell_id']} {view}")
                continue
            if view_root.exists() and any(view_root.iterdir()) and not resume:
                raise SystemExit(f"Collision: evaluation output exists: {view_root}")
            run({
                "PROJECT_ROOT": str(PROJECT_ROOT), "CONFIG": str(config), "FOLD": str(task["fold"]),
                "CHECKPOINT_DIR": str(checkpoint), "OUTPUT_DIR": str(output_root / "evaluation" / view),
                "EXTRA_EVAL_ARGS": override_args({**common_overrides, **view_overrides}),
                "LOG_ROOT": str(logs / view),
            }, "scripts/run_eval_slurm.sh")
        if "matched10_resampled" in task["views"]:
            from src.daic_chunking import matched_k_resamples
            source = output_root / "evaluation" / "all" / "predictions_sample_level.jsonl"
            samples = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
            rows = matched_k_resamples(samples, k=10, iterations=1000, seed=1337)
            target = output_root / "evaluation" / "matched10_resampled" / "predictions_subject_resamples.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        return
    strategy = "joint" if task["protocol_id"].startswith("j") else "all"
    if args.kind == "hidden":
        expected_views = [view for view in next(
            row for row in matrix["tasks"] if row["kind"] == "evaluation" and row["cell_id"] == task["cell_id"]
        )["views"] if view != "matched10_resampled"]
        if resume and all((output_root / "hidden" / view / "extraction_metadata.json").exists() for view in expected_views):
            print(f"RESUME skip completed hidden task {task['task_id']}")
            return
        if (output_root / "hidden").exists() and not resume:
            raise SystemExit(f"Collision: hidden output exists: {output_root / 'hidden'}")
        run({
            "PROJECT_ROOT": str(PROJECT_ROOT), "CHECKPOINT_DIR": str(checkpoint),
            "CACHE_ROOT": str(output_root / "hidden"), "STRATEGY": strategy,
            "PROTOCOL_ID": task["protocol_id"], "EVALUATION_VIEWS": ",".join(expected_views),
            "LOG_ROOT": str(logs),
        }, "scripts/run_daic_chunking_hidden_slurm.sh")
        return
    expected_views = [view for view in next(
        row for row in matrix["tasks"] if row["kind"] == "evaluation" and row["cell_id"] == task["cell_id"]
    )["views"] if view != "matched10_resampled"]
    if resume and all(
        (output_root / "classical" / view / head / "metrics.json").exists()
        for view in expected_views for head in task["heads"]
    ):
        print(f"RESUME skip completed classical task {task['task_id']}")
        return
    if (output_root / "classical").exists() and not resume:
        raise SystemExit(f"Collision: classical output exists: {output_root / 'classical'}")
    for variant in task["heads"]:
        run({
            "PROJECT_ROOT": str(PROJECT_ROOT), "CACHE_ROOT": str(output_root / "hidden"),
            "OUTPUT_ROOT": str(output_root / "classical"), "STRATEGY": strategy,
            "PROTOCOL_ID": task["protocol_id"], "EVALUATION_VIEWS": ",".join(expected_views),
            "VARIANT": variant, "LOG_ROOT": str(logs),
        }, "scripts/run_daic_chunking_classical_slurm.sh")


if __name__ == "__main__":
    main()
