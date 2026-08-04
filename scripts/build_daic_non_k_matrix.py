from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    "src/daic_chunking.py",
    "src/data/runtime.py",
    "src/train.py",
    "src/evaluate.py",
    "scripts/build_daic_non_k_matrix.py",
    "scripts/run_daic_non_k_task.py",
    "scripts/audit_daic_non_k.py",
    "scripts/report_daic_non_k.py",
    "scripts/submit_daic_non_k.sh",
    "scripts/run_daic_non_k_array_slurm.sh",
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def implementation_hash() -> str:
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_PATHS:
        path = ROOT / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def expand(spec: dict[str, Any], run_id: str, stage: str) -> dict[str, Any]:
    if stage not in {"smoke", "production"}:
        raise ValueError(f"Unsupported stage={stage!r}.")
    if not run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one non-empty path component.")
    protocols = spec["protocols"]
    if stage == "smoke":
        protocols = {name: protocols[name] for name in spec["smoke_protocols"]}
    tasks = []
    for protocol_id, protocol in protocols.items():
        overrides = {**spec["common_overrides"], **protocol["overrides"], "seed": int(spec["seed"])}
        if stage == "smoke":
            overrides.update(
                {
                    "split.smoke_subject_limit": 6,
                    "training.num_train_epochs": 1,
                    "training.early_stopping.enabled": False,
                }
            )
        root = f"output_model/experiments/daic_non_k/{run_id}/{stage}/{protocol_id}/fold_0"
        common = {
            "cell_id": protocol_id,
            "protocol_id": protocol_id,
            "group": protocol["group"],
            "stage": stage,
            "fold": 0,
            "base_config": protocol["base_config"],
            "overrides": overrides,
            "output_root": root,
            "config_hash": canonical_hash(
                {
                    "base_config_sha256": hashlib.sha256(
                        (ROOT / protocol["base_config"]).read_bytes()
                    ).hexdigest(),
                    "overrides": overrides,
                }
            ),
        }
        tasks.append({**common, "task_id": f"train__{protocol_id}", "kind": "train"})
        tasks.append(
            {
                **common,
                "task_id": f"evaluation__{protocol_id}",
                "kind": "evaluation",
                "dependencies": [f"train__{protocol_id}"],
            }
        )
    payload = {
        "schema_version": spec["schema_version"],
        "run_id": run_id,
        "stage": stage,
        "seed": int(spec["seed"]),
        "implementation_commit": git_commit(),
        "implementation_hash": implementation_hash(),
        "spec_hash": canonical_hash(spec),
        "expected_training_cells": len(protocols),
        "kind_counts": {kind: sum(task["kind"] == kind for task in tasks) for kind in ("train", "evaluation")},
        "maximum_concurrent_train": int(spec["maximum_concurrent_train"]),
        "maximum_concurrent_evaluation": int(spec["maximum_concurrent_evaluation"]),
        "resources": spec["resources"],
        "tasks": tasks,
    }
    payload["matrix_hash"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=ROOT / "configs/experiments/daic_non_k/matrix.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("smoke", "production"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    payload = expand(spec, args.run_id, args.stage)
    output = args.output or ROOT / "outputs/daic_non_k" / args.run_id / f"matrix_{args.stage}.json"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if not args.resume or existing.get("matrix_hash") != payload["matrix_hash"]:
            raise SystemExit(f"Collision or incompatible resume: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "kind_counts": payload["kind_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
