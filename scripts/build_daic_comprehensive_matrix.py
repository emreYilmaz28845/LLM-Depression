from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def implementation_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _focused_protocols(spec: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    leaders = [str(selection["leading_joint"]), str(selection["leading_independent"])]
    selected: dict[str, Any] = {}
    for leader in leaders:
        base = spec["protocols"][leader]
        selected[f"{leader}_class_balanced"] = {
            **base, "overrides": {**base["overrides"], "training.class_balance": "subject_inverse_frequency"},
            "inclusion_reason": f"class-balanced follow-up of {leader}",
        }
        k_value = base["overrides"].get("data.train_chunks_per_subject")
        if k_value != "all":
            for k in (2, 8):
                selected[f"{leader}_k{k}"] = {
                    **base, "overrides": {**base["overrides"], "data.train_chunks_per_subject": k},
                    "inclusion_reason": f"predeclared K sensitivity for {leader}",
                }
    mil_base = spec["protocols"]["ian"]
    selected["qwen_mil"] = {
        **mil_base,
        "overrides": {**mil_base["overrides"], "data.sample_mode": "subject_mil", "training.objective": "subject_mean_margin_mil", "training.gradient_accumulation_steps": 1},
        "evaluation_views": ["all", "matched10_even", "matched10_resampled"],
        "inclusion_reason": "predeclared true subject mean-margin MIL follow-up",
    }
    return selected


def expand(spec: dict[str, Any], run_id: str, stage: str, selection: dict[str, Any] | None = None) -> dict[str, Any]:
    if stage not in {"smoke", "core", "focused", "final"}:
        raise ValueError(f"Unsupported stage={stage!r}.")
    if stage in {"focused", "final"} and not selection:
        raise ValueError(f"stage={stage} requires a hash-recorded selection artifact.")
    folds = [0] if stage == "smoke" else list(map(int, spec["folds"]))
    seeds = [int(spec["seeds"][0])] if stage == "smoke" else list(map(int, spec["seeds"]))
    protocols = spec["protocols"]
    if stage == "smoke":
        mil_base = spec["protocols"]["ian"]
        protocols = {**protocols, "qwen_mil": {
            **mil_base,
            "overrides": {**mil_base["overrides"], "data.sample_mode": "subject_mil", "training.objective": "subject_mean_margin_mil", "training.gradient_accumulation_steps": 1},
            "evaluation_views": ["all", "matched10_even", "matched10_resampled"],
            "inclusion_reason": "smoke the exact two-pass subject MIL path",
        }}
    elif stage == "focused":
        protocols = _focused_protocols(spec, selection or {})
    elif stage == "final":
        winner = str((selection or {})["winner"])
        winner_protocol = spec["protocols"].get(winner, (selection or {}).get("winner_protocol"))
        if winner_protocol is not None:
            winner_protocol = dict(winner_protocol)
            default_view = "fixed15" if winner.startswith("j") else "all"
            winner_protocol["evaluation_views"] = [str((selection or {}).get("aggregation_view", default_view))]
        protocols = {winner: winner_protocol}
        if protocols[winner] is None:
            raise ValueError("Final selection must embed winner_protocol when winner is a focused condition.")
        folds = [0]
    tasks: list[dict[str, Any]] = []
    for protocol_id, protocol in protocols.items():
        for seed in seeds:
            for fold in folds:
                root = f"output_model/daic_chunking_comprehensive/{run_id}/{stage}/{protocol_id}/seed_{seed}/fold_{fold}"
                overrides = {**spec["common_overrides"], **protocol["overrides"], "seed": seed}
                if stage == "smoke":
                    overrides.update({"split.smoke_subject_limit": 4, "training.num_train_epochs": 1, "training.early_stopping.enabled": False})
                if stage == "final":
                    overrides.update({
                        "split.mode": "full_train", "training.num_train_epochs": int((selection or {})["final_epoch_count"]),
                        "training.early_stopping.enabled": False,
                    })
                common = {
                    "cell_id": f"{protocol_id}__seed_{seed}__fold_{fold}", "stage": stage,
                    "protocol_id": protocol_id, "seed": seed, "fold": fold,
                    "base_config": protocol["base_config"], "overrides": overrides,
                    "config_hash": canonical_hash({
                        "base_config_sha256": hashlib.sha256((PROJECT_ROOT / protocol["base_config"]).read_bytes()).hexdigest(),
                        "overrides": overrides,
                    }),
                    "output_root": root,
                }
                train_id = f"train__{common['cell_id']}"
                tasks.append({**common, "task_id": train_id, "kind": "train", "dependencies": [], "resource_profile": "train"})
                tasks.append({**common, "task_id": f"evaluation__{common['cell_id']}", "kind": "evaluation", "dependencies": [train_id], "resource_profile": "evaluation", "views": protocol["evaluation_views"]})
                tasks.append({**common, "task_id": f"hidden__{common['cell_id']}", "kind": "hidden", "dependencies": [train_id], "resource_profile": "hidden"})
                tasks.append({**common, "task_id": f"classical__{common['cell_id']}", "kind": "classical", "dependencies": [f"hidden__{common['cell_id']}"], "resource_profile": "classical", "heads": ["logreg_raw", "xgb_raw"]})
    return {
        "schema_version": spec["schema_version"], "run_id": run_id, "stage": stage,
        "implementation_commit": implementation_commit(), "spec_hash": canonical_hash(spec),
        "task_count": len(tasks), "kind_counts": {kind: sum(task["kind"] == kind for task in tasks) for kind in ("train", "evaluation", "hidden", "classical")},
        "expected_training_cells": len(protocols) * len(folds) * len(seeds),
        "maximum_concurrent_train": int(spec["maximum_concurrent_train"]),
        "resources": spec["resources"], "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=PROJECT_ROOT / "configs/experiments/daic_chunking/comprehensive_matrix.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("smoke", "core", "focused", "final"), default="core")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8")) if args.selection else None
    payload = expand(spec, args.run_id, args.stage, selection)
    if selection is not None:
        payload["selection_artifact"] = str(args.selection)
        payload["selection_hash"] = hashlib.sha256(args.selection.read_bytes()).hexdigest()
    output = args.output or PROJECT_ROOT / "outputs/daic_chunking_comprehensive" / args.run_id / f"matrix_{args.stage}.json"
    if output.exists() and not args.resume:
        raise SystemExit(f"Collision: {output} exists (use --resume only after verifying the same hashes).")
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("spec_hash") != payload["spec_hash"] or existing.get("implementation_commit") != payload["implementation_commit"]:
            raise SystemExit("Resume rejected: spec or implementation commit differs.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "task_count": payload["task_count"], "kind_counts": payload["kind_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
