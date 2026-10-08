#!/usr/bin/env python
"""Freeze the window15 control, treatment and head contracts.

Inputs: the audited 126 reusable 30-second controls, the current selection map,
the canonical configs they were trained with, and the frozen standalone head
implementation. Outputs three machine-readable contracts under the lane
evidence directory. The treatment contract declares only the window/input
identity differences; the generator test enforces that allowlist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

from tools.qwen3_multiseed_plan import build_selection_map  # noqa: E402

SEEDS = (7, 1337, 2024)
HEAD_VARIANTS = ("logreg_raw", "xgb_raw")


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def route_window_identity(config_path: Path) -> dict:
    import yaml

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data = payload.get("data") or {}
    resources = payload.get("resources") or {}
    return {
        "recipe_id": payload.get("recipe_id"),
        "protocol_id": payload.get("protocol_id"),
        "manifest_variant": payload.get("manifest_variant"),
        "sample_mode": data.get("sample_mode"),
        "segment_seconds": data.get("segment_seconds"),
        "segment_partition": data.get("segment_partition"),
        "participant_chunk_samples": data.get("participant_chunk_samples"),
        "train_nodes": resources.get("train_nodes"),
        "eval_gpus_per_node": resources.get("eval_gpus_per_node"),
    }


def declared_treatment_diff(route_id: str, identity: dict) -> dict:
    if str(identity.get("sample_mode")) == "participant_speech_packed30":
        return {
            "recipe_id": f"{identity['recipe_id']} -> single15 window15 marker",
            "manifest_variant": (
                f"{identity.get('manifest_variant')} -> "
                "unprocessed_participant_speech_packed30_15s_v1"
            ),
            "data.participant_chunk_samples": (
                f"{identity.get('participant_chunk_samples')} -> 240000"
            ),
        }
    return {
        "recipe_id": f"{identity['recipe_id']} -> single15 window15 marker",
        "data.segment_seconds": f"{identity.get('segment_seconds')} -> 15.0",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-audit",
        type=Path,
        default=LANE / "outputs/qwen3_window15_20261008/contracts/control_audit.json",
    )
    parser.add_argument(
        "--contracts-dir",
        type=Path,
        default=LANE / "outputs/qwen3_window15_20261008/contracts",
    )
    args = parser.parse_args()

    audit = read_json(args.control_audit)
    if not audit.get("rows"):
        raise SystemExit("control audit is missing; run qwen3_window15_control_audit.py first")
    selection = build_selection_map()
    routes = {
        str(route["route_id"]): str(route["config"])
        for route in selection["routes"]
        if route.get("language") == "native"
        and route.get("modality") in ("audio_only", "audio_text")
    }

    route_identities = {
        route_id: route_window_identity(LANE / config) for route_id, config in routes.items()
    }
    control_rows = audit["rows"]
    treatment_rows = []
    for row in control_rows:
        route_id = str(row["route_id"])
        base = Path(routes[route_id]).name
        target = f"configs/experiments/window15/{base.replace('.yaml', '_window15.yaml')}"
        treatment_rows.append(
            {
                "registry_key": row["registry_key"],
                "route_id": route_id,
                "seed": row["seed"],
                "fold": row["fold"],
                "control_attempt_id": row["attempt_id"],
                "control_run_name": row["run_name"],
                "treatment_config": target,
                "planned_run_name": f"q3w15_{route_id}_s{row['seed']}_f{row['fold']}",
            }
        )

    control_contract = {
        "schema_version": "audiollm.qwen3_window15_control_contract.v1",
        "campaign": "qwen3_window15_20261008",
        "routes": routes,
        "route_window_identity": route_identities,
        "expected_controls": 126,
        "rows": control_rows,
    }
    treatment_contract = {
        "schema_version": "audiollm.qwen3_window15_treatment_contract.v1",
        "campaign": "qwen3_window15_20261008",
        "routes": {
            route_id: {
                "control_config": config,
                "treatment_config": f"configs/experiments/window15/{Path(config).name.replace('.yaml', '_window15.yaml')}",
                "declared_diff": declared_treatment_diff(route_id, route_identities[route_id]),
                "partition_semantics": (
                    "DAIC participant-only packed fixed-15 (240000-sample chunks, short tail)"
                    if str(route_identities[route_id].get("sample_mode")) == "participant_speech_packed30"
                    else "equal_duration with ceil(unit duration / 15)"
                ),
            }
            for route_id, config in routes.items()
        },
        "expected_treatment_fits": 126,
        "head_training_windows": "treatment (15-second) windows, all windows per subject",
        "rows": treatment_rows,
    }
    classifier_src = LANE / "baselines/qwen_hidden_classifier.py"
    head_contract = {
        "schema_version": "audiollm.qwen3_window15_head_contract.v1",
        "campaign": "qwen3_window15_20261008",
        "variants": list(HEAD_VARIANTS),
        "head_seed": 1337,
        "implementation": {
            "path": "baselines/qwen_hidden_classifier.py",
            "sha256": sha256_file(classifier_src),
        },
        "estimators": {
            "logreg_raw": {
                "pipeline": ["StandardScaler", "LogisticRegression"],
                "params": {
                    "C": 1.0,
                    "class_weight": "balanced",
                    "max_iter": 5000,
                    "solver": "liblinear",
                    "random_state": 1337,
                },
                "pca": None,
            },
            "xgb_raw": {
                "pipeline": ["XGBClassifier"],
                "params": {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "n_estimators": 300,
                    "learning_rate": 0.03,
                    "max_depth": 2,
                    "min_child_weight": 5,
                    "subsample": 0.8,
                    "colsample_bytree": 0.25,
                    "reg_alpha": 1.0,
                    "reg_lambda": 10.0,
                    "scale_pos_weight": 1.0,
                    "tree_method": "hist",
                    "random_state": 1337,
                    "n_jobs": 1,
                },
                "pca": None,
            },
        },
        "recorded_metadata_policy": {
            "prediction_backend": "qwen_hidden_classifier",
            "sampling_mode": "legacy",
            "oversampling_ratio": None,
            "oversampling_seed": 1337,
            "weight_policy": "uniform_rows",
            "threshold": 0.5,
            "aggregation_policy": "sample_majority_probability_margin_tie_break_then_subject",
        },
        "extraction": "full-cohort features from treatment-window inputs; no PCA; no Optuna",
    }
    args.contracts_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("control_contract.json", control_contract),
        ("treatment_contract.json", treatment_contract),
        ("head_contract.json", head_contract),
    ):
        (args.contracts_dir / name).write_text(
            json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "controls": len(control_rows),
                "treatments": len(treatment_rows),
                "routes": len(routes),
                "contracts_dir": str(args.contracts_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
