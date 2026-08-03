from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from functools import cmp_to_key
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.daic_comprehensive_audit import audit_oof_predictions
from src.metrics import binary_auroc, classification_metrics


DEFAULT_SEEDS = {1337, 2027, 3407}
DEFAULT_FOLDS = {0, 1, 2, 3, 4}


def implementation_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _run_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["protocol_id"]), int(row["seed"]), int(row["fold"])


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _selection_hash(payload: dict[str, Any]) -> str:
    without_hash = dict(payload)
    without_hash.pop("selection_hash", None)
    return hashlib.sha256(
        json.dumps(without_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def rank_protocols(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the predeclared macro-F1/AUROC/positive-F1/GPU-hour ordering."""
    passing = [row for row in summaries if row["passing"]]

    def compare(left: dict[str, Any], right: dict[str, Any]) -> int:
        macro_delta = float(left["mean_macro_f1"]) - float(right["mean_macro_f1"])
        if abs(macro_delta) >= 0.01:
            return -1 if macro_delta > 0 else 1
        for field in ("mean_auroc", "mean_positive_f1"):
            delta = float(left[field]) - float(right[field])
            if not math.isclose(delta, 0.0, abs_tol=1e-12):
                return -1 if delta > 0 else 1
        gpu_delta = float(left["gpu_hours"]) - float(right["gpu_hours"])
        if not math.isclose(gpu_delta, 0.0, abs_tol=1e-12):
            return -1 if gpu_delta < 0 else 1
        return -1 if str(left["protocol_id"]) < str(right["protocol_id"]) else 1 if str(left["protocol_id"]) > str(right["protocol_id"]) else 0

    return sorted(passing, key=cmp_to_key(compare))


def build_selection(
    rows: list[dict[str, Any]],
    expected_subjects: set[str],
    spec: dict[str, Any],
    *,
    expected_fold_subjects: dict[int, set[str]] | None = None,
    expected_implementation_hash: str | None = None,
    expected_config_hashes: dict[tuple[str, int, int], str] | None = None,
    require_gpu_hours: bool = False,
) -> dict[str, Any]:
    expected_protocols = set(map(str, (spec.get("protocols") or {}).keys()))
    seeds = set(map(int, spec.get("seeds", sorted(DEFAULT_SEEDS))))
    folds = set(map(int, spec.get("folds", sorted(DEFAULT_FOLDS))))
    if not expected_protocols:
        expected_protocols = {str(row.get("protocol_id")) for row in rows}
    failures = audit_oof_predictions(
        rows,
        expected_subject_ids=expected_subjects,
        protocols=expected_protocols,
        seeds=seeds,
        folds=folds,
        expected_fold_subjects=expected_fold_subjects,
        expected_implementation_hash=expected_implementation_hash,
        expected_config_hashes=expected_config_hashes,
        require_hashes=expected_implementation_hash is not None or expected_config_hashes is not None,
    )
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    epochs: dict[str, dict[tuple[int, int], int]] = defaultdict(dict)
    gpu_hours: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
    subject_labels: dict[str, int] = {}
    for row in rows:
        protocol = str(row.get("protocol_id", ""))
        try:
            seed = int(row["seed"])
            fold = int(row["fold"])
        except (KeyError, TypeError, ValueError):
            continue
        grouped[(protocol, seed)].append(row)
        subject_id = str(row.get("subject_id", ""))
        try:
            label = int(row.get("label", -1))
        except (TypeError, ValueError):
            failures.append(f"invalid_label:{protocol}:{seed}:{fold}")
            label = -1
        if subject_id in subject_labels and subject_labels[subject_id] != label:
            failures.append(f"label_mismatch:{subject_id}")
        subject_labels[subject_id] = label
        run_key = (seed, fold)
        if row.get("selected_epoch") is not None:
            try:
                epoch = int(row["selected_epoch"])
                if not 1 <= epoch <= 20:
                    failures.append(f"invalid_selected_epoch:{protocol}:{seed}:{fold}")
                    continue
                if run_key in epochs[protocol] and epochs[protocol][run_key] != epoch:
                    failures.append(f"selected_epoch_mismatch:{protocol}:{seed}:{fold}")
                epochs[protocol][run_key] = epoch
            except (TypeError, ValueError):
                failures.append(f"invalid_selected_epoch:{protocol}:{seed}:{fold}")
        if row.get("gpu_hours") is not None:
            try:
                value = float(row["gpu_hours"])
                if not _finite(value) or value < 0:
                    raise ValueError
                gpu_hours[protocol].setdefault(run_key, value)
            except (TypeError, ValueError):
                failures.append(f"invalid_gpu_hours:{protocol}:{seed}:{fold}")
        elif require_gpu_hours:
            failures.append(f"missing_gpu_hours:{protocol}:{seed}:{fold}")

    observed_protocols = {protocol for protocol, _ in grouped}
    for protocol in sorted(observed_protocols - expected_protocols):
        failures.append(f"unexpected_protocol:{protocol}")
    global_failures = [
        item for item in failures
        if item.startswith(("oof_malformed", "oof_unexpected_cell", "oof_invalid_", "label_mismatch", "unexpected_protocol"))
    ]
    summaries: list[dict[str, Any]] = []
    for protocol in sorted(expected_protocols | observed_protocols):
        per_seed: list[dict[str, Any]] = []
        protocol_failures = list(global_failures)
        protocol_failures.extend(
            item for item in failures if item.endswith(f":{protocol}") or f":{protocol}:" in item
        )
        for seed in sorted(seeds):
            cell = grouped.get((protocol, seed), [])
            ids = Counter(str(row.get("subject_id", "")) for row in cell)
            if set(ids) != expected_subjects or any(count != 1 for count in ids.values()):
                protocol_failures.append(f"coverage_seed_{seed}")
                continue
            if {int(row["fold"]) for row in cell} != folds:
                protocol_failures.append(f"folds_seed_{seed}")
                continue
            try:
                predictions = [int(row["prediction"]) for row in cell]
                labels = [int(row["label"]) for row in cell]
            except (KeyError, TypeError, ValueError):
                protocol_failures.append(f"invalid_prediction_or_label_seed_{seed}")
                continue
            if any(prediction not in {0, 1} for prediction in predictions) or any(label not in {0, 1} for label in labels):
                protocol_failures.append(f"invalid_prediction_or_label_seed_{seed}")
                continue
            if len(set(predictions)) == 1:
                protocol_failures.append(f"collapse_seed_{seed}")
                continue
            try:
                margins = [float(row["score_margin"]) for row in cell]
            except (KeyError, TypeError, ValueError):
                protocol_failures.append(f"invalid_margin_seed_{seed}")
                continue
            if not all(_finite(value) for value in margins):
                protocol_failures.append(f"invalid_margin_seed_{seed}")
                continue
            metrics = classification_metrics(labels, predictions)
            metrics["auroc"] = binary_auroc(labels, margins)
            per_seed.append({"seed": seed, **metrics})
        required_runs = len(seeds) * len(folds)
        if len(epochs.get(protocol, {})) != required_runs:
            protocol_failures.append(f"selected_epoch_coverage:{len(epochs.get(protocol, {}))}!={required_runs}")
        passing = not protocol_failures and len(per_seed) == len(seeds)
        summaries.append({
            "protocol_id": protocol,
            "passing": passing,
            "failures": sorted(set(protocol_failures)),
            "per_seed": per_seed,
            "mean_macro_f1": statistics.mean(row["macro_f1"] for row in per_seed) if per_seed else -1.0,
            "mean_auroc": statistics.mean(row["auroc"] for row in per_seed) if per_seed else -1.0,
            "mean_positive_f1": statistics.mean(row["positive_f1"] for row in per_seed) if per_seed else -1.0,
            "gpu_hours": sum(gpu_hours[protocol].values()),
        })

    ranked = rank_protocols(summaries)
    if not ranked:
        raise ValueError("No complete non-collapsed protocol is eligible.")
    joint = [row for row in ranked if str(row["protocol_id"]).startswith("j")]
    independent = [row for row in ranked if not str(row["protocol_id"]).startswith("j")]
    if not joint or not independent:
        raise ValueError("Selection requires at least one passing joint and independent protocol.")
    winner = ranked[0]["protocol_id"]
    winner_epochs = list(epochs[winner].values())
    final_epoch_count = max(1, min(20, round(statistics.median(winner_epochs)))) if winner_epochs else None
    winner_config_hashes = {
        f"seed_{seed}_fold_{fold}": str(row["config_hash"])
        for row in rows
        if str(row.get("protocol_id")) == str(winner)
        for seed, fold in [(int(row["seed"]), int(row["fold"]))]
        if row.get("config_hash")
    }
    payload = {
        "schema_version": "daic_comprehensive_selection.v2",
        "winner": winner,
        "leading_joint": joint[0]["protocol_id"],
        "leading_independent": independent[0]["protocol_id"],
        "aggregation_view": "fixed15" if str(winner).startswith("j") else "all",
        "final_epoch_count": final_epoch_count,
        "winner_protocol": (spec.get("protocols") or {}).get(winner),
        "config_hashes": dict(sorted(winner_config_hashes.items())),
        "ranking": ranked,
        "disqualified": [row for row in summaries if not row["passing"]],
        "rule": "mean macro-F1; AUROC within 0.01; positive F1; lower GPU-hours",
        "expected_subject_count": len(expected_subjects),
        "expected_seeds": sorted(seeds),
        "expected_folds": sorted(folds),
        "audit_failures": sorted(set(failures)),
    }
    payload["selection_hash"] = _selection_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True, help="JSONL with protocol_id, seed, fold, subject_id, label, prediction, score_margin.")
    parser.add_argument("--expected-subjects", type=Path, required=True, help="JSON list of the 142 development subject IDs.")
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, help="Core/focused matrix used to verify task hashes and protocol definitions.")
    parser.add_argument("--folds", type=Path, help="Shared DAIC fold JSON used to verify identical OOF membership.")
    parser.add_argument("--require-gpu-hours", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = set(map(str, json.loads(args.expected_subjects.read_text(encoding="utf-8"))))
    spec = __import__("yaml").safe_load(args.protocol_spec.read_text(encoding="utf-8"))
    expected_impl = None
    expected_configs: dict[tuple[str, int, int], str] | None = None
    if args.matrix:
        matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
        expected_impl = str(matrix.get("implementation_hash", "")) or None
        expected_configs = {
            (str(task["protocol_id"]), int(task["seed"]), int(task["fold"])): str(task["config_hash"])
            for task in matrix.get("tasks", [])
            if task.get("kind") == "evaluation"
        }
        matrix_protocols = {
            str(task["protocol_id"]): {
                "base_config": task["base_config"],
                "overrides": task.get("overrides", {}),
                "evaluation_views": task.get("views", []),
            }
            for task in matrix.get("tasks", [])
            if task.get("kind") == "evaluation"
        }
        spec = dict(spec or {})
        spec["protocols"] = {**(spec.get("protocols") or {}), **matrix_protocols}
    expected_fold_subjects = None
    if args.folds:
        fold_payload = json.loads(args.folds.read_text(encoding="utf-8"))
        expected_fold_subjects = {
            int(fold): {str(subject_id) for subject_id in payload.get("final_eval_subject_ids", [])}
            for fold, payload in fold_payload.items()
        }
    payload = build_selection(
        rows,
        expected,
        spec,
        expected_fold_subjects=expected_fold_subjects,
        expected_implementation_hash=expected_impl,
        expected_config_hashes=expected_configs,
        require_gpu_hours=args.require_gpu_hours or args.matrix is not None,
    )
    payload["protocol_spec_sha256"] = hashlib.sha256(args.protocol_spec.read_bytes()).hexdigest()
    payload["implementation_commit"] = implementation_commit()
    payload["selection_hash"] = _selection_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"winner": payload["winner"], "output": str(args.output), "eligible": len(payload["ranking"])}, sort_keys=True))


if __name__ == "__main__":
    main()
