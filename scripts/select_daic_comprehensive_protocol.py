from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from functools import cmp_to_key
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.metrics import binary_auroc, classification_metrics


SEEDS = {1337, 2027, 3407}
FOLDS = {0, 1, 2, 3, 4}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True, help="JSONL with protocol_id, seed, fold, subject_id, label, prediction, score_margin.")
    parser.add_argument("--expected-subjects", type=Path, required=True, help="JSON list of the 142 development subject IDs.")
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    expected = set(map(str, json.loads(args.expected_subjects.read_text())))
    spec = __import__("yaml").safe_load(args.protocol_spec.read_text())
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    epochs: dict[str, list[int]] = defaultdict(list)
    gpu_hours: dict[str, float] = defaultdict(float)
    for row in rows:
        grouped[(str(row["protocol_id"]), int(row["seed"]))].append(row)
        if row.get("selected_epoch") is not None:
            epochs[str(row["protocol_id"])].append(int(row["selected_epoch"]))
        gpu_hours[str(row["protocol_id"])] += float(row.get("gpu_hours", 0.0))
    summaries = []
    for protocol in sorted({key[0] for key in grouped}):
        per_seed = []
        failures = []
        for seed in sorted(SEEDS):
            cell = grouped.get((protocol, seed), [])
            ids = Counter(str(row["subject_id"]) for row in cell)
            if set(ids) != expected or any(count != 1 for count in ids.values()):
                failures.append(f"coverage_seed_{seed}"); continue
            if {int(row["fold"]) for row in cell} != FOLDS:
                failures.append(f"folds_seed_{seed}"); continue
            predictions = [int(row["prediction"]) for row in cell]
            if len(set(predictions)) == 1:
                failures.append(f"collapse_seed_{seed}"); continue
            y = [int(row["label"]) for row in cell]
            metrics = classification_metrics(y, predictions)
            metrics["auroc"] = binary_auroc(y, [float(row["score_margin"]) for row in cell])
            per_seed.append({"seed": seed, **metrics})
        passing = not failures and len(per_seed) == 3
        summaries.append({
            "protocol_id": protocol, "passing": passing, "failures": failures,
            "per_seed": per_seed,
            "mean_macro_f1": statistics.mean(row["macro_f1"] for row in per_seed) if per_seed else -1.0,
            "mean_auroc": statistics.mean(row["auroc"] for row in per_seed) if per_seed else -1.0,
            "mean_positive_f1": statistics.mean(row["positive_f1"] for row in per_seed) if per_seed else -1.0,
            "gpu_hours": gpu_hours[protocol],
        })
    passing = [row for row in summaries if row["passing"]]
    if not passing:
        raise SystemExit("No complete non-collapsed protocol is eligible.")
    def compare(left, right):
        macro_delta = left["mean_macro_f1"] - right["mean_macro_f1"]
        if abs(macro_delta) >= 0.01:
            return -1 if macro_delta > 0 else 1
        for field in ("mean_auroc", "mean_positive_f1"):
            delta = left[field] - right[field]
            if delta:
                return -1 if delta > 0 else 1
        return -1 if left["gpu_hours"] < right["gpu_hours"] else 1 if left["gpu_hours"] > right["gpu_hours"] else 0
    passing.sort(key=cmp_to_key(compare))
    joint = [row for row in passing if row["protocol_id"].startswith("j")]
    independent = [row for row in passing if not row["protocol_id"].startswith("j")]
    winner = passing[0]["protocol_id"]
    selected_epochs = epochs[winner]
    payload = {
        "schema_version": "daic_comprehensive_selection.v1",
        "winner": winner, "leading_joint": joint[0]["protocol_id"],
        "leading_independent": independent[0]["protocol_id"],
        "aggregation_view": "fixed15" if winner.startswith("j") else "all",
        "final_epoch_count": max(1, min(20, round(statistics.median(selected_epochs)))) if selected_epochs else None,
        "winner_protocol": spec["protocols"].get(winner), "ranking": passing,
        "disqualified": [row for row in summaries if not row["passing"]],
        "rule": "mean macro-F1; AUROC within 0.01; positive F1; lower GPU-hours",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"winner": winner, "output": str(args.output), "eligible": len(passing)}, sort_keys=True))


if __name__ == "__main__":
    main()
