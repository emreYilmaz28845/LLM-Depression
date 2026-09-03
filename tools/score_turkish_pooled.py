#!/usr/bin/env python3
"""Three-way pooled scoring for Turkish pos_only + negative_only runs.

Locked plan: docs/TURKISH_POOLED_TRAINING_PLAN.md Step 7.
- Run evaluate.py once per checkpoint on the pooled manifest (per-sample
  teacher-forced predictions do not interact across windows), then aggregate
  subject-level metrics over pos_only rows / negative_only rows / all rows.
- Audio (multi-window) piles replicate evaluate.py response_subject +
  hierarchical-mean semantics: subject prediction = sign of summed
  score margins (>0 Depressed, <0 Non-depressed, ==0 INVALID).
- Text-only pooled pairs use the locked pair rule: pair_margin =
  mean(pos_margin, neg_margin); Depressed iff pair_margin >= 0; any INVALID
  in the pair makes the subject wrong. Per-pile text uses decoded labels.
- INVALID counts as wrong everywhere (binary_strict spirit); invalid rates
  are reported alongside, never hidden.
- This module leaves the locked report module untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import classification_metrics

POS_VARIANT = "pos_only_t17"
NEG_VARIANT = "negative_only_t17"
INVALID = -1


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_prediction(row: dict) -> int:
    for key in ("teacher_forced_prediction", "prediction", "parsed_prediction"):
        value = row.get(key)
        if value in (0, 1, "0", "1"):
            return int(value)
        if isinstance(value, str) and value.strip().upper() == "INVALID":
            return INVALID
    text_keys = ("teacher_forced_prediction_text", "prediction_text")
    for key in text_keys:
        value = row.get(key)
        if isinstance(value, str):
            upper = value.strip().upper()
            if upper == "INVALID":
                return INVALID
    return INVALID


def _parse_margin(row: dict) -> float:
    for key in ("score_margin", "teacher_forced_margin"):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    try:
        return float(row.get("dep_score", 0.0)) - float(row.get("non_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _strict_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Macro-F1 etc. with INVALID mapped to a wrong vote per gold label."""
    strict_pred = [
        pred if pred in (0, 1) else 1 - gold for gold, pred in zip(y_true, y_pred)
    ]
    metrics = classification_metrics(y_true, strict_pred)
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "positive_f1": float(metrics["positive_f1"]),
        "accuracy": float(metrics["accuracy"]),
        "num_subjects": len(y_true),
        "num_invalid_subjects": sum(1 for pred in y_pred if pred not in (0, 1)),
    }


def score_piles(
    sample_rows: list[dict],
    manifest_condition: dict[str, str] | None = None,
    *,
    text_pair_mode: bool = False,
) -> dict:
    """Aggregate per-sample rows into pos / neg / pooled subject metrics.

    manifest_condition maps sample_id -> dataset_variant and is used when a
    prediction row lacks its own dataset_variant column.
    """
    by_subject_pile: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in sample_rows:
        subject = str(row["subject_id"])
        condition = str(row.get("dataset_variant", "") or "").strip()
        if not condition and manifest_condition is not None:
            condition = manifest_condition.get(str(row.get("sample_id", "")), "")
        if condition not in (POS_VARIANT, NEG_VARIANT):
            raise ValueError(
                f"sample {row.get('sample_id')!r} has unknown condition {condition!r}; "
                "expected pos_only_t17 / negative_only_t17"
            )
        by_subject_pile[subject][condition].append(row)

    result: dict = {}
    for pile in (POS_VARIANT, NEG_VARIANT, "pooled"):
        y_true: list[int] = []
        y_pred: list[int] = []
        invalid_subjects = 0
        for subject in sorted(by_subject_pile):
            piles = by_subject_pile[subject]
            if pile == "pooled":
                rows = piles.get(POS_VARIANT, []) + piles.get(NEG_VARIANT, [])
            else:
                rows = piles.get(pile, [])
            if not rows:
                continue
            golds = {int(row["label"]) for row in rows}
            if len(golds) != 1:
                raise ValueError(f"subject {subject} has conflicting labels: {golds}")
            gold = next(iter(golds))
            invalid_here = sum(1 for row in rows if _parse_prediction(row) not in (0, 1))
            if text_pair_mode and pile == "pooled":
                pos_rows = piles.get(POS_VARIANT, [])
                neg_rows = piles.get(NEG_VARIANT, [])
                if not pos_rows or not neg_rows:
                    raise ValueError(
                        f"text pair mode needs both conditions for subject {subject}"
                    )
                pos_margin = sum(_parse_margin(row) for row in pos_rows) / len(pos_rows)
                neg_margin = sum(_parse_margin(row) for row in neg_rows) / len(neg_rows)
                pair_margin = (pos_margin + neg_margin) / 2.0
                if invalid_here:
                    pred = INVALID
                else:
                    pred = 1 if pair_margin >= 0 else 0
            elif text_pair_mode:
                preds = [_parse_prediction(row) for row in rows]
                valid = [pred for pred in preds if pred in (0, 1)]
                if invalid_here or not valid:
                    pred = INVALID
                elif len(set(valid)) == 1:
                    pred = valid[0]
                else:
                    margin_sum = sum(_parse_margin(row) for row in rows)
                    pred = 1 if margin_sum >= 0 else 0
            else:
                margin_sum = sum(_parse_margin(row) for row in rows)
                if margin_sum > 0:
                    pred = 1
                elif margin_sum < 0:
                    pred = 0
                else:
                    pred = INVALID
            y_true.append(gold)
            y_pred.append(pred)
            invalid_subjects += 1 if pred not in (0, 1) else 0
        if not y_true:
            raise ValueError(f"pile {pile} has no subjects")
        metrics = _strict_metrics(y_true, y_pred)
        metrics["invalid_subject_rate"] = invalid_subjects / len(y_true)
        result[pile] = metrics
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="per-sample predictions CSV")
    parser.add_argument("--manifest", default=None, help="pooled turkish_manifest.jsonl (condition fallback)")
    parser.add_argument("--text-pair-mode", action="store_true",
                        help="use the locked text pair rule for the pooled pile")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        sample_rows = [dict(row) for row in csv.DictReader(handle)]
    if not sample_rows:
        raise ValueError(f"no prediction rows in {predictions_path}")

    manifest_condition: dict[str, str] | None = None
    manifest_sha = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_condition = {}
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                manifest_condition[str(row["sample_id"])] = str(
                    row.get("dataset_variant", "")
                ).strip()
        manifest_sha = _sha(manifest_path)

    piles = score_piles(sample_rows, manifest_condition, text_pair_mode=args.text_pair_mode)
    report = {
        "schema": "audiollm.turkish_pooled_score.v1",
        "predictions_path": str(predictions_path),
        "predictions_sha256": _sha(predictions_path),
        "manifest_path": args.manifest,
        "manifest_sha256": manifest_sha,
        "text_pair_mode": bool(args.text_pair_mode),
        "pair_rule": (
            "pair_margin=mean(pos_margin,neg_margin); Depressed iff pair_margin>=0; "
            "any INVALID in pair counts wrong"
            if args.text_pair_mode
            else "mean score-margin sign per pile (>0 Depressed, <0 Non-depressed, ==0 INVALID)"
        ),
        "piles": piles,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    pooled = piles["pooled"]
    print(
        f"pooled macro_f1={pooled['macro_f1']:.4f} pos macro_f1={piles[POS_VARIANT]['macro_f1']:.4f} "
        f"neg macro_f1={piles[NEG_VARIANT]['macro_f1']:.4f} -> {output_path}"
    )


if __name__ == "__main__":
    main()
