#!/usr/bin/env python3
"""Write compact pooled/fold Androids hidden-head result tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    payload = json.loads(args.acceptance.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or payload.get("mode") != "production":
        raise ValueError("The Androids compact summarizer requires a passed production acceptance audit.")
    pooled_rows: list[dict[str, Any]] = []
    for key, report in sorted(payload["pooled_results"].items()):
        metrics = report["metrics"]
        pooled_rows.append(
            {
                "dataset": "androids_interview",
                "modality": report["modality"],
                "head": report["head"],
                "accuracy": metrics["accuracy"],
                "positive_f1": metrics["positive_f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "macro_f1": metrics["macro_f1"],
                "negative_f1": metrics["negative_f1"],
                "auroc": metrics["auroc"],
                "confusion_matrix": json.dumps(metrics["confusion_matrix"], separators=(",", ":")),
                "subject_count": report["subject_count"],
                "source_commit": payload["source_commit"],
            }
        )
    fold_rows: list[dict[str, Any]] = []
    for report in payload["fold_results"]:
        metrics = report["metrics"]
        fold_rows.append(
            {
                "dataset": "androids_interview",
                "modality": report["modality"],
                "fold": report["fold"],
                "head": report["head"],
                "accuracy": metrics["accuracy"],
                "positive_f1": metrics["positive_f1"],
                "macro_f1": metrics["macro_f1"],
                "auroc": metrics["auroc"],
                "subject_count": report["subject_count"],
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pooled_results.json").write_text(
        json.dumps({"schema_version": "androids_hidden_compact.v1", "rows": pooled_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(pooled_rows, args.output_dir / "pooled_results.csv")
    _write_csv(fold_rows, args.output_dir / "fold_results.csv")
    print(json.dumps({"pooled_rows": len(pooled_rows), "fold_rows": len(fold_rows), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
