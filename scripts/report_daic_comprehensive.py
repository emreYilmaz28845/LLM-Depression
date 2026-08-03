from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_FILES = (
    "summary_qwen.csv", "summary_heads.csv", "summary_evaluation_views.csv",
    "summary_k_weighting_mil.csv", "summary_robustness.csv", "summary_final_test.csv",
    "resource_accounting.csv",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "no_passing_artifacts"}])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    audit = json.loads(args.audit.read_text())
    output = args.output_dir or args.matrix.parent
    output.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, Any]] = []
    for path in sorted(args.artifact_root.rglob("metrics*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        metrics_rows.append({
            "artifact": str(path.relative_to(args.artifact_root)),
            **{key: payload.get(key) for key in ("macro_f1", "positive_f1", "auroc", "accuracy", "num_subjects")},
        })
    for filename in SUMMARY_FILES:
        if filename == "summary_heads.csv":
            rows = [row for row in metrics_rows if "classical" in row["artifact"]]
        elif filename == "summary_final_test.csv":
            rows = [row for row in metrics_rows if matrix["stage"] == "final"]
        elif filename == "resource_accounting.csv":
            rows = []
        else:
            rows = metrics_rows
        write_csv(output / filename, rows)
    (output / "paired_comparisons.json").write_text(json.dumps({"status": "requires_complete_oof_predictions", "comparisons": []}, indent=2) + "\n")
    report = [
        f"# DAIC Comprehensive Chunking Report: {matrix['run_id']} ({matrix['stage']})", "",
        f"- Audit passed: `{audit.get('passed')}`", f"- Matrix tasks: `{matrix.get('task_count')}`",
        f"- Implementation commit: `{matrix.get('implementation_commit')}`", f"- Spec hash: `{matrix.get('spec_hash')}`", "",
        "## Results", "",
        f"Discovered {len(metrics_rows)} metric artifacts. The CSV files beside this report are generated from those raw artifacts.", "",
        "## Disclosure", "",
        "The official DAIC test partition was exposed in the historical July 2026 study. Any final-stage result is confirmatory with historical test exposure, not a never-seen test.", "",
        "## Audit failures", "",
        *( ["None."] if not audit.get("failures") else [f"- {item}" for item in audit["failures"]] ), "",
        "Failed, retried, or collapsed conditions remain in the audit and are not eligible for selection.",
    ]
    (output / "experiment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "metrics": len(metrics_rows), "audit_passed": audit.get("passed")}, sort_keys=True))


if __name__ == "__main__":
    main()
