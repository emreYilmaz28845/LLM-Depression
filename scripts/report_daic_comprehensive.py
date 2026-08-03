from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.daic_statistics import exact_mcnemar, holm_adjust, stratified_paired_bootstrap
from src.metrics import binary_auroc, classification_metrics


SUMMARY_FILES = (
    "summary_qwen.csv",
    "summary_heads.csv",
    "summary_evaluation_views.csv",
    "summary_k_weighting_mil.csv",
    "summary_robustness.csv",
    "summary_final_test.csv",
    "resource_accounting.csv",
)
SEED_RE = re.compile(r"^seed_(\d+)$")
FOLD_RE = re.compile(r"^fold_(\d+)$")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "no_artifacts"}])


def _context(path: Path, artifact_root: Path) -> dict[str, Any]:
    parts = path.relative_to(artifact_root).parts
    result: dict[str, Any] = {
        "artifact": str(path.relative_to(artifact_root)),
        "protocol_id": parts[0] if parts else "",
        "seed": "",
        "fold": "",
        "phase": parts[3] if len(parts) > 3 else "",
        "view": parts[4] if len(parts) > 4 else "",
        "head": parts[5] if len(parts) > 5 else "",
    }
    if len(parts) > 1:
        seed_match = SEED_RE.fullmatch(parts[1])
        if seed_match:
            result["seed"] = int(seed_match.group(1))
    if len(parts) > 2:
        fold_match = FOLD_RE.fullmatch(parts[2])
        if fold_match:
            result["fold"] = int(fold_match.group(1))
    return result


def collect_metric_rows(artifact_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qwen_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    for path in sorted(artifact_root.rglob("metrics*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        context = _context(path, artifact_root)
        row = {
            **context,
            "metric_file": path.name,
            "macro_f1": payload.get("macro_f1"),
            "positive_f1": payload.get("positive_f1"),
            "auroc": payload.get("auroc"),
            "accuracy": payload.get("accuracy"),
            "num_subjects": payload.get("num_subjects", payload.get("num_valid_subject_predictions")),
            "prediction_backend": payload.get("prediction_backend"),
            "aggregation_method": payload.get("aggregation_method"),
        }
        if "classical" in path.parts:
            head_rows.append(row)
        else:
            qwen_rows.append(row)
    return qwen_rows, head_rows


def _prediction_groups(artifact_root: Path) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for path in sorted(artifact_root.rglob("predictions_subject_level.jsonl")):
        context = _context(path, artifact_root)
        key = (str(context["protocol_id"]), str(context["seed"]), str(context["fold"]), str(context["view"]))
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if {"subject_id", "label", "prediction"} <= set(row):
                rows.append(row)
        if rows:
            groups[key] = rows
    return groups


def _ensemble_prediction_groups(artifact_root: Path) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Concatenate disjoint outer-fold subject rows into one seed ensemble."""
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for (protocol, seed, _fold, view), rows in _prediction_groups(artifact_root).items():
        target = grouped.setdefault((protocol, seed, view), {})
        for row in rows:
            subject_id = str(row["subject_id"])
            if subject_id in target:
                # Duplicate subjects are retained as an invalid comparison input;
                # exact_mcnemar/bootstrap will reject the duplicate key upstream.
                target[f"{subject_id}#duplicate"] = row
            else:
                target[subject_id] = {**row, "seed": int(seed)}
    return {key: list(rows.values()) for key, rows in grouped.items()}


def paired_comparisons(artifact_root: Path) -> dict[str, Any]:
    groups = _ensemble_prediction_groups(artifact_root)
    by_scope: dict[tuple[str, str], list[tuple[str, list[dict[str, Any]]]]] = {}
    for (protocol, seed, view), rows in groups.items():
        by_scope.setdefault((seed, view), []).append((protocol, rows))
    comparisons: list[dict[str, Any]] = []
    bootstrap_results: list[dict[str, Any]] = []
    p_values_by_view: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for (seed, view), candidates in sorted(by_scope.items()):
        candidates = sorted(candidates)
        for index, (left_name, left_rows) in enumerate(candidates):
            for right_name, right_rows in candidates[index + 1 :]:
                try:
                    result = exact_mcnemar(left_rows, right_rows)
                except ValueError:
                    continue
                result.update({"seed": seed, "fold": "ensemble", "view": view, "baseline": left_name, "comparison": right_name})
                comparisons.append(result)
                p_values_by_view.setdefault(view, []).append((result, float(result["p_value"])))
                try:
                    bootstrap = stratified_paired_bootstrap(
                        left_rows,
                        right_rows,
                        metric="macro_f1",
                        iterations=10000,
                        seed=1337,
                    )
                    bootstrap.update({"seed": seed, "view": view, "baseline": left_name, "comparison": right_name})
                    bootstrap_results.append(bootstrap)
                except ValueError:
                    # Incomplete or malformed paired artifacts remain visible in
                    # the raw audit; they are not converted into a fabricated CI.
                    continue
    for view, entries in p_values_by_view.items():
        adjusted = holm_adjust([value for _, value in entries])
        for (row, _), adjusted_value in zip(entries, adjusted):
            row["holm_adjusted_p_value"] = adjusted_value
    return {
        "method": "exact_mcnemar_on_seed_ensemble",
        "correction": "holm_within_view",
        "comparisons": comparisons,
        "bootstrap": bootstrap_results,
        "bootstrap_iterations": 10000,
        "bootstrap_seed": 1337,
    }


def resource_rows(accounting_path: Path | None) -> list[dict[str, Any]]:
    if accounting_path is None or not accounting_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in accounting_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def final_test_summary_rows(artifact_root: Path, metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add seed mean/std and a predeclared mean-margin ensemble row."""
    final_rows = [
        row for row in metric_rows
        if row.get("phase") == "evaluation"
        and row.get("metric_file") == "metrics_original_teacher_forced.json"
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in final_rows:
        if isinstance(row.get("seed"), int):
            grouped.setdefault((str(row.get("protocol_id")), str(row.get("view"))), []).append(row)
    output = list(final_rows)
    for (protocol, view), rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        summary: dict[str, Any] = {"protocol_id": protocol, "seed": "mean", "fold": "0", "phase": "evaluation", "view": view, "aggregation_method": "seed_mean_std"}
        for metric in ("macro_f1", "positive_f1", "auroc", "accuracy"):
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            if values:
                summary[metric] = statistics.mean(values)
                summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(summary)

    ensembles = _ensemble_prediction_groups(artifact_root)
    by_protocol_view: dict[tuple[str, str], list[list[dict[str, Any]]]] = {}
    for (protocol, seed, view), rows in ensembles.items():
        by_protocol_view.setdefault((protocol, view), []).append(rows)
    for (protocol, view), seed_rows in sorted(by_protocol_view.items()):
        if len(seed_rows) < 2:
            continue
        per_subject: dict[str, list[dict[str, Any]]] = {}
        for rows in seed_rows:
            for row in rows:
                per_subject.setdefault(str(row["subject_id"]), []).append(row)
        ensemble_rows: list[dict[str, Any]] = []
        for subject_id, rows in sorted(per_subject.items()):
            labels = {int(row["label"]) for row in rows}
            if len(labels) != 1 or len(rows) != len(seed_rows):
                ensemble_rows = []
                break
            margins = [
                float(row.get("score_margin", float(row.get("dep_score", 0.0)) - float(row.get("non_score", 0.0))))
                for row in rows
            ]
            margin = statistics.mean(margins)
            ensemble_rows.append({"subject_id": subject_id, "label": next(iter(labels)), "prediction": int(margin > 0.0), "score_margin": margin})
        if not ensemble_rows:
            continue
        labels = [int(row["label"]) for row in ensemble_rows]
        predictions = [int(row["prediction"]) for row in ensemble_rows]
        margins = [float(row["score_margin"]) for row in ensemble_rows]
        metrics = classification_metrics(labels, predictions)
        output.append({
            "protocol_id": protocol,
            "seed": "ensemble",
            "fold": "0",
            "phase": "evaluation",
            "view": view,
            "aggregation_method": "three_seed_mean_margin_ensemble",
            "macro_f1": metrics["macro_f1"],
            "positive_f1": metrics["positive_f1"],
            "auroc": binary_auroc(labels, margins),
            "accuracy": metrics["accuracy"],
            "num_subjects": len(ensemble_rows),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--slurm-accounting", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    output = args.output_dir or args.matrix.parent
    output.mkdir(parents=True, exist_ok=True)
    qwen_rows, head_rows = collect_metric_rows(args.artifact_root)
    all_metric_rows = qwen_rows + head_rows
    write_csv(output / "summary_qwen.csv", qwen_rows)
    write_csv(output / "summary_heads.csv", head_rows)
    write_csv(output / "summary_evaluation_views.csv", qwen_rows)
    followup_rows = [
        row for row in qwen_rows
        if any(token in str(row.get("protocol_id", "")).lower() for token in ("_k", "balanced", "mil"))
    ]
    write_csv(output / "summary_k_weighting_mil.csv", followup_rows)
    write_csv(output / "summary_robustness.csv", all_metric_rows)
    final_rows = [row for row in all_metric_rows if str(matrix.get("stage")) == "final" or "test" in str(row.get("view", "")).lower()]
    if str(matrix.get("stage")) == "final":
        final_rows = final_test_summary_rows(args.artifact_root, final_rows)
    write_csv(output / "summary_final_test.csv", final_rows)
    resources = resource_rows(args.slurm_accounting)
    write_csv(output / "resource_accounting.csv", resources)
    comparisons = paired_comparisons(args.artifact_root)
    (output / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    selection = {}
    if args.selection and args.selection.exists():
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
    report = [
        f"# DAIC Comprehensive Chunking Report: {matrix.get('run_id')} ({matrix.get('stage')})",
        "",
        "## Dataset and leakage audit",
        "",
        f"- Audit passed: `{audit.get('passed')}`",
        f"- Matrix tasks: `{matrix.get('task_count')}`",
        f"- Implementation commit: `{matrix.get('implementation_commit')}`",
        f"- Implementation hash: `{matrix.get('implementation_hash')}`",
        "- The manifest contract is subject-level: 10 chunks for non-depressed subjects and 15 for depressed subjects; development and official-test subjects are disjoint.",
        "",
        "## Protocol definitions and causal comparison map",
        "",
        "- `jr4`, `jt4`, and `ja4` are joint four-chunk prompt schedules.",
        "- `ir4`, `ian`, and `iaf` are independent chunk schedules; `iaf` is the equal-row-weight negative control.",
        "- Mean subject score margin is the operational aggregation; alternative aggregations are robustness analyses.",
        "",
        "## Results",
        "",
        f"- Qwen metric artifacts: `{len(qwen_rows)}`",
        f"- Classical-head metric artifacts: `{len(head_rows)}`",
        f"- Resource/accounting rows: `{len(resources)}`",
        "",
        "## Selection and follow-ups",
        "",
        f"- Selected winner: `{selection.get('winner', 'not supplied')}`",
        f"- Selection rule: `{selection.get('rule', 'mean macro-F1; AUROC within 0.01; positive F1; lower GPU-hours')}`",
        "- K sensitivity, class weighting, aggregation, MIL, and hidden-head results remain secondary to the predeclared Qwen development selection.",
        "",
        "## Statistical comparisons",
        "",
        "Paired comparisons are generated from subject-level prediction artifacts. Exact McNemar p-values are Holm-adjusted within each evaluation view; confidence intervals must not be interpreted as equivalence tests.",
        "",
        "## Final test disclosure",
        "",
        "The official DAIC test partition was exposed in the historical July 2026 study. Any final-stage result is confirmatory with historical test exposure, not a never-seen test.",
        "",
        "## Failures, retries, and limitations",
        "",
        *( ["None recorded by the supplied audit."] if not audit.get("failures") else [f"- {item}" for item in audit["failures"]] ),
        "",
        "All failed, retried, excluded, or collapsed conditions must remain in the audit and are ineligible for selection.",
    ]
    (output / "experiment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "qwen_metrics": len(qwen_rows), "head_metrics": len(head_rows), "audit_passed": audit.get("passed")}, sort_keys=True))


if __name__ == "__main__":
    main()
