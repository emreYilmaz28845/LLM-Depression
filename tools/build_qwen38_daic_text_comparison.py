#!/usr/bin/env python
"""Build the DAIC text-only fold 0 comparison table from verified local evidence.

Rows are model x verified evaluation view on the DAIC text-only fold 0 held-out
test partition. Every value comes from the run's own local artifacts; a metric
that cannot be verified locally stays blank with its reason instead of being
inferred. Smoke runs are refused: only declared production runs enter the table.

Outputs (deterministic, no timestamps):
  comparison.csv  one row per model x evaluation view, the source table
  comparison.md   the short reader-facing report
  comparison.json per-row provenance (paths, hashes, jobs, notes)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_yaml  # noqa: E402

PROJECT_ROOT = Path("/home/emre/Projects/AudioLLM/LLM-Depression")

COLUMNS = [
    "model",
    "run_name",
    "attempt_id",
    "fold",
    "checkpoint_role",
    "selection_metric",
    "selection_metric_mode",
    "eval_backend",
    "evaluation_view",
    "aggregation",
    "macro_f1",
    "positive_f1",
    "uar",
    "jobs",
    "local_evidence",
    "notes",
]

# Declared rows. `backend` is the evaluated backend for that row; the same run may
# appear twice with different backends (the Qwen3.8 likelihood and
# original_teacher_forced views share one best_model checkpoint).
ROWS = [
    {
        "model": "Qwen2-7B-Instruct",
        "run_dir": "output_model/harmonized_v1/text_only/daic/"
        "harmonized_v1_harmonized_v1_prod_20260809T171705Z_d1e8130b_daic_text_only_r1/fold_0",
        "backend": "original_teacher_forced",
    },
    {
        "model": "Gemma 4 12B",
        "run_dir": "output_model/harmonized_v1_gemma4/text_only/daic/"
        "gemma4_harmonized_v1_gemma4_v1_prod_20260812T020449Z_cca3f4ae_daic_text_only/fold_0",
        "backend": "original_teacher_forced",
    },
    {
        "model": "Qwen3.8-27B",
        "run_dir": "output_model/harmonized_v1_qwen38_likelihood/text_only/daic/"
        "qwen38_text_only_fold0_20260920/fold_0",
        "backend": "likelihood",
    },
    {
        "model": "Qwen3.8-27B",
        "run_dir": "output_model/harmonized_v1_qwen38_likelihood/text_only/daic/"
        "qwen38_text_only_fold0_20260920/fold_0",
        "backend": "original_teacher_forced",
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uar_from_confusion(confusion: list[list[int]] | None) -> float | None:
    if not confusion or len(confusion) != 2:
        return None
    true_negative, false_positive = confusion[0][0], confusion[0][1]
    false_negative, true_positive = confusion[1][0], confusion[1][1]
    negatives = true_negative + false_positive
    positives = true_positive + false_negative
    if negatives == 0 or positives == 0:
        return None
    return round((true_positive / positives + true_negative / negatives) / 2.0, 6)


def _metrics_path(run_dir: Path, backend: str) -> Path:
    return run_dir / "best_model" / "standalone_eval" / f"metrics_{backend}.json"


def _job_ids(run_dir: Path) -> str:
    jobs_path = run_dir / "jobs.jsonl"
    if not jobs_path.is_file():
        return ""
    identifiers: list[str] = []
    for line in jobs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        job_id = str(event.get("slurm_job_id") or "").strip()
        if job_id and job_id not in identifiers:
            identifiers.append(job_id)
    return ",".join(identifiers)


def _row(declared: dict, project_root: Path) -> dict:
    run_dir = project_root / declared["run_dir"]
    row = {column: "" for column in COLUMNS}
    row["model"] = declared["model"]
    row["eval_backend"] = declared["backend"]
    row["run_name"] = run_dir.parent.name
    notes: list[str] = []
    if not run_dir.is_dir():
        row["notes"] = f"run directory not present: {declared['run_dir']}"
        return row
    if "smoke" in run_dir.parent.name.lower():
        raise ValueError(
            f"Refusing to include smoke run {run_dir.parent.name} in the comparison table."
        )

    run_config_path = run_dir / "run_config.yaml"
    if run_config_path.is_file():
        run_config = load_yaml(run_config_path)
        selection = run_config.get("selection_protocol", {}) or {}
        row["selection_metric"] = str(selection.get("metric_name") or "")
        row["selection_metric_mode"] = str(selection.get("metric_mode") or "")
        row["aggregation"] = str(
            (run_config.get("evaluation", {}) or {}).get("aggregation_level")
            or run_config.get("final_eval_protocol", {}).get("final_eval_aggregation_level")
            or ""
        )
        row["fold"] = str(run_config.get("fold", ""))
        tracking = run_config.get("tracking", {}) or {}
        row["attempt_id"] = str(tracking.get("attempt_id") or "")
    else:
        notes.append("run_config.yaml missing")

    evaluations_path = run_dir / "evaluations.json"
    evaluation_view = ""
    if evaluations_path.is_file():
        payload = json.loads(evaluations_path.read_text(encoding="utf-8"))
        for evaluation in payload.get("evaluations", []):
            if str(evaluation.get("backend")) != declared["backend"]:
                continue
            evaluation_view = str(evaluation.get("evaluation_view") or "")
            row["aggregation"] = row["aggregation"] or str(evaluation.get("aggregation") or "")
            row["checkpoint_role"] = str(evaluation.get("checkpoint_role") or "")
            if evaluation.get("locally_verified") is False:
                warnings = "; ".join(str(item) for item in evaluation.get("warnings") or [])
                notes.append(f"evaluation not locally verified: {warnings or 'no reason recorded'}")
    else:
        notes.append("evaluations.json missing")
    row["evaluation_view"] = evaluation_view
    if not evaluation_view:
        notes.append("evaluation_view not recorded in local evidence")

    metrics_path = _metrics_path(run_dir, declared["backend"])
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        macro_f1 = metrics.get("binary_strict_macro_f1", metrics.get("macro_f1"))
        positive_f1 = metrics.get("binary_strict_positive_f1", metrics.get("positive_f1"))
        row["macro_f1"] = "" if macro_f1 is None else f"{float(macro_f1):.6f}"
        row["positive_f1"] = "" if positive_f1 is None else f"{float(positive_f1):.6f}"
        uar = _uar_from_confusion(metrics.get("binary_strict_confusion_matrix"))
        recorded_uar = metrics.get("macro_recall")
        if uar is None and recorded_uar is not None:
            uar = round(float(recorded_uar), 6)
            notes.append("uar taken from macro_recall (confusion matrix unavailable)")
        elif uar is not None and recorded_uar is not None and abs(uar - float(recorded_uar)) > 1e-6:
            notes.append(
                "uar mismatch: confusion-derived %s vs recorded macro_recall %s"
                % (uar, round(float(recorded_uar), 6))
            )
        row["uar"] = "" if uar is None else f"{uar:.6f}"
        row["local_evidence"] = str(
            (run_dir / "best_model" / "standalone_eval").relative_to(project_root)
        )
    else:
        notes.append(f"metrics_{declared['backend']}.json missing")

    row["jobs"] = _job_ids(run_dir)
    row["checkpoint_role"] = row["checkpoint_role"] or "best_model"
    row["notes"] = "; ".join(notes)
    return row


def build(project_root: Path) -> tuple[list[dict], dict]:
    rows = [_row(declared, project_root) for declared in ROWS]
    provenance = {
        "schema": "audiollm.qwen38_daic_text_comparison.v1",
        "project_root": str(project_root),
        "rows": rows,
        "sources": [
            {"run_dir": declared["run_dir"], "backend": declared["backend"]} for declared in ROWS
        ],
    }
    return rows, provenance


def _markdown(rows: list[dict]) -> str:
    header = "| Model | Eval backend | View | Aggregation | Macro-F1 | Positive-F1 | UAR | Run / attempt | Notes |"
    separator = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, separator]
    for row in rows:
        view = row["evaluation_view"] or "not recorded"
        aggregate = row["aggregation"] or ""
        run = row["run_name"] + (f" / {row['attempt_id']}" if row["attempt_id"] else "")
        lines.append(
            "| {model} | {backend} | {view} | {aggregation} | {macro} | {positive} | {uar} | {run} | {notes} |".format(
                model=row["model"],
                backend=row["eval_backend"],
                view=view,
                aggregation=aggregate,
                macro=row["macro_f1"] or "",
                positive=row["positive_f1"] or "",
                uar=row["uar"] or "",
                run=run,
                notes=row["notes"],
            )
        )
    body = "\n".join(lines)
    return (
        "# DAIC text-only fold 0: model comparison\n\n"
        "One row per model per verified DAIC text-only fold 0 evaluation view. Values come from each\n"
        "run's `best_model/standalone_eval` artifacts; blanks mean the value is not recorded in local\n"
        "evidence. The comparison is descriptive: checkpoint-selection and evaluation backends differ\n"
        "between rows, so it carries no winner claim.\n\n"
        f"{body}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--output-dir", default="outputs/qwen38_daic_text_comparison/report"
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    rows, provenance = build(project_root)
    output_dir = project_root / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "comparison.md").write_text(_markdown(rows), encoding="utf-8")
    (output_dir / "comparison.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {csv_path}, {output_dir / 'comparison.md'}, {output_dir / 'comparison.json'}")
    for row in rows:
        print(f"  {row['model']:<18} {row['eval_backend']:<24} macro_f1={row['macro_f1'] or '-':<10} notes={row['notes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
