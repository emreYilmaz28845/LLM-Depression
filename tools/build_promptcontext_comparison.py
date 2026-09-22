#!/usr/bin/env python
"""Build the reader-facing comparison for the prompt-context cells.

One row per dataset and model. Every Qwen3.8 row is recomputed from that run's
own local subject-level predictions (INVALID counted as wrong) and checked
against the stored metrics; a mismatch stops the build instead of publishing a
number. Reference rows come from the canonical likelihood derivation artifact and
from the PR #259 Qwen3.8 old-prompt run, each carrying its own provenance.

Outputs (deterministic, no timestamps):
  comparison.csv   one row per dataset and model, the source table
  comparison.md    the short reader-facing report
  comparison.json  per-row provenance (paths, hashes, jobs, notes)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CANONICAL = Path(
    "/home/emre/Projects/AudioLLM/LLM-Depression/outputs/experiment_reports/"
    "likelihood_canonical_values/derived_values.json"
)
PR259_ROOT = Path(
    "/home/emre/Projects/AudioLLM/worktrees/LLM-Depression-feat-qwen38-daic-text"
)
PR259_RUN = (
    PR259_ROOT
    / "output_model/harmonized_v1_qwen38_likelihood/text_only/daic"
    / "qwen38_text_only_fold0_prod_20260921/fold_0"
)
CAMPAIGN_ROOT = PROJECT_ROOT / "output_model/promptcontext_v1_qwen38_likelihood/text_only"

COLUMNS = [
    "dataset",
    "model",
    "prompt",
    "run",
    "folds",
    "checkpoint_role",
    "backend",
    "evaluation_view",
    "aggregation",
    "macro_f1",
    "positive_f1",
    "uar",
    "evidence",
    "notes",
]

# dataset key -> (dataset dir, run name, folds, canonical reference cell selector)
CELLS = (
    ("DAIC", "daic", "qwen38_pc_daic_f0_20260922", (0,), ("DAIC", "text_only", None, None)),
    ("D3TEC", "d3tec", "qwen38_pc_d3tec_f", (0, 1, 2, 3, 4), ("D3TEC", "text_only", None, None)),
    ("Androids", "androids_interview", "qwen38_pc_androids_f", (0, 1, 2, 3, 4), ("Androids Interview", "text_only", None, None)),
    ("CMDC", "cmdc", "qwen38_pc_cmdc_f", (0, 1, 2, 3, 4), ("CMDC", "text_only", None, None)),
    ("Turkish pooled", "turkish", "qwen38_pc_turkish_f", (0, 1, 2, 3, 4), ("Turkish", "Text only", "Q02", "native")),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def strict_metrics(subject_rows: list[dict[str, str]]) -> dict[str, float]:
    """Strict subject-level metrics from local predictions.

    An invalid prediction (anything other than 0/1) counts as a wrong prediction
    for that subject's gold label, matching src/aggregate.py's strict policy.
    """
    true_positive = false_positive = true_negative = false_negative = 0
    for row in subject_rows:
        label = int(row["label"])
        raw = str(row.get("prediction", "")).strip()
        predicted = int(raw) if raw in {"0", "1"} else -1
        if label == 1:
            if predicted == 1:
                true_positive += 1
            else:
                false_negative += 1
        else:
            if predicted == 0:
                true_negative += 1
            else:
                false_positive += 1
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    negative_precision = true_negative / (true_negative + false_negative) if (true_negative + false_negative) else 0.0
    negative_recall = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else 0.0
    positive_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    negative_f1 = (
        2 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if (negative_precision + negative_recall)
        else 0.0
    )
    return {
        "macro_f1": round((positive_f1 + negative_f1) / 2.0, 6),
        "positive_f1": round(positive_f1, 6),
        "uar": round((recall + negative_recall) / 2.0, 6),
    }


# Fold-level run overrides: a fold whose first attempt failed on infrastructure
# and was completed by a bounded retry under a new run name.
FOLD_RUN_OVERRIDES: dict[str, dict[int, str]] = {
    "Androids": {0: "qwen38_pc_androids_f0b_20260922"},
}


def qwen38_row(dataset: str, dataset_dir: str, run_stem: str, folds: tuple[int, ...]) -> dict[str, Any]:
    row = {column: "" for column in COLUMNS}
    row["dataset"] = dataset
    row["model"] = "Qwen3.8-27B"
    row["prompt"] = "promptcontext_v1"
    row["folds"] = ",".join(str(fold) for fold in folds)
    row["checkpoint_role"] = "best_model"
    row["backend"] = "likelihood"
    row["evaluation_view"] = "harmonized_all_windows_full_coverage"
    row["aggregation"] = "subject_level"
    notes: list[str] = []
    per_fold: dict[str, dict[str, float]] = {}
    evidence: list[str] = []
    overrides = FOLD_RUN_OVERRIDES.get(dataset, {})
    for fold in folds:
        run_name = overrides.get(fold) or (
            f"{run_stem}{fold}_20260922" if run_stem.endswith("f") else run_stem
        )
        fold_dir = CAMPAIGN_ROOT / dataset_dir / run_name / f"fold_{fold}"
        metrics_path = fold_dir / "best_model/standalone_eval/metrics_likelihood.json"
        subjects_path = fold_dir / "best_model/standalone_eval/predictions_subject_level.csv"
        if not metrics_path.is_file() or not subjects_path.is_file():
            notes.append(f"fold {fold}: local evidence missing ({fold_dir})")
            continue
        stored = json.loads(metrics_path.read_text(encoding="utf-8"))
        recomputed = strict_metrics(read_csv_rows(subjects_path))
        for name in ("macro_f1", "positive_f1", "uar"):
            stored_value = stored.get(f"binary_strict_{name}")
            if stored_value is None or abs(float(stored_value) - recomputed[name]) > 1e-6:
                raise SystemExit(
                    f"metric mismatch for {run_name} fold {fold} {name}: "
                    f"stored={stored_value} recomputed={recomputed[name]}"
                )
        per_fold[str(fold)] = recomputed
        evidence.append(str(subjects_path))
    if not per_fold:
        row["notes"] = "; ".join(notes) or "no local evidence"
        return row
    if len(per_fold) != len(folds):
        notes.append(f"only {len(per_fold)} of {len(folds)} folds have local evidence")
    for name in ("macro_f1", "positive_f1", "uar"):
        row[name] = round(statistics.fmean(per_fold[str(fold)][name] for fold in folds if str(fold) in per_fold), 6)
    row["run"] = f"{run_stem}<fold>_20260922" if run_stem.endswith("f") else run_stem
    row["evidence"] = ";".join(evidence)
    if len(folds) > 1:
        notes.append("unweighted mean of the per-fold strict subject-level metrics")
    row["notes"] = "; ".join(notes)
    return row


def canonical_reference(
    canonical: Path,
    display: str,
    dataset: str,
    modality: str,
    pooled_cell: str | None,
    condition: str | None,
) -> dict[str, Any]:
    row = {column: "" for column in COLUMNS}
    payload = json.loads(canonical.read_text(encoding="utf-8"))
    for cell in payload["cells"]:
        if (
            cell.get("dataset") != dataset
            or cell.get("modality") != modality
            or cell.get("pooled_cell") != pooled_cell
            or cell.get("transcript_condition") != condition
        ):
            continue
        values = cell["likelihood"]
        row.update(
            {
                "dataset": display,
                "model": "Qwen2-7B-Instruct (canonical)",
                "prompt": "pre-promptcontext",
                "run": "harmonized canonical campaign",
                "backend": "likelihood",
                "evaluation_view": "harmonized_all_windows_full_coverage",
                "aggregation": f"subject_level ({cell.get('aggregation')})",
                "macro_f1": round(values["macro_f1"], 6),
                "positive_f1": round(values["positive_f1"], 6),
                "uar": round(values["uar"], 6),
                "evidence": f"{canonical} (sha256 {sha256_file(canonical)[:16]})",
                "notes": (
                    "canonical likelihood derivation from the run's saved per-subject candidate "
                    "scores; same label, subject set, folds and aggregation as the Qwen3.8 cell, "
                    "different model weights and prompt"
                ),
            }
        )
        return row
    row["dataset"] = dataset
    row["notes"] = "no canonical likelihood cell with matching qualifiers"
    return row


def pr259_row() -> dict[str, Any]:
    row = {column: "" for column in COLUMNS}
    row.update(
        {
            "dataset": "DAIC",
            "model": "Qwen3.8-27B",
            "prompt": "pre-promptcontext (PR #259)",
            "run": PR259_RUN.parent.name,
            "folds": "0",
            "checkpoint_role": "best_model",
            "backend": "likelihood",
            "evaluation_view": "harmonized_all_windows_full_coverage",
            "aggregation": "subject_level",
        }
    )
    metrics_path = PR259_RUN / "best_model/standalone_eval/metrics_likelihood.json"
    subjects_path = PR259_RUN / "best_model/standalone_eval/predictions_subject_level.csv"
    if not metrics_path.is_file() or not subjects_path.is_file():
        row["notes"] = "PR #259 local evidence is not present"
        return row
    recomputed = strict_metrics(read_csv_rows(subjects_path))
    row["macro_f1"] = recomputed["macro_f1"]
    row["positive_f1"] = recomputed["positive_f1"]
    row["uar"] = recomputed["uar"]
    row["evidence"] = str(subjects_path)
    row["notes"] = "same backend, split, evaluation view and official test endpoint as the prompt-context DAIC row"
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/prompt_context_comparison/report",
    )
    args = parser.parse_args(argv)

    if not args.canonical.is_file():
        raise SystemExit(f"canonical likelihood artifact is missing: {args.canonical}")

    rows: list[dict[str, Any]] = []
    for dataset, dataset_dir, run_stem, folds, reference in CELLS:
        rows.append(qwen38_row(dataset, dataset_dir, run_stem, folds))
        rows.append(canonical_reference(args.canonical, dataset, *reference))
        if dataset == "DAIC":
            rows.append(pr259_row())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Qwen3.8 prompt-context standalone cells",
        "",
        "| Dataset | Model | Prompt | Macro-F1 | Positive-F1 | UAR | Folds | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        score = lambda name: f"{row[name]:.4f}" if isinstance(row[name], (int, float)) else "—"
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['prompt']} | {score('macro_f1')} | "
            f"{score('positive_f1')} | {score('uar')} | {row['folds'] or '—'} | {row['notes']} |"
        )
    lines += [
        "",
        "All values are strict subject-level metrics (INVALID counted as wrong) in the likelihood",
        "backend with aggregation `subject_level`; the CV cells report the unweighted mean of their",
        "five folds and DAIC reports its single official-test fold. Outside DAIC the Qwen3.8 rows",
        "change both the model and the prompt, so a difference there is not a prompt-only effect;",
        "the DAIC block is the same-backend prompt comparison. No row is called significant.",
    ]
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "comparison.json").write_text(
        json.dumps({"schema_version": "audiollm.promptcontext_comparison.v1", "rows": rows}, indent=2, sort_keys=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {csv_path}")
    for row in rows:
        print(f"  {row['dataset']:<14} {row['model']:<28} {row['prompt']:<24} macro_f1={row['macro_f1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
