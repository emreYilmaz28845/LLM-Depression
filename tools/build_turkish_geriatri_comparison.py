#!/usr/bin/env python3
"""Build the reader-facing Turkish four-source comparison tables.

Inputs are the derivation audits of the three matched cells and, when available,
the prespecified significance report. The tables keep three populations apart:

* the matched original participants, the only population both arms share;
* the four-source arm's own full evaluation;
* the four-source arm's geriatri-only subgroup.

The pooled out-of-fold significance view is a different aggregation from the
fold-mean headline cells and is only ever reported in its own table, with the
Holm-corrected columns separated from the uncorrected ones.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS = ("macro_f1", "positive_f1", "uar")
METRIC_LABELS = {"macro_f1": "Macro-F1", "positive_f1": "Positive-F1", "uar": "UAR"}
CELL_LABELS = {
    "qwen38_text_only": ("Qwen3.8-27B", "text_only"),
    "qwen3omni_audio_only": ("Qwen3-Omni-30B-A3B Thinker", "audio_only"),
    "qwen3omni_audio_text": ("Qwen3-Omni-30B-A3B Thinker", "audio_text"),
}
POPULATION_LABELS = {
    "original": "original participants",
    "full": "full four-source cohort",
    "geriatri": "geriatri subgroup",
}


class ComparisonError(RuntimeError):
    """Raised when the comparison inputs are incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"cannot read {path}: {exc}") from exc


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def fold_mean(audit: dict[str, Any], arm: str, population: str) -> dict[str, float] | None:
    values = audit["arms"][arm]["fold_mean_metrics"].get(population)
    if not values:
        return None
    missing = [metric for metric in METRICS if values.get(metric) is None]
    if missing:
        raise ComparisonError(f"{audit['cell']} {arm} {population} is missing {missing}")
    return {metric: float(values[metric]) for metric in METRICS}


def primary_table(audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for audit in audits:
        cell = audit["cell"]
        model, modality = CELL_LABELS.get(cell, (cell, cell))
        baseline = fold_mean(audit, "baseline", "original")
        treatment = fold_mean(audit, "treatment", "original")
        if baseline is None or treatment is None:
            raise ComparisonError(f"{cell}: the matched original comparison is incomplete")
        subject_count = int(audit["matched_original_population"]["subject_count"])
        rows.append(
            {
                "model": model,
                "modality": modality,
                "cell": cell,
                "baseline": baseline,
                "treatment": treatment,
                "delta_macro_f1": treatment["macro_f1"] - baseline["macro_f1"],
                "delta_positive_f1": treatment["positive_f1"] - baseline["positive_f1"],
                "delta_uar": treatment["uar"] - baseline["uar"],
                "subject_count": subject_count,
                "macro_f1_winner": (
                    "baseline" if baseline["macro_f1"] >= treatment["macro_f1"] else "four_source"
                ),
            }
        )
    return rows


def secondary_table(audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for audit in audits:
        cell = audit["cell"]
        model, modality = CELL_LABELS.get(cell, (cell, cell))
        rows.append(
            {
                "model": model,
                "modality": modality,
                "cell": cell,
                "full": fold_mean(audit, "treatment", "full"),
                "geriatri": fold_mean(audit, "treatment", "geriatri"),
                "original": fold_mean(audit, "treatment", "original"),
                "counts": {
                    population: int(audit["populations"][population]["treatment_subjects"])
                    for population in ("full", "geriatri", "original")
                },
            }
        )
    return rows


def significance_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for block in report.get("blocks", []):
        for comparison in block.get("comparisons", []):
            for metric in report.get("metrics", []):
                values = comparison.get("metrics", {}).get(metric)
                if not values:
                    continue
                permutation = values.get("permutation", {})
                bootstrap = values.get("bootstrap", {})
                mcnemar = comparison.get("mcnemar", {})
                rows.append(
                    {
                        "comparison": comparison["id"],
                        "family": comparison.get("correction_family", block.get("id")),
                        "n_subjects": comparison.get("n_subjects"),
                        "metric": METRIC_LABELS.get(metric, metric),
                        "delta": permutation.get("observed_delta"),
                        "ci_low": bootstrap.get("ci_low"),
                        "ci_high": bootstrap.get("ci_high"),
                        "p_uncorrected": permutation.get("p_value"),
                        "p_holm_primary_family": permutation.get("p_value_holm_primary_family"),
                        "p_holm_metric_block": permutation.get("p_value_holm_metric_block"),
                        "mcnemar_b": mcnemar.get("b"),
                        "mcnemar_c": mcnemar.get("c"),
                        "mcnemar_p": mcnemar.get("p_value"),
                        "mcnemar_p_holm": mcnemar.get("p_value_holm_primary_family"),
                    }
                )
    return rows


def markdown_report(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    significance: list[dict[str, Any]],
    significance_provenance: dict[str, Any] | None,
    extra_provenance: dict[str, Any],
) -> str:
    lines = [
        "# Turkish four-source versus pooled comparison",
        "",
        "Both arms ran the same five locked patient folds and the same recipe; only the training data "
        "changed (pooled 120 participants versus four-source 222 participants). Every number is a "
        "selected-validation five-fold mean from the Turkish train_val protocol, not a held-out test.",
        "",
        "## Primary comparison: the same original participants",
        "",
        "Rows are model and modality; both arms are scored on the original participants they share. "
        "The condition cell holding the higher Macro-F1 in a row is shown in bold. UAR is the "
        "unweighted average recall (balanced accuracy), with an invalid prediction counted as wrong.",
        "",
        "| Model | Modality | n | Pooled 120: Macro-F1 / Positive-F1 | Pooled UAR | Four-source 120: Macro-F1 / Positive-F1 | Four-source UAR | Δ Macro-F1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in primary:
        baseline = row["baseline"]
        treatment = row["treatment"]
        baseline_cell = f"{fmt(baseline['macro_f1'])} / {fmt(baseline['positive_f1'])}"
        treatment_cell = f"{fmt(treatment['macro_f1'])} / {fmt(treatment['positive_f1'])}"
        if row["macro_f1_winner"] == "baseline":
            baseline_cell = f"**{baseline_cell}**"
        else:
            treatment_cell = f"**{treatment_cell}**"
        lines.append(
            f"| {row['model']} | {row['modality']} | {row['subject_count']} | {baseline_cell} | "
            f"{fmt(baseline['uar'])} | {treatment_cell} | {fmt(treatment['uar'])} | "
            f"{row['delta_macro_f1']:+.4f} |"
        )
    lines += [
        "",
        "## Secondary populations of the four-source arm (not comparable to the pooled column)",
        "",
        "The four-source arm's own headline evaluation covers all 222 participants, and its geriatri "
        "subgroup has no pooled counterpart. These rows answer different questions and never share a "
        "comparison column with the primary table.",
        "",
        "| Model | Modality | Full 222: Macro-F1 / Positive-F1 | Full 222 UAR | Geriatri subgroup: Macro-F1 / Positive-F1 | Geriatri UAR |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in secondary:
        full = row["full"]
        geriatri = row["geriatri"]
        lines.append(
            f"| {row['model']} | {row['modality']} | "
            f"{fmt(full['macro_f1']) if full else 'n/a'} / {fmt(full['positive_f1']) if full else 'n/a'} | "
            f"{fmt(full['uar']) if full else 'n/a'} | "
            f"{fmt(geriatri['macro_f1']) if geriatri else 'n/a'} / "
            f"{fmt(geriatri['positive_f1']) if geriatri else 'n/a'} | "
            f"{fmt(geriatri['uar']) if geriatri else 'n/a'} |"
        )
    if significance:
        lines += [
            "",
            "## Prespecified significance view (pooled out-of-fold subject predictions)",
            "",
            "This view pools each arm's five out-of-fold subject predictions and tests the paired "
            "difference. It is a different aggregation from the fold-mean headline cells above and is "
            "reported separately. The prespecified decision is the Holm column inside the single "
            "family `turkish_four_source_vs_pooled` (Macro-F1 primary, the other metrics corrected "
            "separately with the same membership). The uncorrected p-value and its bootstrap interval "
            "are shown for transparency only and are never the decision.",
            "",
            "| Comparison | n | Metric | Δ (four-source − pooled) | 95% CI | p (uncorrected) | p (Holm, prespecified family) | McNemar b/c | McNemar p (Holm) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in significance:
            ci = (
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
                if row["ci_low"] is not None and row["ci_high"] is not None
                else "n/a"
            )
            lines.append(
                f"| {row['comparison']} | {row['n_subjects']} | {row['metric']} | "
                f"{row['delta']:+.4f} | {ci} | {row['p_uncorrected']:.6g} | "
                f"{row['p_holm_primary_family'] if row['p_holm_primary_family'] is not None else 'n/a'} | "
                f"{row['mcnemar_b']}/{row['mcnemar_c']} | "
                f"{row['mcnemar_p_holm'] if row['mcnemar_p_holm'] is not None else 'n/a'} |"
            )
        if significance_provenance:
            lines += [
                "",
                f"Significance family: `{significance_provenance.get('family_path')}` "
                f"(sha256 `{significance_provenance.get('family_sha256')}`), code commit "
                f"`{significance_provenance.get('code_commit')}`, "
                f"{significance_provenance.get('iterations')} permutation iterations, seed "
                f"{significance_provenance.get('seed')}, alpha {significance_provenance.get('alpha')}.",
            ]
    lines += ["", "## Provenance", ""]
    for key, value in extra_provenance.items():
        lines.append(f"- **{key}**: {value}")
    lines += [
        "",
        "Caveats: on the Turkish train_val protocol the outer fold is both the selection and the "
        "reported partition, so every cell is a selected-validation result. The two arms keep the same "
        "epoch count, so the four-source arm also sees more optimizer steps; that difference is part of "
        "the intervention and is not corrected. The full-222 and geriatri-only rows describe different "
        "evaluation populations from the pooled baseline and are not a comparison against it. "
        "No significance is claimed outside the prespecified family above.",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cell-audit", action="append", required=True, metavar="CELL=PATH")
    parser.add_argument("--significance-report", type=Path, default=None)
    parser.add_argument("--campaign-provenance", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audits: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    for value in args.cell_audit:
        if "=" not in value:
            raise ComparisonError(f"--cell-audit expects CELL=PATH, got {value!r}")
        _cell, raw_path = value.split("=", 1)
        path = Path(raw_path)
        audit = read_json(path)
        audits.append(audit)
        inputs[str(path)] = sha256_file(path)
    audits.sort(key=lambda item: ("qwen38" not in item["cell"], item["cell"]))

    significance_report = read_json(args.significance_report) if args.significance_report else None
    if significance_report:
        inputs[str(args.significance_report)] = sha256_file(args.significance_report)
    extra_provenance = read_json(args.campaign_provenance) if args.campaign_provenance else {}
    if args.campaign_provenance:
        inputs[str(args.campaign_provenance)] = sha256_file(args.campaign_provenance)

    primary = primary_table(audits)
    secondary = secondary_table(audits)
    significance = significance_rows(significance_report) if significance_report else []
    significance_provenance = (
        {
            "family_path": significance_report.get("family_path"),
            "family_sha256": significance_report.get("family_sha256"),
            "code_commit": significance_report.get("code_commit"),
            "iterations": significance_report.get("iterations"),
            "seed": significance_report.get("seed"),
            "alpha": significance_report.get("alpha"),
        }
        if significance_report
        else None
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(
        markdown_report(primary, secondary, significance, significance_provenance, extra_provenance),
        encoding="utf-8",
    )
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "table", "model", "modality", "population", "n_subjects",
                "macro_f1", "positive_f1", "uar",
            ]
        )
        for row in primary:
            for population, values in (("pooled_120", row["baseline"]), ("four_source_120", row["treatment"])):
                writer.writerow(
                    [
                        "primary_matched", row["model"], row["modality"], population, row["subject_count"],
                        f"{values['macro_f1']:.6f}", f"{values['positive_f1']:.6f}", f"{values['uar']:.6f}",
                    ]
                )
        for row in secondary:
            for population in ("full", "geriatri"):
                values = row[population]
                if not values:
                    continue
                writer.writerow(
                    [
                        "secondary", row["model"], row["modality"], population,
                        row["counts"][population],
                        f"{values['macro_f1']:.6f}", f"{values['positive_f1']:.6f}", f"{values['uar']:.6f}",
                    ]
                )
    provenance = {
        "schema_version": "audiollm.turkish_comparison_provenance.v1",
        "inputs": inputs,
        "cells": [audit["cell"] for audit in audits],
        "significance_present": bool(significance_report),
        "campaign": extra_provenance,
        "report_sha256": None,
    }
    report_path = output_dir / "report.md"
    provenance["report_sha256"] = sha256_file(report_path)
    provenance["comparison_csv_sha256"] = sha256_file(output_dir / "comparison.csv")
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cells": provenance["cells"], "significance": provenance["significance_present"],
                      "output_dir": str(output_dir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
