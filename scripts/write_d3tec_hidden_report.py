#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


HEADLINE = {
    "logreg_raw": "LogReg raw",
    "xgb_raw": "XGBoost fixed raw",
    "xgb_optuna_raw_t150_d6_seed1337_inner1337": "XGBoost Optuna raw",
}


def _registry(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _metric(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def _duration(seconds: int | float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m {seconds:02d}s"


def _bytes(value: int | float) -> str:
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def write_report(
    audit_path: Path,
    stability_path: Path,
    registry_path: Path,
    accounting_path: Path,
    output_path: Path,
) -> None:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    stability = json.loads(stability_path.read_text(encoding="utf-8"))
    accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
    jobs = _registry(registry_path)
    if audit.get("status") != "passed":
        raise ValueError("Report generation requires a passed acceptance audit.")
    nonterminal = [
        row
        for row in accounting["jobs"]
        if row.get("state") != "COMPLETED" or row.get("exit_code") != "0:0"
    ]
    if nonterminal:
        raise ValueError(f"Report generation requires successful terminal jobs: {nonterminal[:3]}.")

    lines = [
        "# D3TEC Hidden-State Classifier Report — 2026-07-29",
        "",
        "## Protocol",
        "",
        "Five predeclared outer folds were evaluated for the normalized D3TEC "
        "audio+text and audio-only checkpoints and the D3TEC text-only checkpoints. "
        "Gold labels remained external to Qwen inputs. Audio segments were weighted "
        "by the inverse number of segments in their response, rescaled to mean one. "
        "Predictions were aggregated segment → response → subject with 27 equal "
        "response votes. Text-only used one vector and prediction per subject.",
        "",
        "The primary selection metric was pooled outer-fold Macro-F1. Optuna used "
        "150 `standard_d6` trials, threshold 0.5, three subject-stratified inner "
        "folds, and sampler/model seed 1337. Inner seeds 7 and 2024 were used only "
        "for stability analysis and never selected by outer performance.",
        "",
        "## Slurm execution",
        "",
        f"- Run registry: `{registry_path}`",
        f"- Registered jobs: {len(jobs)}",
        f"- Terminal accounting rows: {len(accounting['jobs'])}",
        f"- Retries: {len(accounting.get('retries', []))}",
        f"- Aggregate allocated job runtime: "
        f"{_duration(accounting.get('total_job_runtime_seconds', 0))}",
        "",
        "| Stage | Jobs | Aggregate runtime |",
        "|---|---:|---:|",
    ]
    stage_counts: dict[str, int] = {}
    for row in jobs:
        stage_counts[row["stage"]] = stage_counts.get(row["stage"], 0) + 1
    stage_runtime = accounting.get("stage_runtime_seconds", {})
    lines.extend(
        f"| {stage} | {count} | {_duration(stage_runtime.get(stage, 0))} |"
        for stage, count in sorted(stage_counts.items())
    )
    if accounting.get("storage"):
        lines.extend(["", "Remote retained artifact footprint:"])
        lines.extend(
            f"- `{name}`: {_bytes(value)}"
            for name, value in sorted(accounting["storage"].items())
        )
    source_commits = sorted({row["source_commit"] for row in jobs})
    lines.extend(
        [
            "",
            "Registered source commits: "
            + ", ".join(f"`{commit}`" for commit in source_commits)
            + ".",
        ]
    )
    lines.extend(
        [
            "",
            "## Headline pooled results",
            "",
            "| Modality | Head | Accuracy | PosF1 | Macro-F1 | Negative F1 | AUROC | Confusion matrix |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    headline_rows = [
        row for row in audit["results"] if row["variant"] in HEADLINE
    ]
    for row in sorted(headline_rows, key=lambda item: (item["modality"], item["variant"])):
        metrics = row["pooled_metrics"]
        lines.append(
            f"| {row['modality']} | {HEADLINE[row['variant']]} | "
            f"{_metric(metrics['accuracy'])} | {_metric(metrics['positive_f1'])} | "
            f"{_metric(metrics['macro_f1'])} | {_metric(metrics['negative_f1'])} | "
            f"{_metric(metrics['auroc'])} | `{json.dumps(metrics['confusion_matrix'])}` |"
        )
    lines.extend(
        [
            "",
            "## Headline fold metrics",
            "",
            "| Modality | Head | Fold | Subjects | Accuracy | PosF1 | Macro-F1 | AUROC |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(headline_rows, key=lambda item: (item["modality"], item["variant"])):
        for fold in row["folds"]:
            metrics = fold["metrics"]
            lines.append(
                f"| {row['modality']} | {HEADLINE[row['variant']]} | "
                f"{fold['fold']} | {fold['subjects']} | {_metric(metrics['accuracy'])} | "
                f"{_metric(metrics['positive_f1'])} | {_metric(metrics['macro_f1'])} | "
                f"{_metric(metrics['auroc'])} |"
            )
    lines.extend(
        [
            "",
            "## Stability",
            "",
            f"The maximum pooled Macro-F1 range across the two pilot modalities was "
            f"{stability['observed_pilot_max_range']:.6f}, against the predeclared "
            f"{stability['gate_threshold']:.2f} gate. Conditional audio-only expansion: "
            f"**{'run' if stability['expand_audio_only'] else 'not triggered'}**.",
            "",
        ]
    )
    for row in stability["stability_rows"]:
        lines.append(
            f"- `{row['condition']}`: range {_metric(row['pooled_macro_f1_range'])}; "
            + ", ".join(
                f"inner seed {item['inner_seed']}={_metric(item['pooled_macro_f1'])}"
                for item in row["seed_results"]
            )
        )
    controls = [
        row
        for row in audit["results"]
        if row["variant"] in {"majority_class", "xgb_raw_shuffled_labels"}
    ]
    lines.extend(
        [
            "",
            "## Controls and baselines",
            "",
            f"- All-positive baseline PosF1: {audit['all_positive_baseline']['fraction']} "
            f"= {_metric(audit['all_positive_baseline']['positive_f1'])}.",
            f"- Female-positive sex-rule Macro-F1: "
            f"{_metric(audit['female_positive_sex_rule_baseline']['macro_f1'])}.",
        ]
    )
    for row in sorted(controls, key=lambda item: (item["modality"], item["variant"])):
        lines.append(
            f"- `{row['modality']}` / `{row['variant']}`: Macro-F1 "
            f"{_metric(row['pooled_metrics']['macro_f1'])}."
        )
    lines.extend(
        [
            "",
            "## Gender-stratified headline errors",
            "",
            "| Modality | Head | Group | Subjects | Errors | Error rate |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(headline_rows, key=lambda item: (item["modality"], item["variant"])):
        analysis = row["gender_analysis"]
        for group in sorted(analysis["subject_counts"]):
            lines.append(
                f"| {row['modality']} | {HEADLINE[row['variant']]} | {group} | "
                f"{analysis['subject_counts'][group]} | "
                f"{analysis['error_counts'][group]} | "
                f"{_metric(analysis['error_rates'][group])} |"
            )
    lines.extend(
        [
            "",
            "## Audit and provenance",
            "",
            f"The local acceptance audit passed for {audit['result_count']} condition/head "
            f"combinations, exactly {audit['unique_subjects']} pooled held-out subjects "
            f"({audit['label_counts']}), subject-disjoint inner and outer partitions, "
            "27 response predictions per audio subject, complete response-weight audits, "
            "and the expected manifest, split, checkpoint, cache, model, prediction, "
            "metric, and provenance artifacts.",
            "",
            "Gender-stratified error counts and rates are stored per result in the "
            f"acceptance audit: `{audit_path}`.",
            "",
            "## Limitations",
            "",
            "- The panel contains 62 participants, so fold and subgroup estimates remain noisy.",
            "- Inner-seed stability does not quantify checkpoint-training seed variability.",
            "- The gender analysis is descriptive and not evidence of a causal relationship.",
            "- Hidden-state heads reuse representations from supervised LoRA checkpoints; "
            "they are downstream probes, not independently trained foundation models.",
            "",
            "## Slurm accounting appendix",
            "",
            "| Job ID | Stage | Modality | Fold | Experiment | State | Exit | Elapsed |",
            "|---:|---|---|---:|---|---|---:|---:|",
        ]
    )
    for row in accounting["jobs"]:
        lines.append(
            f"| {row['job_id']} | {row['stage']} | {row['modality']} | "
            f"{row['fold']} | `{row['experiment_id']}` | {row['state']} | "
            f"{row['exit_code']} | {row['elapsed']} |"
        )
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the audited D3TEC hidden-head report.")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--stability", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/D3TEC_HIDDEN_CLASSIFIER_REPORT_2026-07-29.md"),
    )
    args = parser.parse_args()
    write_report(
        args.audit,
        args.stability,
        args.registry,
        args.accounting,
        args.output,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
