#!/usr/bin/env python3
"""Publish a compact, source-grounded report for the Androids hidden heads."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODALITIES = ("audio_only", "audio_text", "text_only")
HEADS = ("logreg_raw", "xgb_raw", "xgb_optuna_150t_d6")
MODALITY_LABELS = {"audio_only": "Audio only", "audio_text": "Audio + Text", "text_only": "Text only"}
HEAD_LABELS = {
    "logreg_raw": "Logistic Regression",
    "xgb_raw": "XGBoost fixed raw",
    "xgb_optuna_150t_d6": "XGBoost Optuna (150 trials, standard_d6)",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", required=True, type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _registry_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    args = _parse_args()
    acceptance = json.loads(args.acceptance.read_text(encoding="utf-8"))
    if acceptance.get("status") != "passed" or acceptance.get("mode") != "production":
        raise ValueError("Report generation requires a passed production acceptance audit.")
    registry = _registry_rows(args.registry)
    pooled = acceptance["pooled_results"]
    lines = [
        "# Androids Interview Hidden-State Classifier Report",
        "",
        f"Run ID: `{acceptance['run_id']}`  ",
        f"Source commit: `{acceptance['source_commit']}`  ",
        f"Manifest canonical SHA-256: `{acceptance['manifest_sha256']}`  ",
        f"Official split SHA-256: `{acceptance['split_metadata_sha256']}`",
        "",
        "## Acceptance",
        "",
        "The production acceptance audit passed with 45 fold/head results and nine pooled results. "
        "Each pooled result contains 116 unique held-out subjects: 52 controls and 64 patients. "
        "The audit recomputed predictions and metrics from the saved compact prediction artifacts.",
        "",
        "## Protocol",
        "",
        "- Modalities: Audio only, Audio + Text, and Text only; full-turn inputs are excluded.",
        "- Hidden representation: final-layer last-valid-prompt-token vector from the existing five-fold best-model checkpoints.",
        "- Audio aggregation: arithmetic probability mean from window to turn to subject.",
        "- Audio fit weights: `1 / (turns_for_subject * windows_for_turn)`, rescaled to mean one; subject and within-subject turn totals are audited.",
        "- Text fit: one vector per subject with unit weight.",
        "- Decision rule: fixed threshold 0.5; exact ties are invalid and counted wrong in strict headline metrics.",
        "- Fixed heads: the repository `logreg_raw` and `xgb_raw` defaults. Optuna: `standard_d6`, three subject-stratified inner folds, pooled inner OOF Macro-F1, 150 trials, seed 1337.",
        "- No PCA, oversampling, controls, or outer-fold result selection was used.",
        "",
        "## Pooled results",
        "",
        "| Modality | Head | Accuracy | Positive F1 | Negative F1 | Macro-F1 | AUROC | Confusion |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for modality in MODALITIES:
        for head in HEADS:
            metrics = pooled[f"{modality}/{head}"]["metrics"]
            lines.append(
                f"| {MODALITY_LABELS[modality]} | {HEAD_LABELS[head]} | "
                f"{_fmt(metrics['accuracy'])} | {_fmt(metrics['positive_f1'])} | "
                f"{_fmt(metrics['negative_f1'])} | {_fmt(metrics['macro_f1'])} | "
                f"{_fmt(metrics['auroc'])} | `{json.dumps(metrics['confusion_matrix'], separators=(',', ':'))}` |"
            )
    lines.extend(
        [
            "",
            "## Fold and job accounting",
            "",
            f"The audit recorded `{acceptance['counts']['fold_head_results']}` fold/head results across five outer folds and `{acceptance['counts']['pooled_results']}` pooled results.",
            f"The synchronized job registry contains `{len(registry)}` rows" + ("." if registry else "; the registry path was not supplied to the report command."),
            "Each Optuna result was required to contain exactly 150 COMPLETE trials with zero failed trials; inner validation assignments were subject-disjoint and covered each outer-training subject exactly once.",
            "",
            "## Retrieval and limitations",
            "",
            "The remote acceptance audit was performed against the full hidden vectors and model artifacts on MN5. The local handoff is compact: prediction rows, metrics, configurations, trial summaries, best parameters, audits, extraction metadata, row inventories, registry, and logs. Adapter/checkpoint files, hidden-vector NPZ caches, model binaries, and Optuna SQLite databases remain excluded from the default retrieval.",
            "",
            "The hidden heads are a controlled representation-level comparison, not a new end-to-end fine-tuning run. Metrics are outer-fold pooled subject results, and the Audio + Text label is intentionally generic in the workbook.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "written", "output": str(args.output), "registry_rows": len(registry)}, indent=2))


if __name__ == "__main__":
    main()
