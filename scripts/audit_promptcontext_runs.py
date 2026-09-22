#!/usr/bin/env python3
"""Audit collected prompt-context runs against the recipe's run contract.

For every fold directory it checks the recorded prompt identity (version, dataset
context, exact system-prompt hash), the Qwen3.8 training/evaluation settings, and
the standalone likelihood evaluation artifacts. For the pooled Turkish cell it
additionally proves that both question conditions reach the evaluation and that
each participant contributes exactly one positive and one negative row.

Usage:
    python scripts/audit_promptcontext_runs.py --root output_model/<campaign>/text_only
    python scripts/audit_promptcontext_runs.py --fold-dir <fold dir> [--fold-dir ...]

Exit code is non-zero when any checked fold fails, so the audit can gate reporting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROMPT_CONTEXT_VERSION = "promptcontext_v1"
DATASET_CONTEXT_KEYS = {
    "daic": "daic",
    "d3tec": "d3tec",
    "androids_interview": "androids",
    "cmdc": "cmdc",
    "turkish": "turkish_pooled",
}
POOLED_CONDITIONS = ("pos_only_t17", "negative_only_t17")
REQUIRED_EVAL_FILES = (
    "metrics_likelihood.json",
    "predictions_subject_level.csv",
    "predictions_sample_level.csv",
    "confusion_matrix.json",
    "eval_config.yaml",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_fold(fold_dir: Path) -> dict[str, Any]:
    failures: list[str] = []
    notes: list[str] = []
    run_config_path = fold_dir / "run_config.yaml"
    record: dict[str, Any] = {
        "fold_dir": str(fold_dir),
        "run_name": fold_dir.parent.name,
        "fold": fold_dir.name,
        "failures": failures,
        "notes": notes,
    }
    if not run_config_path.is_file():
        failures.append("run_config.yaml is missing")
        return record
    run_config = _read_yaml(run_config_path)
    config = run_config.get("config", {})
    prompt = config.get("prompt", {}) or {}
    record["dataset"] = config.get("dataset")
    record["attempt_id"] = (run_config.get("tracking") or {}).get("attempt_id")

    if prompt.get("version") != PROMPT_CONTEXT_VERSION:
        failures.append(f"config.prompt.version is {prompt.get('version')!r}")
    if "system" in prompt:
        failures.append("config.prompt.system must not be set by the prompt-context recipe")
    expected_key = DATASET_CONTEXT_KEYS.get(str(config.get("dataset")))
    if expected_key is None:
        failures.append(f"dataset {config.get('dataset')!r} is outside the prompt-context cells")
    elif prompt.get("dataset_context") != expected_key:
        failures.append(
            f"config.prompt.dataset_context is {prompt.get('dataset_context')!r}, expected {expected_key!r}"
        )

    prompt_context = run_config.get("prompt_context") or {}
    system_prompt = prompt_context.get("system_prompt")
    if not system_prompt:
        failures.append("run_config.prompt_context.system_prompt is missing")
    else:
        digest = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        if digest != prompt_context.get("system_prompt_sha256"):
            failures.append("prompt_context.system_prompt_sha256 does not match the recorded text")
    if prompt_context.get("input_modality") != "text_only":
        failures.append(f"input modality is {prompt_context.get('input_modality')!r}, expected text_only")

    training = config.get("training", {})
    if training.get("strategy") != "fsdp":
        failures.append(f"training.strategy is {training.get('strategy')!r}")
    if training.get("activation_offload") != "cpu":
        failures.append(f"training.activation_offload is {training.get('activation_offload')!r}")
    if training.get("run_final_eval_in_train") is not False:
        failures.append("training.run_final_eval_in_train must be false")
    if training.get("selection_metric") != "inner_val_macro_f1" or training.get("selection_metric_mode") != "max":
        failures.append("checkpoint selection is not inner_val_macro_f1/max")

    evaluation = config.get("evaluation", {})
    evaluation_view = evaluation.get("evaluation_view")
    if not evaluation_view:
        failures.append("evaluation.evaluation_view is missing")
    if evaluation.get("sample_prediction_mode") != "likelihood":
        failures.append(f"evaluation.sample_prediction_mode is {evaluation.get('sample_prediction_mode')!r}")
    if evaluation.get("inference_dtype") != "bf16":
        failures.append(f"evaluation.inference_dtype is {evaluation.get('inference_dtype')!r}")

    eval_dir = fold_dir / "best_model" / "standalone_eval"
    for name in REQUIRED_EVAL_FILES:
        if not (eval_dir / name).is_file():
            failures.append(f"standalone evaluation artifact is missing: {name}")
    metrics_path = eval_dir / "metrics_likelihood.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        record["metrics"] = {
            key: metrics.get(key)
            for key in (
                "num_subjects",
                "binary_strict_macro_f1",
                "binary_strict_positive_f1",
                "binary_strict_uar",
                "binary_strict_confusion_matrix",
                "evaluation_view",
                "aggregation_level",
            )
        }
        if metrics.get("evaluation_view") != evaluation_view:
            failures.append(
                f"metrics evaluation_view {metrics.get('evaluation_view')!r} != config {evaluation_view!r}"
            )
        if metrics.get("aggregation_level") != "subject":
            failures.append(f"metrics aggregation_level is {metrics.get('aggregation_level')!r}")
    eval_config_path = eval_dir / "eval_config.yaml"
    if eval_config_path.is_file():
        eval_config = _read_yaml(eval_config_path)
        eval_prompt_context = eval_config.get("prompt_context") or {}
        if eval_prompt_context.get("system_prompt_sha256") != prompt_context.get("system_prompt_sha256"):
            failures.append("evaluation rendered a different system prompt than training")

    sample_csv = eval_dir / "predictions_sample_level.csv"
    subject_csv = eval_dir / "predictions_subject_level.csv"
    if sample_csv.is_file() and subject_csv.is_file():
        samples = _read_csv(sample_csv)
        subjects = _read_csv(subject_csv)
        record["sample_rows"] = len(samples)
        record["subject_rows"] = len(subjects)
        if len({row["subject_id"] for row in subjects}) != len(subjects):
            failures.append("subject-level predictions repeat a subject")
        if str(config.get("dataset")) == "turkish":
            conditions = Counter(str(row.get("question_condition", "")) for row in samples)
            record["condition_counts"] = dict(conditions)
            if set(conditions) != set(POOLED_CONDITIONS):
                failures.append(f"pooled evaluation conditions are {sorted(conditions)}")
            per_subject = Counter(row["subject_id"] for row in samples)
            expected = {subject: 2 for subject in {row["subject_id"] for row in samples}}
            if dict(per_subject) != expected:
                failures.append("pooled evaluation does not give every subject exactly two condition rows")
            labels_by_subject: dict[str, set[str]] = {}
            for row in samples:
                labels_by_subject.setdefault(row["subject_id"], set()).add(str(row["label"]))
            inconsistent = {
                subject: sorted(labels)
                for subject, labels in labels_by_subject.items()
                if len(labels) != 1
            }
            if inconsistent:
                failures.append(
                    "pooled participants carry more than one label across conditions: "
                    f"{list(inconsistent)[:5]}"
                )
            if len(subjects) != len({row["subject_id"] for row in samples}):
                failures.append("subject-level rows do not match the evaluated subjects")
    record["passed"] = not failures
    return record


def _fold_dirs_from_root(root: Path) -> list[Path]:
    """Find fold dirs under a campaign root, a modality root or a dataset root."""
    found: dict[Path, None] = {}
    for path in root.rglob("fold_*"):
        if path.is_dir() and (path / "run_config.yaml").is_file():
            found[path] = None
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="campaign/text modality root holding <run>/fold_<n>")
    parser.add_argument("--fold-dir", type=Path, action="append", default=[], help="explicit fold directory")
    parser.add_argument("--output", type=Path, default=None, help="audit JSON path")
    args = parser.parse_args(argv)

    folds = list(args.fold_dir)
    if args.root is not None:
        folds.extend(_fold_dirs_from_root(args.root))
    if not folds:
        print("ERROR: no fold directories to audit", file=sys.stderr)
        return 2

    records = [audit_fold(Path(fold)) for fold in folds]
    passed = [record for record in records if record.get("passed")]
    audit = {
        "schema_version": "audiollm.promptcontext_run_audit.v1",
        "prompt_context_version": PROMPT_CONTEXT_VERSION,
        "folds": records,
        "passed": len(passed),
        "failed": len(records) - len(passed),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")

    for record in records:
        status = "ok" if record.get("passed") else "FAILED"
        detail = "" if record.get("passed") else " | " + "; ".join(record["failures"])
        print(f"{record['run_name']}/{record['fold']}: {status}{detail}")
    print(f"audited {len(records)} fold(s): {len(passed)} passed, {audit['failed']} failed")
    return 0 if audit["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
