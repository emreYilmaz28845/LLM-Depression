#!/usr/bin/env python3
"""Smoke acceptance gate for the Qwen3 DAIC label-vocabulary campaign.

The campaign plan requires fifteen smoke chains to be accepted, separately from
production, before any production submission:

* training and standalone evaluation terminal with exit code 0:0;
* finite losses and scores;
* the resolved LoRA target modules and the frozen audio encoder are visible in
  the training log;
* the recorded distributed shape (world size four) is visible;
* the adapter was saved and reloaded for the standalone evaluation of
  ``best_model``;
* coverage and aggregation match the fixed DAIC test partition;
* the answer-label token audit passed for the config.

This tool reads the locally collected evidence (``exp.py collect`` output), the
lifecycle sidecars and the synced Slurm logs, and fails closed: an unreadable
sidecar, a missing log, a non-finite metric or a missing subject is a failure,
never a warning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_qwen3_daic_label_configs import generated_name  # noqa: E402
from tools.qwen3_daic_label_vocab_matrix import (  # noqa: E402
    SMOKE_CAMPAIGN,
    SOURCES,
    run_name,
)

REQUIRED_SIDECARS = ("run_config.yaml", "metadata.json", "status.json", "jobs.jsonl", "artifacts.json", "evaluations.json")
HEADLINE_KEYS = ("binary_strict_macro_f1", "binary_strict_positive_f1", "binary_strict_uar")
# Every pattern must match at least one line of the training log.
TRAIN_LOG_PATTERNS = {
    "lora_targets": (r"lora", r"target"),
    "audio_encoder_frozen": (r"audio", r"froze"),
    "distributed_shape": (r"world_size",),
}
EVAL_LOG_PATTERNS = {
    "checkpoint_role": (r"best_model",),
}


class SmokeGateError(RuntimeError):
    """Raised when the gate cannot run at all."""


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise SmokeGateError(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _log_matches(text: str, patterns: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    lines = text.splitlines()
    result: dict[str, Any] = {}
    for name, patterns in patterns.items():
        hits = [line.strip() for line in lines if all(re.search(pattern, line, re.IGNORECASE) for pattern in patterns)]
        result[name] = {"matched": bool(hits), "examples": hits[:3]}
    return result


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _train_losses(text: str) -> list[float]:
    losses = []
    for match in re.finditer(r"'loss':\s*([0-9eE+\-.]+)", text):
        losses.append(float(match.group(1)))
    return losses


def expected_test_subjects(split_metadata: Path) -> int:
    payload = _load_json(split_metadata)
    records = payload["subject_partitions"] if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise SmokeGateError(f"{split_metadata}: expected a subject partition list")
    return sum(1 for record in records if str(record.get("partition")) == "test")


def check_chain(
    *,
    evidence_root: Path,
    logs_root: Path,
    modality: str,
    arm: str,
    seed: int,
    test_subjects: int,
    train_log: Path,
    eval_log: Path,
) -> dict[str, Any]:
    run = run_name(smoke=True, modality=modality, tag=arm, seed=seed)
    fold = evidence_root / SMOKE_CAMPAIGN / modality / "daic" / run / "fold_0"
    failures: list[str] = []
    checks: dict[str, Any] = {"run_name": run, "modality": modality, "arm": arm, "seed": seed}

    for name in REQUIRED_SIDECARS:
        if not (fold / name).is_file():
            failures.append(f"{run}: missing {name}")
    status = _load_json(fold / "status.json") if (fold / "status.json").is_file() else {}
    checks["lifecycle_state"] = status.get("state")
    if status.get("state") != "REPORTABLE":
        failures.append(f"{run}: lifecycle state is {status.get('state')!r}, expected REPORTABLE after validate/finish")

    jobs_path = fold / "jobs.jsonl"
    completed: dict[str, str] = {}
    if jobs_path.is_file():
        for line in jobs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if str(event.get("event_type")) in {"COMPLETED", "TERMINAL"} or str(event.get("state")) == "COMPLETED":
                exit_code = event.get("exit_code") or event.get("detail") or ""
                key = str(event.get("job_key", ""))
                if key and str(exit_code).startswith("0:0"):
                    completed[key] = str(exit_code)
    checks["completed_jobs"] = sorted(completed)
    for key in ("train", "best_eval"):
        if key not in completed:
            failures.append(f"{run}: no COMPLETED 0:0 job event for {key}")

    backend = "likelihood"
    metrics_path = fold / "best_model" / "standalone_eval" / f"metrics_{backend}.json"
    metrics = _load_json(metrics_path) if metrics_path.is_file() else {}
    if not metrics:
        failures.append(f"{run}: missing {metrics_path.name}")
    else:
        headline = metrics.get("headline_metrics") or metrics.get("metrics") or {}
        for key in HEADLINE_KEYS:
            value = headline.get(key)
            if not _finite(value):
                failures.append(f"{run}: headline metric {key} is not finite ({value!r})")
        aggregation = metrics.get("aggregation_level") or metrics.get("aggregation")
        checks["aggregation"] = aggregation
        if aggregation not in {"subject", "subject_level"}:
            failures.append(f"{run}: aggregation is {aggregation!r}, expected subject level")

    predictions = fold / "best_model" / "standalone_eval" / "predictions_subject_level.csv"
    if not predictions.is_file():
        failures.append(f"{run}: missing subject-level predictions")
        subject_rows = 0
    else:
        with predictions.open(newline="", encoding="utf-8") as handle:
            subject_rows = sum(1 for _ in csv.DictReader(handle))
    checks["subject_rows"] = subject_rows
    if subject_rows != test_subjects:
        failures.append(f"{run}: {subject_rows} test subjects, expected {test_subjects}")

    run_config_path = fold / "run_config.yaml"
    if run_config_path.is_file():
        config = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
        evaluation = config.get("evaluation", {})
        checks["config_qualifiers"] = {
            "dataset": config.get("dataset"),
            "sample_prediction_mode": evaluation.get("sample_prediction_mode"),
            "evaluation_view": evaluation.get("evaluation_view"),
            "label_vocab_version": (config.get("labels") or {}).get("label_vocab_version"),
            "run_root": (config.get("output_dirs") or {}).get("run_root"),
        }
        if evaluation.get("sample_prediction_mode") != backend:
            failures.append(f"{run}: run_config backend is {evaluation.get('sample_prediction_mode')!r}")
        if evaluation.get("evaluation_view") != "harmonized_all_windows_full_coverage":
            failures.append(f"{run}: run_config evaluation view is {evaluation.get('evaluation_view')!r}")
        if config.get("seed") != seed:
            failures.append(f"{run}: run_config seed is {config.get('seed')!r}, expected {seed}")
        if int((config.get("split") or {}).get("seed", -1)) != 1337:
            failures.append(f"{run}: run_config split.seed is {(config.get('split') or {}).get('seed')!r}, expected 1337")

    if not train_log.is_file():
        failures.append(f"{run}: missing training log {train_log}")
        checks["train_log"] = {}
    else:
        text = train_log.read_text(encoding="utf-8", errors="replace")
        log_checks = _log_matches(text, TRAIN_LOG_PATTERNS)
        checks["train_log"] = log_checks
        for name, result in log_checks.items():
            if not result["matched"]:
                failures.append(f"{run}: training log has no {name} evidence")
        losses = [value for value in _train_losses(text) if _finite(value)]
        checks["train_losses"] = {"count": len(losses), "last": losses[-1] if losses else None}
        if not losses:
            failures.append(f"{run}: training log records no finite loss")

    if not eval_log.is_file():
        failures.append(f"{run}: missing evaluation log {eval_log}")
        checks["eval_log"] = {}
    else:
        text = eval_log.read_text(encoding="utf-8", errors="replace")
        log_checks = _log_matches(text, EVAL_LOG_PATTERNS)
        checks["eval_log"] = log_checks
        for name, result in log_checks.items():
            if not result["matched"]:
                failures.append(f"{run}: evaluation log has no {name} evidence")

    checks["failures"] = failures
    checks["passed"] = not failures
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", default=str(ROOT / "output_model"))
    parser.add_argument("--logs-root", default=str(ROOT / "logs/qwen3_daic_label_vocab_smoke_v1"))
    parser.add_argument("--split-metadata", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    split_metadata = Path(args.split_metadata) if args.split_metadata else Path(args.logs_root) / "daic_subject_partitions.json"
    test_subjects = expected_test_subjects(split_metadata)
    evidence_root = Path(args.evidence_root)
    logs_root = Path(args.logs_root)

    chains: list[dict[str, Any]] = []
    for modality in SOURCES:
        for arm in ("ab", "01", "truefalse", "yesno", "en"):
            run = run_name(smoke=True, modality=modality, tag=arm, seed=args.seed)
            train_logs = sorted((logs_root / "slurm_train" / "daic").glob(f"train-*-{run}.log")) if (logs_root / "slurm_train" / "daic").is_dir() else []
            train_logs += sorted((logs_root / "slurm_train" / "daic").glob(f"*{run}*")) if (logs_root / "slurm_train" / "daic").is_dir() else []
            eval_logs = sorted((logs_root / "slurm_eval" / "daic").glob(f"*{run}*")) if (logs_root / "slurm_eval" / "daic").is_dir() else []
            chains.append(
                check_chain(
                    evidence_root=evidence_root,
                    logs_root=logs_root,
                    modality=modality,
                    arm=arm,
                    seed=args.seed,
                    test_subjects=test_subjects,
                    train_log=train_logs[0] if train_logs else logs_root / "slurm_train" / "daic" / f"train-{run}.log",
                    eval_log=eval_logs[0] if eval_logs else logs_root / "slurm_eval" / "daic" / f"eval-{run}.log",
                )
            )

    failures = [message for chain in chains for message in chain["failures"]]
    report = {
        "schema_version": "audiollm.qwen3_daic_label_vocab.smoke_gate.v1",
        "campaign": SMOKE_CAMPAIGN,
        "cells": len(chains),
        "expected_test_subjects": test_subjects,
        "chains": chains,
        "failures": failures,
        "passed": not failures,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for chain in chains:
        status = "PASS" if chain["passed"] else "FAIL"
        print(f"{status} {chain['modality']:10s} {chain['arm']:9s} state={chain['lifecycle_state']} subjects={chain['subject_rows']} jobs={chain['completed_jobs']}")
    if failures:
        print("\nfailures:")
        for message in failures:
            print(f"- {message}")
        return 1
    print(f"smoke gate passed for {len(chains)} chains")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeGateError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
