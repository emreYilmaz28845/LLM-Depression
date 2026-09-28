#!/usr/bin/env python3
"""Smoke acceptance gate for the Qwen3 DAIC label-vocabulary campaign.

The campaign plan requires fifteen smoke chains to be accepted, separately from
production, before any production submission:

* training and standalone evaluation terminal with exit code 0:0;
* finite strict headline metrics;
* the resolved LoRA targets and the adapter-only trainable share, the frozen
  audio encoder for the audio cells, and the recorded four-rank FSDP shape;
* the adapter was saved and reloaded for the standalone evaluation of
  ``best_model``;
* coverage and aggregation match the fixed DAIC test partition;
* manifest and split identity is the same across all fifteen chains.

The gate reads the locally collected evidence (``exp.py collect`` output), the
lifecycle sidecars and the synced Slurm logs, and fails closed: an unreadable
sidecar, a missing log, a non-finite metric or a missing subject is a failure,
never a warning. It reports the matched log lines as evidence so a weak check is
visible instead of hidden.
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

from tools.qwen3_daic_label_vocab_matrix import (  # noqa: E402
    SMOKE_CAMPAIGN,
    SOURCES,
    run_name,
)

REQUIRED_SIDECARS = ("run_config.yaml", "metadata.json", "status.json", "jobs.jsonl", "artifacts.json", "evaluations.json")
HEADLINE_KEYS = ("binary_strict_macro_f1", "binary_strict_positive_f1", "binary_strict_uar")
AUDIO_CELLS = ("audio_only", "audio_text")
LORA_AUDIT_PATTERNS = (
    r"LoRA audit.*matched_modules=\d+",
    r"Resolved \d+ LoRA target modules",
)
AUDIO_FREEZE_PATTERNS = (
    r"audio.{0,80}frozen",
    r"frozen.{0,80}audio",
    r"encoder.{0,40}frozen",
)
TRAINABLE_PERCENT = re.compile(r"trainable%:\s*([0-9.]+)")
# The repo logs ``epoch=1 step=25 loss=0.178806`` itself; the HuggingFace trainer
# dict line is accepted as well because it can also reach the log.
LOSS_PATTERNS = (
    re.compile(r"\bloss=([0-9][0-9eE+\-.]*)"),
    re.compile(r"'loss':\s*([0-9eE+\-.]+)"),
)
EVAL_LOG_PATTERNS = {
    "checkpoint_role": (r"checkpoint_name': 'best_model'",),
    "final_result": (r"FINAL EVALUATION RESULT",),
    "aggregation": (r"aggregation_level': 'subject'|aggregation=subject",),
}


class SmokeGateError(RuntimeError):
    """Raised when the gate cannot run at all."""


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise SmokeGateError(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _log_matches(text: str, patterns: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    lines = text.splitlines()
    result: dict[str, Any] = {}
    for name, group in patterns.items():
        hits = [line.strip() for line in lines if all(re.search(pattern, line, re.IGNORECASE) for pattern in group)]
        result[name] = {"matched": bool(hits), "examples": hits[:3]}
    return result


def _train_losses(text: str) -> list[float]:
    values: list[float] = []
    for pattern in LOSS_PATTERNS:
        values.extend(float(match.group(1)) for match in pattern.finditer(text))
    return values


def read_fold_config(fold_dir: Path) -> dict[str, Any]:
    """Read ``run_config.yaml``; the resolved config lives under ``config``."""
    document = yaml.safe_load((fold_dir / "run_config.yaml").read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SmokeGateError(f"{fold_dir}/run_config.yaml is not a mapping")
    body = document.get("config")
    if isinstance(body, dict):
        return {**document, **body, "provenance": {key: value for key, value in document.items() if key != "config"}}
    return document


def read_metrics(fold_dir: Path) -> dict[str, Any]:
    path = fold_dir / "best_model" / "standalone_eval" / "metrics_likelihood.json"
    return _load_json(path) if path.is_file() else {}


def job_ids_by_key(fold_dir: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    path = fold_dir / "jobs.jsonl"
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        job_id = event.get("slurm_job_id")
        if job_id:
            result.setdefault(str(event.get("job_key")), []).append(str(job_id))
    return result


def terminal_events(fold_dir: Path) -> dict[str, str]:
    """Map job key to the terminal event type when the exit code was 0:0."""
    completed: dict[str, str] = {}
    path = fold_dir / "jobs.jsonl"
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event_type") not in {"COMPLETED", "FAILED", "CANCELLED"}:
            continue
        exit_code = str(event.get("exit_code") or "")
        if event.get("event_type") == "COMPLETED" and exit_code.startswith("0:0"):
            completed[str(event.get("job_key"))] = exit_code
    return completed


def _find_log(logs_root: Path, job_ids: list[str], prefix: str) -> Path:
    """Find the synced Slurm log by job id.

    Both workers share one LOG_ROOT in the managed submission, so an evaluation
    log can sit beside the training logs; both locations are searched.
    """
    for job_id in job_ids:
        for job_type in ("slurm_train", "slurm_eval"):
            directory = logs_root / job_type / "daic"
            if not directory.is_dir():
                continue
            for candidate in sorted(directory.glob(f"{prefix}-{job_id}*.log")):
                return candidate
    fallback = "train" if prefix == "train" else "eval"
    return logs_root / "slurm_train" / "daic" / f"{fallback}-missing-{'_'.join(job_ids) or 'no-job-id'}.log"


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

    completed = terminal_events(fold)
    checks["completed_jobs"] = sorted(completed)
    for key in ("train", "best_eval"):
        if key not in completed:
            failures.append(f"{run}: no COMPLETED 0:0 job event for {key}")

    job_ids = job_ids_by_key(fold)
    checks["job_ids"] = job_ids

    metrics = read_metrics(fold)
    if not metrics:
        failures.append(f"{run}: missing metrics_likelihood.json")
    else:
        for key in HEADLINE_KEYS:
            if not _finite(metrics.get(key)):
                failures.append(f"{run}: headline metric {key} is not finite ({metrics.get(key)!r})")
        checks["aggregation"] = metrics.get("aggregation_level")
        if metrics.get("aggregation_level") not in {"subject", "subject_level"}:
            failures.append(f"{run}: aggregation is {metrics.get('aggregation_level')!r}, expected subject level")
        checks["evaluation_view"] = metrics.get("evaluation_view")
        if metrics.get("evaluation_view") != "harmonized_all_windows_full_coverage":
            failures.append(f"{run}: evaluation view is {metrics.get('evaluation_view')!r}")
        checks["checkpoint_name"] = metrics.get("checkpoint_name")
        if metrics.get("checkpoint_name") != "best_model":
            failures.append(f"{run}: evaluation checkpoint is {metrics.get('checkpoint_name')!r}, expected best_model")
        checks["reported_subjects"] = metrics.get("num_subjects")
        if metrics.get("num_subjects") != test_subjects:
            failures.append(f"{run}: metrics report {metrics.get('num_subjects')} subjects, expected {test_subjects}")
        checks["validation_view_metrics"] = {
            key: metrics.get(key) for key in HEADLINE_KEYS
        }

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

    if (fold / "run_config.yaml").is_file():
        config = read_fold_config(fold)
        evaluation = config.get("evaluation") or {}
        training_strategy = config.get("training_strategy") or {}
        provenance = config.get("provenance") or {}
        checks["config_qualifiers"] = {
            "dataset": config.get("dataset"),
            "sample_prediction_mode": evaluation.get("sample_prediction_mode"),
            "evaluation_view": evaluation.get("evaluation_view"),
            "label_vocab_version": (config.get("labels") or {}).get("label_vocab_version"),
            "seed": config.get("seed"),
            "split_seed": (config.get("split") or {}).get("seed"),
            "world_size": training_strategy.get("world_size"),
            "manifest_hash": provenance.get("manifest_hash"),
            "split_metadata_hash": provenance.get("split_metadata_hash"),
        }
        if evaluation.get("sample_prediction_mode") != "likelihood":
            failures.append(f"{run}: run_config backend is {evaluation.get('sample_prediction_mode')!r}")
        if evaluation.get("evaluation_view") != "harmonized_all_windows_full_coverage":
            failures.append(f"{run}: run_config evaluation view is {evaluation.get('evaluation_view')!r}")
        if config.get("seed") != seed:
            failures.append(f"{run}: run_config seed is {config.get('seed')!r}, expected {seed}")
        if int((config.get("split") or {}).get("seed", -1)) != 1337:
            failures.append(f"{run}: run_config split.seed is {(config.get('split') or {}).get('seed')!r}, expected 1337")
        if training_strategy.get("world_size") != 4:
            failures.append(f"{run}: run_config world_size is {training_strategy.get('world_size')!r}, expected 4")

    train_log = _find_log(logs_root, job_ids.get("train", []), "train")
    if not train_log.is_file():
        failures.append(f"{run}: missing training log for job ids {job_ids.get('train')}")
        checks["train_log"] = {}
    else:
        text = train_log.read_text(encoding="utf-8", errors="replace")
        log_checks: dict[str, Any] = {"file": str(train_log.name)}
        if run not in text:
            failures.append(f"{run}: training log does not name the run")
        lora_hits = [line.strip() for line in text.splitlines() if any(re.search(p, line) for p in LORA_AUDIT_PATTERNS)]
        log_checks["lora_targets"] = {"matched": bool(lora_hits), "examples": lora_hits[:2]}
        if not lora_hits:
            failures.append(f"{run}: training log has no LoRA target audit evidence")
        freeze_hits = [
            line.strip()
            for line in text.splitlines()
            if any(re.search(p, line, re.IGNORECASE) for p in AUDIO_FREEZE_PATTERNS)
        ]
        log_checks["audio_encoder_frozen"] = {
            "required": modality in AUDIO_CELLS,
            "matched": bool(freeze_hits),
            "examples": freeze_hits[:2],
        }
        if modality in AUDIO_CELLS and not freeze_hits:
            failures.append(f"{run}: training log has no frozen-audio-encoder evidence")
        shape_hits = [line.strip() for line in text.splitlines() if "world_size=4" in line]
        log_checks["distributed_shape"] = {"matched": bool(shape_hits), "examples": shape_hits[:1]}
        if not shape_hits:
            failures.append(f"{run}: training log has no world_size=4 evidence")
        trainable = [float(value) for value in TRAINABLE_PERCENT.findall(text)]
        log_checks["trainable_percent"] = trainable[:1]
        if trainable and min(trainable) >= 1.0:
            failures.append(f"{run}: trainable parameter share is {min(trainable)}%, expected adapter-only training")
        checks["train_log"] = log_checks
        losses = [value for value in _train_losses(text) if _finite(value)]
        checks["train_losses"] = {"count": len(losses), "last": losses[-1] if losses else None}
        if not losses:
            failures.append(f"{run}: training log records no finite loss")

    eval_log = _find_log(logs_root, job_ids.get("best_eval", []), "eval")
    if not eval_log.is_file():
        failures.append(f"{run}: missing evaluation log for job ids {job_ids.get('best_eval')}")
        checks["eval_log"] = {}
    else:
        text = eval_log.read_text(encoding="utf-8", errors="replace")
        log_checks = _log_matches(text, EVAL_LOG_PATTERNS)
        log_checks["file"] = str(eval_log.name)
        coverage = re.search(r"num_subjects':\s*(\d+)", text) or re.search(r"num_units':\s*(\d+)", text)
        reported = int(coverage.group(1)) if coverage else None
        log_checks["test_subject_coverage"] = {"matched": reported == test_subjects, "reported": reported}
        if reported != test_subjects:
            failures.append(f"{run}: evaluation log reports {reported} test subjects, expected {test_subjects}")
        checks["eval_log"] = log_checks
        for name, result in log_checks.items():
            if isinstance(result, dict) and "matched" in result and not result["matched"]:
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
    parser.add_argument("--expected-manifest-hash", default=None)
    parser.add_argument("--expected-split-metadata-hash", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    split_metadata = Path(args.split_metadata) if args.split_metadata else Path(args.logs_root) / "daic_subject_partitions.json"
    test_subjects = expected_test_subjects(split_metadata)
    evidence_root = Path(args.evidence_root)
    logs_root = Path(args.logs_root)

    chains: list[dict[str, Any]] = []
    for modality in SOURCES:
        for arm in ("ab", "01", "truefalse", "yesno", "en"):
            chains.append(
                check_chain(
                    evidence_root=evidence_root,
                    logs_root=logs_root,
                    modality=modality,
                    arm=arm,
                    seed=args.seed,
                    test_subjects=test_subjects,
                )
            )

    failures = [message for chain in chains for message in chain["failures"]]
    manifest_hashes = {
        (chain["config_qualifiers"]["manifest_hash"] if chain.get("config_qualifiers") else None) for chain in chains
    }
    split_hashes = {
        (chain["config_qualifiers"]["split_metadata_hash"] if chain.get("config_qualifiers") else None) for chain in chains
    }
    manifest_hashes.discard(None)
    split_hashes.discard(None)
    if len(manifest_hashes) > 1:
        failures.append(f"chains disagree on the manifest hash: {sorted(manifest_hashes)}")
    if len(split_hashes) > 1:
        failures.append(f"chains disagree on the split metadata hash: {sorted(split_hashes)}")
    if args.expected_manifest_hash and manifest_hashes != {args.expected_manifest_hash}:
        failures.append(f"manifest hash {sorted(manifest_hashes)} differs from the expected {args.expected_manifest_hash}")
    if args.expected_split_metadata_hash and split_hashes != {args.expected_split_metadata_hash}:
        failures.append(
            f"split metadata hash {sorted(split_hashes)} differs from the expected {args.expected_split_metadata_hash}"
        )

    report = {
        "schema_version": "audiollm.qwen3_daic_label_vocab.smoke_gate.v1",
        "campaign": SMOKE_CAMPAIGN,
        "cells": len(chains),
        "expected_test_subjects": test_subjects,
        "manifest_hashes": sorted(manifest_hashes),
        "split_metadata_hashes": sorted(split_hashes),
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
        print(f"\n{len(failures)} failures:")
        for message in failures[:40]:
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
