#!/usr/bin/env python3
"""Post-run acceptance audit for the DAIC official-development campaign.

Model-free. Verifies, for one campaign RUN_ID: all twelve attempts (six
training + six fixed-head children) exist with complete sidecars, the 24
principal jobs reached terminal success (from a sacct capture or live sacct),
extraction coverage matches the locked 86/21/35 and 1312/603 row counts,
classifier backends and evaluation records carry the official-development
qualifiers, and every evaluation record supports 35 subjects. Writes a
task-owned JSON audit. Exit code 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.canonical import read_json, read_jsonl
from src.experiment_tracking.lifecycle import read_job_events, read_status

OFFICIALDEV_SPLIT_PROTOCOL = "daic_official_train_inner_split_dev_evaluation"
EXPECTED_COUNTS = {
    "audio_only": {"fit_rows": 1312, "fit_subjects": 86, "eval_rows": 603, "eval_subjects": 35},
    "audio_text": {"fit_rows": 1312, "fit_subjects": 86, "eval_rows": 603, "eval_subjects": 35},
    "text_only": {"fit_rows": 86, "fit_subjects": 86, "eval_rows": 35, "eval_subjects": 35},
}


class AuditFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def _collect_sacct(job_ids: list[str], user: str | None) -> dict[str, dict[str, str]]:
    if not job_ids:
        return {}
    ids = ",".join(sorted(set(job_ids)))
    cmd = [
        "sacct", "-j", ids, "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList",
        "--parsable2", "--noheader",
    ]
    if user:
        cmd += ["-u", user]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise AuditFailure(f"sacct failed: {result.stderr.strip()[:400]}")
    states: dict[str, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        fields = line.rstrip("|").split("|")
        if len(fields) < 4 or "." in fields[0]:
            continue  # skip step rows
        job_id = fields[0].strip()
        states[job_id] = {
            "state": fields[1].strip(),
            "exit_code": fields[2].strip(),
            "elapsed": fields[3].strip(),
            "nodes": fields[4].strip() if len(fields) > 4 else "",
        }
    return states


def _load_sacct_file(path: Path) -> dict[str, dict[str, str]]:
    """Read a sacct capture. Supports both the --parsable2 pipe format (with
    or without a header) and a TSV with a header row."""
    states: dict[str, dict[str, str]] = {}
    raw = path.read_text(encoding="utf-8")
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        return states
    first = lines[0]
    if "\t" in first and first.lstrip().startswith("JobIDRaw"):
        rows = list(csv.DictReader(lines, delimiter="\t"))
        for row in rows:
            if "." in (row.get("JobIDRaw") or ""):
                continue
            states[str(row["JobIDRaw"]).strip()] = {
                "state": str(row.get("State", "")).strip(),
                "exit_code": str(row.get("ExitCode", "")).strip(),
                "elapsed": str(row.get("Elapsed", "")).strip(),
                "nodes": str(row.get("NodeList", "")).strip(),
            }
        return states
    for line in lines:
        fields = [field.strip() for field in line.rstrip("|").split("|")]
        if len(fields) < 2 or "." in fields[0]:
            continue
        states[fields[0]] = {
            "state": fields[1],
            "exit_code": fields[2] if len(fields) > 2 else "",
            "elapsed": fields[3] if len(fields) > 3 else "",
            "nodes": fields[4] if len(fields) > 4 else "",
        }
    return states


def audit_run(
    *,
    run_id: str,
    submissions_root: Path,
    contexts_root: Path,
    output_model_root: Path,
    sacct_states: dict[str, dict[str, str]],
    retry_registry_dir: Path | None = None,
    transition_completed: bool = False,
) -> dict[str, Any]:
    registry = submissions_root / run_id / "jobs.tsv"
    _require(registry.is_file(), f"missing submission registry: {registry}")
    cells: dict[tuple[str, str], dict[str, str]] = {}
    job_rows: list[dict[str, str]] = []
    with registry.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _require(reader.fieldnames and "kind" in reader.fieldnames, "registry has no header")
        for row in reader:
            job_rows.append(row)
            cells.setdefault((row["backbone"], row["modality"]), {})[row["kind"]] = row["job_id"]
    _require(len(cells) == 6, f"expected six campaign cells, found {len(cells)}")
    _require(len(job_rows) == 24, f"expected 24 principal jobs, found {len(job_rows)}")

    # Merge the bounded retry chains: a failed principal job may have been
    # replaced by one or more retry jobs recorded in the retry registries.
    retry_rows: list[dict[str, str]] = []
    if retry_registry_dir is not None and retry_registry_dir.is_dir():
        for path in sorted(retry_registry_dir.glob(f"{run_id}/retry_*_jobs.tsv")):
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    retry_rows.append(row)
    jobs_by_cell_kind: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in job_rows:
        jobs_by_cell_kind.setdefault((row["backbone"], row["modality"], row["kind"]), []).append(row)
    for row in retry_rows:
        jobs_by_cell_kind.setdefault((row["backbone"], row["modality"], row["kind"]), []).append(row)

    job_states: dict[str, dict[str, str]] = {}
    for cell_key, rows in sorted(jobs_by_cell_kind.items()):
        for row in rows:
            job_id = row["job_id"]
            state = sacct_states.get(job_id)
            _require(state is not None, f"job {job_id} has no terminal sacct record")
            state = dict(state)
            # Slurm appends a reason to some states (e.g. "CANCELLED by 53836").
            state["state"] = str(state["state"]).split()[0]
            job_states[job_id] = state
        # The cell's job kind is successful only if at least one of its jobs
        # (principal or retry) completed and every other one is terminal.
        completed = [r for r in rows if job_states[r["job_id"]]["state"] == "COMPLETED"]
        non_terminal = [
            r for r in rows
            if job_states[r["job_id"]]["state"] not in {"COMPLETED", "FAILED", "CANCELLED"}
        ]
        _require(len(completed) == 1, f"cell {cell_key} must have exactly one COMPLETED job, got {len(completed)}")
        _require(not non_terminal, f"cell {cell_key} has non-terminal jobs: {[r['job_id'] for r in non_terminal]}")
        completed_row = completed[0]
        _require(
            job_states[completed_row["job_id"]]["exit_code"] == "0:0",
            f"job {completed_row['job_id']} exit code is {job_states[completed_row['job_id']]['exit_code']}",
        )

    attempts: dict[str, Any] = {}
    for (backbone, modality), jobs in sorted(cells.items()):
        context_path = contexts_root / run_id / backbone / modality / "fold_0" / "context.json"
        _require(context_path.is_file(), f"missing context: {context_path}")
        context = read_json(context_path)
        training_attempt = context["attempt_id"]

        # Training attempt sidecars and eval evidence.
        registry_row = next(
            row for row in job_rows
            if row["backbone"] == backbone and row["modality"] == modality and row["kind"] == "train"
        )
        run_name = registry_row["run_name"]
        config = _load_cell_config(backbone, modality)
        run_root = str(config["output_dirs"]["run_root"]).replace("${PROJECT_ROOT}", str(PROJECT_ROOT))
        fold_dir = Path(run_root) / run_name / f"fold_{context['fold']}"
        _require(fold_dir.is_dir(), f"training fold dir missing: {fold_dir}")
        status = read_status(fold_dir / "status.json")
        if transition_completed and status["state"] == "RUNNING":
            # The training job completed but the training path never leaves
            # RUNNING; the audit performs the official COMPLETED_ON_MN5
            # transition through the lifecycle API.
            train_job_state = job_states.get(jobs["train"], {}).get("state")
            eval_job_state = job_states.get(jobs["eval"], {}).get("state")
            if train_job_state == "COMPLETED" and eval_job_state == "COMPLETED":
                from src.experiment_tracking.lifecycle import StatusRecord, write_status

                record = StatusRecord.from_dict(read_json(fold_dir / "status.json"))
                record.transition("COMPLETED_ON_MN5", reason="campaign run audit confirmed terminal jobs")
                write_status(fold_dir / "status.json", record)
                status = read_status(fold_dir / "status.json")
        _require(
            status["state"] in {"COMPLETED_ON_MN5", "SYNCED_LOCALLY", "LOCALLY_VALIDATED", "REPORTABLE"},
            f"training attempt {training_attempt} state is {status['state']}",
        )
        _require((fold_dir / "best_model" / "adapter_model.safetensors").is_file(), "best_model adapter missing")
        split_payload = read_json(fold_dir / "logs" / "split_used.json")
        _require(len(split_payload.get("train_subject_ids") or []) == 86, "training split is not 86")
        _require(len(split_payload.get("selection_subject_ids") or []) == 21, "selection split is not 21")
        _require(len(split_payload.get("final_eval_subject_ids") or []) == 35, "final eval split is not 35")
        eval_dir = fold_dir / "best_model" / "standalone_eval"
        _require(eval_dir.is_dir(), "standalone eval dir missing")
        metrics_path = eval_dir / "metrics_original_teacher_forced.json"
        _require(metrics_path.is_file(), "standalone eval metrics missing")
        metrics = read_json(metrics_path)
        _require(metrics.get("evaluation_view") == "harmonized_all_windows_full_coverage", "eval view mismatch")
        predictions = list((eval_dir / "predictions_subject_level.csv").read_text(encoding="utf-8").splitlines())
        _require(len(predictions) >= 36, "subject predictions missing (expected header + 35)")

        # Fixed-head child attempt: with retries there may be several child
        # dirs per cell; the final one is the only one that reached
        # COMPLETED_ON_MN5 (older children are FAILED/SUPERSEDED).
        child_root = (
            output_model_root / "harmonized_v1_officialdev_heads"
            if backbone == "qwen"
            else output_model_root / "harmonized_v1_gemma4_officialdev_heads"
        )
        candidates = sorted(
            (child_root / modality / "daic").glob(
                f"daic_officialdev_{backbone}_{modality}_fixed_heads_seed1337_*/fold_0"
            )
        )
        _require(bool(candidates), f"no fixed-head child dirs for {backbone}/{modality}")
        child_fold = None
        for candidate in candidates:
            if not (candidate / "status.json").is_file():
                continue
            state = read_status(candidate / "status.json")["state"]
            if state in {"COMPLETED_ON_MN5", "SYNCED_LOCALLY", "LOCALLY_VALIDATED", "REPORTABLE"}:
                if child_fold is not None:
                    raise AuditFailure(f"multiple completed child attempts for {backbone}/{modality}")
                child_fold = candidate
        _require(child_fold is not None, f"no completed child attempt for {backbone}/{modality}")
        child_status = read_status(child_fold / "status.json")
        _require(
            child_status["state"] in {"COMPLETED_ON_MN5", "SYNCED_LOCALLY", "LOCALLY_VALIDATED", "REPORTABLE"},
            f"child attempt state is {child_status['state']}",
        )
        child_config = (read_json(child_fold / "run_config.yaml"))["config"]
        _require(child_config["campaign_protocol"] == "officialdev", "child campaign protocol mismatch")
        _require(child_config["evaluation"]["split_name"] == "val", "child split name mismatch")
        _require(
            child_config["evaluation"]["split_protocol"] == OFFICIALDEV_SPLIT_PROTOCOL,
            "child split protocol mismatch",
        )
        _require(child_config["evaluation"]["support"] == 35, "child support mismatch")
        extraction_metadata = read_json(child_fold / "hidden_features" / "extraction_metadata.json")
        partitions = extraction_metadata.get("partitions") or {}
        expected = EXPECTED_COUNTS[modality]
        outer = partitions.get("outer_train") or {}
        final = partitions.get("final_eval") or {}
        _require(
            outer.get("rows") == expected["fit_rows"] and outer.get("subjects") == expected["fit_subjects"],
            f"{backbone}/{modality} outer_train coverage mismatch: {outer}",
        )
        _require(
            final.get("rows") == expected["eval_rows"] and final.get("subjects") == expected["eval_subjects"],
            f"{backbone}/{modality} final_eval coverage mismatch: {final}",
        )
        _require(
            (extraction_metadata.get("evaluation_provenance") or {}).get("evaluation_protocol")
            == OFFICIALDEV_SPLIT_PROTOCOL,
            "extraction evaluation protocol mismatch",
        )
        evaluations = read_json(child_fold / "evaluations.json")["evaluations"]
        _require(len(evaluations) == 2, f"child attempt has {len(evaluations)} evaluation records")
        for entry in evaluations:
            _require(entry["split_name"] == "val", "evaluation split name mismatch")
            _require(entry["split_protocol"] == OFFICIALDEV_SPLIT_PROTOCOL, "evaluation split protocol mismatch")
            _require(entry["aggregation"] == "subject_level", "evaluation aggregation mismatch")
            _require(entry["metric_namespace"] == "headline/binary_strict", "evaluation namespace mismatch")
            _require(all(m["support"] == 35 for m in entry["metrics"]), "evaluation support mismatch")
        for variant in ("logreg_raw", "xgb_raw"):
            variant_dir = child_fold / "hidden_classifiers" / variant
            _require((variant_dir / "metrics.json").is_file(), f"{variant} metrics missing")
            subject_rows = list((variant_dir / "predictions_subject_level.csv").read_text(encoding="utf-8").splitlines())
            _require(len(subject_rows) >= 36, f"{variant} subject predictions missing (expected header + 35)")

        attempts[f"{backbone}/{modality}"] = {
            "training_attempt": training_attempt,
            "training_state": status["state"],
            "training_fold_dir": str(fold_dir),
            "child_state": child_status["state"],
            "child_fold_dir": str(child_fold),
            "jobs": {kind: jobs[kind] for kind in ("train", "eval", "extract", "heads")},
        }

    return {
        "schema_version": "daic_officialdev_run_audit.v1",
        "run_id": run_id,
        "status": "passed",
        "cells": attempts,
        "jobs": job_states,
    }


def _load_cell_config(backbone: str, modality: str) -> dict[str, Any]:
    from src.utils import load_yaml

    suffix = "_gemma4_12b" if backbone == "gemma4" else ""
    path = PROJECT_ROOT / "configs/main" / f"daic_{modality}_harmonized_selmacrof1_tf{suffix}_officialdev.yaml"
    return load_yaml(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--sacct-file", type=Path, default=None,
        help="TSV capture of sacct --parsable2 output (JobIDRaw/State/ExitCode/...).",
    )
    parser.add_argument("--use-live-sacct", action="store_true", help="Query sacct directly.")
    parser.add_argument("--user", default=None)
    parser.add_argument("--submissions-root", type=Path, default=PROJECT_ROOT / "outputs/daic_officialdev_submissions")
    parser.add_argument("--contexts-root", type=Path, default=PROJECT_ROOT / "outputs/daic_officialdev_experiment_contexts")
    parser.add_argument("--output-model-root", type=Path, default=PROJECT_ROOT / "output_model")
    parser.add_argument(
        "--retry-registry-dir", type=Path, default=None,
        help="Directory containing the campaign retry registries (default: submissions-root).",
    )
    parser.add_argument("--transition-completed", action="store_true",
                        help="Transition RUNNING training attempts to COMPLETED_ON_MN5 when their jobs completed.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.sacct_file and not args.use_live_sacct:
        parser.error("provide --sacct-file or --use-live-sacct")

    try:
        if args.sacct_file:
            states = _load_sacct_file(args.sacct_file)
        else:
            states = _collect_sacct([], args.user)
            registry = args.submissions_root / args.run_id / "jobs.tsv"
            if registry.is_file():
                with registry.open(newline="", encoding="utf-8") as handle:
                    job_ids = [row["job_id"] for row in csv.DictReader(handle, delimiter="\t")]
                states.update(_collect_sacct(job_ids, args.user))
        record = audit_run(
            run_id=args.run_id,
            submissions_root=args.submissions_root,
            contexts_root=args.contexts_root,
            output_model_root=args.output_model_root,
            sacct_states=states,
            retry_registry_dir=args.retry_registry_dir,
            transition_completed=args.transition_completed,
        )
    except AuditFailure as error:
        print(f"AUDIT FAILED: {error}", file=sys.stderr)
        return 1

    output = args.output or (
        PROJECT_ROOT / "outputs/daic_officialdev_audits" / f"{args.run_id}_run_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"audit passed: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
