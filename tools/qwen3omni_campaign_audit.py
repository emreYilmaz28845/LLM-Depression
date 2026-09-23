#!/usr/bin/env python
"""Deterministic audits for the Qwen3-Omni standalone prompt-context campaign.

Two subcommands, both reading only local evidence and writing deterministic
outputs (no timestamps unless asked):

``jobs``
    Reconcile the planned matrix against the run sidecars: every expected cell and
    fold, its attempt identity, the SUBMITTED train/evaluation job ids, the
    lifecycle state, any supersession link, and — when a ``sacct`` dump is given —
    the scheduler's terminal state and exit code for each job. Missing attempts,
    jobs without scheduler evidence, and non-COMPLETED terminal states are
    reported explicitly; nothing is invented.

``resources``
    Summarize the per-family probe reports (forward / maxrisk / perf) into a
    compact table: world size, peak GPU and host memory, wall time, losses and
    throughput as recorded by the probes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PROJECT_ROOT = Path("/home/emre/Projects/AudioLLM/LLM-Depression")
RUN_ROOT_REL = "output_model/promptcontext_v1_qwen3omni_likelihood"
GROUP_ID = "qwen3omni-standalone-promptcontext-20260923"

# (dataset directory, modality, folds) of the ten campaign cells.
EXPECTED_CELLS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("d3tec", "audio_only", (0, 1, 2, 3, 4)),
    ("d3tec", "audio_text", (0, 1, 2, 3, 4)),
    ("androids_interview", "audio_only", (0, 1, 2, 3, 4)),
    ("androids_interview", "audio_text", (0, 1, 2, 3, 4)),
    ("cmdc", "audio_only", (0, 1, 2, 3, 4)),
    ("cmdc", "audio_text", (0, 1, 2, 3, 4)),
    ("turkish", "audio_only", (0, 1, 2, 3, 4)),
    ("turkish", "audio_text", (0, 1, 2, 3, 4)),
    ("daic", "audio_only", (0,)),
    ("daic", "audio_text", (0,)),
)
TERMINAL_OK = {"COMPLETED"}
# DAIC is carried from PR #261 instead of being rerun, so those two folds belong
# to the pilot group and are expected to keep its identity.
CARRIED_GROUPS = {
    ("daic", "audio_only"): "qwen3omni-daic-promptcontext-20260922",
    ("daic", "audio_text"): "qwen3omni-daic-promptcontext-20260922",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _job_events(fold_dir: Path) -> list[dict[str, Any]]:
    path = fold_dir / "jobs.jsonl"
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _parse_sacct(dump: Path) -> dict[str, dict[str, str]]:
    """JobIDRaw -> {state, exit_code, elapsed, node} from a sacct table dump."""
    rows: dict[str, dict[str, str]] = {}
    for line in dump.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[0] in {"JobIDRaw", "JobID"} or "." in parts[0]:
            continue
        rows[parts[0]] = {
            "state": parts[1],
            "exit_code": parts[2],
            "elapsed": parts[3] if len(parts) > 3 else "",
        }
    return rows


def audit_jobs(project_root: Path, sacct: dict[str, dict[str, str]]) -> dict[str, Any]:
    run_root = project_root / RUN_ROOT_REL
    records: list[dict[str, Any]] = []
    for dataset, modality, folds in EXPECTED_CELLS:
        dataset_dir = run_root / modality / dataset
        for fold in folds:
            fold_dirs = sorted(dataset_dir.glob(f"*/fold_{fold}")) if dataset_dir.is_dir() else []
            for fold_dir in fold_dirs:
                run_name = fold_dir.parent.name
                if "smoke" in run_name.lower():
                    continue
                metadata = _read_json(fold_dir / "metadata.json")
                status = _read_json(fold_dir / "status.json")
                events = _job_events(fold_dir)
                jobs: dict[str, dict[str, Any]] = {}
                for event in events:
                    job_id = str(event.get("slurm_job_id") or "")
                    if not job_id:
                        continue
                    entry = jobs.setdefault(
                        job_id,
                        {
                            "job_key": event.get("job_key"),
                            "job_type": event.get("job_type"),
                            "events": [],
                            "dependency_job_ids": event.get("dependency_job_ids") or [],
                        },
                    )
                    entry["events"].append(str(event.get("event_type")))
                    if sacct.get(job_id):
                        entry["sacct"] = sacct[job_id]
                record = {
                    "dataset": dataset,
                    "modality": modality,
                    "fold": fold,
                    "run_name": run_name,
                    "attempt_id": metadata.get("attempt_id", ""),
                    "group_id": metadata.get("group_id", ""),
                    "supersedes_attempt_id": metadata.get("supersedes_attempt_id", ""),
                    "lifecycle_state": status.get("state", ""),
                    "jobs": jobs,
                    "problems": [],
                }
                if metadata.get("group_id") not in {
                    GROUP_ID,
                    CARRIED_GROUPS.get((dataset, modality)),
                }:
                    record["problems"].append(
                        f"group_id {metadata.get('group_id')!r} is neither the campaign nor the carried group"
                    )
                if not record["attempt_id"]:
                    record["problems"].append("metadata.json missing or has no attempt_id")
                if not jobs:
                    record["problems"].append("no SUBMITTED job events recorded")
                for job_id, job in jobs.items():
                    if sacct:
                        info = job.get("sacct")
                        if info is None:
                            record["problems"].append(f"job {job_id} has no sacct record")
                        elif info["state"] not in TERMINAL_OK or info["exit_code"] != "0:0":
                            record["problems"].append(
                                f"job {job_id} terminal {info['state']} exit {info['exit_code']}"
                            )
                records.append(record)
    expected_attempts = sum(len(folds) for _dataset, _modality, folds in EXPECTED_CELLS)
    campaign_attempts = [record for record in records if record["group_id"] == GROUP_ID]
    superseded = sorted(
        {record["supersedes_attempt_id"] for record in records if record["supersedes_attempt_id"]}
    )
    return {
        "schema_version": "audiollm.qwen3omni_campaign_job_audit.v1",
        "group_id": GROUP_ID,
        "run_root": str(run_root),
        "expected_attempts": expected_attempts,
        "attempts_found": len(records),
        "campaign_attempts_found": len(campaign_attempts),
        "supersedes_links": superseded,
        "records": records,
        "status": (
            "passed"
            if len(records) == expected_attempts and not any(record["problems"] for record in records)
            else "failed"
        ),
    }


def _smoke_evidence(logs_root: Path | None) -> dict[str, Any]:
    """Peak per-rank memory and the selected checkpoint line of each training log.

    The training logs are the real-path evidence: the probes measure a synthetic
    stress loop, while these come from the production-shape runs themselves.
    """
    if logs_root is None or not logs_root.is_dir():
        return {}
    evidence: dict[str, Any] = {}
    for log_path in sorted(logs_root.glob("*/train-*.log")):
        dataset = log_path.parent.name
        peak_lines = [
            line
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if "Peak GPU memory [final]" in line
        ]
        selection_lines = [
            line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if "Selected checkpoint epoch" in line
        ]
        if not peak_lines and not selection_lines:
            continue
        job_id = log_path.name.split("-")[1] if "-" in log_path.name else ""
        entry: dict[str, Any] = {"job": job_id, "log": str(log_path)}
        if peak_lines:
            parts = peak_lines[-1].split("|")[-2:]
            entry["peak"] = " | ".join(part.strip() for part in parts)
        if selection_lines:
            entry["selection"] = selection_lines[-1].split("|", 3)[-1].strip()[:220]
        evidence.setdefault(dataset, []).append(entry)
    return evidence


def audit_resources(probes_root: Path, logs_root: Path | None = None) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}
    for probe_path in sorted(probes_root.glob("*/*.json")):
        dataset = probe_path.parent.name
        mode = probe_path.stem
        if mode not in {"forward", "maxrisk", "perf"}:
            continue
        payload = _read_json(probe_path)
        summary: dict[str, Any] = {
            "path": str(probe_path),
            "mode": payload.get("mode"),
            "world_size": payload.get("world_size"),
            "prompt_version": payload.get("prompt_version"),
        }
        ranks = payload.get("ranks") or []
        # FSDP probes report one entry per rank; the single-process device-map
        # forward probe reports the same fields at the top level.
        if not ranks and payload.get("mode") in {"forward", "tree"}:
            ranks = [payload]
        if ranks:
            summary["ranks"] = len(ranks)

            def _per_device_memory(rank: dict[str, Any]) -> list[dict[str, Any]]:
                gpu = rank.get("gpu_memory") or {}
                if not gpu:
                    return []
                if all(isinstance(value, dict) for value in gpu.values()):
                    return list(gpu.values())
                return [gpu]

            devices = [device for rank in ranks for device in _per_device_memory(rank)]
            if devices:
                summary["peak_allocated_gib"] = max(
                    (device.get("peak_allocated_gib") or 0) for device in devices
                )
                summary["peak_reserved_gib"] = max(
                    (device.get("peak_reserved_gib") or 0) for device in devices
                )
                summary["min_free_gib"] = min((device.get("free_gib") or 0) for device in devices)
                summary["gpu_total_gib"] = max((device.get("total_gib") or 0) for device in devices)
            host_usage = [
                (rank.get("host_memory") or {}).get("MemTotal_gb", 0)
                - (rank.get("host_memory") or {}).get("MemAvailable_gb", 0)
                for rank in ranks
                if rank.get("host_memory")
            ]
            if host_usage:
                summary["peak_host_used_gb"] = round(max(host_usage), 2)
                summary["host_total_gb"] = max(
                    (rank.get("host_memory") or {}).get("MemTotal_gb", 0) for rank in ranks
                )
            audit = (ranks[0].get("model_load_audit") or {})
            if audit:
                summary["model_load_audit"] = {
                    key: audit.get(key)
                    for key in (
                        "loaded_class",
                        "load_mode",
                        "talker_parameters",
                        "matched_modules",
                        "expected_modules",
                        "lora_trainable_params",
                        "audit_passed",
                        "evaluation_device_map",
                        "evaluation_resource_shape",
                    )
                    if key in audit
                }
            results = ranks[0].get("results") or {}
            if results:
                summary["finite_results"] = {}
                for modality, entries in results.items():
                    if not isinstance(entries, list):
                        continue
                    summary["finite_results"][modality] = {
                        "examples": len(entries),
                        "all_finite": all(bool(entry.get("finite")) for entry in entries),
                        "max_audio_seconds": max(
                            (entry.get("audio_seconds") or 0) for entry in entries
                        ),
                    }
            structure = ranks[0].get("structure") or {}
            if structure:
                summary["training_shape"] = {
                    "world_size": structure.get("world_size"),
                    "activation_offload": structure.get("activation_offload"),
                    "gradient_accumulation_steps": structure.get("gradient_accumulation_steps"),
                    "effective_global_batch_size": structure.get("effective_global_batch_size"),
                    "fsdp_units": structure.get("fsdp_units"),
                    "gradient_checkpointing_active": structure.get("gradient_checkpointing_active"),
                }
                audit = structure.get("model_load_audit") or {}
                if audit:
                    summary["model_load_audit"] = {
                        key: audit.get(key)
                        for key in (
                            "loaded_class",
                            "load_mode",
                            "talker_parameters",
                            "matched_modules",
                            "expected_modules",
                            "lora_trainable_params",
                            "audit_passed",
                        )
                        if key in audit
                    }
            timings = ranks[0].get("timings")
            if timings:
                summary["timings"] = timings
            for key in (
                "losses",
                "microbatch_seconds_mean",
                "global_examples_per_second",
                "derived_optimizer_step_seconds",
                "derived_optimizer_steps_per_hour",
                "host_memory_before_gb",
                "host_memory_after_gb",
            ):
                if key in ranks[0]:
                    summary[key] = ranks[0][key]
        families.setdefault(dataset, {})[mode] = summary
    return {
        "schema_version": "audiollm.qwen3omni_campaign_resource_report.v1",
        "probes_root": str(probes_root),
        "families": families,
        "smoke_evidence": _smoke_evidence(logs_root),
    }


DATASET_LABELS = {
    "d3tec": "D3TEC",
    "androids_interview": "Androids Interview",
    "cmdc": "CMDC",
    "turkish": "Turkish pooled t17",
    "daic": "DAIC",
}
MODALITY_LABELS = {"audio_only": "Audio only", "audio_text": "Audio + Text"}
WORKBOOK_SHEET = "Qwen3-Omni promptcontext"


def workbook_cell(dataset: str, modality: str) -> str:
    """The workbook cell label a campaign run fills (must match the generator)."""
    return (
        f"{WORKBOOK_SHEET}|{DATASET_LABELS[dataset]} — {MODALITY_LABELS[modality]} "
        "— Qwen3-Omni Thinker, promptcontext_v1"
    )


def build_selections(project_root: Path) -> dict[str, Any]:
    """Attempt-pinned per-fold workbook selections for the campaign's Omni cells."""
    run_root = project_root / RUN_ROOT_REL
    selections: list[dict[str, Any]] = []
    missing: list[str] = []
    for dataset, modality, folds in EXPECTED_CELLS:
        dataset_dir = run_root / modality / dataset
        for fold in folds:
            fold_dirs = [
                path
                for path in sorted(dataset_dir.glob(f"*/fold_{fold}"))
                if "smoke" not in path.parent.name.lower()
            ]
            if len(fold_dirs) != 1:
                missing.append(f"{dataset}/{modality}/fold_{fold}: {len(fold_dirs)} run directories")
                continue
            metadata = _read_json(fold_dirs[0] / "metadata.json")
            attempt_id = str(metadata.get("attempt_id") or "")
            if not attempt_id:
                missing.append(f"{dataset}/{modality}/fold_{fold}: no attempt id in metadata.json")
                continue
            selections.append(
                {
                    "cell": workbook_cell(dataset, modality),
                    "dataset": dataset,
                    "modality": modality,
                    "fold": fold,
                    "metric": "macro_f1",
                    "namespace": "headline/binary_strict",
                    "backend": "likelihood",
                    "view": "harmonized_all_windows_full_coverage",
                    "aggregation": "subject_level",
                    "attempt_id": attempt_id,
                }
            )
    return {"selections": selections, "missing": missing}


def _yaml_dump(payload: dict[str, Any]) -> str:
    lines = [
        "schema_version: audiollm.workbook_selection.v1",
        "description: >-",
        "  Explicit, attempt-pinned selections for the Qwen3-Omni prompt-context campaign",
        f"  cells in the \"{WORKBOOK_SHEET}\" sheet. Each cross-validated cell is selected per fold;",
        "  --validate-selected compares the workbook cell with the unweighted mean of the",
        "  resolved folds, and the DAIC cells (official test, fold 0) resolve directly.",
        "selections:",
    ]
    for selection in payload["selections"]:
        lines.append(f'  - cell: "{selection["cell"]}"')
        lines.append(f'    dataset: {selection["dataset"]}')
        lines.append(f'    modality: {selection["modality"]}')
        lines.append(f'    fold: {selection["fold"]}')
        lines.append(f'    metric: {selection["metric"]}')
        lines.append(f'    namespace: {selection["namespace"]}')
        lines.append(f'    backend: {selection["backend"]}')
        lines.append(f'    view: {selection["view"]}')
        lines.append(f'    aggregation: {selection["aggregation"]}')
        lines.append(f'    attempt_id: {selection["attempt_id"]}')
    return "\n".join(lines) + "\n"


def _resources_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-Omni standalone prompt-context campaign — resource and memory report",
        "",
        "All numbers are measured on MareNostrum 5 by "
        "`scripts/qwen3omni_backend_probe.py` in the campaign runtime; every probe reuses the "
        "risk inventory's selected stress examples.",
        "",
        "| Dataset | Probe | World size | Peak allocated (GiB) | Peak reserved (GiB) | Peak host (GB) | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for dataset, modes in payload["families"].items():
        for mode, summary in modes.items():
            note_bits = []
            shape = summary.get("training_shape") or {}
            if shape:
                note_bits.append(
                    f"offload {shape.get('activation_offload')}, accum {shape.get('gradient_accumulation_steps')}, "
                    f"effective batch {shape.get('effective_global_batch_size')}"
                )
            if summary.get("global_examples_per_second"):
                note_bits.append(f"{summary['global_examples_per_second']} examples/s global")
            if summary.get("min_free_gib"):
                note_bits.append(f"{summary['min_free_gib']} GiB free at peak")
            finite = summary.get("finite_results") or {}
            if finite:
                note_bits.append(
                    "; ".join(
                        f"{modality}: {entry['examples']} examples finite={entry['all_finite']}"
                        for modality, entry in finite.items()
                    )
                )
            audit = summary.get("model_load_audit") or {}
            if audit:
                bits = [f"talker params {audit.get('talker_parameters')}"]
                if audit.get("matched_modules") is not None:
                    bits.append(
                        f"LoRA {audit.get('matched_modules')}/{audit.get('expected_modules')}"
                    )
                if audit.get("evaluation_device_map") is not None:
                    bits.append(f"device map {audit.get('evaluation_device_map')}")
                note_bits.append(", ".join(bits))
            lines.append(
                "| {dataset} | {mode} | {world} | {alloc} | {reserved} | {host} | {notes} |".format(
                    dataset=dataset,
                    mode=mode,
                    world=summary.get("world_size", ""),
                    alloc=summary.get("peak_allocated_gib", ""),
                    reserved=summary.get("peak_reserved_gib", ""),
                    host=summary.get("peak_host_used_gb", ""),
                    notes="; ".join(note_bits),
                )
            )
    smoke = payload.get("smoke_evidence") or {}
    if smoke:
        lines += [
            "",
            "## Real training path (fold-0 smokes at the production shape)",
            "",
            "| Dataset | Job | Final peak | Selected checkpoint |",
            "| --- | --- | --- | --- |",
        ]
        for dataset, entries in smoke.items():
            for entry in entries:
                lines.append(
                    "| {dataset} | {job} | {peak} | {selection} |".format(
                        dataset=dataset,
                        job=entry.get("job", ""),
                        peak=entry.get("peak", ""),
                        selection=(entry.get("selection") or "")[:160],
                    )
                )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    jobs = sub.add_parser("jobs", help="reconcile planned cells against run sidecars and sacct")
    jobs.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    jobs.add_argument("--sacct", type=Path, default=None, help="sacct table dump to join")
    jobs.add_argument("--output", required=True, type=Path)

    resources = sub.add_parser("resources", help="summarize the probe reports")
    resources.add_argument("--probes-root", required=True, type=Path)
    resources.add_argument(
        "--logs-root",
        type=Path,
        default=None,
        help="slurm_train log root (adds the real-path peak memory and checkpoint selection per run)",
    )
    resources.add_argument("--output", required=True, type=Path)

    selections = sub.add_parser(
        "selections",
        help="write the attempt-pinned workbook selection YAML for the campaign cells",
    )
    selections.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    selections.add_argument("--output", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "selections":
        payload = build_selections(Path(args.project_root))
        if payload["missing"]:
            for entry in payload["missing"]:
                print(f"ERROR: {entry}", file=sys.stderr)
            return 1
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(_yaml_dump(payload), encoding="utf-8")
        print(f"wrote {len(payload['selections'])} selections to {args.output}")
        return 0
    if args.command == "jobs":
        sacct = _parse_sacct(args.sacct) if args.sacct else {}
        audit = audit_jobs(Path(args.project_root), sacct)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        print(
            f"attempts found {audit['attempts_found']}/{audit['expected_attempts']} "
            f"(campaign {audit['campaign_attempts_found']}) status={audit['status']}"
        )
        for record in audit["records"]:
            if record["problems"]:
                print(f"  {record['dataset']}/{record['modality']}/fold_{record['fold']}: {record['problems']}")
        return 0 if audit["status"] == "passed" else 1

    payload = audit_resources(args.probes_root, args.logs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(_resources_markdown(payload), encoding="utf-8")
    print(f"wrote {args.output} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
