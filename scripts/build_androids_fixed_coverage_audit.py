#!/usr/bin/env python3
"""Evidence-derived, fail-closed coverage audit for the Qwen3 multiseed merged campaign.

A group counts as complete only when all four evidence pillars agree:

1. Registry: ``registries/qmsm_<slug>_s<seed>.json`` exists, parses, matches the run id,
   and contains every expected unit (cv: 5 folds x {train, postprocess, head};
   final: fold_0 x the same three) with the active attempt COMPLETED and exit 0:0.
2. Stage audits: ``outputs/symmetric_merged/<campaign>/<modality>/<run_id>/<stage>/acceptance_audit.json``
   (falling back to ``acceptance_audit.synced.json``) with status ``passed``, matching stage,
   the expected fold count (cv=5, final=1), and no failures.
3. Collected evidence: per fold, qwen strict metrics/predictions for the expected datasets
   (cv: androids_interview, cmdc, d3tec, daic, turkish; final: daic), merged head
   metrics/predictions for logreg and xgb_fixed, train-side identity/complete files, the
   postprocess identity/complete files, and the cv selected_checkpoint log.
4. Local verification: ``local_metric_verification.json`` (v2) entry for run_id+stage with
   exactly the expected fold ids (cv 0..4, final 0), a nonempty, duplicate-free,
   finite-numeric check for every required qwen/logreg/xgb_fixed dataset/metric triplet,
   frozen epoch passed for final, and per-fold input sha256 hashes that still match the
   current collected artifacts (stale checks cannot approve replaced artifacts).

Missing, unreadable, mismatched or unverified evidence is reported as incomplete/unknown and
is never treated as complete. Superseded attempts recorded in ``registries/*.pre_retry*.json``
are documented with exact job ids/states but never counted as active failures or as coverage.

The previous ``coverage_audit.json`` is snapshotted into ``coverage_snapshots/`` before every
rewrite; prior snapshots and failures are never deleted.

Usage:
    python build_coverage_audit.py [--root DIR] [--out PATH]
    python build_coverage_audit.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

EVIDENCE_REL = "outputs/qwen3_androids_official_folds_20261008"
UNITS = ("train", "postprocess", "head")
CV_FOLDS = 5
CV_DATASETS = ("androids_interview", "cmdc", "d3tec", "daic", "turkish")
FINAL_DATASETS = ("daic",)
HEADS = ("logreg", "xgb_fixed")
QWEN_METRICS = ("binary_strict_macro_f1", "binary_strict_positive_f1", "binary_strict_uar")
HEAD_METRICS = ("macro_f1", "positive_f1", "macro_recall")
TRAIN_FILES = (
    "training_identity.json",
    "training_complete.json",
    "resolved_merged_config.json",
    "slurm_provenance.json",
)
MERGED_FILES = (
    "postprocess_complete.json",
    "postprocess_identity.json",
    "resolved_merged_config.json",
    "slurm_provenance.json",
)

# The approved campaign: 5 routes x seeds 7/1337/2024; every group requires cv AND final.
GROUPS = (
    ("native_text_only", 7),
    ("native_text_only", 1337),
    ("native_text_only", 2024),
    ("english_text_only", 7),
    ("english_text_only", 1337),
    ("english_text_only", 2024),
    ("native_audio_only", 7),
    ("native_audio_only", 1337),
    ("native_audio_only", 2024),
    ("native_audio_text", 7),
    ("native_audio_text", 1337),
    ("native_audio_text", 2024),
    ("english_audio_text", 7),
    ("english_audio_text", 1337),
    ("english_audio_text", 2024),
)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def mtime_utc(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except OSError:
        return None


def derive(slug: str) -> tuple[str, str, str]:
    """Return (kind, campaign, modality) for a route slug."""

    kind = "text" if slug.endswith("_text_only") else "audio"
    campaign = (
        "qwen3_androids_official_folds_20261008_native"
        if slug.startswith("native_")
        else "qwen3_androids_official_folds_20261008_english"
    )
    modality = "_".join(slug.split("_")[-2:])
    return kind, campaign, modality


def expected_units(modality: str, stage: str) -> list[str]:
    folds = range(CV_FOLDS) if stage == "cv" else (0,)
    return [f"{modality}:{stage}:fold_{fold}:{unit}" for fold in folds for unit in UNITS]


def datasets_for(stage: str) -> tuple[str, ...]:
    return CV_DATASETS if stage == "cv" else FINAL_DATASETS


def required_check_keys(stage: str) -> set[tuple[str, str, str]]:
    """Exact (kind, dataset, metric) triplets the verification must contain."""
    keys: set[tuple[str, str, str]] = set()
    for dataset in datasets_for(stage):
        for metric in QWEN_METRICS:
            keys.add(("qwen", dataset, metric))
        for method in HEADS:
            for metric in HEAD_METRICS:
                keys.add((method, dataset, metric))
    return keys


def required_verification_inputs(
    campaign: str, modality: str, run_id: str, stage: str, fold: int
) -> list[str]:
    """Artifacts whose hashes must be recorded in the verification entry."""
    merged = f"outputs/symmetric_merged/{campaign}/{modality}/{run_id}/{stage}/fold_{fold}"
    train = (
        f"output_model/symmetric_merged/{campaign}_likelihood/{modality}/{run_id}/{stage}/fold_{fold}"
    )
    rels: list[str] = []
    for dataset in datasets_for(stage):
        rels.append(f"{merged}/qwen/{dataset}/metrics_likelihood.json")
        rels.append(f"{merged}/qwen/{dataset}/predictions_subject_level.csv")
    for method in HEADS:
        rels.append(f"{merged}/heads/{method}/metrics_by_dataset.json")
        rels.append(f"{merged}/heads/{method}/predictions_subject_level.csv")
    for name in TRAIN_FILES:
        rels.append(f"{train}/{name}")
    if stage == "cv":
        rels.append(f"{train}/logs/selected_checkpoint.json")
    return rels


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def classify_group(
    root: Path, slug: str, seed: int, verif_index: dict, verif_path: Path
) -> dict:
    kind, campaign, modality = derive(slug)
    run_id = f"qmsm_{slug}_s{seed}"
    ev = root / EVIDENCE_REL
    reg_dir = ev / "registries"
    reg_path = reg_dir / f"{run_id}.json"
    merged_base = root / "outputs/symmetric_merged" / campaign / modality / run_id
    train_base = (
        root / "output_model/symmetric_merged" / f"{campaign}_likelihood" / modality / run_id
    )
    checks: list[tuple[str | None, str]] = []

    registry: dict = {
        "path": str(reg_path),
        "present": reg_path.is_file(),
        "sha256": sha256_file(reg_path),
        "mtime_utc": mtime_utc(reg_path),
        "parse_ok": False,
        "run_id_match": None,
        "source_commit": None,
        "registry_status": None,
        "plan_hash": None,
    }
    jobs_by_key: dict[str, dict] = {}
    if not reg_path.is_file():
        checks.append((None, f"registry not collected: {reg_path}"))
    else:
        payload = read_json(reg_path)
        if payload is None:
            checks.append((None, f"registry parse error: {reg_path}"))
        else:
            registry["parse_ok"] = True
            registry["run_id_match"] = payload.get("run_id") == run_id
            registry["source_commit"] = payload.get("source_commit")
            registry["registry_status"] = payload.get("registry_status")
            registry["plan_hash"] = payload.get("plan_hash")
            if not registry["run_id_match"]:
                checks.append(
                    (None, f"registry run_id mismatch: {payload.get('run_id')!r} != {run_id!r}")
                )
            for job in payload.get("jobs") or []:
                jobs_by_key[str(job.get("job_key"))] = job

    exp_all = set(expected_units(modality, "cv")) | set(expected_units(modality, "final"))
    stages: dict[str, dict] = {}
    for stage in ("cv", "final"):
        expected = expected_units(modality, stage)
        units: list[dict] = []
        missing_units: list[str] = []
        incomplete_units: list[str] = []
        fits_complete = 0
        for key in expected:
            job = jobs_by_key.get(key)
            if job is None:
                missing_units.append(key)
                units.append(
                    {
                        "unit": key,
                        "job_id": None,
                        "state": None,
                        "exit_code": None,
                        "retry": None,
                        "complete": False,
                    }
                )
                continue
            state = str(job.get("observed_state") or "")
            exit_code = str(job.get("exit_code") or "")
            complete = state.startswith("COMPLETED") and exit_code == "0:0"
            units.append(
                {
                    "unit": key,
                    "job_id": str(job.get("job_id")),
                    "state": state,
                    "exit_code": exit_code,
                    "retry": job.get("retry"),
                    "complete": complete,
                }
            )
            if not complete:
                incomplete_units.append(
                    f"{key} id={job.get('job_id')} state={state} exit={exit_code} "
                    f"retry={job.get('retry')}"
                )
        if missing_units:
            checks.append(
                (stage, f"missing units ({len(missing_units)}): {', '.join(missing_units)}")
            )
        for line in incomplete_units:
            checks.append((stage, f"unit not complete: {line}"))

        # Fit-level coverage (train+postprocess+head all complete per fold).
        fold_ids = range(CV_FOLDS) if stage == "cv" else (0,)
        for fold in fold_ids:
            fit_units = [u for u in units if f":fold_{fold}:" in u["unit"]]
            if fit_units and all(u["complete"] for u in fit_units):
                fits_complete += 1

        # Stage audit evidence.
        stage_dir = merged_base / stage
        primary = stage_dir / "acceptance_audit.json"
        synced = stage_dir / "acceptance_audit.synced.json"
        chosen = primary if primary.is_file() else (synced if synced.is_file() else None)
        audit_entry: dict = {
            "path": str(chosen) if chosen else str(primary),
            "sha256": sha256_file(chosen) if chosen else None,
            "source": "primary" if chosen == primary and chosen else ("synced" if chosen else None),
            "status": None,
            "stage_field": None,
            "expected_folds": None,
            "failures": None,
            "ok": False,
            "also_present": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "status": (read_json(path) or {}).get("status") if path.is_file() else None,
                }
                for name, path in (
                    ("acceptance_audit.json", primary),
                    ("acceptance_audit.synced.json", synced),
                )
            },
        }
        if chosen is None:
            checks.append((stage, f"audit missing ({primary})"))
        else:
            audit_payload = read_json(chosen)
            if audit_payload is None:
                checks.append((stage, f"audit parse error ({chosen})"))
            else:
                audit_entry["status"] = audit_payload.get("status")
                audit_entry["stage_field"] = audit_payload.get("stage")
                audit_entry["expected_folds"] = audit_payload.get("expected_folds")
                audit_entry["failures"] = audit_payload.get("failures")
                expected_folds = CV_FOLDS if stage == "cv" else 1
                if audit_payload.get("status") != "passed":
                    checks.append(
                        (stage, f"audit not passed (status={audit_payload.get('status')}) ({chosen})")
                    )
                elif audit_payload.get("stage") != stage:
                    checks.append(
                        (stage, f"audit stage mismatch ({audit_payload.get('stage')!r}) ({chosen})")
                    )
                elif audit_payload.get("expected_folds") != expected_folds:
                    checks.append(
                        (
                            stage,
                            f"audit expected_folds mismatch "
                            f"({audit_payload.get('expected_folds')} != {expected_folds}) ({chosen})",
                        )
                    )
                elif audit_payload.get("failures"):
                    checks.append(
                        (stage, f"audit failures non-empty: {audit_payload.get('failures')} ({chosen})")
                    )
                else:
                    audit_entry["ok"] = True

        # Collected evidence.
        collection: dict = {"ok": True, "missing": [], "folds": []}
        if not stage_dir.is_dir():
            collection["ok"] = False
            collection["missing"].append(f"stage dir not collected ({stage_dir})")
            checks.append((stage, f"collection missing: stage dir not collected ({stage_dir})"))
        else:
            fold_ids = range(CV_FOLDS) if stage == "cv" else (0,)
            for fold in fold_ids:
                fold_dir = stage_dir / f"fold_{fold}"
                missing_paths: list[str] = []
                if not fold_dir.is_dir():
                    missing_paths.append(f"fold_{fold} dir not collected ({fold_dir})")
                else:
                    for dataset in datasets_for(stage):
                        for name in ("metrics_likelihood.json", "predictions_subject_level.csv"):
                            path = fold_dir / "qwen" / dataset / name
                            if not path.is_file():
                                missing_paths.append(str(path.relative_to(root)))
                    for head in HEADS:
                        for name in ("metrics_by_dataset.json", "predictions_subject_level.csv"):
                            path = fold_dir / "heads" / head / name
                            if not path.is_file():
                                missing_paths.append(str(path.relative_to(root)))
                    for name in MERGED_FILES:
                        path = fold_dir / name
                        if not path.is_file():
                            missing_paths.append(str(path.relative_to(root)))
                train_fold = train_base / stage / f"fold_{fold}"
                for name in TRAIN_FILES:
                    path = train_fold / name
                    if not path.is_file():
                        missing_paths.append(str(path.relative_to(root)))
                if stage == "cv":
                    path = train_fold / "logs" / "selected_checkpoint.json"
                    if not path.is_file():
                        missing_paths.append(str(path.relative_to(root)))
                collection["folds"].append(
                    {"fold": fold, "ok": not missing_paths, "missing": missing_paths}
                )
                if missing_paths:
                    collection["ok"] = False
                    collection["missing"].extend(missing_paths)
            if not collection["ok"]:
                preview = "; ".join(collection["missing"][:8])
                more = len(collection["missing"]) - 8
                suffix = f"; +{more} more" if more > 0 else ""
                checks.append(
                    (
                        stage,
                        f"collection missing {len(collection['missing'])} file(s): {preview}{suffix}",
                    )
                )

        # Local verification (strict, hash-tied to the current artifacts).
        verification: dict = {
            "path": str(verif_path),
            "sha256": sha256_file(verif_path),
            "entry_present": False,
            "fold_ids": None,
            "checks_total": None,
            "coverage_missing": [],
            "inputs_checked": 0,
            "inputs_mismatched": [],
            "frozen_epoch_status": None,
            "ok": False,
            "problems": [],
        }
        problems: list[str] = []
        entry = verif_index.get((run_id, stage))
        if entry is None:
            problems.append(f"entry missing for {run_id}/{stage}")
        else:
            verification["entry_present"] = True
            expected_fold_ids = list(range(CV_FOLDS)) if stage == "cv" else [0]
            folds_raw = entry.get("folds") or []
            fold_ids = [fold.get("fold") for fold in folds_raw]
            verification["fold_ids"] = fold_ids
            int_ids = [
                fid for fid in fold_ids if isinstance(fid, int) and not isinstance(fid, bool)
            ]
            if (
                len(fold_ids) != len(int_ids)
                or sorted(int_ids) != expected_fold_ids
                or len(set(int_ids)) != len(int_ids)
            ):
                problems.append(f"fold ids mismatch: {fold_ids} != {expected_fold_ids}")
            required_checks = required_check_keys(stage)
            total_checks = 0
            for fold in folds_raw:
                fid = fold.get("fold")
                if not isinstance(fid, int) or fid not in expected_fold_ids:
                    continue
                fold_checks = fold.get("checks")
                if not isinstance(fold_checks, list) or not fold_checks:
                    problems.append(f"fold {fid}: empty checks")
                    continue
                seen: set[tuple] = set()
                for check in fold_checks:
                    total_checks += 1
                    check_kind = check.get("kind")
                    check_dataset = check.get("dataset")
                    check_metric = check.get("metric")
                    key = (check_kind, check_dataset, check_metric)
                    if check.get("ok") is not True:
                        problems.append(
                            f"fold {fid}: check not ok: {check_kind}/{check_dataset}/{check_metric} "
                            f"ok={check.get('ok')!r}"
                        )
                    if key in seen:
                        problems.append(
                            f"fold {fid}: duplicate check {check_kind}/{check_dataset}/{check_metric}"
                        )
                    seen.add(key)
                    for field in ("recorded", "recomputed"):
                        value = check.get(field)
                        if (
                            not isinstance(value, (int, float))
                            or isinstance(value, bool)
                            or not math.isfinite(float(value))
                        ):
                            problems.append(
                                f"fold {fid}: {check_kind}/{check_dataset}/{check_metric} {field} "
                                f"not finite numeric ({value!r})"
                            )
                missing = sorted(required_checks - seen)
                if missing:
                    verification["coverage_missing"].extend(
                        f"fold {fid}: {check_kind}/{check_dataset}/{check_metric}"
                        for check_kind, check_dataset, check_metric in missing
                    )
                inputs = fold.get("inputs")
                if not isinstance(inputs, dict) or not inputs:
                    problems.append(f"fold {fid}: verification inputs missing (rerun verification)")
                else:
                    for rel, expected_sha in sorted(inputs.items()):
                        current = sha256_file(root / rel)
                        verification["inputs_checked"] += 1
                        if current is None:
                            problems.append(
                                f"fold {fid}: verification input missing on disk: {rel}"
                            )
                        elif current != expected_sha:
                            verification["inputs_mismatched"].append(rel)
                            problems.append(f"fold {fid}: verification input hash mismatch: {rel}")
                    for rel in required_verification_inputs(
                        campaign, modality, run_id, stage, fid
                    ):
                        if rel not in inputs:
                            problems.append(
                                f"fold {fid}: verification inputs missing required artifact: {rel}"
                            )
            verification["checks_total"] = total_checks
            if verification["coverage_missing"]:
                preview = "; ".join(verification["coverage_missing"][:6])
                more = len(verification["coverage_missing"]) - 6
                problems.append(
                    f"required checks missing ({len(verification['coverage_missing'])}): "
                    f"{preview}" + (f"; +{more} more" if more > 0 else "")
                )
            if stage == "final":
                frozen = (entry.get("frozen_epoch") or {}).get("status")
                verification["frozen_epoch_status"] = frozen
                if frozen != "passed":
                    problems.append(f"frozen_epoch={frozen}")
        verification["problems"] = problems[:20]
        if problems:
            preview = "; ".join(problems[:6])
            more = len(problems) - 6
            checks.append(
                (
                    stage,
                    f"local verification failed ({len(problems)} problem(s)): {preview}"
                    + (f"; +{more} more" if more > 0 else ""),
                )
            )
        else:
            verification["ok"] = True

        stages[stage] = {
            "expected_units": len(expected),
            "units_complete": sum(1 for unit in units if unit["complete"]),
            "fits_expected": len(list(fold_ids)),
            "fits_complete": fits_complete,
            "units": units,
            "missing_units": missing_units,
            "audit": audit_entry,
            "collection": collection,
            "verification": verification,
            "ok": not [c for c in checks if c[0] == stage],
        }

    # Superseded attempts: documented from pre-retry snapshots, never counted as coverage.
    superseded: list[dict] = []
    retired: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for snap in sorted(reg_dir.glob(f"{run_id}.pre_retry*.json")):
        payload = read_json(snap)
        if payload is None:
            checks.append((None, f"pre-retry snapshot parse error: {snap}"))
            continue
        for job in payload.get("jobs") or []:
            key = str(job.get("job_key"))
            job_id = str(job.get("job_id"))
            record = {
                "snapshot": snap.name,
                "unit": key,
                "job_id": job_id,
                "state": job.get("observed_state"),
                "exit_code": job.get("exit_code"),
                "retry": job.get("retry"),
            }
            token = (key, job_id)
            if token in seen:
                continue
            seen.add(token)
            if key in exp_all:
                current = jobs_by_key.get(key)
                if current is None or str(current.get("job_id")) != job_id:
                    superseded.append(record)
            else:
                retired.append(record)

    extra_active_jobs = [
        {
            "unit": key,
            "job_id": str(job.get("job_id")),
            "state": job.get("observed_state"),
            "exit_code": job.get("exit_code"),
        }
        for key, job in sorted(jobs_by_key.items())
        if key not in exp_all
    ]

    return {
        "run_id": run_id,
        "slug": slug,
        "seed": seed,
        "kind": kind,
        "campaign": campaign,
        "modality": modality,
        "status": "complete" if not checks else "incomplete",
        "checks": [message for _, message in checks],
        "registry": registry,
        "stages": stages,
        "superseded_attempts": superseded,
        "retired_jobs": retired,
        "extra_active_jobs": extra_active_jobs,
    }


def build_report(root: Path) -> dict:
    ev = root / EVIDENCE_REL
    verif_path = ev / "local_metric_verification.json"
    verif_payload = read_json(verif_path) or {}
    verif_index = {
        (str(entry.get("run_id")), str(entry.get("stage"))): entry
        for entry in verif_payload.get("groups") or []
    }

    protocol_files: list[dict] = []
    seen_protocols: set[tuple[str, str]] = set()
    for slug, _ in GROUPS:
        _, campaign, modality = derive(slug)
        if (campaign, modality) in seen_protocols:
            continue
        seen_protocols.add((campaign, modality))
        path = root / "outputs/symmetric_merged" / campaign / modality / "merged_protocol.json"
        protocol_files.append(
            {
                "campaign": campaign,
                "modality": modality,
                "path": str(path),
                "present": path.is_file(),
                "sha256": sha256_file(path),
                "mtime_utc": mtime_utc(path),
            }
        )

    groups = [classify_group(root, slug, seed, verif_index, verif_path) for slug, seed in GROUPS]

    complete = [g for g in groups if g["status"] == "complete"]
    incomplete = [g for g in groups if g["status"] != "complete"]
    reg_mtimes = [g["registry"]["mtime_utc"] for g in groups if g["registry"]["mtime_utc"]]

    def _stage_ok(group: dict, stage: str) -> bool:
        return bool(group["stages"][stage]["ok"])

    fits_verified = sum(
        (CV_FOLDS if _stage_ok(g, "cv") else 0) + (1 if _stage_ok(g, "final") else 0)
        for g in groups
    )
    postprocess_verified = sum(
        (CV_FOLDS if _stage_ok(g, "cv") else 0) + (1 if _stage_ok(g, "final") else 0)
        for g in groups
    )
    head_verified = postprocess_verified
    completion_control = {
        "fits_expected": len(GROUPS) * (CV_FOLDS + 1),
        "fits_verified": fits_verified,
        "downstream_expected": len(GROUPS) * (CV_FOLDS + 1) * 2,
        "downstream_verified": postprocess_verified + head_verified,
        "downstream_breakdown": {
            "postprocess": postprocess_verified,
            "head": head_verified,
        },
        "internal_cells": {
            "component_evaluation_cells_expected": len(GROUPS) * (CV_FOLDS + 1) * 5,
            "component_evaluation_cells_verified": postprocess_verified * 5,
            "feature_extraction_per_postprocess": "required (enforced by the collection checks)",
            "fixed_head_variant_fits_expected": len(GROUPS) * (CV_FOLDS + 1) * 2,
            "fixed_head_variant_fits_verified": head_verified * 2,
            "summary_jobs": 0,
        },
        "binding": f"{EVIDENCE_REL}/frozen_downstream_graph.json",
        "eligible_for_marker": (
            len(complete) == len(GROUPS)
            and fits_verified == len(GROUPS) * (CV_FOLDS + 1)
            and postprocess_verified + head_verified == len(GROUPS) * (CV_FOLDS + 1) * 2
        ),
    }
    report = {
        "schema_version": "audiollm.androids_fixed_merged.coverage_audit.v1",
        "generated_utc": now_utc(),
        "evidence_root": str(root),
        "rule": (
            "fail-closed evidence-derived classification: a group is complete only when its "
            "registry shows all 18 units COMPLETED 0:0, both stage audits passed, collected "
            "evidence is present for every fold, and local verification passed with exact "
            "fold ids, complete nonempty finite-numeric check coverage (qwen/logreg/xgb_fixed "
            "triplets for every protocol dataset) and sha256-tied inputs matching the current "
            "artifacts; missing, unreadable, partial or stale evidence is incomplete/unknown, "
            "never complete; superseded attempts are documented but not counted as active "
            "failures or coverage"
        ),
        "scope": {
            "routes": 5,
            "training_seeds": [7, 1337, 2024],
            "groups": len(GROUPS),
            "cv_fits": len(GROUPS) * CV_FOLDS,
            "final_fits": len(GROUPS),
            "units": len(GROUPS) * (CV_FOLDS + 1) * len(UNITS),
        },
        "local_registry_freshness_utc": max(reg_mtimes) if reg_mtimes else None,
        "protocol_files": protocol_files,
        "groups": groups,
        "incomplete_keys": {g["run_id"]: g["checks"] for g in incomplete},
        "summary": {
            "groups_expected": len(GROUPS),
            "groups_complete": len(complete),
            "groups_incomplete": len(incomplete),
            "text_groups_complete": sum(1 for g in complete if g["kind"] == "text"),
            "audio_groups_complete": sum(1 for g in complete if g["kind"] == "audio"),
            "cv_stages_complete": sum(1 for g in groups if g["stages"]["cv"]["ok"]),
            "final_stages_complete": sum(1 for g in groups if g["stages"]["final"]["ok"]),
            "units_expected": len(GROUPS) * (CV_FOLDS + 1) * len(UNITS),
            "units_complete": sum(
                g["stages"][stage]["units_complete"] for g in groups for stage in ("cv", "final")
            ),
            "fits_expected": len(GROUPS) * (CV_FOLDS + 1),
            "fits_complete": sum(
                g["stages"][stage]["fits_complete"] for g in groups for stage in ("cv", "final")
            ),
            "complete_groups": [g["run_id"] for g in complete],
            "incomplete_groups": [g["run_id"] for g in incomplete],
        },
        "completion_control": completion_control,
        "blocker": (
            None
            if not incomplete
            else f"{len(incomplete)} group(s) incomplete; see incomplete_keys for exact evidence gaps"
        ),
    }
    return report


def snapshot_previous(out_path: Path) -> str | None:
    if not out_path.is_file():
        return None
    snap_dir = out_path.parent / "coverage_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    dest = snap_dir / f"coverage_audit.{now_utc().replace('-', '').replace(':', '')}.json"
    shutil.copy2(out_path, dest)
    return str(dest)


def print_report(report: dict) -> None:
    summary = report["summary"]
    print(json.dumps(summary, indent=2))
    print()
    for group in report["groups"]:
        parts = []
        for stage in ("cv", "final"):
            entry = group["stages"][stage]
            parts.append(
                f"{stage} units={entry['units_complete']}/{entry['expected_units']} "
                f"audit={'ok' if entry['audit']['ok'] else 'FAIL'} "
                f"coll={'ok' if entry['collection']['ok'] else 'FAIL'} "
                f"verify={'ok' if entry['verification']['ok'] else 'FAIL'}"
            )
        print(f"{group['run_id']:34s} {group['status']:10s} | " + " | ".join(parts))
        for check in group["checks"]:
            print(f"    - {check}")


def self_test() -> int:
    base = Path(tempfile.mkdtemp(prefix="cov_selftest_", dir="/tmp/opencode"))
    failures: list[str] = []

    def wj(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def mk_registry(ev: Path, run_id: str, modality: str, stages: dict, retry_keys=()) -> None:
        jobs = []
        counter = 9000000
        for stage, folds in stages.items():
            for fold in folds:
                for unit in UNITS:
                    counter += 1
                    key = f"{modality}:{stage}:fold_{fold}:{unit}"
                    jobs.append(
                        {
                            "job_key": key,
                            "job_id": str(counter),
                            "stage": stage,
                            "observed_state": "COMPLETED",
                            "exit_code": "0:0",
                            "retry": 1 if key in retry_keys else None,
                        }
                    )
        wj(
            ev / "registries" / f"{run_id}.json",
            {
                "run_id": run_id,
                "source_commit": "selftest",
                "registry_status": "terminal_success",
                "jobs": jobs,
            },
        )

    def mk_stage(root: Path, campaign: str, modality: str, run_id: str, stage: str, audit=True):
        folds = list(range(CV_FOLDS)) if stage == "cv" else [0]
        base_dir = root / "outputs/symmetric_merged" / campaign / modality / run_id / stage
        if audit:
            wj(
                base_dir / "acceptance_audit.json",
                {
                    "status": "passed",
                    "stage": stage,
                    "expected_folds": len(folds),
                    "failures": [],
                },
            )
        for fold in folds:
            fold_dir = base_dir / f"fold_{fold}"
            for dataset in datasets_for(stage):
                for name in ("metrics_likelihood.json", "predictions_subject_level.csv"):
                    wj(fold_dir / "qwen" / dataset / name, {"ok": True})
            for head in HEADS:
                for name in ("metrics_by_dataset.json", "predictions_subject_level.csv"):
                    wj(fold_dir / "heads" / head / name, {"ok": True})
            for name in MERGED_FILES:
                wj(fold_dir / name, {"ok": True})
            train_fold = (
                root
                / "output_model/symmetric_merged"
                / f"{campaign}_likelihood"
                / modality
                / run_id
                / stage
                / f"fold_{fold}"
            )
            for name in TRAIN_FILES:
                wj(train_fold / name, {"ok": True})
            if stage == "cv":
                wj(train_fold / "logs" / "selected_checkpoint.json", {"selected_epoch": 1})

    def build_entry(run_id, campaign, modality, stage, frozen, anomalies=None):
        anomalies = anomalies or {}
        folds = list(range(CV_FOLDS)) if stage == "cv" else [0]
        fold_records = []
        for fold in folds:
            if anomalies.get("missing_fold") == fold:
                continue
            checks = []
            for dataset in datasets_for(stage):
                for metric in QWEN_METRICS:
                    checks.append(
                        {
                            "kind": "qwen",
                            "dataset": dataset,
                            "metric": metric,
                            "recorded": 0.5,
                            "recomputed": 0.5,
                            "ok": True,
                        }
                    )
                for method in HEADS:
                    for metric in HEAD_METRICS:
                        checks.append(
                            {
                                "kind": method,
                                "dataset": dataset,
                                "metric": metric,
                                "recorded": 0.5,
                                "recomputed": 0.5,
                                "ok": True,
                            }
                        )
            if anomalies.get("drop_check") and fold == 0:
                checks = [
                    c
                    for c in checks
                    if (c["kind"], c["dataset"], c["metric"]) != anomalies["drop_check"]
                ]
            if anomalies.get("empty_fold") == fold:
                checks = []
            inputs = {}
            for rel in required_verification_inputs(campaign, modality, run_id, stage, fold):
                path = root / rel
                if path.is_file():
                    inputs[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
            if anomalies.get("stale_fold") == fold and inputs:
                first = sorted(inputs)[0]
                inputs[first] = "0" * 64
            fold_records.append({"fold": fold, "checks": checks, "inputs": inputs})
        entry = {"run_id": run_id, "stage": stage, "folds": fold_records}
        if frozen:
            entry["frozen_epoch"] = {"status": frozen}
        return entry

    def write_verif(ev: Path, specs) -> None:
        groups = [build_entry(*spec) for spec in specs]
        wj(
            ev / "local_metric_verification.json",
            {"schema_version": "selftest-v2", "groups": groups, "status": "passed"},
        )

    root = base
    ev = root / EVIDENCE_REL
    verif_entries: list[tuple[str, str, str | None]] = []

    # Case 1: fully evidenced group with a superseded failed attempt on cv fold_1 train.
    mk_registry(
        ev,
        "qmsm_native_text_only_s7",
        "text_only",
        {"cv": range(CV_FOLDS), "final": [0]},
        retry_keys={"text_only:cv:fold_1:train"},
    )
    mk_stage(root, "qwen3_androids_official_folds_20261008_native", "text_only", "qmsm_native_text_only_s7", "cv")
    mk_stage(root, "qwen3_androids_official_folds_20261008_native", "text_only", "qmsm_native_text_only_s7", "final")
    wj(
        ev / "registries" / "qmsm_native_text_only_s7.pre_retry.json",
        {
            "run_id": "qmsm_native_text_only_s7",
            "jobs": [
                {
                    "job_key": "text_only:cv:fold_1:train",
                    "job_id": "1111111",
                    "observed_state": "FAILED",
                    "exit_code": "1:0",
                    "retry": None,
                }
            ],
        },
    )
    verif_entries += [
        ("qmsm_native_text_only_s7", "qwen3_androids_official_folds_20261008_native", "text_only", "cv", None, None),
        ("qmsm_native_text_only_s7", "qwen3_androids_official_folds_20261008_native", "text_only", "final", "passed", None),
    ]

    # Case 2: registry missing entirely.
    # Case 3: audit missing for cv.
    mk_registry(
        ev, "qmsm_native_text_only_s1337", "text_only", {"cv": range(CV_FOLDS), "final": [0]}
    )
    mk_stage(root, "qwen3_androids_official_folds_20261008_native", "text_only", "qmsm_native_text_only_s1337", "cv", audit=False)
    mk_stage(root, "qwen3_androids_official_folds_20261008_native", "text_only", "qmsm_native_text_only_s1337", "final")
    verif_entries += [
        ("qmsm_native_text_only_s1337", "qwen3_androids_official_folds_20261008_native", "text_only", "cv", None, None),
        ("qmsm_native_text_only_s1337", "qwen3_androids_official_folds_20261008_native", "text_only", "final", "passed", None),
    ]

    # Case 4: collected evidence missing for cv fold_2.
    mk_registry(
        ev, "qmsm_native_text_only_s2024", "text_only", {"cv": range(CV_FOLDS), "final": [0]}
    )
    mk_stage(root, "qwen3_androids_official_folds_20261008_native", "text_only", "qmsm_native_text_only_s2024", "cv")
    mk_stage(root, "qwen3_androids_official_folds_20261008_native", "text_only", "qmsm_native_text_only_s2024", "final")
    shutil.rmtree(
        root
        / "outputs/symmetric_merged/qwen3_androids_official_folds_20261008_native/text_only/qmsm_native_text_only_s2024/cv/fold_2/qwen"
    )
    verif_entries += [
        ("qmsm_native_text_only_s2024", "qwen3_androids_official_folds_20261008_native", "text_only", "cv", None, None),
        ("qmsm_native_text_only_s2024", "qwen3_androids_official_folds_20261008_native", "text_only", "final", "passed", None),
    ]

    # Case 5: local verification entry missing for final.
    mk_registry(
        ev, "qmsm_english_text_only_s1337", "text_only", {"cv": range(CV_FOLDS), "final": [0]}
    )
    mk_stage(root, "qwen3_androids_official_folds_20261008_english", "text_only", "qmsm_english_text_only_s1337", "cv")
    mk_stage(root, "qwen3_androids_official_folds_20261008_english", "text_only", "qmsm_english_text_only_s1337", "final")
    verif_entries += [
        ("qmsm_english_text_only_s1337", "qwen3_androids_official_folds_20261008_english", "text_only", "cv", None, None)
    ]

    # Case 6: cv-only registry (final not submitted).
    mk_registry(ev, "qmsm_native_audio_only_s7", "audio_only", {"cv": range(CV_FOLDS)})
    mk_stage(root, "qwen3_androids_official_folds_20261008_native", "audio_only", "qmsm_native_audio_only_s7", "cv")
    verif_entries += [
        ("qmsm_native_audio_only_s7", "qwen3_androids_official_folds_20261008_native", "audio_only", "cv", None, None)
    ]

    # Cases 7-10: strict verification gates.
    for slug, seed, anomalies in (
        ("english_audio_only", 7, {"empty_fold": 0}),
        ("english_audio_only", 1337, {"missing_fold": 4}),
        ("english_audio_only", 2024, {"drop_check": ("qwen", "daic", "binary_strict_uar")}),
        ("english_audio_text", 1337, {"stale_fold": 0}),
    ):
        run_id = f"qmsm_{slug}_s{seed}"
        modality = "_".join(slug.split("_")[-2:])
        mk_registry(ev, run_id, modality, {"cv": range(CV_FOLDS), "final": [0]})
        mk_stage(root, "qwen3_androids_official_folds_20261008_english", modality, run_id, "cv")
        mk_stage(root, "qwen3_androids_official_folds_20261008_english", modality, run_id, "final")
        verif_entries += [
            (run_id, "qwen3_androids_official_folds_20261008_english", modality, "cv", None, anomalies),
            (run_id, "qwen3_androids_official_folds_20261008_english", modality, "final", "passed", None),
        ]

    write_verif(ev, verif_entries)

    verif_path = ev / "local_metric_verification.json"
    verif_payload = read_json(verif_path) or {}
    verif_index = {
        (str(entry.get("run_id")), str(entry.get("stage"))): entry
        for entry in verif_payload.get("groups") or []
    }

    def check(name: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"PASS {name}")
        else:
            failures.append(name)
            print(f"FAIL {name} {detail}")

    results = {
        (slug, seed): classify_group(root, slug, seed, verif_index, verif_path)
        for slug, seed in (
            ("native_text_only", 7),
            ("english_text_only", 7),
            ("native_text_only", 1337),
            ("native_text_only", 2024),
            ("english_text_only", 1337),
            ("native_audio_only", 7),
            ("english_audio_only", 7),
            ("english_audio_only", 1337),
            ("english_audio_only", 2024),
            ("english_audio_text", 1337),
        )
    }

    r1 = results[("native_text_only", 7)]
    check("full evidence group complete", r1["status"] == "complete", str(r1["checks"]))
    check(
        "superseded attempt recorded",
        len(r1["superseded_attempts"]) == 1
        and r1["superseded_attempts"][0]["job_id"] == "1111111"
        and not any("FAILED" in c for c in r1["checks"]),
        str(r1["superseded_attempts"]),
    )
    r2 = results[("english_text_only", 7)]
    check(
        "missing registry stays incomplete",
        r2["status"] == "incomplete"
        and any("registry not collected" in c for c in r2["checks"]),
        str(r2["checks"]),
    )
    r3 = results[("native_text_only", 1337)]
    check(
        "missing audit stays incomplete",
        r3["status"] == "incomplete" and any("audit missing" in c for c in r3["checks"]),
        str(r3["checks"]),
    )
    r4 = results[("native_text_only", 2024)]
    check(
        "missing collection stays incomplete",
        r4["status"] == "incomplete" and any("collection missing" in c for c in r4["checks"]),
        str(r4["checks"]),
    )
    r5 = results[("english_text_only", 1337)]
    check(
        "missing verification stays incomplete",
        r5["status"] == "incomplete"
        and any("entry missing" in c for c in r5["checks"]),
        str(r5["checks"]),
    )
    r6 = results[("native_audio_only", 7)]
    check(
        "final stage required even if not submitted",
        r6["status"] == "incomplete"
        and any("final" in c and "missing units" in c for c in r6["checks"]),
        str(r6["checks"]),
    )
    r7 = results[("english_audio_only", 7)]
    check(
        "empty checks stay incomplete",
        r7["status"] == "incomplete" and any("empty checks" in c for c in r7["checks"]),
        str(r7["checks"]),
    )
    r8 = results[("english_audio_only", 1337)]
    check(
        "missing fold stays incomplete",
        r8["status"] == "incomplete" and any("fold ids mismatch" in c for c in r8["checks"]),
        str(r8["checks"]),
    )
    r9 = results[("english_audio_only", 2024)]
    check(
        "missing required metric stays incomplete",
        r9["status"] == "incomplete"
        and any("required checks missing" in c for c in r9["checks"]),
        str(r9["checks"]),
    )
    r10 = results[("english_audio_text", 1337)]
    check(
        "stale verification hash stays incomplete",
        r10["status"] == "incomplete" and any("hash mismatch" in c for c in r10["checks"]),
        str(r10["checks"]),
    )

    shutil.rmtree(base, ignore_errors=True)
    if failures:
        print(f"SELF_TEST FAILED ({len(failures)}): {failures}")
        return 1
    print("SELF_TEST PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="evidence root (default: repo worktree root)")
    parser.add_argument("--out", default=None, help="output JSON path (default: coverage_audit.json)")
    parser.add_argument("--self-test", action="store_true", help="run focused synthetic checks")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    out_path = Path(args.out) if args.out else root / EVIDENCE_REL / "coverage_audit.json"
    report = build_report(root)
    snapshot = snapshot_previous(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"previous snapshot: {snapshot}" if snapshot else "no previous snapshot")
    print(f"written: {out_path}")
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
