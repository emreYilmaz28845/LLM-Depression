#!/usr/bin/env python3
"""Plan or submit the smoke, CV, and final symmetric merged job chains."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.merged.configuration import (
    validate_evaluation_contract,
    validate_merged_resources,
)
from src.merged.protocol import canonical_sha256
from src.merged.provenance import source_commits_match
from src.merged.runtime import load_merged_config, load_protocol_artifact
from src.utils import read_json, resolve_project_path, save_json, sha256_file


# The guard is keyed on the resolved contract identity (config name + backend +
# modality) and the recorded per-route readiness table below, so it cannot be
# bypassed by editing a documentation field. A Qwen3-backed contract is refused
# before any GPU job is submitted unless it is one of the declared routes; the
# bounded ``smoke`` stage is how a declared route is verified, while the
# multi-fold ``cv`` and ``final`` stages require ``production_ready``. The head
# kind stays deferred for every Qwen3 route until Qwen3 hidden features land.
NEW_MODEL_BACKENDS = frozenset({"qwen38", "qwen3omni"})
QWEN3_MERGED_PREREQUISITE = "Qwen3 merged FSDP/postprocess prerequisite incomplete"
QWEN3_HEAD_PREREQUISITE = "Qwen3 merged head support prerequisite incomplete"

# Readiness of the five declared Qwen3 pooled merged contracts. Only a route
# that passed its own bounded GPU smoke chain may be marked production-ready;
# the others keep their production guard (and their CPU/config/processor route
# tests) until they pass their own chain.
QWEN3_CONTRACT_READINESS: dict[str, dict[str, Any]] = {
    "symmetric_merged_qwen3_pooled_native_text_only": {
        "backend": "qwen38",
        "modality": "text_only",
        "verified_shape": {
            "train_nodes": 1,
            "gpus_per_node": 4,
            "gradient_accumulation_steps": 32,
        },
        "production_ready": True,
        "production_block_reason": None,
        "head_ready": True,
        "evidence": (
            "GPU smoke chain passed: run qwen3_merged_smoke_text_only_20260930_r3. Train job "
            "46844006 COMPLETED 0:0 and postprocess job 46845076 COMPLETED 0:0 both ran from the "
            "immutable deployment "
            "feat-qwen3-merged-fsdp-postprocess-20260930-20260930T142929Z-c41d8e76-91bf3e5a "
            "(source c41d8e76c12979c358499a8b84fce035f67b6478). The completed train output was "
            "relocated from that deployment's "
            "output_model/symmetric_merged/qwen3_pooled_native_likelihood/text_only/"
            "qwen3_merged_smoke_text_only_20260930_r3/smoke/fold_0 to the same relative path under "
            "the permanent checkout (identical files; earlier failed attempts and their logs are "
            "preserved). Likelihood subject-level evidence was locally verified for all five "
            "components."
            " Hidden-feature audit passed: postprocess job 46853892 and head job 46855535 COMPLETED 0:0 under the isolated run qwen3_heads_audit_native_text_only_20260930_r1 (feature dimension 5120, 24 train and 24 holdout rows); the first head attempt 46853893 failed on the missing project-local dependency path and is preserved."
        ),
    },
    "symmetric_merged_qwen3_pooled_native_audio_text": {
        "backend": "qwen3omni",
        "modality": "audio_text",
        "verified_shape": {
            "train_nodes": 2,
            "gpus_per_node": 4,
            "gradient_accumulation_steps": 16,
        },
        "production_ready": True,
        "production_block_reason": None,
        "head_ready": True,
        "evidence": (
            "Two-node smoke chain passed (run qwen3_merged2n_smoke_native_audio_text_20261001_r1): train "
            "46877582 COMPLETED 0:0 (52:10) and postprocess 46877583 COMPLETED 0:0 (9:33), and the smoke "
            "stage's fixed head 46877584 COMPLETED 0:0, all from the immutable deployment "
            "feat-qwen3-multiseed-matrix-readiness-20260930-20261001T112633Z-6a49be82-3313b433 (source "
            "6a49be820b3d88c1426d1df9bbc206fd95cac6a6). The train job ran on two four-GPU nodes "
            "(NumNodes=2-2 NumTasks=2 NumCPUs=160, rendezvous nnodes=2 master=as07r3b01:29517) with the "
            "declared CPU activation offload active, and recorded per-rank peaks of 22.53-26.84 GiB "
            "allocated / 34.36-36.90 GiB reserved / about 50.5 GiB free on every rank. Likelihood "
            "subject-level evidence was collected locally for all five components and the strict metrics "
            "were recomputed from the stored subject predictions with an exact match (status passed)."
            " Hidden-feature audit passed in the same two-node shape: postprocess 46885550 and head "
            "46885551 COMPLETED 0:0 under the isolated run qwen3_heads_audit_native_audio_text_20261001_r2 "
            "(feature dimension 2048). Superseded one-node chain 46846648/46846649 is kept for the record."
        ),
    },
    "symmetric_merged_qwen3_pooled_native_audio_only": {
        "backend": "qwen3omni",
        "modality": "audio_only",
        "verified_shape": {
            "train_nodes": 2,
            "gpus_per_node": 4,
            "gradient_accumulation_steps": 16,
        },
        "production_ready": True,
        "production_block_reason": None,
        "head_ready": True,
        "evidence": (
            "Two-node smoke chain passed (run qwen3_merged2n_smoke_native_audio_only_20261001_r1): train "
            "46880498 COMPLETED 0:0 (51:30) and postprocess 46880499 COMPLETED 0:0 (7:49), and the smoke "
            "stage's fixed head 46880500 COMPLETED 0:0, all from the immutable deployment "
            "feat-qwen3-multiseed-matrix-readiness-20260930-20261001T112633Z-6a49be82-3313b433 (source "
            "6a49be820b3d88c1426d1df9bbc206fd95cac6a6). The train job ran on two four-GPU nodes "
            "(rendezvous nnodes=2 master=as07r4b26:29517) with the declared CPU activation offload active, "
            "and recorded per-rank peaks of 21.40-21.60 GiB allocated / 29.21-29.41 GiB reserved / about "
            "50.7-50.9 GiB free on every rank. Likelihood subject-level evidence was collected locally for "
            "all five components and the strict metrics were recomputed from the stored subject predictions "
            "with an exact match (status passed)."
            " Hidden-feature audit passed in the same two-node shape: postprocess 46885552 and head "
            "46885553 COMPLETED 0:0 under the isolated run qwen3_heads_audit_native_audio_only_20261001_r2 "
            "(feature dimension 2048). Superseded one-node chain 46852252/46852253 is kept for the record."
        ),
    },
    "symmetric_merged_qwen3_pooled_english_text_only": {
        "backend": "qwen38",
        "modality": "text_only",
        "verified_shape": {
            "train_nodes": 1,
            "gpus_per_node": 4,
            "gradient_accumulation_steps": 32,
        },
        "production_ready": True,
        "production_block_reason": None,
        "head_ready": True,
        "evidence": (
            "GPU smoke chain passed: run qwen3_multiseed_smoke_en_text_20260930_r1. Train job "
            "46852254 COMPLETED 0:0 (6:40) and postprocess job 46852255 COMPLETED 0:0 (2:37), both "
            "from the immutable deployment "
            "feat-qwen3-multiseed-matrix-readiness-20260930-20260930T183953Z-4ff77c53-521d2e6a "
            "(source 4ff77c53ebd3808671af551a58287136bd1726e5). The four translated components "
            "render the versioned translation notice and DAIC keeps its native English input; "
            "likelihood subject-level evidence was locally verified for all five components."
            " Hidden-feature audit passed: postprocess job 46853898 and head job 46855540 COMPLETED 0:0 under the isolated run qwen3_heads_audit_english_text_only_20260930_r1 (feature dimension 5120, 24 train and 24 holdout rows); the first head attempt 46853899 failed on the missing project-local dependency path and is preserved."
        ),
    },
    "symmetric_merged_qwen3_pooled_english_audio_text": {
        "backend": "qwen3omni",
        "modality": "audio_text",
        "verified_shape": {
            "train_nodes": 2,
            "gpus_per_node": 4,
            "gradient_accumulation_steps": 16,
        },
        "production_ready": True,
        "production_block_reason": None,
        "head_ready": True,
        "evidence": (
            "Two-node smoke chain passed (run qwen3_merged2n_smoke_en_audio_text_20261001_r2): train "
            "46889195 COMPLETED 0:0 (51:44) and postprocess 46889196 COMPLETED 0:0 (10:00), and the smoke "
            "stage's fixed head 46889197 COMPLETED 0:0, all from the immutable deployment "
            "feat-qwen3-multiseed-matrix-readiness-20260930-20261001T112633Z-6a49be82-3313b433 (source "
            "6a49be820b3d88c1426d1df9bbc206fd95cac6a6). The train job ran on two four-GPU nodes "
            "(rendezvous nnodes=2 master=as03r5b22:29517) with the declared CPU activation offload active "
            "and per-rank peaks of 22.53-26.84 GiB allocated / 34.36-36.90 GiB reserved / about 50.5 GiB "
            "free on every rank. Likelihood subject-level evidence was collected locally for all five "
            "components and the strict metrics were recomputed from the stored subject predictions with an "
            "exact match (status passed)."
            " Hidden-feature audit passed in the same two-node shape: postprocess 46891829 and head "
            "46891830 COMPLETED 0:0 under the isolated run qwen3_heads_audit_english_audio_text_20261001_r2 "
            "(feature dimension 2048)."
            " The first attempt r1 (train 46880501) failed on an NCCL collective timeout of a 1-element "
            "ALLREDUCE that never completed inside the process-group timeout; it is classified transient "
            "infrastructure, its dependent jobs were cancelled with DependencyNeverSatisfied, and the "
            "unchanged retry r2 above passed. The superseded one-node chain 46852256/46852257 is kept for "
            "the record."
        ),
    },
}


def merged_blocked_backends(config: dict[str, Any]) -> list[str]:
    """Qwen3-family backends among the merged config and its component configs.

    A component that points at an archived or renamed path cannot be resolved;
    the planner's own gates report that later, so the guard only blocks what it
    can actually resolve. The Qwen3 pooled contracts also declare the backend on
    the merged config itself, so a missing component can never hide them.
    """
    found: set[str] = set()
    own = str(config.get("model_backend") or "")
    if own in NEW_MODEL_BACKENDS:
        found.add(own)
    for component in config.get("components") or []:
        path = resolve_project_path(str(component.get("config", "")))
        if not path.is_file():
            continue
        component_config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        backend = str(component_config.get("model_backend") or "")
        if backend in NEW_MODEL_BACKENDS:
            found.add(backend)
    return sorted(found)


def merged_component_contract_records(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load the five component configs for contract validation without manifests.

    The planner must validate the evaluation contract before any job is
    submitted, and a dry-run may legitimately run before the component
    manifests exist. Only the component YAML declarations are needed for that,
    so the records carry the dataset identity and the loaded config. Component
    paths that cannot be resolved are reported separately so the Qwen3 gate can
    fail closed while legacy contracts keep their historical behavior.
    """
    records: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for item in config.get("components") or []:
        dataset = str(item.get("name") or item.get("dataset") or "").strip().lower()
        raw_path = str(item.get("config") or "")
        path = resolve_project_path(raw_path) if raw_path else None
        if path is None or not path.is_file():
            unresolved.append(dataset or raw_path or "<unnamed>")
            continue
        component_config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        records.append({"dataset": dataset, "config": component_config})
    return records, unresolved


def merged_route_decision(config: dict[str, Any], *, stage: str) -> dict[str, Any]:
    """Gate one merged contract for one stage against the Qwen3 readiness table.

    Legacy (non-Qwen3) contracts keep today's behavior: they are allowed. A
    Qwen3-backed contract must be one of the four declared pooled contracts with
    a matching backend and modality; the bounded ``smoke`` stage is allowed for
    a declared route, and ``cv``/``final`` additionally require the recorded
    production readiness *and* the declared lane shape the route was verified in,
    so an override that only restores the effective global batch (for example one
    node with accumulation 32 instead of two with 16) cannot run production on
    the strength of evidence recorded for another shape.
    """
    if stage not in {"smoke", "cv", "final"}:
        raise ValueError(f"Unsupported merged stage: {stage!r}")
    name = str(config.get("name") or "")
    modality = str(config.get("modality") or "").strip().lower()
    execution = config.get("execution") or {}
    training = config.get("training") or {}
    declared_shape = {
        "train_nodes": int(execution.get("train_nodes") or 1),
        "gpus_per_node": int(execution.get("qwen_gpus") or 4),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps") or 1),
    }
    backends = merged_blocked_backends(config)
    if not backends:
        return {
            "qwen3": False,
            "declared": False,
            "allowed": True,
            "stage": stage,
            "contract": name,
            "backends": [],
            "modality": modality,
            "declared_shape": declared_shape,
            "verified_shape": None,
            "shape_verified": True,
            "head_ready": True,
            "head_deferred_reason": None,
            "reason": None,
            "evidence": None,
        }
    resolved_backend = "+".join(backends)
    entry = QWEN3_CONTRACT_READINESS.get(name)
    if entry is None or entry["backend"] != resolved_backend or entry["modality"] != modality:
        return {
            "qwen3": True,
            "declared": False,
            "allowed": False,
            "stage": stage,
            "contract": name,
            "backends": backends,
            "modality": modality,
            "declared_shape": declared_shape,
            "verified_shape": None,
            "shape_verified": False,
            "head_ready": False,
            "head_deferred_reason": QWEN3_HEAD_PREREQUISITE,
            "reason": (
                f"{QWEN3_MERGED_PREREQUISITE}: {name or '<unnamed contract>'} is not a declared "
                f"Qwen3 merged contract with recorded readiness for backend={resolved_backend} "
                f"modality={modality!r}."
            ),
            "evidence": None,
        }
    verified_shape = entry.get("verified_shape")
    shape_verified = verified_shape is None or declared_shape == dict(verified_shape)
    if stage == "smoke":
        allowed = True
        reason = None
    else:
        allowed = bool(entry["production_ready"]) and shape_verified
        if not entry["production_ready"]:
            reason = (
                f"{QWEN3_MERGED_PREREQUISITE}: {name} {stage} needs a passed GPU smoke chain; "
                f"{entry['production_block_reason']}."
            )
        elif not shape_verified:
            reason = (
                f"{QWEN3_MERGED_PREREQUISITE}: {name} {stage} declares "
                f"{declared_shape} but the route was verified as {entry.get('verified_shape')}; "
                "a smoke chain in the declared shape is required."
            )
        else:
            reason = None
    return {
        "qwen3": True,
        "declared": True,
        "allowed": allowed,
        "stage": stage,
        "contract": name,
        "backends": backends,
        "modality": modality,
        "declared_shape": declared_shape,
        "verified_shape": entry.get("verified_shape"),
        "shape_verified": shape_verified,
        "head_ready": bool(entry["head_ready"]),
        "head_deferred_reason": None if entry["head_ready"] else QWEN3_HEAD_PREREQUISITE,
        "reason": reason,
        "evidence": entry.get("evidence"),
    }


CONFIG_BY_MODALITY = {
    "audio_text": PROJECT_ROOT / "configs/experiments/merged/symmetric_merged_audio_text.yaml",
    "audio_only": PROJECT_ROOT / "configs/experiments/merged/symmetric_merged_audio_only.yaml",
    "text_only": PROJECT_ROOT / "configs/experiments/merged/symmetric_merged_text_only.yaml",
}


def _source_commit() -> str:
    explicit = str(os.environ.get("SYMMETRIC_MERGED_SOURCE_COMMIT", "")).strip()
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _reservation() -> str:
    return str(os.environ.get("SYMMETRIC_MERGED_RESERVATION", "")).strip()


def _job_id(run_id: str, modality: str, stage: str, fold: int, kind: str) -> str:
    value = f"{run_id}|{modality}|{stage}|{fold}|{kind}"
    return "dry_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _head_trials(config: dict[str, Any], *, stage: str, smoke_trials: int) -> int:
    optuna = (config.get("heads") or {}).get("optuna") or {}
    if optuna.get("enabled") is False:
        return 0
    if stage == "smoke":
        return int(smoke_trials)
    return int(optuna.get("target_trials", 150))


def _apply_concurrency_lanes(jobs: list[dict[str, Any]], *, kind: str, limit: int) -> None:
    """Add afterany lane dependencies without changing scientific dependencies."""

    if int(limit) < 0:
        raise ValueError(f"Concurrency limit for {kind} cannot be negative.")
    if int(limit) == 0:
        return
    previous_by_lane: list[str | None] = [None] * int(limit)
    candidates = [job for job in jobs if job.get("kind") == kind]
    for index, job in enumerate(candidates):
        lane = index % int(limit)
        previous = previous_by_lane[lane]
        if previous:
            job["throttle_dependency_job_key"] = previous
        job["concurrency_lane"] = lane
        previous_by_lane[lane] = str(job["job_key"])


def _run_roots(config: dict[str, Any], run_id: str, stage: str, fold: int) -> dict[str, Path]:
    return {
        "train": Path(config["output_dirs"]["run_root"]) / run_id / stage / f"fold_{fold}",
        "post": Path(config["output_dirs"]["merged_root"]) / run_id / stage / f"fold_{fold}",
    }


def _expected_protocol_identity(
    config: dict[str, Any], config_path: str | Path, fold: int
) -> dict[str, str] | None:
    """Resolve the hashes required before a completed artifact may be reused."""

    try:
        protocol = load_protocol_artifact(config)
        fold_payload = protocol["protocol"]["folds"][str(int(fold))]
        expected = {
            "merged_config_sha256": sha256_file(config_path),
            "manifest_hash": str(protocol["manifest"]["manifest_hash"]),
            "split_hash": str(protocol["protocol"]["split_hash"]),
            "fold_hash": str(fold_payload["fold_hash"]),
        }
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        # A dry-run may be planned before the generated protocol artifact has
        # been materialized. In that case there is no evidence strong enough
        # to skip an existing output, so force the normal compatibility gate.
        return None
    if any(not value or value == "None" for value in expected.values()):
        return None
    return expected


def _provenance_matches(path: Path) -> bool:
    """Require the artifact's recorded source commit to match this submission."""

    if not path.is_file():
        return False
    current = _source_commit()
    if not current or current == "unknown":
        return False
    try:
        return source_commits_match(read_json(path).get("source_commit"), current)
    except (OSError, TypeError, ValueError):
        return False


def _identity_hashes_match(
    identity: dict[str, Any], expected: dict[str, str], *, split_key: str
) -> bool:
    return (
        identity.get("merged_config_sha256") == expected["merged_config_sha256"]
        and identity.get("manifest_hash") == expected["manifest_hash"]
        and identity.get(split_key) == expected["split_hash"]
        and identity.get("fold_hash") == expected["fold_hash"]
    )


def _completed(
    config: dict[str, Any],
    config_path: str | Path,
    run_id: str,
    stage: str,
    fold: int,
    kind: str,
    *,
    epochs: int | None = None,
    subjects_per_class: int | None = None,
    trials: int | None = None,
    head_ready: bool = True,
) -> bool:
    roots = _run_roots(config, run_id, stage, fold)
    expected_identity = _expected_protocol_identity(config, config_path, fold)
    if expected_identity is None:
        return False
    if kind == "train":
        complete = roots["train"] / "training_complete.json"
        if not complete.is_file() or not (roots["train"] / "best_model").is_dir():
            return False
        if not _provenance_matches(roots["train"] / "slurm_provenance.json"):
            return False
        payload = read_json(complete)
        identity = payload.get("identity", {})
        expected_epochs = int(epochs if epochs is not None else config["training"].get("num_train_epochs", 20))
        return (
            payload.get("status") == "completed"
            and identity.get("config_name") == config.get("name")
            and identity.get("stage") == stage
            and int(identity.get("fold", -1)) == int(fold)
            and identity.get("run_id") == run_id
            and int(identity.get("epochs", -1)) == expected_epochs
            and identity.get("subjects_per_class") == subjects_per_class
            and _identity_hashes_match(
                identity, expected_identity, split_key="protocol_split_hash"
            )
        )
    if kind == "postprocess":
        complete = roots["post"] / "postprocess_complete.json"
        identity_path = roots["post"] / "postprocess_identity.json"
        feature_metadata = roots["post"] / "features" / "feature_metadata.json"
        if not complete.is_file() or not identity_path.is_file():
            return False
        if head_ready and not feature_metadata.is_file():
            # The hidden feature matrix feeds the head stage; a route whose head
            # support is deferred completes with its evaluation evidence only.
            return False
        if not _provenance_matches(roots["post"] / "slurm_provenance.json"):
            return False
        identity = read_json(identity_path)
        return (
            read_json(complete).get("status") == "completed"
            and identity.get("config_name") == config.get("name")
            and identity.get("stage") == stage
            and int(identity.get("fold", -1)) == int(fold)
            and identity.get("run_id") == run_id
            and identity.get("modality") == config.get("modality")
            and identity.get("checkpoint_dir") == str((roots["train"] / "best_model").resolve())
            and identity.get("subjects_per_class") == (subjects_per_class if stage == "smoke" else None)
            and _identity_hashes_match(identity, expected_identity, split_key="split_hash")
        )
    if kind == "head":
        complete = roots["post"] / "heads" / "heads_complete.json"
        identity_path = roots["post"] / "heads" / "heads_identity.json"
        if not complete.is_file() or not identity_path.is_file():
            return False
        if not _provenance_matches(roots["post"] / "heads" / "slurm_provenance.json"):
            return False
        identity = read_json(identity_path)
        expected_trials = int(trials if trials is not None else 150)
        return (
            read_json(complete).get("status") == "completed"
            and identity.get("stage") == stage
            and int(identity.get("fold", -1)) == int(fold)
            and identity.get("run_id") == run_id
            and identity.get("feature_metadata") == str((roots["post"] / "features" / "feature_metadata.json").resolve())
            and int(identity.get("optuna_trials", -1)) == expected_trials
            and _identity_hashes_match(identity, expected_identity, split_key="split_hash")
        )
    raise ValueError(kind)


def _final_epoch(config: dict[str, Any], run_id: str, modality: str) -> int:
    values: list[int] = []
    for fold in range(5):
        path = _run_roots(config, run_id, "cv", fold)["train"] / "logs" / "selected_checkpoint.json"
        if not path.is_file():
            raise FileNotFoundError(f"CV selection artifact is missing for final {modality} fold {fold}: {path}")
        values.append(int(read_json(path)["selected_epoch"]))
    result = int(math.floor(float(median(values)) + 0.5))
    if result < 1 or result > 20:
        raise ValueError(f"Invalid rounded median selected epoch for {modality}: {values} -> {result}")
    return result


def _final_epoch_for_dry_run(
    config: dict[str, Any], run_id: str, modality: str
) -> int | None:
    """Use the frozen CV epoch when a post-CV dry-run can resolve it.

    Dry-runs remain useful before CV exists, so missing gate artifacts do not
    fail planning. Once the passed CV audit and all five selections exist,
    however, using the real median epoch lets restart checks recognize a
    completed final training job instead of comparing it with the default 20.
    """

    audit_path = Path(config["output_dirs"]["merged_root"]) / run_id / "cv" / "acceptance_audit.json"
    if not audit_path.is_file() or read_json(audit_path).get("status") != "passed":
        return None
    selection_paths = [
        _run_roots(config, run_id, "cv", fold)["train"]
        / "logs"
        / "selected_checkpoint.json"
        for fold in range(5)
    ]
    if not all(path.is_file() for path in selection_paths):
        return None
    return _final_epoch(config, run_id, modality)


def _check_final_gate(config: dict[str, Any], run_id: str, modality: str) -> Path:
    path = _run_roots(config, run_id, "cv", 0)["post"] / "acceptance_audit.json"
    # The audit is written once per modality at the stage root, not per fold.
    path = Path(config["output_dirs"]["merged_root"]) / run_id / "cv" / "acceptance_audit.json"
    if not path.is_file():
        raise FileNotFoundError(f"Final stage requires a passed CV audit for {modality}: {path}")
    payload = read_json(path)
    if payload.get("status") != "passed":
        raise ValueError(f"Final stage refused because the CV audit did not pass: {path}")
    return path


def _resolved_route_resources(
    config: dict[str, Any], component_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Resolve the train/postprocess GPU shape for one Qwen3 merged contract.

    The merged contract declares its shape: ``execution.train_nodes`` and
    ``execution.qwen_gpus`` (GPUs per node) for the FSDP training lane, and
    ``execution.postprocess_gpus`` for the sharded evaluation, which must agree
    with the components' ``resources.eval_gpus_per_node``. The FSDP recipe keeps
    an effective global batch of 128, so the accumulated per-rank batch must
    match the declared rank count (nodes x GPUs per node).
    """
    execution = config.get("execution") or {}
    training = config.get("training") or {}
    train_nodes = int(execution.get("train_nodes") or 1)
    train_gpus = int(execution.get("qwen_gpus") or 4)
    if train_nodes < 1 or train_gpus < 1:
        raise ValueError("execution.train_nodes and execution.qwen_gpus must be positive")
    resources = validate_merged_resources(config, component_records)
    postprocess_gpus = int(resources["eval_gpus_per_node"])
    per_device = int(training.get("per_device_train_batch_size", 1))
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    world_size = train_nodes * train_gpus
    effective_batch = per_device * accumulation * world_size
    if effective_batch != 128:
        raise ValueError(
            f"Qwen3 merged FSDP keeps an effective global batch of 128; {world_size} rank(s) "
            f"({train_nodes} node(s) x {train_gpus}) with per_device={per_device} "
            f"accumulation={accumulation} give {effective_batch}."
        )
    return {
        "train_nodes": train_nodes,
        "train_gpus": train_gpus,
        "world_size": world_size,
        "postprocess_gpus": postprocess_gpus,
        "evaluation_resources": resources,
    }


def _runtime_override_tokens(
    config: dict[str, Any],
    *,
    input_root: str | Path | None,
    pooled_runtime_root: str | Path | None,
) -> list[str]:
    """Repoint the read-only component inputs for an isolated deployment.

    Relative component manifest/metadata paths resolve against PROJECT_ROOT,
    which in a managed deployment is the immutable source directory. The
    prebuilt inputs live outside it, so ``input_root`` rewrites every relative
    component path, and ``pooled_runtime_root`` repoints the pooled Turkish
    component at its task runtime exactly like the harmonized preparation flow
    does. The tokens travel to every stage through OVERRIDES_JSON_B64.
    """
    tokens: list[str] = []
    components = list(config.get("components") or [])
    if input_root is not None:
        root = Path(input_root)
        for index, component in enumerate(components):
            for field in ("manifest_path", "metadata_path"):
                raw = str(component.get(field) or "")
                if raw and not Path(raw).is_absolute():
                    tokens.append(f"--set=components.{index}.{field}={root / raw}")
    if pooled_runtime_root is not None:
        root = Path(pooled_runtime_root)
        for index, component in enumerate(components):
            if str(component.get("name")) != "turkish":
                continue
            english = "harmonized_en" in str(component.get("manifest_path", ""))
            manifest_root = root / ("manifests_en" if english else "manifests")
            split_root = root / ("splits_en" if english else "splits")
            tokens.append(
                f"--set=components.{index}.manifest_path="
                f"{manifest_root / 'turkish' / 'turkish_manifest.jsonl'}"
            )
            tokens.append(
                f"--set=components.{index}.metadata_path="
                f"{split_root / 'turkish' / 'turkish_manifest_metadata.json'}"
            )
    return tokens


def build_job_specs(
    configs: list[Path], *, stage: str, run_id: str, dry_run: bool, smoke_subjects: int,
    smoke_epochs: int, smoke_trials: int, max_concurrent_trains: int = 0,
    max_concurrent_postprocess: int = 0, github_issue: int | None = None,
    github_pr: int | None = None, overrides: list[str] | None = None,
    log_root: str | None = None, input_root: str | Path | None = None,
    pooled_runtime_root: str | Path | None = None,
) -> dict[str, Any]:
    if stage not in {"smoke", "cv", "final"}:
        raise ValueError(stage)
    if (github_issue is None) != (github_pr is None):
        raise ValueError("GitHub Issue and PR provenance must be provided together.")
    if github_issue is not None and (github_issue < 1 or github_pr is None or github_pr < 1):
        raise ValueError("GitHub Issue and PR provenance must use positive integers.")
    override_tokens = list(overrides or [])
    if stage == "smoke":
        configs = [path for path in configs]
        if len(configs) < 1:
            raise ValueError("Merged smoke requires at least one merged config.")
        folds = [0]
    elif stage == "cv":
        folds = list(range(5))
    else:
        folds = [0]
    jobs: list[dict[str, Any]] = []
    config_identities: list[dict[str, Any]] = []
    route_readiness: dict[str, Any] = {}
    for config_path in configs:
        declared_config = load_merged_config(config_path, override_tokens)
        resolved_tokens = override_tokens + _runtime_override_tokens(
            declared_config,
            input_root=input_root,
            pooled_runtime_root=pooled_runtime_root,
        )
        config = load_merged_config(config_path, resolved_tokens)
        modality = str(config["modality"])
        head_trials = _head_trials(config, stage=stage, smoke_trials=smoke_trials)
        decision = merged_route_decision(config, stage=stage)
        evaluation_contract: dict[str, Any] | None = None
        route_resources: dict[str, Any] | None = None
        if decision["qwen3"] and decision["declared"]:
            component_records, unresolved_components = merged_component_contract_records(config)
            if unresolved_components:
                raise ValueError(
                    f"{QWEN3_MERGED_PREREQUISITE}: cannot resolve merged components "
                    f"{unresolved_components} for {config_path}; a Qwen3 contract must keep all "
                    "five component configs resolvable."
                )
            # The decision rule and the evidence view are validated separately,
            # per component, before any job is planned or submitted.
            evaluation_contract = validate_evaluation_contract(config, component_records)
            route_resources = _resolved_route_resources(config, component_records)
            decision["evaluation_contract"] = evaluation_contract
            decision["resources"] = route_resources
        if not decision["allowed"]:
            if not dry_run:
                raise ValueError(decision["reason"])
            print(
                f"WARNING: {decision['reason']} Dry-run plan only, execution is refused.",
                file=sys.stderr,
            )
        route_readiness[str(config_path)] = {
            "contract": decision["contract"],
            "qwen3": decision["qwen3"],
            "backends": decision["backends"],
            "allowed": decision["allowed"],
            "production_ready": (
                bool((QWEN3_CONTRACT_READINESS.get(decision["contract"]) or {}).get("production_ready", False))
                if decision["qwen3"]
                else None
            ),
            "head_ready": decision["head_ready"],
            "head_deferred_reason": decision["head_deferred_reason"],
            "reason": decision["reason"],
            "evidence": decision["evidence"],
            "sample_prediction_mode": (evaluation_contract or {}).get("sample_prediction_mode"),
            "evaluation_view": (evaluation_contract or {}).get("evaluation_view"),
        }
        config_identities.append(
            {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
                "modality": modality,
                "blocked_backends": decision["backends"],
                "qwen3": decision["qwen3"],
                "allowed": decision["allowed"],
                "head_ready": decision["head_ready"],
            }
        )
        if stage == "final":
            # This gate is evaluated by the real submit path. A dry-run still
            # reports the exact deterministic epoch input when available.
            final_epochs = (
                _final_epoch_for_dry_run(config, run_id, modality)
                if dry_run
                else None
            )
            if not dry_run:
                _check_final_gate(config, run_id, modality)
                final_epochs = _final_epoch(config, run_id, modality)
        else:
            final_epochs = None
        model_backend = str(config.get("model_backend") or "")
        train_nodes = int(route_resources["train_nodes"]) if route_resources else 1
        train_gpus = int(route_resources["train_gpus"]) if route_resources else 4
        postprocess_gpus = int(route_resources["postprocess_gpus"]) if route_resources else 1
        for fold in folds:
            roots = _run_roots(config, run_id, stage, fold)
            chain = [
                {
                    "kind": "train",
                    "config": str(config_path),
                    "modality": modality,
                    "stage": stage,
                    "fold": fold,
                    "run_id": run_id,
                    "run_root": str(roots["train"]),
                    "model_backend": model_backend,
                    "resource": {
                        "nodes": train_nodes,
                        "gpus": train_gpus,
                        "cpus": 20 * train_gpus,
                        "time": config["execution"]["qwen_time"],
                    },
                    "epochs": final_epochs if stage == "final" else (smoke_epochs if stage == "smoke" else None),
                    "subjects_per_class": smoke_subjects if stage == "smoke" else None,
                },
                {
                    "kind": "postprocess",
                    "config": str(config_path),
                    "modality": modality,
                    "stage": stage,
                    "fold": fold,
                    "run_id": run_id,
                    "run_root": str(roots["post"]),
                    "model_backend": model_backend,
                    "checkpoint_dir": str(roots["train"] / "best_model"),
                    "subjects_per_class": smoke_subjects if stage == "smoke" else None,
                    "resource": {"gpus": postprocess_gpus, "cpus": 20 * postprocess_gpus, "time": config["execution"]["postprocess_time"]},
                },
            ]
            if decision["head_ready"]:
                chain.append(
                    {
                        "kind": "head",
                        "config": str(config_path),
                        "modality": modality,
                        "stage": stage,
                        "fold": fold,
                        "run_id": run_id,
                        "run_root": str(roots["post"] / "heads"),
                        "model_backend": model_backend,
                        "features_dir": str(roots["post"] / "features"),
                        "resource": {"gpus": 0, "cpus": 20, "time": config["execution"]["head_time"]},
                        "trials": head_trials,
                    }
                )
            previous_id: str | None = None
            for job in chain:
                job_key = f"{modality}:{stage}:fold_{fold}:{job['kind']}"
                job["job_key"] = job_key
                job["expected_job_id"] = _job_id(run_id, modality, stage, fold, job["kind"])
                job["dependency_job_key"] = (
                    f"{modality}:{stage}:fold_{fold}:{'train' if job['kind'] == 'postprocess' else 'postprocess'}"
                    if job["kind"] != "train" else None
                )
                job["qwen3_route"] = decision["qwen3"]
                job["head_deferred"] = None if decision["head_ready"] else decision["head_deferred_reason"]
                job["overrides"] = list(resolved_tokens) if resolved_tokens else None
                if log_root:
                    job["log_root"] = str(log_root)
                job["completed_before_submission"] = _completed(
                    config,
                    config_path,
                    run_id,
                    stage,
                    fold,
                    job["kind"],
                    epochs=job.get("epochs"),
                    subjects_per_class=job.get("subjects_per_class"),
                    trials=job.get("trials"),
                    head_ready=decision["head_ready"],
                )
                if job["completed_before_submission"]:
                    job["state"] = "skipped_compatible_complete"
                    previous_id = None
                else:
                    job["state"] = "planned"
                    if not dry_run and previous_id:
                        job["dependency_job_id"] = previous_id
                    previous_id = job["expected_job_id"]
                jobs.append(job)
    blocked_paths = {
        str(record["path"]) for record in config_identities if not record.get("allowed")
    }
    for job in jobs:
        job["blocked_prerequisite"] = str(job["config"]) in blocked_paths
    for order, job in enumerate(jobs):
        job["submission_order"] = order
    _apply_concurrency_lanes(jobs, kind="train", limit=int(max_concurrent_trains))
    _apply_concurrency_lanes(jobs, kind="postprocess", limit=int(max_concurrent_postprocess))
    # The default production invocation has three modalities (45 CV or 9
    # final jobs), while targeted retries may intentionally pass one or more
    # configs. Count the actual planned chain so retry registries remain
    # truthful without changing the default protocol scope. A route whose head
    # kind is deferred plans train + postprocess only.
    expected = sum(3 if record["head_ready"] else 2 for record in config_identities) * len(folds)
    plan_identity = {
        "stage": stage,
        "configs": config_identities,
        "smoke_subjects": int(smoke_subjects),
        "smoke_epochs": int(smoke_epochs),
        "smoke_trials": int(smoke_trials),
        "max_concurrent_trains": int(max_concurrent_trains),
        "max_concurrent_postprocess": int(max_concurrent_postprocess),
        "research": {"github_issue": github_issue, "github_pr": github_pr},
    }
    # The new identity fields only appear when they differ from the historical
    # defaults, so an unchanged legacy rerun keeps its recorded plan hash.
    if override_tokens:
        plan_identity["overrides"] = override_tokens
    if any(record["qwen3"] for record in config_identities):
        plan_identity["route_readiness"] = route_readiness
    stage_plan = {
        "stage": stage,
        "plan_identity": plan_identity,
        "plan_hash": canonical_sha256(plan_identity),
        "expected_fresh_job_count": expected,
    }
    blocked_reasons = sorted(
        {
            str(route_readiness[str(record["path"])]["reason"])
            for record in config_identities
            if not record.get("allowed")
        }
    )
    return {
        "schema_version": "symmetric_merged_job_registry.v2",
        "run_id": run_id,
        "stage": stage,
        "source_commit": _source_commit(),
        "reservation": _reservation() or None,
        "research": {"github_issue": github_issue, "github_pr": github_pr},
        "overrides": override_tokens,
        "log_root": str(log_root) if log_root else None,
        "route_readiness": route_readiness,
        "plan_identity": plan_identity,
        "plan_hash": stage_plan["plan_hash"],
        "stages": [stage],
        "stage_plans": {stage: stage_plan},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "expected_fresh_job_count": expected,
        "planned_job_count": sum(job["state"] == "planned" for job in jobs),
        "skipped_job_count": sum(job["state"] != "planned" for job in jobs),
        "blocked_prerequisite": sorted(
            str(record["path"]) for record in config_identities if not record.get("allowed")
        ),
        "blocked_reason": "; ".join(blocked_reasons) if blocked_reasons else None,
        "jobs": jobs,
    }


def _submit_job(
    job: dict[str, Any], *, worker: Path, dependency_id: str | None,
    throttle_dependency_id: str | None, overrides: list[str] | None = None,
) -> str:
    export_values = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "CONFIG": job["config"],
        "STAGE": job["stage"],
        "FOLD": str(job["fold"]),
        "RUN_ID": job["run_id"],
        "SOURCE_COMMIT": _source_commit(),
    }
    backend = str(job.get("model_backend") or "")
    if backend == "gemma4":
        export_values["ENV_ACTIVATE"] = os.environ.get(
            "GEMMA_ENV",
            "/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1",
        ) + "/bin/activate"
        export_values["MODEL_PATH"] = os.environ.get(
            "GEMMA4_MODEL_PATH",
            "/gpfs/projects/etur92/ozu647717/models/gemma-4-12B-it/"
            "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        )
    elif backend == "qwen38":
        export_values["ENV_ACTIVATE"] = os.environ.get(
            "QWEN38_ENV_ACTIVATE",
            "/gpfs/projects/etur92/ozu647717/venvs/qwen38_fsdp_fastpath_20260921",
        ) + "/bin/activate"
    elif backend == "qwen3omni":
        export_values["ENV_ACTIVATE"] = os.environ.get(
            "QWEN3OMNI_ENV_ACTIVATE",
            "/gpfs/projects/etur92/ozu647717/venvs/qwen3omni",
        ) + "/bin/activate"
    override_tokens = list(overrides or [])
    if override_tokens:
        export_values["OVERRIDES_JSON_B64"] = base64.b64encode(
            json.dumps(override_tokens).encode("utf-8")
        ).decode("ascii")
    if job.get("log_root"):
        export_values["LOG_ROOT"] = str(job["log_root"])
    if job["kind"] == "train":
        export_values["NPROC_PER_NODE"] = str(int(job["resource"]["gpus"]))
        export_values["NNODES"] = str(int(job["resource"].get("nodes", 1)))
    if job["kind"] == "postprocess":
        export_values["POSTPROCESS_GPUS"] = str(int(job["resource"]["gpus"]))
    for key in ("epochs", "subjects_per_class", "trials", "checkpoint_dir", "features_dir"):
        if job.get(key) is not None:
            export_values[key.upper()] = str(job[key])
    export_text = "ALL," + ",".join(f"{key}={value}" for key, value in export_values.items())
    arguments = ["sbatch", "--parsable", f"--job-name=sym-{job['modality'][:4]}-{job['stage'][:4]}-{job['fold']}-{job['kind'][:4]}"]
    gpus = int(job["resource"]["gpus"])
    cpus = int(job["resource"]["cpus"])
    nodes = int(job["resource"].get("nodes", 1))
    if nodes > 1 and gpus > 0:
        # A multi-node training lane: one task per node, each holding that node's
        # GPUs and CPUs, and the worker expands the node rank through srun. The
        # single-node branch below stays byte-identical to the verified shape.
        arguments.append(f"--nodes={nodes}")
        arguments.append(f"--ntasks={nodes}")
        arguments.append("--ntasks-per-node=1")
        arguments.append(f"--cpus-per-task={cpus}")
        arguments.append(f"--gres=gpu:{gpus}")
    elif gpus > 0:
        # Sbatch command-line flags override the worker's script defaults, so
        # the configured resource shape travels with the job contract.
        arguments.append(f"--gres=gpu:{gpus}")
        arguments.append(f"--cpus-per-task={cpus}")
    dependencies: list[str] = []
    if dependency_id:
        dependencies.append(f"afterok:{dependency_id}")
    if throttle_dependency_id:
        dependencies.append(f"afterany:{throttle_dependency_id}")
    if dependencies:
        arguments.append(f"--dependency={','.join(dependencies)}")
    reservation = _reservation()
    if reservation:
        arguments.append(f"--reservation={reservation}")
    arguments.extend([f"--export={export_text}", str(worker)])
    output = subprocess.check_output(arguments, cwd=PROJECT_ROOT, text=True).strip()
    return output.split(";", 1)[0]


def submit_registry(registry: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    def _successful_slurm_job(job: dict[str, Any]) -> bool:
        state_tokens = str(job.get("observed_state", "")).upper().split(None, 1)
        return bool(state_tokens) and state_tokens[0] == "COMPLETED" and str(job.get("exit_code", "")) == "0:0"

    previous_ids: dict[str, str] = {
        str(job["job_key"]): str(job["job_id"])
        for job in registry.get("jobs", [])
        if job.get("job_id") and not str(job.get("job_id")).startswith("dry_")
        and not _successful_slurm_job(job)
    }
    worker_by_kind = {
        "train": PROJECT_ROOT / "scripts/run_symmetric_merged_train_slurm.sh",
        "postprocess": PROJECT_ROOT / "scripts/run_symmetric_merged_postprocess_slurm.sh",
        "head": PROJECT_ROOT / "scripts/run_symmetric_merged_head_slurm.sh",
    }
    active_before = sum(job.get("state") == "planned" for job in registry.get("jobs", []))
    for job in registry["jobs"]:
        if job["state"] != "planned":
            continue
        dependency_job_key = job.get("dependency_job_key")
        dependency_id = previous_ids.get(dependency_job_key) if dependency_job_key else None
        throttle_dependency_job_key = job.get("throttle_dependency_job_key")
        throttle_dependency_id = (
            previous_ids.get(throttle_dependency_job_key)
            if throttle_dependency_job_key else None
        )
        if dry_run:
            submitted_id = job["expected_job_id"]
        else:
            submitted_id = _submit_job(
                job,
                worker=worker_by_kind[job["kind"]],
                dependency_id=dependency_id,
                throttle_dependency_id=throttle_dependency_id,
                # The per-config resolved tokens carry the isolated-runtime
                # component input paths; the registry-level list is only the
                # explicit extra --set overrides. Prefer the job's own tokens so
                # a deployed worker never falls back to PROJECT_ROOT-relative
                # component paths.
                overrides=job.get("overrides") or registry.get("overrides") or [],
            )
        job["job_id"] = submitted_id
        if dependency_id:
            job["dependency_job_id"] = dependency_id
        if throttle_dependency_id:
            job["throttle_dependency_job_id"] = throttle_dependency_id
        job["submission_time_utc"] = datetime.now(timezone.utc).isoformat()
        job["state"] = "planned_dry_run" if dry_run else "submitted"
        previous_ids[job["job_key"]] = submitted_id
    # A dry-run or a restart can carry already-submitted jobs forward without
    # visiting them in the loop above.  Refresh their recorded dependency IDs
    # from the authoritative job-key map so the registry cannot retain a
    # stale dry-run ID even though Slurm received the real dependency.
    for job in registry["jobs"]:
        dependency_key = job.get("dependency_job_key")
        dependency_id = previous_ids.get(str(dependency_key)) if dependency_key else None
        if dependency_id and not str(dependency_id).startswith("dry_"):
            job["dependency_job_id"] = dependency_id
        elif not dependency_id:
            job.pop("dependency_job_id", None)
        throttle_key = job.get("throttle_dependency_job_key")
        throttle_id = previous_ids.get(str(throttle_key)) if throttle_key else None
        if throttle_id and not str(throttle_id).startswith("dry_"):
            job["throttle_dependency_job_id"] = throttle_id
        elif not throttle_id:
            job.pop("throttle_dependency_job_id", None)
    registry["submission_mode"] = "dry_run" if dry_run else "sbatch"
    registry["terminal"] = False
    registry["planned_job_count"] = active_before
    registry["skipped_job_count"] = len(registry.get("jobs", [])) - active_before
    return registry


def merge_existing_registry(registry: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    """Carry forward submitted/terminal jobs so reruns are restart-safe."""

    old_jobs = {str(job.get("job_key")): job for job in existing.get("jobs", [])}
    # Slurm can expose dependency failures with several spellings (for
    # example, ``DependencyNeverSatisfied``).  Normalize the state before
    # deciding whether an old job can be carried forward.  A failed train
    # also invalidates any already-submitted descendants: leaving those old
    # jobs in place would keep them attached to the failed job ID forever.
    failed_states = {
        "failed",
        "cancelled",
        "canceled",
        "timeout",
        "oom",
        "outofmemory",
        "nodefail",
        "preempted",
        "dependencyneversatisfied",
    }

    def state_token(value: Any) -> str:
        return "".join(character for character in str(value).lower() if character.isalnum())

    retry_job_keys: set[str] = set()
    for job in registry.get("jobs", []):
        old = old_jobs.get(str(job["job_key"]))
        if not old:
            continue
        # Preserve retry metadata and the last known dependency while a
        # planned dry-run registry is promoted to a real submission.  The
        # dependency is refreshed from current job IDs in submit_registry.
        if "retry" in old:
            job["retry"] = old["retry"]
        if "dependency_job_id" in old:
            job["dependency_job_id"] = old["dependency_job_id"]
        old_state = state_token(old.get("observed_state", old.get("state", "")))
        old_job_id = old.get("job_id")
        terminal_success_without_artifact = (
            old_state == "completed" and not job.get("completed_before_submission")
        )
        if (
            old_job_id
            and not str(old_job_id).startswith("dry_")
            and old_state not in failed_states
            and not terminal_success_without_artifact
        ):
            job["state"] = old.get("state", "submitted")
            job["job_id"] = old_job_id
            for key in ("submission_time_utc", "observed_state", "exit_code"):
                if key in old:
                    job[key] = old[key]
        elif old_state in failed_states or terminal_success_without_artifact:
            job["retry"] = int(old.get("retry", 0)) + 1
            retry_job_keys.add(str(job["job_key"]))

    def reset_for_retry(job: dict[str, Any]) -> None:
        job["state"] = "planned"
        for key in (
            "job_id",
            "dependency_job_id",
            "throttle_dependency_job_id",
            "submission_time_utc",
            "observed_state",
            "exit_code",
            "elapsed",
            "node_list",
            "allocated_cpus",
        ):
            job.pop(key, None)

    # Propagate a retry through the dependency chain.  The loop is
    # intentionally order-independent so a registry loaded from an older
    # run cannot submit a postprocess/head job before its replacement train
    # job.  Compatible artifacts remain reusable if they already exist.
    changed = True
    while changed:
        changed = False
        for job in registry.get("jobs", []):
            job_key = str(job["job_key"])
            dependency_key = job.get("dependency_job_key")
            if not dependency_key or str(dependency_key) not in retry_job_keys:
                continue
            if job.get("completed_before_submission"):
                continue
            old = old_jobs.get(job_key)
            if old:
                job["retry"] = int(old.get("retry", 0)) + 1
            reset_for_retry(job)
            if job_key not in retry_job_keys:
                retry_job_keys.add(job_key)
                changed = True
    current_keys = {str(job.get("job_key")) for job in registry.get("jobs", [])}
    for old in existing.get("jobs", []):
        if str(old.get("job_key")) not in current_keys:
            registry.setdefault("jobs", []).append(old)
    dependency_order = {"train": 0, "postprocess": 1, "head": 2}
    registry["jobs"].sort(
        key=lambda job: (
            int(job.get("submission_order", 10**9)),
            str(job.get("stage", "")),
            str(job.get("modality", "")),
            int(job.get("fold", 0)),
            dependency_order.get(str(job.get("kind", "")), 99),
            str(job.get("kind", "")),
        )
    )
    return registry


def _stage_plans(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Read stage plans from v2 registries and legacy single-stage files."""

    plans = registry.get("stage_plans")
    if isinstance(plans, dict) and plans:
        return {
            str(stage): dict(payload)
            for stage, payload in plans.items()
            if isinstance(payload, dict)
        }
    stage = str(registry.get("stage", "")).strip()
    if stage and stage != "multi":
        return {
            stage: {
                "stage": stage,
                "plan_identity": registry.get("plan_identity", {}),
                "plan_hash": registry.get("plan_hash"),
                "expected_fresh_job_count": registry.get("expected_fresh_job_count"),
            }
        }
    return {}


def _set_combined_registry_metadata(
    registry: dict[str, Any], stage_plans: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Make one registry authoritative for all stages sharing a run ID."""

    ordered = {stage: stage_plans[stage] for stage in sorted(stage_plans)}
    stages = list(ordered)
    registry["schema_version"] = "symmetric_merged_job_registry.v2"
    registry["stages"] = stages
    registry["stage"] = stages[0] if len(stages) == 1 else "multi"
    registry["stage_plans"] = ordered
    registry["plan_identity"] = {
        "stages": {
            stage: ordered[stage].get("plan_identity", {})
            for stage in stages
        }
    }
    research_values = {
        canonical_sha256((ordered[stage].get("plan_identity") or {}).get("research")):
            (ordered[stage].get("plan_identity") or {}).get("research")
        for stage in stages
    }
    if len(research_values) != 1:
        raise ValueError("Merged stages have incompatible GitHub Issue/PR provenance.")
    registry["research"] = next(iter(research_values.values()))
    registry["plan_hash"] = canonical_sha256(
        {stage: ordered[stage].get("plan_hash") for stage in stages}
    )
    registry["expected_job_count"] = sum(
        int(ordered[stage].get("expected_fresh_job_count", 0) or 0)
        for stage in stages
    )
    return registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "cv", "final"), required=True)
    parser.add_argument("--config", action="append", type=Path, dest="configs")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--smoke-subjects", type=int, default=2)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--smoke-trials", type=int, default=2)
    parser.add_argument("--max-concurrent-trains", type=int, default=0)
    parser.add_argument("--max-concurrent-postprocess", type=int, default=0)
    parser.add_argument("--github-issue", type=int)
    parser.add_argument("--github-pr", type=int)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="set_overrides",
        metavar="KEY=VALUE",
        help="Extra config override applied to every stage and its workers (repeatable).",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Root that relative component manifest/metadata paths resolve against; an "
        "isolated deployment reads its prebuilt inputs outside the source directory.",
    )
    parser.add_argument(
        "--pooled-runtime-root",
        type=Path,
        default=None,
        help="Runtime root holding the prebuilt pooled Turkish manifests/splits.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help="Writable log root exported to every worker (isolated runtime).",
    )
    return parser.parse_args()


def _normalized_extra_overrides(raw: list[str]) -> list[str]:
    tokens: list[str] = []
    for token in raw:
        text = str(token).strip()
        if not text:
            continue
        tokens.append(text if text.startswith("--set") else f"--set={text}")
    return tokens


def main() -> None:
    args = parse_args()
    configs = [resolve_project_path(value) for value in (args.configs or list(CONFIG_BY_MODALITY.values()))]
    extra_overrides = _normalized_extra_overrides(args.set_overrides)
    run_id = args.run_id
    if not run_id:
        identity = {
            "stage": args.stage,
            "configs": [str(path) + ":" + sha256_file(path) for path in configs],
            "source_commit": _source_commit(),
        }
        if extra_overrides:
            identity["overrides"] = extra_overrides
        run_id = f"symmetric_merged_{args.stage}_{canonical_sha256(identity)[:12]}"
    registry = build_job_specs(
        configs,
        stage=args.stage,
        run_id=run_id,
        dry_run=args.dry_run,
        smoke_subjects=args.smoke_subjects,
        smoke_epochs=args.smoke_epochs,
        smoke_trials=args.smoke_trials,
        max_concurrent_trains=args.max_concurrent_trains,
        max_concurrent_postprocess=args.max_concurrent_postprocess,
        github_issue=args.github_issue,
        github_pr=args.github_pr,
        overrides=extra_overrides,
        log_root=str(args.log_root) if args.log_root else None,
        input_root=args.input_root,
        pooled_runtime_root=args.pooled_runtime_root,
    )
    registry_path = resolve_project_path(args.registry) if args.registry else PROJECT_ROOT / "outputs/symmetric_merged_jobs" / f"{run_id}.json"
    if registry_path.exists():
        existing = read_json(registry_path)
        if existing.get("run_id") != run_id:
            raise ValueError(f"Refusing colliding symmetric merged registry: {registry_path}")
        # A rerun may reuse a stage only when it is the same protocol plan.
        # A new stage may be appended to the same run ID: final needs the CV
        # artifacts and epoch selections under that shared run root.
        existing_stage_plans = _stage_plans(existing)
        current_stage_plan = _stage_plans(registry)[args.stage]
        if args.stage in existing_stage_plans:
            if existing_stage_plans[args.stage].get("plan_hash") != current_stage_plan.get("plan_hash"):
                raise ValueError(f"Existing registry has an incompatible protocol plan: {registry_path}")
        existing_configs = {
            str(job.get("job_key")): str(job.get("config"))
            for job in existing.get("jobs", [])
            if job.get("job_key") and str(job.get("stage")) == args.stage
        }
        current_configs = {
            str(job.get("job_key")): str(job.get("config"))
            for job in registry.get("jobs", [])
            if job.get("job_key")
        }
        if existing_configs and existing_configs != current_configs:
            raise ValueError(f"Existing registry has incompatible job/config identities: {registry_path}")
        registry = merge_existing_registry(registry, existing)
        existing_stage_plans[args.stage] = current_stage_plan
        registry = _set_combined_registry_metadata(registry, existing_stage_plans)
    else:
        registry = _set_combined_registry_metadata(registry, _stage_plans(registry))
    registry = submit_registry(registry, dry_run=args.dry_run)
    save_json(registry, registry_path)
    print(json.dumps({
        "status": "dry_run_complete" if args.dry_run else "submitted",
        "registry": str(registry_path),
        "run_id": run_id,
        "stage": args.stage,
        "expected_fresh_job_count": registry["expected_fresh_job_count"],
        "planned_or_submitted_job_count": registry["planned_job_count"],
        "skipped_job_count": registry["skipped_job_count"],
        "job_ids": [job.get("job_id") for job in registry["jobs"] if job.get("job_id")],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
