#!/usr/bin/env python3
"""Incremental compact collection and official validation for the window-cap lane.

Runs every watcher cycle, before head dispatch and refills, so mask or output
defects surface as soon as a unit completes instead of after all 378 fits.

Selection rules (fail closed):

* fits: exact owned latest attempts whose train and best_eval job ids are
  scheduler-confirmed ``COMPLETED 0:0`` (the same gate the head planner uses);
  superseded attempts and unknown states are preserved, never processed;
* heads: production registry entries whose keys are inside the 378 arm-explicit
  expected set (smokes/controls/legacy excluded) and whose extract and
  classifier job ids are scheduler-confirmed ``COMPLETED 0:0``;
* a unit already validated in the ledger is skipped; every processed unit is
  appended to the append-only ledger with its exact commands and results.

Official flow per fit: ``exp.py collect --execute`` (compact evidence, no
adapters), then one ``exp.py status`` reconciliation for the batch (the remote
sidecar only records RUNNING; the official reconciler appends terminal events
and advances the fold to COMPLETED_ON_MN5), then ``exp.py validate``. A fit is
marked validated only when the local sidecar state is actually
``LOCALLY_VALIDATED``; a rc-0 tool call alone is never enough.

Official flow per head: ``qwen3_heads_dispatch.py collect --registry <r>``,
then the both-variant train-mask membership and parent-binding checks, then
``qwen3_heads_dispatch.py validate --registry <r>``. The checks use the exact
canonical contract: the parent fold's ``run_config.yaml`` (with
``config.training.window_cap``) and ``window_cap_mask.json``, the extraction
cache's ``extraction_metadata.json`` from ``entry.cache_dir`` (fetched compact
with its hash), and the head attempt's ``metadata.json`` plus both
``classifier/<variant>/classifier_metadata.json``. Parent checks bind
``entry.parent_attempt_id``/fold/seed; the head attempt identity is checked
separately.

``collection_progress.json`` records ``fits_submitted``, ``fits_validated``,
``heads_dispatched`` and ``heads_validated``; ``--status`` prints it. The
watcher must not exit while ``heads_validated < 378``.

Usage:
  python tools/qwen3_window_cap_collect_validate.py --campaign-dir <dir> [--execute] [--max-fits 4] [--max-heads 4]
  python tools/qwen3_window_cap_collect_validate.py --campaign-dir <dir> --status
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_CAMPAIGN_DIR = PROJECT_ROOT / "outputs" / "qwen3_train_window_cap_20261008"
SLUG = "feat-qwen3-train-window-cap-20261008"
EXPECTED_CHAINS = 378
VARIANTS = ("logreg_raw", "xgb_raw")
DEFAULT_SCHEDULER_HOST = "ozu647717@alogin2.bsc.es"
DEFAULT_TRANSFER_HOST = "ozu647717@transfer1.bsc.es"

try:  # package import (tests, lane shim)
    from . import qwen3_window_cap_head_plan as planner
except ImportError:  # direct script execution
    sys.path.insert(0, str(TOOLS_DIR))
    import qwen3_window_cap_head_plan as planner

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configured per campaign dir by configure(); defaults keep the operator path.
LANE = DEFAULT_CAMPAIGN_DIR
MATRIX = LANE / "production_matrix.json"
SUBMISSIONS = LANE / "production_submissions.jsonl"
REGISTRY = LANE / "head_submissions.jsonl"
LEDGER = LANE / "incremental_collection.jsonl"
PROGRESS = LANE / "collection_progress.json"
RAW = LANE / "effective_fractions_raw"
LOCAL_RUN_ROOT = PROJECT_ROOT / "output_model" / "qwen3_train_window_cap_20261008"
DISPATCH_TOOL = TOOLS_DIR / "qwen3_heads_dispatch.py"
EXP_TOOL = TOOLS_DIR / "exp.py"
SCHEDULER_HOST = DEFAULT_SCHEDULER_HOST
TRANSFER_HOST = DEFAULT_TRANSFER_HOST
SUBMISSION_LOCK = LANE / "submission.lock"
CONFOUNDED = LANE / "head_confounded_attempts.jsonl"


def configure(
    campaign_dir: Path | str | None = None,
    *,
    local_run_root: Path | str | None = None,
    scheduler_host: str | None = None,
    transfer_host: str | None = None,
) -> None:
    global LANE, MATRIX, SUBMISSIONS, REGISTRY, LEDGER, PROGRESS, RAW, LOCAL_RUN_ROOT
    global SCHEDULER_HOST, TRANSFER_HOST, SUBMISSION_LOCK
    LANE = Path(campaign_dir or DEFAULT_CAMPAIGN_DIR).resolve()
    MATRIX = LANE / "production_matrix.json"
    SUBMISSIONS = LANE / "production_submissions.jsonl"
    REGISTRY = LANE / "head_submissions.jsonl"
    LEDGER = LANE / "incremental_collection.jsonl"
    PROGRESS = LANE / "collection_progress.json"
    RAW = LANE / "effective_fractions_raw"
    SUBMISSION_LOCK = LANE / "submission.lock"
    global CONFOUNDED
    CONFOUNDED = LANE / "head_confounded_attempts.jsonl"
    if local_run_root is not None:
        LOCAL_RUN_ROOT = Path(local_run_root)
    if scheduler_host is not None:
        SCHEDULER_HOST = str(scheduler_host)
    if transfer_host is not None:
        TRANSFER_HOST = str(transfer_host)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# append-only ledger
# ---------------------------------------------------------------------------


def load_ledger(path: Path | None = None) -> dict[str, dict]:
    """Last record per unit key (append-only file, last wins for selection)."""
    records: dict[str, dict] = {}
    ledger = path or LEDGER
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("key")
            if key:
                records[str(key)] = record
    return records


def append_ledger(record: dict, path: Path | None = None) -> None:
    ledger = path or LEDGER
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def fit_key(row: dict) -> str:
    return f"fit|{row['run_name']}|{row['attempt_id']}"


def select_ready_fits(
    rows: list[dict],
    scheduler_map: dict[str, dict],
    ledger: dict[str, dict],
    *,
    raw_root: Path | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Exact owned latest attempts with scheduler-confirmed train+eval 0:0."""
    ready: list[dict] = []
    reasons: dict[str, int] = {}

    def note(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    for row in rows:
        key = fit_key(row)
        record = ledger.get(key) or {}
        if record.get("validated") and str(record.get("attempt_id")) == str(row.get("attempt_id")):
            note("already validated")
            continue
        ok, reason = planner.readiness_local(row, raw_root=raw_root)
        if not ok:
            note(reason)
            continue
        ok, reason = planner.readiness_scheduler(row, scheduler_map, raw_root=raw_root)
        if not ok:
            note(reason)
            continue
        ready.append(row)
    return ready, reasons


def confounded_attempts() -> dict[str, dict]:
    """Exact attempts durably recorded as confounded, with full evidence.

    A record only counts with the exact attempt id, the full old deployment
    identity, the registry key and a non-empty mismatch-evidence object; a
    bare boolean flag is never accepted.
    """
    records: dict[str, dict] = {}
    if CONFOUNDED.is_file():
        for line in CONFOUNDED.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            attempt = str(entry.get("attempt_id") or "")
            deployment = str(entry.get("deployment_id") or "")
            key = str(entry.get("registry_key") or "")
            evidence = entry.get("mismatch_evidence")
            if (
                entry.get("confounded") is not True
                or not attempt
                or not deployment
                or not key
                or not isinstance(evidence, dict)
                or not evidence
            ):
                continue
            records[attempt] = entry
    return records


def head_job_ids(entry: dict) -> dict[str, str]:
    ids: dict[str, str] = {}
    for field in ("extract_job_id", "classifier_job_id"):
        value = str(entry.get(field) or "").strip()
        if value.isdigit():
            ids[field] = value
    return ids


def select_ready_heads(
    entries: list[dict],
    expected_keys: set[str],
    scheduler_map: dict[str, dict],
    ledger: dict[str, dict],
) -> tuple[list[dict], dict[str, int]]:
    """Production chains only, extract+classifier scheduler-confirmed 0:0."""
    entries = list(latest_registry_entries(entries).values())
    confounded = confounded_attempts()
    ready: list[dict] = []
    reasons: dict[str, int] = {}

    def note(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    for entry in entries:
        key = str(entry.get("registry_key") or "")
        if key not in expected_keys:
            note("not a production treatment chain")
            continue
        if str(entry.get("attempt_id")) in confounded:
            note("confounded old attempt (superseded)")
            continue
        record = ledger.get(key) or {}
        if record.get("validated") and str(record.get("attempt_id")) == str(entry.get("attempt_id")):
            note("already validated")
            continue
        ids = head_job_ids(entry)
        if set(ids) != {"extract_job_id", "classifier_job_id"}:
            note("missing numeric job ids")
            continue
        blocked = False
        for field, job_id in ids.items():
            info = scheduler_map.get(job_id)
            if info is None:
                note(f"scheduler UNKNOWN for {field} {job_id}")
                blocked = True
                break
            if info["state"] != "COMPLETED" or info["exit"] != "0:0":
                note(f"scheduler {info['state']} {info['exit']} for {field} {job_id}")
                blocked = True
                break
        if not blocked:
            ready.append(entry)
    return ready, reasons


# ---------------------------------------------------------------------------
# mask membership and parent binding (both variants)
# ---------------------------------------------------------------------------


WEIGHT_POLICY_D3TEC = "inverse_segments_per_response_rescaled_to_mean_one"
WEIGHT_POLICY_TEXT = "one_vector_per_subject_unweighted"
WEIGHT_POLICY_UNIFORM = "uniform_rows"
WEIGHT_POLICY_DAIC = "inverse_chunks_per_subject_rescaled_to_mean_one"


def expected_weight_policy(extraction: dict) -> str:
    """Canonical route weight policy from the extraction/parent metadata.

    Mirrors ``src.features.hidden_classifier_policy.response_normalized_sample_weights``
    branch order exactly (D3TEC audio, DAIC, pooled Turkish / D3TEC text,
    otherwise uniform rows).
    """
    dataset = str(extraction.get("dataset", "")).lower()
    modality = str(extraction.get("input_modality", "")).strip()
    variant = str(extraction.get("dataset_variant", "")).strip()
    if dataset == "d3tec" and modality in {"audio_only", "audio_text"}:
        return WEIGHT_POLICY_D3TEC
    if dataset == "daic":
        return WEIGHT_POLICY_DAIC
    if (
        dataset == "turkish" and variant == "pooled_t17" and modality == "text_only"
    ) or (dataset == "d3tec" and modality == "text_only"):
        return WEIGHT_POLICY_TEXT
    return WEIGHT_POLICY_UNIFORM


def membership_issues(
    *,
    parent_attempt_id: str,
    head_attempt_id: str,
    fold: int,
    parent_training_seed: int | None,
    mask: dict,
    window_cap: dict,
    extraction: dict,
    head_metadata: dict,
    variants: dict[str, dict],
    extraction_sha256: str,
    expected_head_deployment: str | None = None,
    train_subjects: set[str] | None = None,
    val_subjects: set[str] | None = None,
    pool_rows: list[dict] | None = None,
    pool_rows_sha256: str | None = None,
) -> list[str]:
    """Fail-closed checks that both variants trained the exact mask membership.

    Parent identity is bound to ``entry.parent_attempt_id``/fold/seed; the head
    attempt identity is checked separately against ``head_metadata.attempt_id``.
    The ``window_cap`` block comes from the parent fit's run_config.
    """
    from src.data.window_cap import compute_selection_sha256

    issues: list[str] = []
    membership = {item for ids in mask["subjects"].values() for item in ids}
    mask_subjects = {str(subject) for subject in mask["subjects"]}
    recomputed = compute_selection_sha256(
        mask["algorithm_version"],
        mask["sampling_seed"],
        mask["fraction"],
        mask["subjects"],
    )
    if not (recomputed == mask.get("selection_sha256") == window_cap.get("selection_sha256")):
        issues.append("mask selection hash chain mismatch")
    if len(membership) != int(mask.get("total_selected", -1)):
        issues.append("mask membership is not unique at total_selected")
    if train_subjects is not None and mask_subjects != train_subjects:
        issues.append("mask subjects do not equal the authoritative train subjects")
    if pool_rows is not None and train_subjects is not None and val_subjects is not None:
        pool_subjects = {str(row["subject_id"]) for row in pool_rows}
        if pool_subjects != train_subjects | val_subjects:
            issues.append("cached outer_train subjects do not equal the expected canonical pool")
    if extraction.get("parent_attempt_id") != parent_attempt_id:
        issues.append("extraction parent attempt mismatch")
    if (head_metadata.get("parent") or {}).get("parent_attempt_id") != parent_attempt_id:
        issues.append("head metadata parent attempt mismatch")
    if extraction.get("fold") is not None and int(extraction["fold"]) != int(fold):
        issues.append("extraction fold mismatch")
    if head_metadata.get("fold") is not None and int(head_metadata["fold"]) != int(fold):
        issues.append("head metadata fold mismatch")
    if (
        parent_training_seed is not None
        and head_metadata.get("seed") is not None
        and int(head_metadata["seed"]) != int(parent_training_seed)
    ):
        issues.append("head metadata seed mismatch")
    if head_metadata.get("attempt_id") != head_attempt_id:
        issues.append("head attempt identity mismatch")
    if expected_head_deployment:
        source = head_metadata.get("source") or {}
        if str(source.get("deployment_id") or "") != str(expected_head_deployment):
            issues.append("head source deployment mismatch")
    for variant in VARIANTS:
        metadata = variants.get(variant)
        if metadata is None:
            issues.append(f"{variant}: classifier metadata missing")
            continue
        train_mask = metadata.get("train_mask") or {}
        training_rows = list(metadata.get("training_row_ids") or [])
        capped = list(train_mask.get("capped_train_row_ids") or [])
        retained = list(train_mask.get("retained_val_row_ids") or [])
        weight_audit = metadata.get("fit_weight_audit")
        if not isinstance(weight_audit, dict) or not weight_audit:
            issues.append(f"{variant}: fit weight audit missing")
        else:
            mean_weight = weight_audit.get("mean_weight")
            if (
                not isinstance(mean_weight, (int, float))
                or isinstance(mean_weight, bool)
                or not math.isfinite(float(mean_weight))
                or abs(float(mean_weight) - 1.0) > 1e-9
            ):
                issues.append(f"{variant}: weight audit mean_weight is not finite 1.0")
            row_count = weight_audit.get("row_count")
            if (
                not isinstance(row_count, int)
                or isinstance(row_count, bool)
                or row_count < 0
                or row_count != len(training_rows)
            ):
                issues.append(f"{variant}: weight audit row_count is not the exact selected row count")
            subject_count = weight_audit.get("subject_count")
            if (
                not isinstance(subject_count, int)
                or isinstance(subject_count, bool)
                or subject_count < 0
                or subject_count != len(mask_subjects) + len(val_subjects or ())
            ):
                issues.append(f"{variant}: weight audit subject_count is not the exact fit subject count")
            policy = weight_audit.get("policy")
            if str(policy or "") != expected_weight_policy(extraction):
                issues.append(f"{variant}: weight audit policy {policy!r} != canonical {expected_weight_policy(extraction)!r}")
        for field in (
            "selection_sha256",
            "baseline_input_sha256",
            "fraction",
            "sampling_seed",
            "algorithm_version",
        ):
            if train_mask.get(field) != window_cap.get(field):
                issues.append(f"{variant}: train_mask {field} mismatch")
        if not capped:
            issues.append(f"{variant}: capped_train_row_ids missing")
        elif set(capped) != membership:
            issues.append(f"{variant}: capped train rows do not equal mask membership")
        if len(capped) != int(mask.get("total_selected", -1)):
            issues.append(f"{variant}: capped train row count mismatch")
        if val_subjects is None:
            issues.append(f"{variant}: authoritative inner-val subjects unavailable")
        else:
            if int(train_mask.get("retained_val_subject_count", -1)) != len(val_subjects):
                issues.append(f"{variant}: retained val subject count mismatch")
            if val_subjects and not retained:
                issues.append(f"{variant}: retained_val_row_ids missing")
            if pool_rows is None:
                issues.append(f"{variant}: cache pool rows unavailable for acceptance")
            else:
                expected_val_ids = {
                    str(row["sample_id"])
                    for row in pool_rows
                    if str(row["subject_id"]) in val_subjects
                }
                if set(retained) != expected_val_ids:
                    issues.append(f"{variant}: retained val rows do not equal the full inner-val rows")
        if set(training_rows) != set(capped) | set(retained):
            issues.append(f"{variant}: training rows are not the capped+retained union")
        if len(training_rows) != len(set(training_rows)):
            issues.append(f"{variant}: training rows not unique")
        pool_link = (metadata.get("cache_identity") or {}).get("outer_train_rows.jsonl") or {}
        if pool_rows_sha256 and pool_link.get("sha256") != pool_rows_sha256:
            issues.append(f"{variant}: outer_train rows link mismatch")
        if metadata.get("parent_attempt_id") != parent_attempt_id:
            issues.append(f"{variant}: parent attempt mismatch")
        if metadata.get("fold") is not None and int(metadata["fold"]) != int(fold):
            issues.append(f"{variant}: fold mismatch")
        checkpoint = metadata.get("checkpoint_hashes") or {}
        if (
            checkpoint.get("adapter_config_sha256") != extraction.get("adapter_config_sha256")
            or checkpoint.get("adapter_sha256") != extraction.get("adapter_sha256")
        ):
            issues.append(f"{variant}: checkpoint hash mismatch")
        cache_link = (metadata.get("cache_identity") or {}).get("extraction_metadata.json") or {}
        if cache_link.get("sha256") != extraction_sha256:
            issues.append(f"{variant}: extraction metadata link mismatch")
    return issues


def parent_fold_remote(entry: dict) -> str | None:
    checkpoint = str(entry.get("parent_checkpoint_dir") or "")
    if not checkpoint.endswith("/best_model"):
        return None
    return checkpoint[: -len("/best_model")]


def parent_fold_local(entry: dict) -> Path | None:
    """Collected parent fold evidence from the official local run root."""
    remote = parent_fold_remote(entry)
    if remote is None:
        return None
    parts = remote.rstrip("/").split("/")
    try:
        fold_index = next(i for i, part in enumerate(parts) if part.startswith("fold_"))
    except StopIteration:
        return None
    run_name = parts[fold_index - 1]
    fold = parts[fold_index]
    dataset = str(entry.get("dataset"))
    modality = str(entry.get("modality"))
    candidate = LOCAL_RUN_ROOT / modality / dataset / run_name / fold
    if (candidate / "window_cap_mask.json").is_file() and (candidate / "run_config.yaml").is_file():
        return candidate
    return None


def remote_sha256(remote_path: str, *, host: str | None = None) -> str | None:
    """Exact remote sha256 for one compact file, or None when unavailable."""
    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            host or TRANSFER_HOST,
            f"sha256sum {remote_path}",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        return None
    token = (proc.stdout or "").strip().split()
    if not token or len(token[0]) != 64:
        return None
    return token[0]


def rsync_download(remote: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["rsync", "-a", "--no-motd", f"{TRANSFER_HOST}:{remote}", str(dest)],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and dest.is_file()


def fetch_remote_file(
    remote: str,
    local: Path,
    *,
    remote_hasher=remote_sha256,
    downloader=rsync_download,
) -> tuple[bool, str]:
    """Hash-verified single-file fetch with atomic publish and no overwrite.

    The exact remote SHA-256 must match the local bytes before the file is
    used: an existing mismatch or a partial previous fetch is refused and
    preserved, never overwritten; a fresh download goes to a ``.part`` target
    and is published atomically only after the hash matches.
    """
    expected = remote_hasher(remote)
    if not expected:
        return False, "remote hash unavailable"
    if local.is_file():
        local_sha = sha256_file(local)
        if local_sha == expected:
            return True, local_sha
        return False, f"existing evidence hash mismatch ({local_sha[:12]} != {expected[:12]})"
    temp = local.with_suffix(local.suffix + ".part")
    if temp.exists():
        temp.unlink()
    if not downloader(remote, temp) or not temp.is_file():
        return False, "download failed"
    local_sha = sha256_file(temp)
    if local_sha != expected:
        temp.unlink(missing_ok=True)
        return False, f"downloaded hash mismatch ({local_sha[:12]} != {expected[:12]})"
    os.replace(temp, local)
    return True, local_sha


def resolve_parent_evidence(
    entry: dict, *, fetcher=fetch_remote_file
) -> tuple[Path | None, Path | None, str]:
    """(run_config, mask, reason) for the parent fit from official evidence."""
    local = parent_fold_local(entry)
    if local is not None:
        return local / "run_config.yaml", local / "window_cap_mask.json", "collected fold"
    remote = parent_fold_remote(entry)
    if remote is None:
        return None, None, "parent fold path unavailable"
    cache = LANE / "head_parent_evidence" / str(entry["attempt_id"])
    run_config = cache / "run_config.yaml"
    mask = cache / "window_cap_mask.json"
    for name, path in (("run_config.yaml", run_config), ("window_cap_mask.json", mask)):
        ok, reason = fetcher(f"{remote}/{name}", path)
        if not ok:
            return None, None, f"{name}: {reason}"
    return run_config, mask, "verified remote fetch"


def resolve_extraction(entry: dict, *, fetcher=fetch_remote_file) -> tuple[Path | None, str]:
    """(path, reason) for the exact compact cache metadata from entry.cache_dir."""
    cache_dir = str(entry.get("cache_dir") or "")
    if not cache_dir:
        return None, "cache_dir missing"
    local = LANE / "head_cache_evidence" / str(entry["attempt_id"]) / "extraction_metadata.json"
    ok, reason = fetcher(f"{cache_dir}/extraction_metadata.json", local)
    if not ok:
        return None, reason
    return local, "verified remote fetch"


def resolve_authoritative_split(entry: dict, *, fetcher=fetch_remote_file) -> tuple[Path | None, str]:
    """The parent fit's logs/split_used.json (local fold first, verified fetch second)."""
    local_fold = parent_fold_local(entry)
    if local_fold is not None:
        candidate = local_fold / "logs" / "split_used.json"
        if candidate.is_file():
            return candidate, "collected fold"
    remote = parent_fold_remote(entry)
    if remote is None:
        return None, "parent fold path unavailable"
    local = LANE / "head_parent_evidence" / str(entry["attempt_id"]) / "split_used.json"
    ok, reason = fetcher(f"{remote}/logs/split_used.json", local)
    if not ok:
        return None, reason
    return local, "verified remote fetch"


def resolve_outer_train_rows(entry: dict, *, fetcher=fetch_remote_file) -> tuple[Path | None, str]:
    """The extraction cache's outer_train rows (hash-verified fetch)."""
    cache_dir = str(entry.get("cache_dir") or "")
    if not cache_dir:
        return None, "cache_dir missing"
    local = LANE / "head_cache_evidence" / str(entry["attempt_id"]) / "outer_train_rows.jsonl"
    ok, reason = fetcher(f"{cache_dir}/outer_train_rows.jsonl", local)
    if not ok:
        return None, reason
    return local, "verified remote fetch"


def head_local_paths(entry: dict) -> dict[str, Path]:
    mirror = Path(str(entry["local_mirror"]))
    return {
        "mirror": mirror,
        "metadata": mirror / "metadata.json",
    }


def variant_metadata_paths(mirror: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for variant in VARIANTS:
        for candidate in (mirror / "classifier" / variant, mirror / variant):
            metadata = candidate / "classifier_metadata.json"
            if metadata.is_file():
                paths[variant] = metadata
                break
    return paths


def verify_head_membership(
    entry: dict,
    *,
    parent_evidence: tuple[Path, Path] | None = None,
    extraction_file: Path | None = None,
    split_file: Path | None = None,
    pool_rows_file: Path | None = None,
    fetcher=fetch_remote_file,
) -> list[str]:
    paths = head_local_paths(entry)
    if not paths["metadata"].is_file():
        return ["missing collected head metadata"]
    if parent_evidence is None:
        run_config_path, mask_path, reason = resolve_parent_evidence(entry, fetcher=fetcher)
    else:
        run_config_path, mask_path = parent_evidence
        reason = "injected"
    if (
        run_config_path is None
        or mask_path is None
        or not run_config_path.is_file()
        or not mask_path.is_file()
    ):
        return [f"parent run_config/mask evidence not found ({reason})"]
    if extraction_file is None:
        extraction_file, extraction_reason = resolve_extraction(entry, fetcher=fetcher)
        if extraction_file is None:
            return [f"extraction cache metadata not found ({extraction_reason})"]
    if not extraction_file.is_file():
        return ["extraction cache metadata not found"]
    if split_file is None:
        split_file, split_reason = resolve_authoritative_split(entry, fetcher=fetcher)
        if split_file is None:
            return [f"authoritative parent split not found ({split_reason})"]
    if pool_rows_file is None:
        pool_rows_file, pool_reason = resolve_outer_train_rows(entry, fetcher=fetcher)
        if pool_rows_file is None:
            return [f"cache outer_train rows not found ({pool_reason})"]
    import yaml

    parent_config = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
    window_cap = ((parent_config.get("config") or {}).get("training") or {}).get(
        "window_cap"
    ) or {}
    if (parent_config.get("tracking") or {}).get("attempt_id") != str(entry["parent_attempt_id"]):
        return ["parent run_config attempt mismatch"]
    mask = json.loads(mask_path.read_text(encoding="utf-8"))
    extraction = json.loads(extraction_file.read_text(encoding="utf-8"))
    head_metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    variants = {
        variant: json.loads(path.read_text(encoding="utf-8"))
        for variant, path in variant_metadata_paths(paths["mirror"]).items()
    }
    split = json.loads(split_file.read_text(encoding="utf-8"))
    if sha256_file(split_file) != str(extraction.get("saved_split_sha256") or ""):
        return ["authoritative parent split hash does not match the extraction record"]
    from src.features.hidden_classifier_policy import expected_outer_train_subjects

    dataset = str(extraction.get("dataset") or entry.get("dataset") or "")
    try:
        expected_pool = expected_outer_train_subjects(split, dataset)
    except ValueError as exc:
        return [str(exc)]
    train_subjects = {str(subject) for subject in (split.get("train_subject_ids") or [])}
    val_subjects = expected_pool - train_subjects
    pool_rows = [
        json.loads(line)
        for line in pool_rows_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return membership_issues(
        parent_attempt_id=str(entry["parent_attempt_id"]),
        head_attempt_id=str(entry["attempt_id"]),
        fold=int(entry["fold"]),
        parent_training_seed=entry.get("parent_training_seed"),
        mask=mask,
        window_cap=window_cap,
        extraction=extraction,
        head_metadata=head_metadata,
        variants=variants,
        extraction_sha256=sha256_file(extraction_file),
        expected_head_deployment=str(entry.get("deployment_id") or "") or None,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        pool_rows=pool_rows,
        pool_rows_sha256=sha256_file(pool_rows_file),
    )


# ---------------------------------------------------------------------------
# official tool runners
# ---------------------------------------------------------------------------


def run_command(argv: list[str], timeout: int = 3600):
    proc = subprocess.run(
        argv, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=timeout
    )
    return proc.returncode, (proc.stdout or "")[-600:], (proc.stderr or "")[-300:]


def acquire_submission_lock():
    """Non-blocking exclusive lane submission lock; None when busy."""
    handle = SUBMISSION_LOCK.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def local_fold_dir(attempt_id: str) -> Path | None:
    """Local fold dir from the recorded submit contract, if it exists."""
    contract = PROJECT_ROOT / "outputs" / "exp_submit" / attempt_id / "contract.json"
    if not contract.is_file():
        return None
    try:
        data = json.loads(contract.read_text(encoding="utf-8"))
    except ValueError:
        return None
    rel = data.get("local_fold_rel")
    if not rel:
        return None
    fold = PROJECT_ROOT / str(rel)
    return fold if fold.is_dir() else None


def collect_fit(row: dict, *, runner=run_command, fold_resolver=local_fold_dir) -> dict:
    attempt = str(row["attempt_id"])
    record = {
        "at_utc": now(),
        "kind": "fit",
        "key": fit_key(row),
        "attempt_id": attempt,
        "run_name": row["run_name"],
        "steps": [],
        "collect_ok": False,
        "validated": False,
    }
    fold = fold_resolver(attempt)
    if fold is not None and (fold / "status.json").is_file() and (fold / "run_config.yaml").is_file():
        # Locally verified sidecars must not be overwritten by a re-collect;
        # proceed with the official reconcile + validate on the existing copy.
        record["collect_ok"] = True
        record["note"] = "already collected locally; skipping re-collect"
        record["steps"].append(
            {"cmd": ["skip-collect"], "rc": 0, "stdout_tail": "already collected", "stderr_tail": ""}
        )
        return record
    argv = [sys.executable, str(EXP_TOOL), "collect", SLUG, "--attempt-id", attempt, "--execute"]
    rc, out, err = runner(argv)
    record["steps"].append({"cmd": argv, "rc": rc, "stdout_tail": out, "stderr_tail": err})
    record["collect_ok"] = rc == 0
    if rc != 0:
        record["note"] = "collect failed"
    return record


def reconcile_fits(records: list[dict], *, runner=run_command) -> None:
    """One official status reconciliation for the batch (terminal events + lifecycle)."""
    if not any(record["collect_ok"] for record in records):
        return
    argv = [sys.executable, str(EXP_TOOL), "status", SLUG]
    rc, out, err = runner(argv)
    for record in records:
        if record["collect_ok"]:
            record["steps"].append({"cmd": argv, "rc": rc, "stdout_tail": out, "stderr_tail": err})


def validate_fit(record: dict, *, runner=run_command, fold_resolver=local_fold_dir) -> None:
    attempt = record["attempt_id"]
    argv = [sys.executable, str(EXP_TOOL), "validate", SLUG, "--attempt-id", attempt]
    rc, out, err = runner(argv)
    record["steps"].append({"cmd": argv, "rc": rc, "stdout_tail": out, "stderr_tail": err})
    if rc != 0:
        record["note"] = "validate failed"
        return
    fold = fold_resolver(attempt)
    if fold is None:
        record["note"] = "local fold not found after collect"
        return
    status_path = fold / "status.json"
    if not status_path.is_file():
        record["note"] = "local status.json missing after validate"
        return
    state = str(json.loads(status_path.read_text(encoding="utf-8")).get("state"))
    record["final_state"] = state
    record["validated"] = state == "LOCALLY_VALIDATED"
    if not record["validated"]:
        record["note"] = f"authoritative lifecycle not advanced (state={state})"


def process_fits_batch(rows: list[dict], *, runner=run_command, fold_resolver=local_fold_dir) -> list[dict]:
    records = [collect_fit(row, runner=runner, fold_resolver=fold_resolver) for row in rows]
    reconcile_fits(records, runner=runner)
    for record in records:
        if record["collect_ok"]:
            validate_fit(record, runner=runner, fold_resolver=fold_resolver)
        append_ledger(record)
    return records


def process_head(entry: dict, *, runner=run_command, membership_checker=verify_head_membership) -> dict:
    attempt = str(entry["attempt_id"])
    record = {
        "at_utc": now(),
        "kind": "head",
        "key": str(entry["registry_key"]),
        "attempt_id": attempt,
        "steps": [],
        "membership_ok": False,
        "validated": False,
    }
    collect = [
        sys.executable,
        str(DISPATCH_TOOL),
        "collect",
        "--registry",
        str(REGISTRY),
        "--attempt-id",
        attempt,
    ]
    rc, out, err = runner(collect)
    record["steps"].append({"cmd": collect, "rc": rc, "stdout_tail": out, "stderr_tail": err})
    if rc != 0:
        record["note"] = "head collect failed"
        return record
    issues = membership_checker(entry)
    record["membership_issues"] = issues
    record["membership_ok"] = not issues
    if issues:
        record["note"] = "mask membership checks failed"
        return record
    validate = [
        sys.executable,
        str(DISPATCH_TOOL),
        "validate",
        "--registry",
        str(REGISTRY),
        "--attempt-id",
        attempt,
    ]
    rc, out, err = runner(validate)
    record["steps"].append({"cmd": validate, "rc": rc, "stdout_tail": out, "stderr_tail": err})
    if rc != 0:
        record["note"] = "head validate failed"
        return record
    status_path = head_local_paths(entry)["mirror"] / "status.json"
    if not status_path.is_file():
        record["note"] = "head status.json missing after validate"
        return record
    status = json.loads(status_path.read_text(encoding="utf-8"))
    state = str(status.get("state"))
    status_attempt = str(status.get("attempt_id"))
    record["final_state"] = state
    if state != "LOCALLY_VALIDATED":
        record["note"] = f"authoritative lifecycle not advanced (state={state})"
        return record
    if status_attempt != attempt:
        record["note"] = f"head status attempt mismatch ({status_attempt})"
        return record
    record["validated"] = True
    return record


# ---------------------------------------------------------------------------
# progress and watcher gate
# ---------------------------------------------------------------------------


def scheduler_map_for_ids(ids: list[str], *, host: str | None = None) -> dict[str, dict]:
    """One batched sacct query for exact numeric ids (no step records)."""
    wanted = sorted({job_id for job_id in ids if str(job_id).isdigit()})
    if not wanted:
        return {}
    command = f"sacct -j {','.join(wanted)} -n -P -o JobIDRaw,State,ExitCode"
    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            host or SCHEDULER_HOST,
            command,
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        return {}
    return planner.parse_sacct(proc.stdout, set(wanted))


def latest_registry_entries(entries: list[dict]) -> dict[str, dict]:
    """Latest (last) registry entry per canonical key; old attempts are not current."""
    latest: dict[str, dict] = {}
    for entry in entries:
        key = str(entry.get("registry_key") or "")
        if key:
            latest[key] = entry
    return latest


def latest_rows(rows: list[dict]) -> dict[str, dict]:
    """Latest submission row per run name; superseded attempts are not current."""
    latest: dict[str, dict] = {}
    for row in rows:
        latest[str(row["run_name"])] = row
    return latest


def production_entries(entries: list[dict], expected_keys: set[str]) -> list[dict]:
    return [entry for entry in entries if str(entry.get("registry_key") or "") in expected_keys]


def progress(rows: list[dict], entries: list[dict], expected_keys: set[str], ledger: dict[str, dict]) -> dict:
    rows_latest = latest_rows(rows)
    fits_validated = 0
    for row in rows_latest.values():
        record = ledger.get(fit_key(row)) or {}
        if record.get("validated") and str(record.get("attempt_id")) == str(row.get("attempt_id")):
            fits_validated += 1
    prod = production_entries(list(latest_registry_entries(entries).values()), expected_keys)
    confounded = confounded_attempts()
    heads_validated = 0
    confounded_latest = 0
    old_technical_validated = 0
    for entry in prod:
        key = str(entry["registry_key"])
        attempt = str(entry["attempt_id"])
        record = ledger.get(key) or {}
        if attempt in confounded:
            confounded_latest += 1
            if record.get("validated") and str(record.get("attempt_id")) == attempt:
                old_technical_validated += 1
            continue
        if record.get("validated") and str(record.get("attempt_id")) == attempt:
            heads_validated += 1
    return {
        "fits_submitted": len(rows_latest),
        "fits_expected": EXPECTED_CHAINS,
        "fits_validated": fits_validated,
        "heads_dispatched": len(prod),
        "heads_expected": EXPECTED_CHAINS,
        "heads_validated": heads_validated,
        "confounded_latest_attempts": confounded_latest,
        "heads_old_technical_validated": old_technical_validated,
        "at_utc": now(),
    }


def watcher_done(progress_doc: dict) -> bool:
    """The watcher may exit only when every required chain and fit is validated."""
    return (
        int(progress_doc.get("fits_submitted", 0)) >= EXPECTED_CHAINS
        and int(progress_doc.get("fits_validated", 0)) >= EXPECTED_CHAINS
        and int(progress_doc.get("heads_dispatched", 0)) >= EXPECTED_CHAINS
        and int(progress_doc.get("heads_validated", 0)) >= EXPECTED_CHAINS
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def load_registry(path: Path | None = None) -> list[dict]:
    registry = path or REGISTRY
    entries: list[dict] = []
    if registry.is_file():
        for line in registry.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=None)
    parser.add_argument("--local-run-root", type=Path, default=None)
    parser.add_argument("--scheduler-host", default=None)
    parser.add_argument("--transfer-host", default=None)
    parser.add_argument("--max-fits", type=int, default=4)
    parser.add_argument("--max-heads", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    configure(
        args.campaign_dir,
        local_run_root=args.local_run_root,
        scheduler_host=args.scheduler_host,
        transfer_host=args.transfer_host,
    )
    if args.status:
        doc = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.is_file() else {}
        print(json.dumps(doc, indent=1, sort_keys=True))
        return 0
    if not args.no_fetch:
        rc = planner.fetch_evidence()
        if rc != 0:
            return rc
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    expected = planner.assert_matrix_keys(matrix)
    rows = list(latest_rows(planner.load_rows()).values())
    entries = load_registry()
    ledger = load_ledger()

    local_pass: list[dict] = []
    fit_reasons: dict[str, int] = {}
    for row in rows:
        if ledger.get(fit_key(row), {}).get("validated"):
            continue
        ok, reason = planner.readiness_local(row)
        if ok:
            local_pass.append(row)
        else:
            fit_reasons[reason] = fit_reasons.get(reason, 0) + 1
    fit_sched, _ = planner.scheduler_confirm(local_pass) if local_pass else ({}, {})
    ready_fits, reasons = select_ready_fits(rows, fit_sched, ledger)
    for reason, count in fit_reasons.items():
        reasons[reason] = reasons.get(reason, 0) + count

    head_ids = [
        job_id
        for entry in production_entries(entries, expected)
        for job_id in head_job_ids(entry).values()
    ]
    head_sched = scheduler_map_for_ids(head_ids) if head_ids else {}
    ready_heads, head_reasons = select_ready_heads(entries, expected, head_sched, ledger)

    doc = progress(rows, entries, expected, ledger)
    doc["ready_fits"] = len(ready_fits)
    doc["ready_heads"] = len(ready_heads)
    doc["fit_reasons"] = reasons
    doc["head_reasons"] = head_reasons
    doc["watcher_done"] = watcher_done(doc)
    if not args.execute:
        PROGRESS.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
        print(json.dumps(doc, indent=1, sort_keys=True))
        print("dry-run only; nothing collected or validated")
        return 0
    lock = acquire_submission_lock()
    if lock is None:
        print("lane submission lock busy; skipping collection this cycle")
        return 0

    fit_records = process_fits_batch(ready_fits[: max(0, args.max_fits)])
    for record in fit_records:
        ledger[record["key"]] = record
    processed_heads = 0
    for entry in ready_heads[: max(0, args.max_heads)]:
        record = process_head(entry)
        append_ledger(record)
        ledger[str(entry["registry_key"])] = record
        processed_heads += 1

    doc = progress(rows, entries, expected, ledger)
    doc["processed_fits"] = len(fit_records)
    doc["processed_heads"] = processed_heads
    doc["watcher_done"] = watcher_done(doc)
    PROGRESS.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(doc, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
