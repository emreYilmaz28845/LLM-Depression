#!/usr/bin/env python3
"""Prepare a derived evidence view for a mixed-attempt managed fold.

Context: two concurrent submissions of the same logical run share one fold
directory. One attempt is canonical (recorded in the submission contract,
metadata, status, evaluations and run_config); a foreign attempt left job
events and/or artifacts identity in the same directory. The original fold
fails ``read_modern_sidecars`` on identity contradictions.

This tool NEVER modifies the original fold. It builds a *derived evidence
view* in a new directory:

  * verbatim copies of the canonical sidecars that are already consistent
    (metadata.json, status.json, evaluations.json, run_config.yaml);
  * a derived ``jobs.jsonl`` containing only the canonical attempt's events,
    copied line-for-line (no invented scheduler events);
  * a derived ``artifacts.json`` re-registered under the canonical attempt
    with hashes recomputed from the real files (no fabricated hashes);
  * verbatim copies of every other evidence file (logs, eval,
    best_model/standalone_eval);
  * ``recovery/`` with an immutable checksummed pre-recovery snapshot of the
    original fold, the quarantined foreign events, the source/lineage
    manifest and a human-readable derived-evidence statement.

Dry-run is the default and writes only a plan (no view). ``--execute``
refuses unless the plan hash is approved explicitly. Ambiguous writer or
hash evidence, unknown foreign attempts, missing files, stale inputs or an
existing output directory are rejected.

The derived view is validated with the UNCHANGED gates via
``tools/exp.py validate --fold-dir <view>`` and ``finish --fold-dir <view>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIDECARS = ("metadata.json", "status.json", "jobs.jsonl", "artifacts.json", "evaluations.json")
VERBATIM_SIDECARS = ("metadata.json", "status.json", "evaluations.json", "run_config.yaml")
DERIVED_SIDECARS = ("jobs.jsonl", "artifacts.json")
TOOL_PATH = Path(__file__).resolve()


class RecoveryError(RuntimeError):
    """Raised when recovery must fail closed."""


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RecoveryError(f"unreadable JSON {path}: {error}") from error


def file_inventory(fold: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in fold.rglob("*") if p.is_file()):
        stat = path.stat()
        inventory[path.relative_to(fold).as_posix()] = {
            "sha256": sha256_file(path),
            "size": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
        }
    return inventory


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryError(message)


def check_isolation(fold: Path, view_dir: Path) -> None:
    """The derived view must be fully outside the original evidence tree."""
    fold_resolved = fold.resolve()
    view_resolved = view_dir.resolve()
    _require(view_resolved != fold_resolved, "view_dir must not be the original fold")
    _require(not view_resolved.is_relative_to(fold_resolved), "view_dir must not be inside the original fold")
    _require(
        not fold_resolved.is_relative_to(view_resolved),
        "view_dir must not be an ancestor of the original fold",
    )


def check_plan_outside_fold(fold: Path, plan_out: Path) -> None:
    fold_resolved = fold.resolve()
    plan_resolved = plan_out.resolve()
    _require(
        plan_resolved != fold_resolved and not plan_resolved.is_relative_to(fold_resolved),
        "plan output must not be inside the original fold",
    )


def load_lineage(path: Path) -> dict[str, Any]:
    lineage = load_json(path)
    _require(isinstance(lineage, dict), "lineage evidence must be a JSON object")
    canonical = lineage.get("canonical_attempt_id")
    foreign = lineage.get("foreign_attempt_ids")
    _require(isinstance(canonical, str) and canonical, "lineage evidence needs canonical_attempt_id")
    _require(isinstance(foreign, list) and foreign, "lineage evidence needs non-empty foreign_attempt_ids")
    _require(all(isinstance(item, str) and item for item in foreign), "foreign attempt ids must be strings")
    _require(canonical not in foreign, "canonical attempt must not be listed as foreign")
    _require(
        isinstance(lineage.get("adapter_sha256"), str) and len(lineage["adapter_sha256"]) == 64,
        "lineage evidence needs the adapter sha256 fingerprint",
    )
    return lineage


def cross_check_head_fingerprint(head_attempt: Path, lineage: dict[str, Any]) -> dict[str, Any]:
    metadata_path = head_attempt / "metadata.json"
    _require(metadata_path.is_file(), f"head attempt metadata missing: {metadata_path}")
    metadata = load_json(metadata_path)
    parent = metadata.get("parent") or {}
    _require(
        parent.get("parent_attempt_id") == lineage["canonical_attempt_id"],
        "head attempt declares a different parent attempt than the lineage evidence",
    )
    _require(
        parent.get("adapter_sha256") == lineage["adapter_sha256"],
        "head attempt adapter fingerprint does not match the lineage evidence",
    )
    config_sha = parent.get("adapter_config_sha256")
    if lineage.get("adapter_config_sha256") is not None:
        _require(
            config_sha == lineage["adapter_config_sha256"],
            "head attempt adapter_config fingerprint does not match the lineage evidence",
        )
    return {
        "head_attempt_id": metadata.get("attempt_id"),
        "adapter_sha256": parent.get("adapter_sha256"),
        "adapter_config_sha256": config_sha,
        "parent_checkpoint_path": parent.get("parent_checkpoint_path"),
    }


def check_pre_snapshot(fold: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    rows = manifest.get("files") if isinstance(manifest, dict) else None
    _require(isinstance(rows, list) and rows, "pre-snapshot manifest has no files list")
    expected = {row["path"]: row["sha256"] for row in rows if isinstance(row, dict) and "path" in row}
    current = file_inventory(fold)
    missing = sorted(set(expected) - set(current))
    added = sorted(set(current) - set(expected))
    changed = sorted(p for p in set(expected) & set(current) if expected[p] != current[p]["sha256"])
    _require(not missing, f"pre-snapshot files missing in the live fold: {missing[:5]}")
    _require(not changed, f"pre-snapshot files changed in the live fold: {changed[:5]}")
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "verified_files": len(expected),
        "added_since_snapshot": added,
    }


def build_plan(
    fold: Path,
    canonical: str,
    lineage: dict[str, Any],
    head_attempt: Path,
    view_dir: Path,
    pre_snapshot_manifest: Path | None,
) -> dict[str, Any]:
    from src.experiment_tracking.schemas import (
        validate_artifacts,
        validate_evaluations,
        validate_job_event,
        validate_metadata,
        validate_status,
    )

    _require((fold / "metadata.json").is_file(), f"not a modern tracked fold: {fold}")
    check_isolation(fold, view_dir)
    fold_root = fold.resolve()

    def ensure_inside_tree(full: Path, label: str) -> None:
        resolved = full.resolve()
        _require(
            resolved == fold_root or resolved.is_relative_to(fold_root),
            f"{label} resolves outside the original evidence tree: {full}",
        )

    metadata = load_json(fold / "metadata.json")
    status = load_json(fold / "status.json")
    evaluations = load_json(fold / "evaluations.json")
    run_config_text = (fold / "run_config.yaml").read_text(encoding="utf-8")
    for name, validator, record in (
        ("metadata.json", validate_metadata, metadata),
        ("status.json", validate_status, status),
        ("evaluations.json", validate_evaluations, evaluations),
    ):
        ok, errors = validator(record)
        _require(ok, f"{name} is not schema-valid: {'; '.join(errors[:3])}")
    for name, record in (("metadata.json", metadata), ("status.json", status), ("evaluations.json", evaluations)):
        _require(
            record.get("attempt_id") == canonical,
            f"{name} attempt_id {record.get('attempt_id')!r} is not the canonical attempt",
        )
    _require(f"attempt_id: {canonical}" in run_config_text, "run_config.yaml does not carry the canonical attempt id")

    # Jobs: canonical events only; every foreign event must belong to the
    # exactly-one declared foreign attempt set.
    jobs_lines = (fold / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
    kept: list[int] = []
    foreign_lines: list[int] = []
    foreign_events: list[dict[str, Any]] = []
    foreign_seen: set[str] = set()
    for index, line in enumerate(jobs_lines):
        if not line.strip():
            continue
        event = json.loads(line)
        ok, errors = validate_job_event(event)
        _require(ok, f"jobs.jsonl[{index}] invalid: {'; '.join(errors[:3])}")
        attempt = event.get("attempt_id")
        if attempt == canonical:
            kept.append(index)
        else:
            foreign_lines.append(index)
            foreign_events.append({"line": index, "event": event})
            if isinstance(attempt, str):
                foreign_seen.add(attempt)
    _require(kept, "jobs.jsonl has no canonical attempt events")
    _require(foreign_lines, "jobs.jsonl has no foreign events; nothing to recover")
    _require(
        foreign_seen == set(lineage["foreign_attempt_ids"]),
        f"foreign events reference {sorted(foreign_seen)} but lineage declares {lineage['foreign_attempt_ids']}",
    )

    # Artifacts: rebuild under the canonical attempt with recomputed hashes.
    artifacts = load_json(fold / "artifacts.json")
    ok, errors = validate_artifacts(artifacts)
    _require(ok, f"artifacts.json is not schema-valid: {'; '.join(errors[:3])}")
    from src.experiment_tracking.identity import artifact_id

    fold_number = int(metadata.get("fold", 0) or 0)
    entries: list[dict[str, Any]] = []
    referenced: set[str] = set()
    changed_hashes = 0
    for index, entry in enumerate(artifacts.get("artifacts", [])):
        path = entry.get("path")
        _require(isinstance(path, str) and path, f"artifacts[{index}] has no path")
        full = fold / path
        _require(full.exists(), f"artifacts[{index}] path missing on disk: {path}")
        ensure_inside_tree(full, f"artifacts[{index}] {path!r}")
        recorded = entry.get("sha256")
        if full.is_dir():
            _require(recorded is None, f"artifacts[{index}] directory entry has a file hash: {path}")
            derived_sha = None
            size = None
        else:
            derived_sha = sha256_file(full)
            size = full.stat().st_size
            referenced.add(path)
            if recorded != derived_sha:
                changed_hashes += 1
        new_id = artifact_id(
            attempt_id=canonical,
            fold=fold_number,
            role=str(entry.get("role")),
            relative_path=path,
            artifact_sha256=derived_sha,
        )
        entries.append(
            {
                "path": path,
                "artifact_type": entry.get("artifact_type"),
                "role": entry.get("role"),
                "recorded_sha256": recorded,
                "derived_sha256": derived_sha,
                "size_bytes": size,
                "exists_on_mn5": entry.get("exists_on_mn5"),
                "artifact_id_old": entry.get("artifact_id"),
                "artifact_id_new": new_id,
            }
        )

    copied_files = sorted(
        relative
        for relative in file_inventory(fold)
        if relative not in SIDECARS
    )
    for relative in copied_files:
        ensure_inside_tree(fold / relative, f"copied file {relative!r}")
    plan: dict[str, Any] = {
        "tool": {"path": str(TOOL_PATH.relative_to(ROOT)), "sha256": sha256_file(TOOL_PATH)},
        "created_at_utc": now_utc(),
        "fold_dir": str(fold.resolve()),
        "view_dir": str(view_dir),
        "canonical_attempt_id": canonical,
        "foreign_attempt_ids": list(lineage["foreign_attempt_ids"]),
        "lineage_evidence": {"path": str(lineage.get("_path", "")), "sha256": lineage.get("_sha256")},
        "head_fingerprint": cross_check_head_fingerprint(head_attempt, lineage),
        "pre_snapshot": check_pre_snapshot(fold, pre_snapshot_manifest)
        if pre_snapshot_manifest is not None
        else None,
        "original_files": file_inventory(fold),
        "jobs": {
            "total_lines": len(jobs_lines),
            "kept_lines": kept,
            "foreign_lines": foreign_lines,
            "foreign_events": foreign_events,
        },
        "artifacts": {
            "entries": entries,
            "recomputed_hashes": changed_hashes,
            "referenced_files": sorted(referenced),
        },
        "verbatim_sidecars": list(VERBATIM_SIDECARS),
        "derived_sidecars": list(DERIVED_SIDECARS),
        "copied_files": copied_files,
    }
    plan["plan_sha256"] = stable_sha256(
        {key: value for key, value in plan.items() if key not in {"plan_sha256", "created_at_utc"}}
    )
    return plan


def write_plan(plan: dict[str, Any], plan_out: Path) -> None:
    plan_out.parent.mkdir(parents=True, exist_ok=True)
    plan_out.write_text(json.dumps(plan, indent=1, sort_keys=True), encoding="utf-8")
    lines = [
        "# Mixed-attempt recovery plan (dry run)",
        "",
        f"- created: {plan['created_at_utc']}",
        f"- original fold: {plan['fold_dir']}",
        f"- derived view: {plan['view_dir']}",
        f"- canonical attempt: {plan['canonical_attempt_id']}",
        f"- foreign attempts: {', '.join(plan['foreign_attempt_ids'])}",
        f"- plan sha256: `{plan['plan_sha256']}`",
        "",
        f"- jobs.jsonl: keep lines {plan['jobs']['kept_lines']}, quarantine lines {plan['jobs']['foreign_lines']} (no invented events)",
        f"- artifacts.json: {len(plan['artifacts']['entries'])} entries, {plan['artifacts']['recomputed_hashes']} hash(es) re-registered from real files",
        f"- verbatim sidecars: {', '.join(plan['verbatim_sidecars'])}",
        f"- copied evidence files: {len(plan['copied_files'])}",
        f"- pre-recovery snapshot check: {plan['pre_snapshot']}",
        f"- head fingerprint: {plan['head_fingerprint']['adapter_sha256']}",
        "",
        "Execute (after coordinator approval) with:",
        "",
        "```",
        f"python {plan['tool']['path']} --fold-dir {plan['fold_dir']} \\",
        f"  --canonical-attempt {plan['canonical_attempt_id']} \\",
        "  --lineage-evidence <lineage.json> --head-attempt <head attempt dir> \\",
        f"  --view-dir {plan['view_dir']} --pre-snapshot-manifest <snapshot_manifest.json> \\",
        f"  --execute --approve-plan {plan['plan_sha256']}",
        "```",
        "",
        "The derived view is then validated with the unchanged gates:",
        f"`python tools/exp.py validate --fold-dir {plan['view_dir']}` and",
        f"`python tools/exp.py finish --fold-dir {plan['view_dir']}`.",
    ]
    plan_out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(plan: dict[str, Any], approve_plan: str) -> None:
    _require(approve_plan == plan["plan_sha256"], "approve-plan does not match the dry-run plan hash")
    fold = Path(plan["fold_dir"])
    view = Path(plan["view_dir"])
    _require(not view.exists(), f"derived view already exists, refusing to overwrite: {view}")
    _require(
        file_inventory(fold) == plan["original_files"],
        "original fold changed since the approved dry-run; regenerate the plan",
    )
    view.mkdir(parents=True)
    recovery = view / "recovery"
    recovery.mkdir()

    # 1. Immutable pre-recovery snapshot of the complete original fold.
    snapshot = recovery / "snapshot" / fold.name
    shutil.copytree(fold, snapshot)
    snapshot_inventory = file_inventory(snapshot)
    _require(
        snapshot_inventory == plan["original_files"],
        "pre-recovery snapshot does not match the planned original inventory",
    )
    (recovery / "snapshot_manifest.json").write_text(
        json.dumps({"created_at_utc": now_utc(), "live_fold": str(fold), "files": plan["original_files"]}, indent=1),
        encoding="utf-8",
    )

    # 2. Verbatim sidecars and evidence files.
    for relative in VERBATIM_SIDECARS:
        shutil.copy2(fold / relative, view / relative)
    for relative in plan["copied_files"]:
        target = view / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fold / relative, target)

    # 3. Derived jobs.jsonl: canonical lines verbatim.
    jobs_lines = (fold / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
    derived_jobs = [jobs_lines[index] for index in plan["jobs"]["kept_lines"]]
    (view / "jobs.jsonl").write_text("\n".join(derived_jobs) + "\n", encoding="utf-8")

    # 4. Derived artifacts.json under the canonical attempt.
    artifacts = load_json(fold / "artifacts.json")
    for index, entry in enumerate(plan["artifacts"]["entries"]):
        target = artifacts["artifacts"][index]
        target["artifact_id"] = entry["artifact_id_new"]
        target["sha256"] = entry["derived_sha256"]
        target["size_bytes"] = entry["size_bytes"]
        target["exists_locally"] = False
        target["locally_verified"] = False
    artifacts["attempt_id"] = plan["canonical_attempt_id"]
    (view / "artifacts.json").write_text(json.dumps(artifacts, indent=1) + "\n", encoding="utf-8")

    # 5. Quarantined foreign events verbatim, recovery manifest and statement.
    foreign = plan["jobs"]["foreign_events"]
    (recovery / "quarantined_foreign_events.jsonl").write_text(
        "".join(json.dumps(item["event"]) + "\n" for item in foreign), encoding="utf-8"
    )
    manifest = {
        "schema": "audiollm.mixed_attempt_recovery.v1",
        "created_at_utc": now_utc(),
        "plan": plan,
        "derived_files": {relative: sha256_file(view / relative) for relative in VERBATIM_SIDECARS + DERIVED_SIDECARS},
        "quarantined_event_lines": [item["line"] for item in foreign],
        "statement": (
            "Derived evidence view: canonical sidecars copied verbatim, jobs.jsonl restricted to "
            "the canonical attempt's existing events, artifacts.json re-registered under the "
            "canonical attempt with hashes recomputed from the real files. No scheduler event, "
            "attempt, training result or hash was invented. The original mixed fold is unmodified "
            "and snapshotted under recovery/snapshot/."
        ),
    }
    (recovery / "recovery_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (recovery / "DERIVED_EVIDENCE.md").write_text(
        "# Derived evidence view\n\n" + manifest["statement"] + "\n\n"
        f"Source lineage: {plan['lineage_evidence']}\n"
        f"Head fingerprint: {plan['head_fingerprint']}\n"
        f"Plan sha256: {plan['plan_sha256']}\n",
        encoding="utf-8",
    )

    # 6. Verify the original fold is unchanged and the view is identity-consistent.
    _require(file_inventory(fold) == plan["original_files"], "original fold changed during recovery")
    from src.experiment_tracking.sidecars import read_modern_sidecars

    sidecars = read_modern_sidecars(view)
    _require(sidecars is not None, "derived view is not readable as modern tracked evidence")
    print(
        json.dumps(
            {
                "executed": True,
                "view_dir": str(view),
                "plan_sha256": plan["plan_sha256"],
                "derived_files": manifest["derived_files"],
            },
            indent=1,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--canonical-attempt", required=True)
    parser.add_argument("--lineage-evidence", type=Path, required=True)
    parser.add_argument("--head-attempt", type=Path, required=True)
    parser.add_argument("--view-dir", type=Path, required=True)
    parser.add_argument("--pre-snapshot-manifest", type=Path)
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve-plan")
    args = parser.parse_args()

    try:
        lineage = load_lineage(args.lineage_evidence)
        lineage["_path"] = str(args.lineage_evidence)
        lineage["_sha256"] = sha256_file(args.lineage_evidence)
        plan = build_plan(
            args.fold_dir,
            args.canonical_attempt,
            lineage,
            args.head_attempt,
            args.view_dir,
            args.pre_snapshot_manifest,
        )
        if not args.execute:
            plan_out = args.plan_out or (args.view_dir.parent / "recovery_plan.json")
            check_plan_outside_fold(args.fold_dir, plan_out)
            write_plan(plan, plan_out)
            print(json.dumps({"dry_run": True, "plan_out": str(plan_out), "plan_sha256": plan["plan_sha256"]}, indent=1))
            return 0
        _require(args.pre_snapshot_manifest is not None, "--execute requires --pre-snapshot-manifest")
        _require(bool(args.approve_plan), "--execute requires --approve-plan <dry-run plan sha256>")
        execute(plan, args.approve_plan)
        return 0
    except RecoveryError as error:
        print(f"RECOVERY REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
