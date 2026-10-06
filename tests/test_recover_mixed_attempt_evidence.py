"""Tests for the mixed-attempt derived-evidence recovery tool.

The synthetic fold mirrors the real cmdc|audio_text|s7|f4 shape: canonical
metadata/status/evaluations/run_config, a foreign attempt's artifacts identity
and job events, and stale artifact hashes. All tests exercise dry-run planning
and derived-view execution in temporary directories; the unchanged validation
gates must accept the derived view.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_tracking import lifecycle  # noqa: E402
from src.experiment_tracking.sidecars import read_modern_sidecars  # noqa: E402
from src.experiment_tracking.validate import validate_attempt  # noqa: E402
from tests.test_parallel_workflow_validate import _build_attempt  # noqa: E402
from tools import recover_mixed_attempt_evidence as recov  # noqa: E402

CANON = "20260821T000000Z-run1-abcdef01-12345678"
FOREIGN = "20260821T000000Z-run1-abcdef01-deadbeef"
OTHER_FOREIGN = "20260821T000000Z-run1-abcdef01-feedface"
FINGERPRINT = "d" * 64
CONFIG_FINGERPRINT = "e" * 64


def _head_and_lineage(tmp_path: Path) -> tuple[Path, Path]:
    head = tmp_path / "head_attempt"
    head.mkdir(exist_ok=True)
    (head / "metadata.json").write_text(
        json.dumps(
            {
                "attempt_id": "head-1",
                "parent": {
                    "parent_attempt_id": CANON,
                    "parent_checkpoint_role": "best_model",
                    "parent_checkpoint_path": "/gpfs/x/best_model",
                    "adapter_sha256": FINGERPRINT,
                    "adapter_config_sha256": CONFIG_FINGERPRINT,
                },
            }
        ),
        encoding="utf-8",
    )
    lineage = {
        "schema": "audiollm.mixed_attempt_lineage_evidence.v1",
        "canonical_attempt_id": CANON,
        "foreign_attempt_ids": [FOREIGN],
        "adapter_sha256": FINGERPRINT,
        "adapter_config_sha256": CONFIG_FINGERPRINT,
    }
    lineage_path = tmp_path / "lineage.json"
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
    return head, lineage_path


def _mixed_fold(tmp_path: Path) -> tuple[Path, Path, Path]:
    fold = _build_attempt(tmp_path)
    run_config = fold / "run_config.yaml"
    run_config.write_text(
        run_config.read_text(encoding="utf-8")
        + "tracking:\n  schema_version: audiollm.tracking.v1\n"
        + f"  attempt_id: {CANON}\n",
        encoding="utf-8",
    )
    foreign_started = lifecycle.new_job_event(
        job_key="train", job_type="train", event_type="STARTED",
        attempt_id=FOREIGN, fold=0, status="RUNNING",
    )
    foreign_completed = lifecycle.new_job_event(
        job_key="train", job_type="train", event_type="COMPLETED",
        attempt_id=FOREIGN, fold=0, status="COMPLETED",
    )
    with (fold / "jobs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(foreign_started) + "\n")
        handle.write(json.dumps(foreign_completed) + "\n")
    artifacts = json.loads((fold / "artifacts.json").read_text(encoding="utf-8"))
    artifacts["attempt_id"] = FOREIGN
    for entry in artifacts["artifacts"]:
        entry["sha256"] = "0" * 64
    (fold / "artifacts.json").write_text(json.dumps(artifacts, indent=1), encoding="utf-8")
    head, lineage_path = _head_and_lineage(tmp_path)
    return fold, head, lineage_path


def _pre_snapshot(tmp_path: Path, fold: Path) -> Path:
    manifest = {
        "created_at_utc": "2026-10-06T00:00:00Z",
        "live_fold": str(fold),
        "files": [
            {"path": relative, **record}
            for relative, record in recov.file_inventory(fold).items()
        ],
    }
    path = tmp_path / "pre_snapshot_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _plan(tmp_path: Path, fold: Path, head: Path, lineage: Path, view: Path, snapshot: Path | None = None):
    return recov.build_plan(
        fold, CANON, recov.load_lineage(lineage), head, view, snapshot
    )


def test_dry_run_plan_filters_foreign_and_recomputes(tmp_path: Path) -> None:
    fold, head, lineage = _mixed_fold(tmp_path)
    before = recov.file_inventory(fold)
    plan = _plan(tmp_path, fold, head, lineage, tmp_path / "view", _pre_snapshot(tmp_path, fold))
    assert plan["jobs"]["kept_lines"] == [0, 1]
    assert plan["jobs"]["foreign_lines"] == [2, 3]
    assert len(plan["jobs"]["foreign_events"]) == 2
    assert plan["artifacts"]["recomputed_hashes"] == 2
    assert plan["head_fingerprint"]["adapter_sha256"] == FINGERPRINT
    assert recov.file_inventory(fold) == before
    assert not (tmp_path / "view").exists()
    rebuilt = _plan(tmp_path, fold, head, lineage, tmp_path / "view", _pre_snapshot(tmp_path, fold))
    assert rebuilt["plan_sha256"] == plan["plan_sha256"]


def test_dry_run_cli_writes_plan_and_no_view(tmp_path: Path, monkeypatch, capsys) -> None:
    fold, head, lineage = _mixed_fold(tmp_path)
    view = tmp_path / "view"
    plan_out = tmp_path / "plan.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recover",
            "--fold-dir", str(fold),
            "--canonical-attempt", CANON,
            "--lineage-evidence", str(lineage),
            "--head-attempt", str(head),
            "--view-dir", str(view),
            "--pre-snapshot-manifest", str(_pre_snapshot(tmp_path, fold)),
            "--plan-out", str(plan_out),
        ],
    )
    assert recov.main() == 0
    assert plan_out.is_file()
    payload = json.loads(plan_out.read_text(encoding="utf-8"))
    assert payload["plan_sha256"]
    assert not view.exists()
    assert "dry_run" in capsys.readouterr().out


def test_execute_builds_derived_view_accepted_by_unchanged_gates(tmp_path: Path) -> None:
    fold, head, lineage = _mixed_fold(tmp_path)
    view = tmp_path / "view" / "fold_0"
    plan = _plan(tmp_path, fold, head, lineage, view, _pre_snapshot(tmp_path, fold))
    originals = recov.file_inventory(fold)
    recov.execute(plan, plan["plan_sha256"])

    for name in ("metadata.json", "status.json", "evaluations.json", "run_config.yaml"):
        assert (view / name).read_bytes() == (fold / name).read_bytes()
    original_jobs = (fold / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
    derived_jobs = (view / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
    assert derived_jobs == [original_jobs[i] for i in plan["jobs"]["kept_lines"]]

    artifacts = json.loads((view / "artifacts.json").read_text(encoding="utf-8"))
    assert artifacts["attempt_id"] == CANON
    for entry in artifacts["artifacts"]:
        if entry["sha256"] is not None:
            assert entry["sha256"] == recov.sha256_file(view / entry["path"])
    assert read_modern_sidecars(view) is not None
    quarantine = (view / "recovery" / "quarantined_foreign_events.jsonl").read_text(encoding="utf-8")
    assert quarantine.count("\n") == 2
    snapshot = view / "recovery" / "snapshot" / fold.name
    assert recov.file_inventory(snapshot) == originals
    assert recov.file_inventory(fold) == originals
    assert (view / "recovery" / "recovery_manifest.json").is_file()

    result = validate_attempt(view)
    assert result["ok"] is True, result["issues"]
    assert result["state"] == "COMPLETED_ON_MN5"


def test_rejects_canonical_mismatch(tmp_path: Path) -> None:
    fold, head, lineage = _mixed_fold(tmp_path)
    metadata = json.loads((fold / "metadata.json").read_text(encoding="utf-8"))
    metadata["attempt_id"] = FOREIGN
    (fold / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(recov.RecoveryError, match="not the canonical attempt"):
        _plan(tmp_path, fold, head, lineage, tmp_path / "view")


def test_rejects_unknown_foreign_attempt(tmp_path: Path) -> None:
    fold, head, lineage = _mixed_fold(tmp_path)
    payload = json.loads(lineage.read_text(encoding="utf-8"))
    payload["foreign_attempt_ids"] = [OTHER_FOREIGN]
    lineage.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(recov.RecoveryError, match="foreign events reference"):
        _plan(tmp_path, fold, head, lineage, tmp_path / "view")


def test_rejects_two_foreign_attempts(tmp_path: Path) -> None:
    fold, head, lineage = _mixed_fold(tmp_path)
    extra = lifecycle.new_job_event(
        job_key="train", job_type="train", event_type="STARTED",
        attempt_id=OTHER_FOREIGN, fold=0, status="RUNNING",
    )
    with (fold / "jobs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(extra) + "\n")
    with pytest.raises(recov.RecoveryError, match="foreign events reference"):
        _plan(tmp_path, fold, head, lineage, tmp_path / "view")


def test_rejects_head_fingerprint_mismatch(tmp_path: Path) -> None:
    fold, head, lineage = _mixed_fold(tmp_path)
    payload = json.loads(lineage.read_text(encoding="utf-8"))
    payload["adapter_sha256"] = "1" * 64
    lineage.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(recov.RecoveryError, match="adapter fingerprint does not match"):
        _plan(tmp_path, fold, head, lineage, tmp_path / "view")


def test_rejects_missing_artifact_file(tmp_path: Path) -> None:
    fold, head, lineage = _mixed_fold(tmp_path)
    (fold / "best_model" / "standalone_eval" / "predictions_subject_level.csv").unlink()
    with pytest.raises(recov.RecoveryError, match="path missing on disk"):
        _plan(tmp_path, fold, head, lineage, tmp_path / "view")


def test_rejects_stale_pre_snapshot(tmp_path: Path) -> None:
    fold, head, lineage = _mixed_fold(tmp_path)
    snapshot = _pre_snapshot(tmp_path, fold)
    run_config = fold / "run_config.yaml"
    run_config.write_text(run_config.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
    with pytest.raises(recov.RecoveryError, match="changed in the live fold"):
        _plan(tmp_path, fold, head, lineage, tmp_path / "view", snapshot)


def test_rejects_no_foreign_events(tmp_path: Path) -> None:
    fold = _build_attempt(tmp_path)
    run_config = fold / "run_config.yaml"
    run_config.write_text(
        run_config.read_text(encoding="utf-8") + f"tracking:\n  attempt_id: {CANON}\n",
        encoding="utf-8",
    )
    head, lineage = _head_and_lineage(tmp_path)
    with pytest.raises(recov.RecoveryError, match="no foreign events"):
        _plan(tmp_path, fold, head, lineage, tmp_path / "view")


def test_execute_requires_approval_and_refuses_overwrite(tmp_path: Path) -> None:
    fold, head, lineage = _mixed_fold(tmp_path)
    view = tmp_path / "view" / "fold_0"
    plan = _plan(tmp_path, fold, head, lineage, view, _pre_snapshot(tmp_path, fold))
    with pytest.raises(recov.RecoveryError, match="approve-plan does not match"):
        recov.execute(plan, "0" * 64)
    view.mkdir(parents=True)
    with pytest.raises(recov.RecoveryError, match="refusing to overwrite"):
        recov.execute(plan, plan["plan_sha256"])


def test_rejects_view_overlapping_original_fold(tmp_path: Path) -> None:
    fold, head, lineage = _mixed_fold(tmp_path)
    with pytest.raises(recov.RecoveryError, match="must not be the original fold"):
        _plan(tmp_path, fold, head, lineage, fold)
    with pytest.raises(recov.RecoveryError, match="must not be inside the original fold"):
        _plan(tmp_path, fold, head, lineage, fold / "recovered")
    with pytest.raises(recov.RecoveryError, match="must not be an ancestor of the original fold"):
        _plan(tmp_path, fold, head, lineage, tmp_path)


def test_rejects_plan_out_inside_fold(tmp_path: Path, monkeypatch, capsys) -> None:
    fold, head, lineage = _mixed_fold(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recover",
            "--fold-dir", str(fold),
            "--canonical-attempt", CANON,
            "--lineage-evidence", str(lineage),
            "--head-attempt", str(head),
            "--view-dir", str(tmp_path / "view"),
            "--plan-out", str(fold / "plan.json"),
        ],
    )
    assert recov.main() == 2
    assert "plan output must not be inside the original fold" in capsys.readouterr().err


def test_rejects_symlinked_artifact_outside_tree(tmp_path: Path) -> None:
    fold, head, lineage = _mixed_fold(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_text("subject_id,label\n", encoding="utf-8")
    predictions = fold / "best_model" / "standalone_eval" / "predictions_subject_level.csv"
    predictions.unlink()
    predictions.symlink_to(outside)
    with pytest.raises(recov.RecoveryError, match="resolves outside the original evidence tree"):
        _plan(tmp_path, fold, head, lineage, tmp_path / "view")


def test_execute_rejects_fold_changed_after_plan(tmp_path: Path) -> None:
    fold, head, lineage = _mixed_fold(tmp_path)
    view = tmp_path / "view" / "fold_0"
    plan = _plan(tmp_path, fold, head, lineage, view, _pre_snapshot(tmp_path, fold))
    metadata = json.loads((fold / "metadata.json").read_text(encoding="utf-8"))
    metadata["created_at_utc"] = "2026-10-07T00:00:00.000000Z"
    (fold / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(recov.RecoveryError, match="changed since the approved dry-run"):
        recov.execute(plan, plan["plan_sha256"])


def test_real_cli_validate_and_finish_accept_derived_view(tmp_path: Path) -> None:
    import subprocess

    fold, head, lineage = _mixed_fold(tmp_path)
    view = tmp_path / "derived" / "fold_0"
    plan = _plan(tmp_path, fold, head, lineage, view, _pre_snapshot(tmp_path, fold))
    recov.execute(plan, plan["plan_sha256"])

    validate = subprocess.run(
        [sys.executable, "tools/exp.py", "validate", "--fold-dir", str(view)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert validate.returncode == 0, validate.stdout + validate.stderr
    finish = subprocess.run(
        [sys.executable, "tools/exp.py", "finish", "--fold-dir", str(view)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert finish.returncode == 0, finish.stdout + finish.stderr
    status = json.loads((view / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "REPORTABLE"
    assert recov.file_inventory(fold) == plan["original_files"]
