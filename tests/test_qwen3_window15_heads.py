"""Focused tests for the window15 head planner and its guarded dispatch wrapper."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
import yaml

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

from tools import qwen3_window15_heads_guard as guard  # noqa: E402
from tools.qwen3_window15_heads_plan import PlanError, build_plan  # noqa: E402


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fixture_rows(count: int = 126) -> list[dict]:
    rows = []
    routes = [
        "daic_audio_only",
        "d3tec_audio_only",
        "androids_interview_audio_text",
        "cmdc_audio_text",
        "turkish_audio_only",
    ]
    for route in routes:
        for seed in (7, 1337, 2024):
            for fold in range(10):
                rows.append(
                    {
                        "registry_key": f"{route}|{seed}|{fold}",
                        "route_id": route,
                        "seed": seed,
                        "fold": fold,
                        "planned_run_name": f"q3w15_{route}_s{seed}_f{fold}",
                        "treatment_config": f"configs/experiments/window15/{route}_harmonized_selmacrof1_likelihood_v1_window15.yaml",
                    }
                )
    return rows[:count]


def _write_config(config_dir: Path, relative: str, dataset: str, use_text: bool) -> None:
    target = config_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            {
                "dataset": dataset,
                "data": {
                    "use_audio": True,
                    "use_text": use_text,
                    "segment_seconds": 15.0,
                    "processor_min_audio_samples": 201,
                },
            }
        ),
        encoding="utf-8",
    )


def _canonical_fold(tmp_path: Path, attempt: str = "a0") -> Path:
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "metadata.json").write_text(json.dumps({"attempt_id": attempt}), encoding="utf-8")
    (fold_dir / "run_config.yaml").write_text(
        yaml.safe_dump(
            {
                "tracking": {"attempt_id": attempt, "fold": 0},
                "input_modality": "audio_only",
                "manifest_hash": "abc",
                "split_metadata_hash": "def",
                "config": {
                    "dataset": "daic",
                    "manifest_variant": "unprocessed_participant_speech_packed30_15s_v1",
                    "data": {"processor_min_audio_samples": 201, "segment_seconds": 15.0},
                },
            }
        ),
        encoding="utf-8",
    )
    # Canonical sidecar schema: SUBMITTED/STARTED/COMPLETED, no exit codes.
    events = [
        {"job_key": "train", "event_type": "SUBMITTED", "slurm_job_id": "900001", "attempt_id": attempt, "status": "PENDING"},
        {"job_key": "best_eval", "event_type": "SUBMITTED", "slurm_job_id": "900002", "attempt_id": attempt, "status": "PENDING"},
        {"job_key": "train", "event_type": "STARTED", "slurm_job_id": None, "attempt_id": attempt, "status": "RUNNING"},
        {"job_key": "train", "event_type": "COMPLETED", "slurm_job_id": None, "attempt_id": attempt, "status": "COMPLETED"},
    ]
    (fold_dir / "jobs.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )
    return fold_dir


def _item(fold_dir: Path, attempt: str = "a0") -> dict:
    return {
        "key": "daic_audio_only_native|7|0",
        "route_id": "daic_audio_only_native",
        "seed": 7,
        "fold": 0,
        "attempt_id": attempt,
        "dataset": "daic",
        "modality": "audio_only",
        "local_fold_dir": str(fold_dir),
    }


def _ledger(attempt: str = "a0") -> dict[str, dict]:
    return {"daic_audio_only_native|7|0": {"status": "submitted", "attempt_id": attempt}}


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------


def test_planner_binds_exact_treatment_attempts_and_rejects_control_configs(tmp_path, monkeypatch) -> None:
    work = tmp_path
    monkeypatch.setattr("tools.qwen3_window15_heads_plan.LANE", work)
    monkeypatch.setattr("tools.qwen3_window15_heads_plan.RUN_ROOT", work / "output_model")
    rows = _fixture_rows()
    for row in rows:
        _write_config(work, row["treatment_config"], str(row["route_id"]).split("_audio")[0], use_text="audio_text" in row["route_id"])
    contract = work / "contract.json"
    contract.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    ledger = work / "submissions.jsonl"
    ledger.write_text(
        "\n".join(
            json.dumps(
                {
                    "key": row["registry_key"],
                    "status": "submitted",
                    "attempt_id": f"attempt-{index}",
                    "job_ids": {"train": str(1000 + index), "best_eval": str(2000 + index)},
                }
            )
            for index, row in enumerate(rows[:2])
        )
        + "\n",
        encoding="utf-8",
    )
    for index, row in enumerate(rows[:2]):
        attempt_dir = work / "outputs" / "exp_submit" / f"attempt-{index}"
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "contract.json").write_text(
            json.dumps(
                {
                    "dataset": str(row["route_id"]).split("_audio")[0],
                    "config_path_remote": f"/remote/code/{row['treatment_config']}",
                    "deployment_id": "feat-qwen3-window15-20261008-xyz",
                    "fold_dir": f"/remote/output_model/qwen3_window15_20261008/{row['route_id']}/{row['planned_run_name']}/fold_{row['fold']}",
                }
            ),
            encoding="utf-8",
        )
    fold_dir = (
        work
        / "output_model"
        / ("audio_text" if "audio_text" in rows[0]["route_id"] else "audio_only")
        / str(rows[0]["route_id"]).split("_audio")[0]
        / rows[0]["planned_run_name"]
        / f"fold_{rows[0]['fold']}"
    )
    fold_dir.mkdir(parents=True)
    (fold_dir / "status.json").write_text(json.dumps({"state": "LOCALLY_VALIDATED"}), encoding="utf-8")

    parent_map, audit = build_plan(contract, ledger, exp_submit_dir=work / "outputs" / "exp_submit")
    assert audit["keys_total"] == 126
    assert len(parent_map["entries"]) == 2
    assert all(entry["config"].startswith("configs/experiments/window15/") for entry in parent_map["entries"])
    assert all("configs/main/" not in entry["config"] for entry in parent_map["entries"])
    assert audit["status_counts"]["eligible"] == 1
    assert audit["status_counts"]["waiting_training"] == 125

    bad_rows = [dict(row) for row in rows]
    bad_rows[0]["treatment_config"] = "configs/main/daic_audio_only_harmonized_selmacrof1_likelihood_v1.yaml"
    bad_contract = work / "bad_contract.json"
    bad_contract.write_text(json.dumps({"rows": bad_rows}), encoding="utf-8")
    with pytest.raises(PlanError, match="not a window15 treatment config"):
        build_plan(bad_contract, ledger, exp_submit_dir=work / "outputs" / "exp_submit")


def test_planner_requires_processor_minimum(tmp_path, monkeypatch) -> None:
    work = tmp_path
    monkeypatch.setattr("tools.qwen3_window15_heads_plan.LANE", work)
    monkeypatch.setattr("tools.qwen3_window15_heads_plan.RUN_ROOT", work / "output_model")
    rows = _fixture_rows()
    for row in rows:
        _write_config(work, row["treatment_config"], str(row["route_id"]).split("_audio")[0], use_text="audio_text" in row["route_id"])
    stripped = work / rows[0]["treatment_config"]
    payload = yaml.safe_load(stripped.read_text())
    del payload["data"]["processor_min_audio_samples"]
    stripped.write_text(yaml.safe_dump(payload), encoding="utf-8")
    contract = work / "contract.json"
    contract.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    with pytest.raises(PlanError, match="processor_min_audio_samples"):
        build_plan(contract, tmp_path / "missing.jsonl")


def test_planner_eligible_parent_map_filters_ineligible_keys() -> None:
    from tools.qwen3_window15_heads_plan import eligible_parent_map

    parent_map = {
        "entries": [
            {"route_id": "a", "parent_training_seed": 7, "fold": 0, "config": "c", "fold_dir": "d", "parent_attempt_id": "x"},
            {"route_id": "b", "parent_training_seed": 7, "fold": 0, "config": "c", "fold_dir": "d", "parent_attempt_id": "y"},
        ]
    }
    audit = {"keys": [{"key": "a|7|0", "status": "eligible"}, {"key": "b|7|0", "status": "waiting_training"}]}
    filtered = eligible_parent_map(parent_map, audit)
    assert [entry["route_id"] for entry in filtered["entries"]] == ["a"]


# ---------------------------------------------------------------------------
# guard: identity + accounting
# ---------------------------------------------------------------------------


def test_guard_bind_parent_identity_realistic_sidecar(tmp_path) -> None:
    fold_dir = _canonical_fold(tmp_path)
    item = _item(fold_dir)
    bound = guard.bind_parent_identity(item, _ledger())
    assert bound == {"attempt": "a0", "ids": {"train": "900001", "best_eval": "900002"}}

    accounting = {
        "900001": {"state": "COMPLETED", "exit": "0:0"},
        "900002": {"state": "COMPLETED", "exit": "0:0"},
    }
    guard.assert_completed(item["key"], bound["ids"], accounting)
    with pytest.raises(guard.GuardError, match="COMPLETED 0:0"):
        guard.assert_completed(item["key"], bound["ids"], {"900001": {"state": "COMPLETED", "exit": "1:0"}, "900002": accounting["900002"]})
    with pytest.raises(guard.GuardError, match="missing from live accounting"):
        guard.assert_completed(item["key"], bound["ids"], {"900001": accounting["900001"]})

    # stale audit attempt
    with pytest.raises(guard.GuardError, match="does not match the ledger attempt"):
        guard.bind_parent_identity(_item(fold_dir, attempt="other"), _ledger())

    # missing best_eval submission id
    (fold_dir / "jobs.jsonl").write_text(
        json.dumps({"job_key": "train", "event_type": "SUBMITTED", "slurm_job_id": "900001", "attempt_id": "a0"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(guard.GuardError, match="no numeric best_eval submission id"):
        guard.bind_parent_identity(item, _ledger())

    # wrong window identity
    _canonical_fold(tmp_path)
    run_config = yaml.safe_load((fold_dir / "run_config.yaml").read_text())
    run_config["config"]["manifest_variant"] = "unprocessed_participant_speech_packed30_v1"
    run_config["config"]["data"]["segment_seconds"] = 30.0
    (fold_dir / "run_config.yaml").write_text(yaml.safe_dump(run_config), encoding="utf-8")
    with pytest.raises(guard.GuardError, match="not the 15-second treatment identity"):
        guard.bind_parent_identity(item, _ledger())


def test_guard_accounting_parser_ignores_steps_and_rejects_contradictions() -> None:
    class FakeSsh:
        def __init__(self, output):
            self.output = output

        def __call__(self, argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, self.output, "")

    raw = "\n".join(
        [
            "900001|COMPLETED|0:0",
            "900001.batch|COMPLETED|0:0",
            "900001.extern|COMPLETED|0:0",
            "900002|COMPLETED|0:0",
            "999999|RUNNING|0:0",
        ]
    )
    states = guard.query_top_level_accounting(["900001", "900002"], FakeSsh(raw))
    assert states == {
        "900001": {"state": "COMPLETED", "exit": "0:0"},
        "900002": {"state": "COMPLETED", "exit": "0:0"},
    }
    contradictory = "900001|COMPLETED|0:0\n900001|FAILED|1:0\n"
    with pytest.raises(guard.GuardError, match="contradictory"):
        guard.query_top_level_accounting(["900001"], FakeSsh(contradictory))


def test_guard_blocks_prior_head_keys_until_reconciled() -> None:
    ledger = {
        "head:daic_audio_only_native|7|0": {"status": "uncertain", "key": "head:daic_audio_only_native|7|0"},
        "head:d3tec_audio_only_native|7|0": {"status": "reconciled", "key": "head:d3tec_audio_only_native|7|0"},
    }
    blocked = guard._head_blocked_keys(ledger)
    assert "daic_audio_only_native|7|0" in blocked
    assert "d3tec_audio_only_native|7|0" not in blocked


def test_guard_valid_ids_rejects_same_and_stale() -> None:
    assert guard._valid_ids({"extract": "1", "classifier": "2"}, set()) == ("1", "2")
    assert guard._valid_ids({"extract": "1", "classifier": "1"}, set()) is None
    assert guard._valid_ids({"extract": "1", "classifier": "2"}, {"1"}) is None
    assert guard._valid_ids({"extract": "x", "classifier": "2"}, set()) is None


# ---------------------------------------------------------------------------
# guard: flow
# ---------------------------------------------------------------------------


def _flow_fixture(tmp_path, monkeypatch):
    registry = tmp_path / "head_submissions.jsonl"
    plan = tmp_path / "plan.json"
    matrix = tmp_path / "matrix.json"
    matrix.write_text("{}", encoding="utf-8")
    plan.write_text(
        json.dumps(
            {
                "routes": [
                    {
                        "route_id": "daic_audio_only_native",
                        "jobs": [
                            {
                                "seed": 7,
                                "fold": 0,
                                "parent_status": "resolved",
                                "parent": {
                                    "attempt_id": "a0",
                                    "adapter_sha256": "aa",
                                    "adapter_config_sha256": "bb",
                                    "manifest_hash": "cc",
                                    "split_fingerprint": {"sha256": "dd"},
                                },
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    item = {
        "key": "daic_audio_only_native|7|0",
        "route_id": "daic_audio_only_native",
        "seed": 7,
        "fold": 0,
        "attempt_id": "a0",
        "run_name": "r",
        "dataset": "daic",
        "modality": "audio_only",
        "local_fold_dir": str(tmp_path),
    }
    audit = {"keys_total": 126, "keys": [dict(item, status="eligible")], "status_counts": {}}
    recorded: list[dict] = []

    def record(entry, *a, **k):
        recorded.append(entry)

    monkeypatch.setattr(guard, "REGISTRY", registry)
    monkeypatch.setattr(guard, "build_plan", lambda *a, **k: ({}, audit))
    monkeypatch.setattr(guard, "lane_submission_lock", lambda *a, **k: _NullLock())
    monkeypatch.setattr(guard, "append_record", record)
    monkeypatch.setattr(guard, "own_job_ids", lambda: ([], 0))
    monkeypatch.setattr(guard, "query_job_states", lambda ids: ({}, 10))
    monkeypatch.setattr(guard, "storage_admission", lambda: {"gpfs_projects_remaining_gb": 1000.0})
    monkeypatch.setattr(guard, "bind_parent_identity", lambda it, latest: {"attempt": "a0", "ids": {"train": "900001", "best_eval": "900002"}})
    monkeypatch.setattr(
        guard,
        "query_top_level_accounting",
        lambda ids, runner=None: {"900001": {"state": "COMPLETED", "exit": "0:0"}, "900002": {"state": "COMPLETED", "exit": "0:0"}},
    )

    def latest():
        out: dict[str, dict] = {}
        if not any(rec.get("key") == "daic_audio_only_native|7|0" for rec in recorded):
            out["daic_audio_only_native|7|0"] = {"status": "submitted", "attempt_id": "a0"}
        for rec in recorded:
            out[str(rec.get("key"))] = rec
        return out

    monkeypatch.setattr(guard, "_ledger_latest", latest)
    args = types.SimpleNamespace(
        matrix=matrix,
        plan=plan,
        deployment_id="feat-qwen3-window15-20261008-xyz",
        limit=None,
        only=None,
        reconcile_key=None,
        reason=None,
        dry_run=False,
    )
    return args, registry, recorded, latest


def test_guard_submits_only_with_bound_fresh_registry(tmp_path, monkeypatch) -> None:
    args, registry, recorded, latest = _flow_fixture(tmp_path, monkeypatch)

    def fake_run(cmd):
        if "submit" in cmd and "plan" not in cmd:
            with registry.open("a", encoding="utf-8") as handle:
                # fully bound row: head attempt, parent attempt, deployment, key
                handle.write(
                    json.dumps(
                        {
                            "registry_key": "daic_audio_only_native|7|0",
                            "attempt_id": "head-attempt-1",
                            "parent_attempt_id": "a0",
                            "deployment_id": "feat-qwen3-window15-20261008-xyz",
                            "extract_job_id": "910001",
                            "classifier_job_id": "910002",
                        }
                    )
                    + "\n"
                )
            return subprocess.CompletedProcess(cmd, 0, "human summary only\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(guard, "_run", fake_run)
    rc = guard._submit_locked(args)
    assert rc == 0
    finals = [rec for rec in recorded if rec["key"] == "head:daic_audio_only_native|7|0" and rec["status"] == "submitted"]
    assert finals and finals[-1]["job_ids"] == {"extract": "910001", "classifier": "910002"}


def test_guard_unbound_registry_row_is_not_promoted(tmp_path, monkeypatch) -> None:
    args, registry, recorded, latest = _flow_fixture(tmp_path, monkeypatch)

    def fake_run(cmd):
        if "submit" in cmd and "plan" not in cmd:
            with registry.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"registry_key": "daic_audio_only_native|7|0", "extract_job_id": "920001", "classifier_job_id": "920002"})
                    + "\n"
                )
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(guard, "_run", fake_run)
    rc = guard._submit_locked(args)
    assert rc == 1
    finals = [rec for rec in recorded if rec["key"] == "head:daic_audio_only_native|7|0"]
    assert finals[-1]["status"] == "uncertain"
    assert finals[-1]["reservation"] == 2


def test_guard_timeout_still_reserves_and_second_call_makes_zero_submits(tmp_path, monkeypatch) -> None:
    args, registry, recorded, latest = _flow_fixture(tmp_path, monkeypatch)
    submits: list[list[str]] = []

    def fake_run(cmd):
        if "submit" in cmd and "plan" not in cmd:
            submits.append(cmd)
            raise subprocess.TimeoutExpired(cmd, 3600)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(guard, "_run", fake_run)
    rc = guard._submit_locked(args)
    assert rc == 1
    finals = [rec for rec in recorded if rec["key"] == "head:daic_audio_only_native|7|0"]
    assert finals[-1]["status"] == "uncertain"
    assert finals[-1]["reservation"] == 2
    assert len(submits) == 1

    # second call after the timeout: blocked key, zero new submits
    rc = guard._submit_locked(args)
    assert rc == 0
    assert len(submits) == 1


def test_guard_fresh_log_requires_content_change(tmp_path) -> None:
    log = tmp_path / "submit_output.log"
    log.write_text("stale content\n", encoding="utf-8")
    pre = {str(log): guard._log_digest(log)}
    unchanged = guard._fresh_log_ids([log], pre)
    assert unchanged == {}
    log.write_text("=== JOB k ===\nEXTRACT_ID=1\nCLASSIFIER_ID=2\n", encoding="utf-8")
    changed = guard._fresh_log_ids([log], pre)
    assert changed == {"k": {"extract": "1", "classifier": "2"}}


def test_guard_shared_lock_refuses_second_submitter_across_processes(tmp_path, monkeypatch) -> None:
    lock_path = tmp_path / "submission.lock"
    monkeypatch.setattr("tools.qwen3_window15_dispatch.LANE_SUBMISSION_LOCK", lock_path)
    script = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(LANE)!r})\n"
        "from tools.qwen3_window15_dispatch import lane_submission_lock\n"
        "with lane_submission_lock(__import__('pathlib').Path(sys.argv[1])):\n"
        "    print('held', flush=True)\n"
        "    time.sleep(10)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", script, str(lock_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "held"
        with pytest.raises(Exception) as exc:
            with guard.lane_submission_lock(lock_path):
                pass
        assert "another lane submitter" in str(exc.value)
    finally:
        proc.terminate()
        proc.wait(timeout=10)
