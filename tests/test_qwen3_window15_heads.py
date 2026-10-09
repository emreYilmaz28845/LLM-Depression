"""Focused tests for the window15 head planner and its guarded dispatch wrapper."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

from tools import qwen3_window15_heads_guard as guard  # noqa: E402
from tools.qwen3_window15_heads_plan import PlanError, build_plan  # noqa: E402


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


def test_planner_binds_exact_treatment_attempts_and_rejects_control_configs(tmp_path, monkeypatch) -> None:
    work = tmp_path

    monkeypatch.setattr("tools.qwen3_window15_heads_plan.LANE", work)
    monkeypatch.setattr("tools.qwen3_window15_heads_plan.RUN_ROOT", work / "output_model")
    rows = _fixture_rows()
    for row in rows:
        _write_config(
            work,
            row["treatment_config"],
            str(row["route_id"]).split("_audio")[0],
            use_text="audio_text" in row["route_id"],
        )
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
    assert {entry["parent_attempt_id"] for entry in parent_map["entries"]} == {"attempt-0", "attempt-1"}
    assert all(entry["config"].startswith("configs/experiments/window15/") for entry in parent_map["entries"])
    assert all("configs/main/" not in entry["config"] for entry in parent_map["entries"])
    assert all(entry["fold_dir"].startswith("/remote/output_model/") for entry in parent_map["entries"])
    counts = audit["status_counts"]
    assert counts["eligible"] == 1
    assert counts["waiting_training"] == 125
    assert counts["eligible"] + counts["waiting_training"] + counts["waiting_validation"] + counts["blocked_failed"] == 126

    bad_rows = [dict(row) for row in rows]
    bad_rows[0]["treatment_config"] = "configs/main/daic_audio_only_harmonized_selmacrof1_likelihood_v1.yaml"
    bad_contract = work / "bad_contract.json"
    bad_contract.write_text(json.dumps({"rows": bad_rows}), encoding="utf-8")
    with pytest.raises(PlanError, match="not a window15 treatment config"):
        build_plan(bad_contract, ledger, exp_submit_dir=work / "outputs" / "exp_submit")

    dup_rows = [dict(rows[0]), dict(rows[0])] + [dict(row) for row in rows[2:]]
    dup_contract = work / "dup_contract.json"
    dup_contract.write_text(json.dumps({"rows": dup_rows}), encoding="utf-8")
    with pytest.raises(PlanError, match="duplicate treatment key"):
        build_plan(dup_contract, ledger, exp_submit_dir=work / "outputs" / "exp_submit")


def test_planner_requires_processor_minimum(tmp_path, monkeypatch) -> None:
    work = tmp_path
    monkeypatch.setattr("tools.qwen3_window15_heads_plan.LANE", work)
    monkeypatch.setattr("tools.qwen3_window15_heads_plan.RUN_ROOT", work / "output_model")
    rows = _fixture_rows()
    for row in rows:
        _write_config(
            work,
            row["treatment_config"],
            str(row["route_id"]).split("_audio")[0],
            use_text="audio_text" in row["route_id"],
        )
    stripped = work / rows[0]["treatment_config"]
    payload = yaml.safe_load(stripped.read_text())
    del payload["data"]["processor_min_audio_samples"]
    stripped.write_text(yaml.safe_dump(payload), encoding="utf-8")
    contract = work / "contract.json"
    contract.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    with pytest.raises(PlanError, match="processor_min_audio_samples"):
        build_plan(contract, tmp_path / "missing.jsonl")


def _eligible_item(work: Path, attempt: str = "attempt-0") -> tuple[dict, Path]:
    fold_dir = work / "fold_0"
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
    (fold_dir / "jobs.jsonl").write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"job_key": "train", "event_type": "TERMINAL", "status": "COMPLETED", "exit_code": "0:0"},
                {"job_key": "best_eval", "event_type": "TERMINAL", "status": "COMPLETED", "exit_code": "0:0"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    item = {
        "key": "daic_audio_only_native|7|0",
        "route_id": "daic_audio_only_native",
        "seed": 7,
        "fold": 0,
        "attempt_id": attempt,
        "dataset": "daic",
        "modality": "audio_only",
        "local_fold_dir": str(fold_dir),
    }
    return item, fold_dir


def test_guard_verify_eligible_parent_binds_and_refuses(tmp_path) -> None:
    item, fold_dir = _eligible_item(tmp_path)
    latest = {"daic_audio_only_native|7|0": {"status": "submitted", "attempt_id": "attempt-0"}}
    assert guard.verify_eligible_parent(item, latest) == "attempt-0"

    # stale/wrong audit attempt: ledger attempt differs
    with pytest.raises(guard.GuardError, match="does not match the ledger attempt"):
        guard.verify_eligible_parent(dict(item, attempt_id="attempt-other"), latest)

    # metadata attempt mismatch
    (fold_dir / "metadata.json").write_text(json.dumps({"attempt_id": "other"}), encoding="utf-8")
    with pytest.raises(guard.GuardError, match="metadata attempt"):
        guard.verify_eligible_parent(item, latest)
    (fold_dir / "metadata.json").write_text(json.dumps({"attempt_id": "attempt-0"}), encoding="utf-8")

    # step-only / incomplete terminal evidence is refused
    (fold_dir / "jobs.jsonl").write_text(
        json.dumps({"job_key": "train", "event_type": "SUBMITTED", "status": "PENDING"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(guard.GuardError, match="TERMINAL COMPLETED 0:0"):
        guard.verify_eligible_parent(item, latest)

    # wrong window identity refused
    (fold_dir / "jobs.jsonl").write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"job_key": "train", "event_type": "TERMINAL", "status": "COMPLETED", "exit_code": "0:0"},
                {"job_key": "best_eval", "event_type": "TERMINAL", "status": "COMPLETED", "exit_code": "0:0"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    run_config = yaml.safe_load((fold_dir / "run_config.yaml").read_text())
    run_config["config"]["manifest_variant"] = "unprocessed_participant_speech_packed30_v1"
    run_config["config"]["data"]["segment_seconds"] = 30.0
    (fold_dir / "run_config.yaml").write_text(yaml.safe_dump(run_config), encoding="utf-8")
    with pytest.raises(guard.GuardError, match="not the 15-second treatment identity"):
        guard.verify_eligible_parent(item, latest)


def test_guard_valid_ids_rejects_same_and_stale() -> None:
    assert guard._valid_ids({"extract": "1", "classifier": "2"}, set()) == ("1", "2")
    assert guard._valid_ids({"extract": "1", "classifier": "1"}, set()) is None
    assert guard._valid_ids({"extract": "1", "classifier": "2"}, {"1"}) is None
    assert guard._valid_ids({"extract": "1", "classifier": "2"}, {"2"}) is None
    assert guard._valid_ids({"extract": "x", "classifier": "2"}, set()) is None


def test_guard_partial_and_finalize_reservations_are_durable(tmp_path, monkeypatch) -> None:
    ledger = tmp_path / "submissions.jsonl"
    recorded: list[dict] = []

    def record(entry: dict) -> None:
        recorded.append(entry)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    monkeypatch.setattr(guard, "append_record", record)
    batch = [{"key": "daic_audio_only_native|7|0", "run_name": "r", "attempt_id": "a0"}]
    guard._finalize_batch(batch, {}, set(), "generic plan failed rc=1", {"daic_audio_only_native|7|0": "a0"})
    assert recorded[-1]["status"] == "uncertain"
    assert recorded[-1]["reservation"] == 2
    assert recorded[-1]["parent_attempt_id"] == "a0"

    # a partial delivery keeps the known id and the reservation
    from tools.qwen3_window15_dispatch import own_job_ids, own_nonterminal_count

    exec_ledger = tmp_path / "exec.json"
    exec_ledger.write_text(json.dumps({"jobs": []}), encoding="utf-8")
    partial = {
        "key": "head:daic_audio_only_native|7|0",
        "status": "uncertain",
        "job_ids": {"extract": "555"},
        "attempt_id": None,
        "reservation": 2,
    }
    record(partial)
    ids, uncertain = own_job_ids(ledger, tmp_path / "no-run-root", exec_ledger, tmp_path / "no-logs")
    assert "555" in ids
    assert uncertain == 1
    assert own_nonterminal_count(ids, {"555": "RUNNING"}, uncertain) == 3  # known + reservation 2

    # a later authoritative delivery for the same key supersedes the reservation
    record(
        {
            "key": "head:daic_audio_only_native|7|0",
            "status": "submitted",
            "job_ids": {"extract": "555", "classifier": "556"},
            "attempt_id": None,
        }
    )
    ids, uncertain = own_job_ids(ledger, tmp_path / "no-run-root", exec_ledger, tmp_path / "no-logs")
    assert {"555", "556"} <= set(ids)
    assert uncertain == 0  # last record per key wins


def test_guard_shared_lock_refuses_second_submitter_across_processes(tmp_path, monkeypatch) -> None:
    lock_path = tmp_path / "submission.lock"
    monkeypatch.setattr("tools.qwen3_window15_dispatch.LANE_SUBMISSION_LOCK", lock_path)
    script = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(LANE)!r})\n"
        "from tools.qwen3_window15_dispatch import lane_submission_lock\n"
        "path = sys.argv[1]\n"
        "with lane_submission_lock(__import__('pathlib').Path(path)):\n"
        "    print('held', flush=True)\n"
        "    time.sleep(10)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(lock_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        line = proc.stdout.readline().strip()
        assert line == "held"
        with pytest.raises(Exception) as exc:
            with guard.lane_submission_lock(lock_path):
                pass
        assert "another lane submitter" in str(exc.value)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_guard_reconciles_only_authoritative_fresh_sources(tmp_path, monkeypatch) -> None:
    from tools.qwen3_window15_dispatch import AdmissionError

    registry = tmp_path / "head_submissions.jsonl"
    plan = tmp_path / "plan.json"
    matrix = tmp_path / "matrix.json"
    plan.write_text(json.dumps({"routes": []}), encoding="utf-8")
    matrix.write_text("{}", encoding="utf-8")

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
    monkeypatch.setattr(guard, "REGISTRY", registry)
    monkeypatch.setattr(guard, "build_plan", lambda *a, **k: ({}, audit))
    monkeypatch.setattr(guard, "verify_eligible_parent", lambda it, latest: "a0")
    monkeypatch.setattr(guard, "own_job_ids", lambda: ([], 0))
    monkeypatch.setattr(guard, "query_job_states", lambda ids: ({}, 10))
    monkeypatch.setattr(guard, "storage_admission", lambda: {"gpfs_projects_remaining_gb": 1000.0})
    monkeypatch.setattr(guard, "append_record", lambda entry, *a, **k: recorded.append(entry))
    monkeypatch.setattr(guard, "lane_submission_lock", lambda *a, **k: _nullcontext())

    # the plan entry must be resolved with adapter/manifest/split hashes
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

    runs: list[list[str]] = []

    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess:
        runs.append(cmd)
        if "submit" in cmd:
            # authoritative delivery appears in the registry, never in stdout
            with registry.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "registry_key": "daic_audio_only_native|7|0",
                            "extract_job_id": "900001",
                            "classifier_job_id": "900002",
                        }
                    )
                    + "\n"
                )
            return subprocess.CompletedProcess(cmd, 0, "human summary only: wrote /tmp/plan.json\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(guard, "_run", fake_run)

    import types

    args = types.SimpleNamespace(
        matrix=matrix,
        plan=plan,
        deployment_id="feat-qwen3-window15-20261008-xyz",
        limit=None,
        only=None,
        dry_run=False,
    )
    rc = guard._submit_locked(args)
    assert rc == 0
    assert any(entry.get("status") == "held" for entry in recorded)
    finals = [entry for entry in recorded if entry["key"] == "head:daic_audio_only_native|7|0"]
    assert finals[-1]["status"] == "submitted"
    assert finals[-1]["job_ids"] == {"extract": "900001", "classifier": "900002"}


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_planner_eligible_parent_map_filters_ineligible_keys() -> None:
    from tools.qwen3_window15_heads_plan import eligible_parent_map

    parent_map = {
        "entries": [
            {"route_id": "a", "parent_training_seed": 7, "fold": 0, "config": "c", "fold_dir": "d", "parent_attempt_id": "x"},
            {"route_id": "b", "parent_training_seed": 7, "fold": 0, "config": "c", "fold_dir": "d", "parent_attempt_id": "y"},
        ]
    }
    audit = {
        "keys": [
            {"key": "a|7|0", "status": "eligible"},
            {"key": "b|7|0", "status": "waiting_training"},
        ]
    }
    filtered = eligible_parent_map(parent_map, audit)
    assert [entry["route_id"] for entry in filtered["entries"]] == ["a"]
