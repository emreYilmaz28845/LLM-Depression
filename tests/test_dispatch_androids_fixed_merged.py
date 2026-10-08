from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.dispatch_androids_fixed_merged import (
    AdmissionError,
    leg_admission,
    ledger_reservations,
    registry_job_ids,
    select_legs,
    submit_leg,
)


def _result(returncode: int, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _append_registry(runtime: Path, run_id: str, jobs: list[dict], mode: str = "sbatch") -> list[str]:
    path = runtime / "registries" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "submission_mode": mode, "jobs": []}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("jobs", [])
    payload["jobs"].extend(jobs)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return [str(job["job_id"]) for job in payload["jobs"]]


def _job(job_id: str, stage: str, kind: str = "train") -> dict:
    return {"job_id": job_id, "stage": stage, "kind": kind}


def _runner(runtime: Path, run_id: str, jobs: list[dict], *, rc: int = 0, raise_exc=None):
    def run(command, **kwargs):
        ids = _append_registry(runtime, run_id, jobs)
        if raise_exc is not None:
            if isinstance(raise_exc, subprocess.TimeoutExpired):
                raise subprocess.TimeoutExpired(
                    cmd="submit", timeout=1200, output=json.dumps({"job_ids": ids})
                )
            raise raise_exc
        return _result(rc, json.dumps({"status": "submitted", "run_id": run_id, "job_ids": ids}))

    return run


def _submit(tmp_path, runner, *, stage="smoke", route="native_text_only", seed=1337):
    (tmp_path / "rt" / "registries").mkdir(parents=True, exist_ok=True)
    return submit_leg(
        stage=stage,
        route=route,
        seed=seed,
        deployment_code=str(tmp_path),
        source_commit="deadbeef",
        input_root="/perm",
        pooled_runtime_root="/pool",
        runtime=str(tmp_path / "rt"),
        runner=runner,
    )


def _round_trip(tmp_path, record: dict) -> tuple[set[str], int]:
    ledger = tmp_path / "rt" / "submissions.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return ledger_reservations(ledger)


def test_full_leg_delivery_submitted(tmp_path) -> None:
    runner = _runner(tmp_path / "rt", "qmsm_native_text_only_s1337", [_job(str(i), "smoke") for i in (11, 12, 13)])
    record = _submit(tmp_path, runner)
    assert record["status"] == "submitted"
    assert record["job_ids"] == ["11", "12", "13"]
    assert record["registry_job_total"] == 3


def test_env_construction_uses_os(tmp_path) -> None:
    runtime = tmp_path / "rt"
    captured: dict = {}
    base = _runner(runtime, "qmsm_native_text_only_s1337", [_job("21", "smoke"), _job("22", "smoke"), _job("23", "smoke")])

    def runner(command, **kwargs):
        captured["env"] = kwargs.get("env")
        return base(command, **kwargs)

    record = _submit(tmp_path, runner)
    assert record["status"] == "submitted"
    env = captured["env"]
    assert env["SYMMETRIC_MERGED_SOURCE_COMMIT"] == "deadbeef"
    assert env["QWEN38_ENV_ACTIVATE"].endswith("qwen38_fsdp_fastpath_20260921")
    assert env["QWEN3OMNI_ENV_ACTIVATE"].endswith("qwen3omni")


def test_round_trip_partial_delivery_reservation(tmp_path) -> None:
    runner = _runner(
        tmp_path / "rt",
        "qmsm_native_text_only_s1337",
        [_job(str(i), "cv") for i in (31, 32, 33)],
    )
    record = _submit(tmp_path, runner, stage="cv")
    assert record["status"] == "uncertain"
    assert record["expected"] == 15
    assert record["remaining"] == 12
    assert record["job_ids"] == ["31", "32", "33"]
    ids, reservation = _round_trip(tmp_path, record)
    assert ids == {"31", "32", "33"}
    assert reservation == 12


def test_round_trip_timeout_known_ids(tmp_path) -> None:
    runner = _runner(
        tmp_path / "rt",
        "qmsm_native_text_only_s1337",
        [_job(str(i), "cv") for i in (41, 42, 43, 44, 45)],
        raise_exc=subprocess.TimeoutExpired(cmd="submit", timeout=1200),
    )
    record = _submit(tmp_path, runner, stage="cv")
    assert record["status"] == "uncertain"
    assert record["remaining"] == 10
    assert record["job_ids"] == ["41", "42", "43", "44", "45"]
    ids, reservation = _round_trip(tmp_path, record)
    assert reservation == 10
    assert ids == {"41", "42", "43", "44", "45"}


def test_rc_nonzero_unparsed_fully_reserved(tmp_path) -> None:
    runner = _runner(tmp_path / "rt", "qmsm_native_text_only_s1337", [], rc=2)
    record = _submit(tmp_path, runner)
    assert record["status"] == "uncertain"
    assert record["remaining"] == 3
    ids, reservation = _round_trip(tmp_path, record)
    assert reservation == 3 and ids == set()


def test_duplicate_stage_ids_uncertain(tmp_path) -> None:
    runner = _runner(
        tmp_path / "rt",
        "qmsm_native_text_only_s1337",
        [_job("51", "smoke"), _job("51", "smoke"), _job("52", "smoke")],
    )
    record = _submit(tmp_path, runner)
    assert record["status"] == "uncertain"
    assert "duplicate stage ids" in record["reason"]
    assert record["remaining"] == 1
    ids, reservation = _round_trip(tmp_path, record)
    assert reservation == 1 and ids == {"51", "52"}


def test_cv_final_shared_registry_preserves_history(tmp_path) -> None:
    runtime = tmp_path / "rt"
    run_id = "qmsm_native_text_only_s1337"
    _append_registry(runtime, run_id, [_job(str(i), "smoke") for i in (61, 62, 63)])
    _append_registry(runtime, run_id, [_job(str(i), "cv") for i in range(100, 115)])
    runner = _runner(runtime, run_id, [_job(str(i), "final") for i in (200, 201, 202)])
    record = _submit(tmp_path, runner, stage="final")
    assert record["status"] == "submitted"
    assert record["job_ids"] == ["200", "201", "202"]
    assert record["registry_job_total"] == 21
    assert registry_job_ids(runtime / "registries") == {str(i) for i in list(range(61, 64)) + list(range(100, 115)) + [200, 201, 202]}


def test_malformed_registry_refuses(tmp_path) -> None:
    registries = tmp_path / "registries"
    registries.mkdir()
    (registries / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(AdmissionError):
        registry_job_ids(registries)


def test_planning_registry_skipped_and_numeric_collected(tmp_path) -> None:
    registries = tmp_path / "registries"
    (registries / "archive").mkdir(parents=True)
    (registries / "delivered.json").write_text(
        json.dumps({"run_id": "r", "submission_mode": "sbatch", "jobs": [{"job_id": "47072301"}]}),
        encoding="utf-8",
    )
    (registries / "archive" / "dry.json").write_text(
        json.dumps({"submission_mode": "dry_run", "jobs": [{"job_id": "dry_abc"}]}),
        encoding="utf-8",
    )
    assert registry_job_ids(registries) == {"47072301"}


def test_non_numeric_delivered_id_refuses(tmp_path) -> None:
    registries = tmp_path / "registries"
    registries.mkdir()
    (registries / "delivered.json").write_text(
        json.dumps({"submission_mode": "sbatch", "jobs": [{"job_id": "not-a-number"}]}),
        encoding="utf-8",
    )
    with pytest.raises(AdmissionError):
        registry_job_ids(registries)


def test_unparsed_success_output_uncertain(tmp_path) -> None:
    runner = lambda *args, **kwargs: _result(0, "no json here")
    record = _submit(tmp_path, runner)
    assert record["status"] == "uncertain"
    assert record["remaining"] == 3


def test_ledger_legacy_reservation_not_double_subtracted(tmp_path) -> None:
    ledger = tmp_path / "submissions.jsonl"
    ledger.write_text(
        json.dumps({"status": "uncertain", "reservation": 12, "job_ids": ["1", "2", "3"]}) + "\n",
        encoding="utf-8",
    )
    ids, reservation = ledger_reservations(ledger)
    assert ids == {"1", "2", "3"}
    assert reservation == 12


def test_round_trip_final_partial1_with_old_cv_stdout(tmp_path) -> None:
    runtime = tmp_path / "rt"
    run_id = "qmsm_native_text_only_s1337"
    cv_ids = [str(i) for i in range(400, 415)]
    _append_registry(runtime, run_id, [_job(value, "cv") for value in cv_ids])
    runner = _runner(runtime, run_id, [_job("500", "final")])
    record = _submit(tmp_path, runner, stage="final")
    assert record["status"] == "uncertain"
    assert record["expected"] == 3
    assert record["remaining"] == 2
    assert record["job_ids"] == ["500"]
    assert set(record["historical_job_ids"]) == set(cv_ids)
    ids, reservation = _round_trip(tmp_path, record)
    assert reservation == 2
    assert ids == {"500"}


def test_round_trip_final_rc1_only_old_cv_stdout(tmp_path) -> None:
    runtime = tmp_path / "rt"
    run_id = "qmsm_native_text_only_s1337"
    cv_ids = [str(i) for i in range(600, 615)]
    _append_registry(runtime, run_id, [_job(value, "cv") for value in cv_ids])
    runner = _runner(runtime, run_id, [], rc=1)
    record = _submit(tmp_path, runner, stage="final")
    assert record["status"] == "uncertain"
    assert record["expected"] == 3
    assert record["remaining"] == 3
    assert record["job_ids"] == []
    assert set(record["historical_job_ids"]) == set(cv_ids)
    ids, reservation = _round_trip(tmp_path, record)
    assert reservation == 3
    assert ids == set()


def test_own_lane_cap_enforced() -> None:
    leg_admission(65, 100, 15)  # exactly at the 80-job lane allocation
    with pytest.raises(AdmissionError):
        leg_admission(66, 100, 15)


def test_leg_selection_and_handled_skipping() -> None:
    planned = [
        ("native_text_only", 7),
        ("native_text_only", 1337),
        ("native_audio_only", 7),
    ]
    handled = {("native_text_only", 7)}
    to_process, skipped = select_legs(planned, handled)
    assert to_process == [("native_text_only", 1337), ("native_audio_only", 7)]
    assert skipped == [("native_text_only", 7)]

    to_process, skipped = select_legs(planned, set(), "native_audio_only:7,native_text_only:1337")
    assert to_process == [("native_audio_only", 7), ("native_text_only", 1337)]
    assert skipped == []

    to_process, skipped = select_legs(planned, handled, "native_text_only:7")
    assert to_process == []
    assert skipped == [("native_text_only", 7)]

    with pytest.raises(AdmissionError):
        select_legs(planned, set(), "not_a_route:7")
    with pytest.raises(AdmissionError):
        select_legs(planned, set(), "native_text_only:abc")


def test_user_threshold_enforced() -> None:
    leg_admission(0, 349, 3)
    with pytest.raises(AdmissionError):
        leg_admission(0, 350, 3)
