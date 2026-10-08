from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts.dispatch_androids_fixed_merged import (
    AdmissionError,
    leg_admission,
    ledger_reservations,
    registry_job_ids,
    submit_leg,
)


def _result(returncode: int, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _submission_stdout(job_ids: list[str], run_id: str = "qmsm_native_text_only_s1337") -> str:
    return json.dumps({"status": "submitted", "run_id": run_id, "job_ids": job_ids})


def _submit(tmp_path, runner, *, stage="smoke", route="native_text_only", seed=1337):
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


def test_full_leg_delivery_submitted(tmp_path) -> None:
    runner = lambda *args, **kwargs: _result(0, _submission_stdout(["11", "12", "13"]))
    record = _submit(tmp_path, runner)
    assert record["status"] == "submitted"
    assert record["job_ids"] == ["11", "12", "13"]
    assert record["registry"].endswith("registries/qmsm_native_text_only_s1337.json")


def test_env_construction_uses_os(tmp_path) -> None:
    captured: dict = {}

    def runner(command, **kwargs):
        captured["env"] = kwargs.get("env")
        return _result(0, _submission_stdout(["21", "22", "23"]))

    record = _submit(tmp_path, runner)
    assert record["status"] == "submitted"
    env = captured["env"]
    assert env["SYMMETRIC_MERGED_SOURCE_COMMIT"] == "deadbeef"
    assert env["QWEN38_ENV_ACTIVATE"].endswith("qwen38_fsdp_fastpath_20260921")
    assert env["QWEN3OMNI_ENV_ACTIVATE"].endswith("qwen3omni")


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


def test_partial_delivery_uncertain_remainder(tmp_path) -> None:
    runner = lambda *args, **kwargs: _result(0, _submission_stdout(["31", "32"]))
    record = _submit(tmp_path, runner)
    assert record["status"] == "uncertain"
    assert record["job_ids"] == ["31", "32"]
    assert record["reservation"] == 1
    assert "partial delivery 2/3" in record["reason"]


def test_timeout_full_reservation(tmp_path) -> None:
    def runner(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="submit", timeout=1200)

    record = _submit(tmp_path, runner, stage="cv")
    assert record["status"] == "uncertain"
    assert record["reservation"] == 15


def test_unexpected_exception_full_reservation(tmp_path) -> None:
    def runner(*args, **kwargs):
        raise RuntimeError("boom")

    record = _submit(tmp_path, runner, stage="final", route="english_audio_text", seed=2024)
    assert record["status"] == "uncertain"
    assert record["reservation"] == 3
    assert "unexpected runner error" in record["reason"]


def test_unparsed_success_output_uncertain(tmp_path) -> None:
    runner = lambda *args, **kwargs: _result(0, "no json here")
    record = _submit(tmp_path, runner)
    assert record["status"] == "uncertain"
    assert record["reservation"] == 3


def test_ledger_remainder_reserved(tmp_path) -> None:
    ledger = tmp_path / "submissions.jsonl"
    ledger.write_text(
        json.dumps({"status": "uncertain", "reservation": 15, "job_ids": ["1", "2", "3"]}) + "\n",
        encoding="utf-8",
    )
    ids, reservation = ledger_reservations(ledger)
    assert ids == {"1", "2", "3"}
    assert reservation == 12


def test_own_lane_cap_enforced() -> None:
    leg_admission(65, 100, 15)  # exactly at the 80-job lane allocation
    with pytest.raises(AdmissionError):
        leg_admission(66, 100, 15)


def test_user_threshold_enforced() -> None:
    leg_admission(0, 349, 3)
    with pytest.raises(AdmissionError):
        leg_admission(0, 350, 3)
