"""Tests for the fail-closed window-cap admission and guarded submission tool."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import qwen3_window_cap_admission as admission  # noqa: E402


def _paths(tmp_path: Path) -> dict:
    return {
        "ledger_path": tmp_path / "ledger.json",
        "state_path": tmp_path / "state.json",
        "head_registry": tmp_path / "head_submissions.jsonl",
        "reservations_path": tmp_path / "reservations.jsonl",
        "local_run_root": tmp_path / "output_model" / "campaign",
        "scheduler": "user@scheduler",
        "user": "user",
    }


def _write_sources(tmp_path: Path, *, ledger: dict | None = None, state: dict | None = None) -> dict:
    paths = _paths(tmp_path)
    paths["ledger_path"].write_text(
        json.dumps(ledger if ledger is not None else {"jobs": []}), encoding="utf-8"
    )
    paths["state_path"].write_text(
        json.dumps(
            state
            if state is not None
            else {"job_inventory": [], "deployments": {}}
        ),
        encoding="utf-8",
    )
    return paths


def _fake_ssh(queue: str = "", sacct: str = ""):
    def fake(command: str, *, scheduler: str, user: str) -> str:
        if "squeue -u" in command:
            return queue
        if "sacct" in command:
            return sacct
        raise AssertionError(f"unexpected ssh command: {command}")

    return fake


def test_missing_ledger_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    paths["state_path"].write_text(json.dumps({"jobs": []}), encoding="utf-8")
    monkeypatch.setattr(admission, "run_ssh", _fake_ssh())
    with pytest.raises(admission.AdmissionError, match="submission ledger is missing"):
        admission.snapshot(**paths)


def test_malformed_ledger_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    paths["ledger_path"].write_text("{not json", encoding="utf-8")
    paths["state_path"].write_text(json.dumps({"jobs": []}), encoding="utf-8")
    monkeypatch.setattr(admission, "run_ssh", _fake_ssh())
    with pytest.raises(admission.AdmissionError, match="unreadable or malformed"):
        admission.snapshot(**paths)


def test_missing_state_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    paths["ledger_path"].write_text(json.dumps({"jobs": []}), encoding="utf-8")
    monkeypatch.setattr(admission, "run_ssh", _fake_ssh())
    with pytest.raises(admission.AdmissionError, match="lane state is missing"):
        admission.snapshot(**paths)


def test_malformed_head_registry_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_sources(tmp_path)
    paths["head_registry"].write_text("{oops\n", encoding="utf-8")
    monkeypatch.setattr(admission, "run_ssh", _fake_ssh())
    with pytest.raises(admission.AdmissionError, match="head registry malformed"):
        admission.snapshot(**paths)


def test_ssh_failure_is_not_an_empty_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    class Fake:
        returncode = 255
        stdout = ""
        stderr = "ssh: connect to host scheduler port 22: Connection timed out"

    monkeypatch.setattr(admission.subprocess, "run", lambda *a, **k: Fake())
    with pytest.raises(admission.AdmissionError, match="scheduler query failed"):
        admission.run_ssh("squeue -u user", scheduler="user@scheduler", user="user")


def test_malformed_scheduler_output_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_sources(tmp_path)
    monkeypatch.setattr(admission, "run_ssh", _fake_ssh(queue="garbage-line\n"))
    with pytest.raises(admission.AdmissionError, match="unparseable scheduler line"):
        admission.snapshot(**paths)


def test_unresolved_own_ids_block_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_sources(
        tmp_path,
        state={"job_inventory": [{"attempt_id": "a1", "train_job": "111", "eval_job": "112"}],
               "deployments": {}},
    )
    monkeypatch.setattr(admission, "run_ssh", _fake_ssh(queue="", sacct=""))
    snap = admission.snapshot(**paths)
    assert snap["own_unresolved_jobs"] == ["111", "112"]
    assert snap["own_nonterminal_count"] == 2
    with pytest.raises(admission.AdmissionError, match="unresolved"):
        admission.admission_check(snap, 2)


def test_partial_delivery_reserves_conservatively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_sources(tmp_path)
    monkeypatch.setattr(admission, "run_ssh", _fake_ssh(queue="", sacct=""))
    command = [
        sys.executable,
        "-c",
        "print(\"submitted jobs: {'train': '9001'}); raise SystemExit(2)",
    ]
    with pytest.raises(admission.AdmissionError, match="partial delivery"):
        admission.run_submit(
            command,
            jobs_this_submit=2,
            wave_fits=1,
            kind="fit",
            attempt_hint=None,
            submit_timeout=30,
            **paths,
        )
    entries = admission.read_reservations(paths["reservations_path"])
    assert [entry["event"] for entry in entries] == ["reserved", "partial"]
    snap = admission.snapshot(**paths)
    assert snap["reserved_jobs"] == 2
    assert snap["own_nonterminal_count"] >= 2
    assert snap["active_reservations"]


def test_timeout_reserves_conservatively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_sources(tmp_path)
    monkeypatch.setattr(admission, "run_ssh", _fake_ssh(queue="", sacct=""))
    command = [sys.executable, "-c", "import time; time.sleep(5)"]
    with pytest.raises(admission.AdmissionError, match="timed out"):
        admission.run_submit(
            command,
            jobs_this_submit=2,
            wave_fits=1,
            kind="fit",
            attempt_hint=None,
            submit_timeout=1,
            **paths,
        )
    entries = admission.read_reservations(paths["reservations_path"])
    assert [entry["event"] for entry in entries] == ["reserved", "uncertain"]
    assert admission.snapshot(**paths)["reserved_jobs"] == 2


def test_full_delivery_records_ids_and_stops_reserving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_sources(tmp_path)
    monkeypatch.setattr(
        admission, "run_ssh", _fake_ssh(queue="", sacct="9001|COMPLETED\n9002|COMPLETED\n")
    )
    command = [
        sys.executable,
        "-c",
        "print(\"submitted jobs: {'train': '9001', 'best_eval': '9002'}\")",
    ]
    rc = admission.run_submit(
        command,
        jobs_this_submit=2,
        wave_fits=1,
        kind="fit",
        attempt_hint="a1",
        submit_timeout=30,
        **paths,
    )
    assert rc == 0
    entries = admission.read_reservations(paths["reservations_path"])
    assert [entry["event"] for entry in entries] == ["reserved", "delivered"]
    snap = admission.snapshot(**paths)
    assert snap["reserved_jobs"] == 0
    assert snap["own_nonterminal_count"] == 0
    assert snap["states"] == {"9001": "COMPLETED", "9002": "COMPLETED"}


def test_refused_admission_never_runs_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_sources(tmp_path)
    marker = tmp_path / "ran.marker"
    queue = "".join(f"{job}|PENDING\n" for job in range(1000, 1350))
    monkeypatch.setattr(admission, "run_ssh", _fake_ssh(queue=queue, sacct=""))
    command = [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('x')"]
    with pytest.raises(admission.AdmissionError, match="at or above"):
        admission.run_submit(
            command,
            jobs_this_submit=2,
            wave_fits=1,
            kind="fit",
            attempt_hint=None,
            submit_timeout=30,
            **paths,
        )
    assert not marker.exists()
    assert not paths["reservations_path"].exists()


def test_reconcile_clears_terminal_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_sources(tmp_path)
    admission.append_reservation(
        paths["reservations_path"],
        {
            "event": "reserved",
            "reservation_id": "r1",
            "at_utc": "2026-10-08T00:00:00Z",
            "kind": "fit",
            "jobs_reserved": 2,
        },
    )
    admission.append_reservation(
        paths["reservations_path"],
        {
            "event": "partial",
            "reservation_id": "r1",
            "at_utc": "2026-10-08T00:00:01Z",
            "job_ids": {"train": "7001", "best_eval": "7002"},
            "error": "rc=2",
        },
    )
    monkeypatch.setattr(
        admission, "run_ssh", _fake_ssh(queue="", sacct="7001|FAILED\n7002|CANCELLED\n")
    )
    results = admission.reconcile(
        paths["reservations_path"],
        reservation_id=None,
        manual_note=None,
        scheduler=paths["scheduler"],
        user=paths["user"],
    )
    assert results[0]["resolution"] == "terminal"
    snap = admission.snapshot(**paths)
    assert snap["reserved_jobs"] == 0
    assert snap["active_reservations"] == {}


def test_reconcile_keeps_uncertain_without_ids_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_sources(tmp_path)
    admission.append_reservation(
        paths["reservations_path"],
        {
            "event": "reserved",
            "reservation_id": "r2",
            "at_utc": "2026-10-08T00:00:00Z",
            "kind": "fit",
            "jobs_reserved": 2,
        },
    )
    admission.append_reservation(
        paths["reservations_path"],
        {"event": "uncertain", "reservation_id": "r2", "at_utc": "2026-10-08T00:00:02Z"},
    )
    monkeypatch.setattr(admission, "run_ssh", _fake_ssh(queue="", sacct=""))
    results = admission.reconcile(
        paths["reservations_path"],
        reservation_id=None,
        manual_note=None,
        scheduler=paths["scheduler"],
        user=paths["user"],
    )
    assert results[0]["resolution"] == "unresolved"
    assert admission.snapshot(**paths)["reserved_jobs"] == 2


def test_oversized_wave_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_sources(tmp_path)
    monkeypatch.setattr(admission, "run_ssh", _fake_ssh(queue="", sacct=""))
    snap = admission.snapshot(**paths)
    with pytest.raises(admission.AdmissionError, match="no lane headroom"):
        admission.admission_check(snap, 2, wave_fits=41)


def test_cli_run_submit_is_integrated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_sources(tmp_path)
    monkeypatch.setattr(
        admission, "run_ssh", _fake_ssh(queue="", sacct="9001|COMPLETED\n9002|COMPLETED\n")
    )
    rc = admission.main(
        [
            "--ledger",
            str(paths["ledger_path"]),
            "--state",
            str(paths["state_path"]),
            "--head-registry",
            str(paths["head_registry"]),
            "--reservations",
            str(paths["reservations_path"]),
            "--local-run-root",
            str(paths["local_run_root"]),
            "--scheduler",
            paths["scheduler"],
            "--user",
            paths["user"],
            "run-submit",
            "--jobs-this-submit",
            "2",
            "--kind",
            "fit",
            "--attempt-hint",
            "cli-a1",
            "--",
            sys.executable,
            "-c",
            "print(\"submitted jobs: {'train': '9001', 'best_eval': '9002'}\")",
        ]
    )
    assert rc == 0
    entries = admission.read_reservations(paths["reservations_path"])
    assert entries[0]["event"] == "reserved"
    assert entries[0]["attempt_hint"] == "cli-a1"
    assert entries[1]["event"] == "delivered"


def test_parse_delivered_job_ids_handles_head_output() -> None:
    output = "=== JOB x ===\nEXTRACT_ID=5001\nCLASSIFIER_ID=5002\n"
    assert admission.parse_delivered_job_ids(output) == {
        "extract_id:5001": "5001",
        "classifier_id:5002": "5002",
    }
