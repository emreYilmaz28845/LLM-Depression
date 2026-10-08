"""Fail-closed admission tests for the Worker 4 production dispatcher.

These tests pin the binding semantics: scheduler/SSH failures never count as
"no jobs", own nonterminal job accounting includes ledger and sidecar IDs with
conservative handling of uncertain submissions, planned waves cannot exceed
the lane allocation, the per-fit condition is exactly own+2 <= 80 and
user < 350 (never processed-count vs remaining capacity), and every uncertain
submit outcome is preserved.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import qwen3_legacy_prompt_dispatch as dispatch

ROOT = Path(__file__).resolve().parents[1]


def fake_runner(returncode: int = 0, stdout: str = "", stderr: str = ""):
    calls: list[list[str]] = []

    def runner(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    return runner, calls


def fits(count: int) -> list[dict]:
    return [
        {
            "key": f"route|s7|f{index}",
            "run_name": f"run_{index}",
            "config": "configs/main/x.yaml",
            "dataset": "daic",
            "modality": "text_only",
            "seed": 7,
            "fold": 0,
        }
        for index in range(count)
    ]


def test_user_queue_parses_delimited_counts() -> None:
    runner, calls = fake_runner(stdout="47071730|PENDING\n47071733|RUNNING\n")
    assert dispatch.user_queue_count(runner) == 2
    assert "squeue" in calls[0][-1]


def test_ssh_failure_refuses_and_is_not_zero_jobs() -> None:
    runner, _ = fake_runner(returncode=255, stderr="ssh: connect failed")
    with pytest.raises(dispatch.AdmissionError):
        dispatch.user_queue_count(runner)


def test_unparseable_scheduler_output_refuses() -> None:
    runner, _ = fake_runner(stdout="not a scheduler line\n")
    with pytest.raises(dispatch.AdmissionError):
        dispatch.user_queue_count(runner)


def test_oversized_wave_is_rejected() -> None:
    with pytest.raises(dispatch.AdmissionError):
        dispatch.validate_wave_size(41)
    assert dispatch.validate_wave_size(40) is None


def test_per_fit_condition_and_user_stop_threshold() -> None:
    assert dispatch.per_fit_admission(78, 10) is None
    with pytest.raises(dispatch.AdmissionError):
        dispatch.per_fit_admission(79, 10)
    with pytest.raises(dispatch.AdmissionError):
        dispatch.per_fit_admission(80, 10)
    with pytest.raises(dispatch.AdmissionError):
        dispatch.per_fit_admission(0, 350)


def test_wave_simulation_from_own0_submits_forty_fits() -> None:
    state = {"own": 0}

    def reconcile():
        return state["own"], 0

    def submit(fit):
        state["own"] += 2
        return {"status": "submitted", "job_ids": {"train": "1", "best_eval": "2"}}

    assert dispatch.run_wave(fits(100), 40, reconcile=reconcile, submit=submit) == 40


def test_wave_simulation_from_own40_submits_twenty_fits() -> None:
    state = {"own": 40}

    def reconcile():
        return state["own"], 0

    def submit(fit):
        state["own"] += 2
        return {"status": "submitted", "job_ids": {"train": "1", "best_eval": "2"}}

    assert dispatch.run_wave(fits(100), 40, reconcile=reconcile, submit=submit) == 20


def test_wave_simulation_from_own79_submits_zero_fits() -> None:
    def reconcile():
        return 79, 0

    def submit(fit):
        raise AssertionError("must not submit without headroom")

    assert dispatch.run_wave(fits(10), 40, reconcile=reconcile, submit=submit) == 0


def test_uncertain_and_unresolved_ids_count_conservatively() -> None:
    assert dispatch.own_nonterminal_count([], {}, uncertain_records=1) == 2
    assert dispatch.own_nonterminal_count(["99999999"], {}) == 1
    terminal = {"1": "COMPLETED", "2": "FAILED", "3": "CANCELLED", "4": "TIMEOUT"}
    assert dispatch.own_nonterminal_count(list(terminal), terminal) == 0
    running = {"1": "PENDING", "2": "RUNNING", "3": "COMPLETED"}
    assert dispatch.own_nonterminal_count(list(running), running) == 2


def test_own_ids_from_ledger_and_fold_sidecars(tmp_path: Path) -> None:
    ledger = tmp_path / "submissions.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "key": "daic_text_only|s7|f0",
                "status": "submitted",
                "job_ids": {"train": "47073846", "best_eval": "47073847"},
            }
        )
        + "\n"
        + json.dumps({"key": "daic_text_only|s2024|f0", "status": "uncertain"})
        + "\n",
        encoding="utf-8",
    )
    run_root = tmp_path / "output_model/qwen3_legacy_prompt_20261008"
    fold = run_root / "text_only/daic/smoke/fold_0"
    fold.mkdir(parents=True)
    (fold / "jobs.jsonl").write_text(
        json.dumps({"job_key": "train", "event_type": "SUBMITTED", "slurm_job_id": "47071730"})
        + "\n",
        encoding="utf-8",
    )
    ids, uncertain, unknown = dispatch.own_job_ids(ledger, run_root)
    assert "47073846" in ids and "47073847" in ids and "47071730" in ids
    assert uncertain == 1 and unknown == []


def test_oversized_three_id_uncertain_counts_at_least_three_without_sidecars(
    tmp_path: Path,
) -> None:
    """Known IDs keep ownership: 3 delivered ids + full 2 reservation (>= 3)."""
    record, ids, uncertain, own, _ = _roundtrip(
        tmp_path,
        _submit_stdout(jobs="{'train': '4701', 'best_eval': '4702', 'extra': '4703'}"),
    )
    assert record["status"] == "uncertain"
    assert ids == ["4701", "4702", "4703"]
    assert uncertain == 1 and own == 5 and own >= 3


def test_unknown_non_numeric_id_stops_admission(tmp_path: Path) -> None:
    record, ids, uncertain, own, _ = _roundtrip(
        tmp_path, _submit_stdout(jobs="{'train': 'abc', 'best_eval': '4702'}")
    )
    assert record["status"] == "uncertain"
    assert ids == ["4702"] and uncertain == 1 and own == 3
    ledger = tmp_path / "submissions.jsonl"
    run_root = tmp_path / "empty"
    job_ids, uncertain2, unknown = dispatch.own_job_ids(ledger, run_root)
    with pytest.raises(dispatch.AdmissionError):
        dispatch.require_reconciled(job_ids, {job_id: "RUNNING" for job_id in job_ids}, unknown)


def test_unresolved_numeric_id_stops_admission() -> None:
    with pytest.raises(dispatch.AdmissionError):
        dispatch.require_reconciled(["99999999"], {}, [])
    # Resolved ids pass.
    assert dispatch.require_reconciled(["4701"], {"4701": "COMPLETED"}, []) is None


def test_query_job_states_uses_full_queue_then_sacct() -> None:
    def runner(args, **kwargs):
        command = args[-1]
        assert "squeue -j" not in command  # old-ID queue queries are forbidden
        if "squeue -u" in command:
            return SimpleNamespace(returncode=0, stdout="100|RUNNING\n101|PENDING\n", stderr="")
        if "sacct" in command:
            return SimpleNamespace(returncode=0, stdout="200|COMPLETED\n", stderr="")
        raise AssertionError(command)

    states, user = dispatch.query_job_states(["100", "200", "300"], runner)
    assert states == {"100": "RUNNING", "200": "COMPLETED"}
    assert user == 2


def test_sacct_failure_refuses() -> None:
    def runner(args, **kwargs):
        command = args[-1]
        if "squeue -u" in command:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="sacct: error")

    with pytest.raises(dispatch.AdmissionError):
        dispatch.query_job_states(["99999999"], runner)


def test_submit_fit_timeout_preserves_partial_delivery(tmp_path: Path) -> None:
    def runner(command, **kwargs):
        raise subprocess.TimeoutExpired(
            command,
            1200,
            output='garbage before {"attempt_id": "20261008T000000Z-x-abc-1234"} after',
            stderr="partial stderr",
        )

    fit = fits(1)[0]
    record = dispatch.submit_fit(fit, runner)
    assert record["status"] == "uncertain"
    assert record["reason"] == "timeout"
    assert record["attempt_id"] == "20261008T000000Z-x-abc-1234"
    assert record["job_ids"] == {}
    assert "partial stderr" in record["tail"]


def test_submit_fit_oserror_and_unexpected_error_are_uncertain() -> None:
    def os_runner(command, **kwargs):
        raise OSError("ssh exploded")

    def weird_runner(command, **kwargs):
        raise ValueError("odd failure")

    fit = fits(1)[0]
    assert dispatch.submit_fit(fit, os_runner)["status"] == "uncertain"
    assert dispatch.submit_fit(fit, weird_runner)["status"] == "uncertain"


def test_submit_fit_nonzero_rc_preserves_ids_and_is_uncertain() -> None:
    def runner(command, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout='{"attempt_id": "20261008T000000Z-x-abc-9999"}\n'
            "submitted jobs: {'train': '4701', 'best_eval': '4702'}\n",
            stderr="late failure",
        )

    fit = fits(1)[0]
    record = dispatch.submit_fit(fit, runner)
    assert record["status"] == "uncertain"
    assert record["attempt_id"] == "20261008T000000Z-x-abc-9999"
    assert record["job_ids"] == {"train": "4701", "best_eval": "4702"}


def test_cli_rejects_oversized_wave_before_any_network_call() -> None:
    result = subprocess.run(
        [sys.executable, "tools/qwen3_legacy_prompt_dispatch.py", "--max-fits", "41"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 2
    assert "REFUSED" in result.stdout


def test_head_chain_headroom_uses_the_same_lane_budget() -> None:
    assert dispatch.head_chain_headroom(0) == 40
    assert dispatch.head_chain_headroom(78) == 1
    assert dispatch.head_chain_headroom(79) == 0
    assert dispatch.head_chain_headroom(80) == 0


def _roundtrip(tmp_path: Path, stdout: str, stderr: str = "", rc: int = 0):
    """submit -> append -> reconcile, the exact production path."""
    ledger = tmp_path / "submissions.jsonl"
    run_root = tmp_path / "empty"

    def runner(command, **kwargs):
        return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)

    record = dispatch.submit_fit(fits(1)[0], runner)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    ids, uncertain, _unknown = dispatch.own_job_ids(ledger, run_root)
    states = {job_id: "RUNNING" for job_id in ids}
    own = dispatch.own_nonterminal_count(ids, states, uncertain)
    return record, ids, uncertain, own, dispatch.settled_keys(ledger)


DEFAULT_JOBS = "{'train': '4701', 'best_eval': '4702'}"


def _submit_stdout(attempt_id: str = "20261008T000000Z-x-abc-1234", jobs: str = "") -> str:
    delivery = jobs or DEFAULT_JOBS
    return f'{{"attempt_id": "{attempt_id}"}}\nsubmitted jobs: {delivery}\n'


def test_roundtrip_valid_pair_is_submitted_and_counted(tmp_path: Path) -> None:
    record, ids, uncertain, own, settled = _roundtrip(tmp_path, _submit_stdout())
    assert record["status"] == "submitted"
    assert ids == ["4701", "4702"] and uncertain == 0 and own == 2
    assert settled == {"route|s7|f0"}


def test_roundtrip_single_id_stays_uncertain_with_reservation(tmp_path: Path) -> None:
    record, ids, uncertain, own, settled = _roundtrip(
        tmp_path, _submit_stdout(jobs="{'train': '4701'}")
    )
    assert record["status"] == "uncertain"
    assert "invalid delivered job ids" in record["reason"]
    assert ids == ["4701"] and uncertain == 1 and own == 3
    assert settled == {"route|s7|f0"}


def test_roundtrip_duplicate_ids_stay_uncertain(tmp_path: Path) -> None:
    record, ids, uncertain, own, _ = _roundtrip(
        tmp_path, _submit_stdout(jobs="{'train': '4701', 'best_eval': '4701'}")
    )
    assert record["status"] == "uncertain"
    assert ids == ["4701"] and uncertain == 1 and own == 3


def test_roundtrip_malformed_ids_stay_uncertain(tmp_path: Path) -> None:
    record, ids, uncertain, own, _ = _roundtrip(
        tmp_path, _submit_stdout(jobs="{'train': 'abc', 'best_eval': '4702'}")
    )
    assert record["status"] == "uncertain"
    assert ids == ["4702"] and uncertain == 1 and own == 3


def test_roundtrip_extra_or_missing_keys_stay_uncertain(tmp_path: Path) -> None:
    jobs_cases = (
        ("{'train': '4701', 'best_eval': '4702', 'extra': '4703'}", 5),
        ("{'best_eval': '4702'}", 3),
    )
    for index, (jobs, expected_own) in enumerate(jobs_cases):
        case = tmp_path / f"case_{index}"
        case.mkdir()
        record, _, uncertain, own, _ = _roundtrip(case, _submit_stdout(jobs=jobs))
        assert record["status"] == "uncertain", jobs
        assert uncertain == 1 and own == expected_own


def test_roundtrip_missing_attempt_id_stays_uncertain(tmp_path: Path) -> None:
    stdout = "submitted jobs: {'train': '4701', 'best_eval': '4702'}\n"
    record, ids, uncertain, own, _ = _roundtrip(tmp_path, stdout)
    assert record["status"] == "uncertain"
    assert ids == ["4701", "4702"] and uncertain == 1 and own == 4


def test_wave_stops_on_uncertain_without_retry() -> None:
    calls = {"count": 0}

    def reconcile():
        return 0, 0

    def submit(fit):
        calls["count"] += 1
        return {"status": "uncertain", "job_ids": {}, "reason": "invalid"}

    processed = dispatch.run_wave(fits(10), 40, reconcile=reconcile, submit=submit)
    assert processed == 1 and calls["count"] == 1
