"""Fail-closed admission tests for the Worker 4 production dispatcher.

These tests pin the binding semantics: scheduler/SSH failures never count as
"no jobs", own nonterminal job accounting includes ledger and sidecar IDs with
conservative handling of uncertain submissions, planned waves cannot exceed
the lane allocation, and refill capacity is limited by actual headroom.
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
        dispatch.admission(own_nonterminal=0, user_queue=0, max_fits=41)
    assert dispatch.admission(own_nonterminal=0, user_queue=0, max_fits=40) == 40


def test_user_stop_threshold_and_headroom() -> None:
    with pytest.raises(dispatch.AdmissionError):
        dispatch.admission(own_nonterminal=0, user_queue=350, max_fits=1)
    assert dispatch.admission(own_nonterminal=78, user_queue=10, max_fits=40) == 1
    with pytest.raises(dispatch.AdmissionError):
        dispatch.admission(own_nonterminal=79, user_queue=10, max_fits=40)
    with pytest.raises(dispatch.AdmissionError):
        dispatch.admission(own_nonterminal=80, user_queue=10, max_fits=40)


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
    ids, uncertain = dispatch.own_job_ids(ledger, run_root)
    assert "47073846" in ids and "47073847" in ids and "47071730" in ids
    assert uncertain == 1


def test_refill_simulation_limits_by_terminal_reconciliation(tmp_path: Path) -> None:
    ledger = tmp_path / "submissions.jsonl"
    fit_keys = [f"route|s7|f{n}" for n in range(40)]
    records = []
    for index, key in enumerate(fit_keys):
        records.append(
            {
                "key": key,
                "status": "submitted",
                "job_ids": {"train": str(47070000 + 2 * index), "best_eval": str(47070001 + 2 * index)},
            }
        )
    ledger.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    ids, uncertain = dispatch.own_job_ids(ledger, tmp_path / "empty")
    assert len(ids) == 80 and uncertain == 0
    # All still nonterminal: no headroom.
    states = {job_id: "RUNNING" for job_id in ids}
    own = dispatch.own_nonterminal_count(ids, states, uncertain)
    with pytest.raises(dispatch.AdmissionError):
        dispatch.admission(own, user_queue=5, max_fits=1)
    # Half terminal: refill limited to the exact remaining headroom.
    states = {job_id: ("COMPLETED" if int(job_id) % 4 < 2 else "RUNNING") for job_id in ids}
    own = dispatch.own_nonterminal_count(ids, states, uncertain)
    assert own == 40
    assert dispatch.admission(own, user_queue=5, max_fits=40) == 20


def test_submit_fit_records_uncertain_without_automatic_retry(tmp_path: Path) -> None:
    fit = {
        "key": "daic_text_only|s7|f0",
        "run_name": "q3lp_daic_text_only_s7_f0",
        "config": "configs/main/daic_text_only_harmonized_selmacrof1_likelihood_v1_legacyprompt_v1.yaml",
        "dataset": "daic",
        "modality": "text_only",
        "seed": 7,
        "fold": 0,
    }
    runner, _ = fake_runner(returncode=1, stdout="", stderr="boom")
    record = dispatch.submit_fit(fit, runner)
    assert record["status"] == "uncertain" and record["job_ids"] == {}
    assert dispatch.settled_keys  # callable exists for append-only settlement


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
