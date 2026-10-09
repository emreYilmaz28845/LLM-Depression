"""Focused tests for the window15 admission-gated dispatch driver.

Covers the binding admission semantics: raw SSH/scheduler return codes,
authoritative own-ID reconciliation, conservative unresolved/partial
delivery counting, and the per-leg own+leg<=80 / user<350 condition.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

from tools.qwen3_window15_dispatch import (  # noqa: E402
    AdmissionError,
    _extract_delivery,
    build_matrix,
    lane_headroom,
    own_job_ids,
    own_nonterminal_count,
    parse_delimited,
    per_fit_admission,
    query_job_states,
    reconcile_evidence,
    run_ssh,
    run_wave,
    submit_fit,
    unresolved_delivery_reservations,
    validate_fit,
    validate_wave_size,
)


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["fake"], returncode=returncode, stdout=stdout, stderr=stderr)


class _FakeSsh:
    def __init__(self, queue: str = "", sacct: str = "", quota: str = "", returncode: int = 0, stderr: str = ""):
        self.queue = queue
        self.sacct = sacct
        self.quota = quota
        self.returncode = returncode
        self.stderr = stderr

    def __call__(self, argv, **kwargs):
        command = argv[-1]
        if "squeue" in command:
            return _completed(self.returncode, self.queue, self.stderr)
        if "bsc_quota" in command:
            return _completed(self.returncode, self.quota, self.stderr)
        return _completed(self.returncode, self.sacct, self.stderr)


def test_run_ssh_refuses_nonzero_returncode() -> None:
    with pytest.raises(AdmissionError, match="scheduler query failed"):
        run_ssh("squeue -u ozu647717", _FakeSsh(returncode=255, stderr="ssh: connect refused"))
    assert run_ssh("squeue -u ozu647717", _FakeSsh(queue="1|RUNNING\n")) == "1|RUNNING\n"


def test_parse_delimited_refuses_unparseable() -> None:
    assert parse_delimited("47072528|FAILED\n47072529|CANCELLED\n") == {
        "47072528": "FAILED",
        "47072529": "CANCELLED",
    }
    assert parse_delimited("") == {}
    with pytest.raises(AdmissionError, match="unparseable"):
        parse_delimited("this is not a scheduler line")
    with pytest.raises(AdmissionError, match="unparseable"):
        parse_delimited("47072528|")


def test_user_queue_failure_is_never_zero() -> None:
    from tools.qwen3_window15_dispatch import user_queue

    with pytest.raises(AdmissionError):
        user_queue(_FakeSsh(returncode=1, stderr="boom"))


def test_own_job_ids_union_and_uncertain(tmp_path: Path) -> None:
    ledger = tmp_path / "submissions.jsonl"
    ledger.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"key": "a", "status": "submitted", "job_ids": {"train": "100", "best_eval": "101"}},
                {"key": "b", "status": "uncertain", "job_ids": {"train": "102"}, "reason": "rc=1"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    exec_ledger = tmp_path / "state.json"
    exec_ledger.write_text(
        json.dumps(
            {
                "jobs": [
                    {"attempt_id": "x-other", "deployment_id": "other-lane", "slurm_job_id": "103"},
                    {"attempt_id": "y-q3w15_marker", "deployment_id": "unknown-dep", "slurm_job_id": "104"},
                    {"attempt_id": "z", "deployment_id": "feat-qwen3-window15-20261008-abc", "slurm_job_id": "105"},
                    {"attempt_id": "unrelated", "deployment_id": "unrelated", "slurm_job_id": "999"},
                ]
            }
        ),
        encoding="utf-8",
    )
    exp_submit = tmp_path / "exp_submit"
    (exp_submit / "attempt1").mkdir(parents=True)
    (exp_submit / "attempt1" / "submit_output.log").write_text(
        '"attempt_id": "attempt1"\nsubmitted jobs: {\'train\': \'106\', \'best_eval\': \'107\'}\n',
        encoding="utf-8",
    )
    run_root = tmp_path / "output_model"
    sidecar_dir = run_root / "audio_only" / "daic" / "run" / "fold_0"
    sidecar_dir.mkdir(parents=True)
    (sidecar_dir / "jobs.jsonl").write_text(
        json.dumps({"slurm_job_id": "108", "event_type": "SUBMITTED"}) + "\n", encoding="utf-8"
    )
    ids, uncertain = own_job_ids(ledger, run_root, exec_ledger, exp_submit)
    # Lane prefix and the q3w15 attempt marker both select own jobs; the
    # unrelated deployment and the unrelated attempt stay out.
    assert ids == ["100", "101", "102", "104", "105", "106", "107", "108"]
    assert uncertain == 1


def test_unresolved_delivery_reservations(tmp_path: Path) -> None:
    ledger = tmp_path / "submissions.jsonl"
    ledger.write_text(
        json.dumps({"key": "a", "status": "submitted", "attempt_id": "settled", "job_ids": {"train": "1", "best_eval": "2"}})
        + "\n",
        encoding="utf-8",
    )
    exec_ledger = tmp_path / "state.json"
    exec_ledger.write_text(
        json.dumps(
            {
                "jobs": [
                    {"attempt_id": "in-exec", "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "train", "slurm_job_id": "3"},
                    {"attempt_id": "in-exec", "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "best_eval", "slurm_job_id": "4"},
                ]
            }
        ),
        encoding="utf-8",
    )
    exp_submit = tmp_path / "exp_submit"
    for name, attempt in (("a", "settled"), ("b", "in-exec"), ("c", "partial")):
        (exp_submit / name).mkdir(parents=True)
        (exp_submit / name / "submit_output.log").write_text(
            f'"attempt_id": "{attempt}"\nsubmitted jobs: {{}}\n', encoding="utf-8"
        )
    assert unresolved_delivery_reservations(ledger, exec_ledger, exp_submit) == 1


def test_query_job_states_queue_then_sacct(tmp_path: Path) -> None:
    runner = _FakeSsh(
        queue="100|RUNNING\n999|PENDING\n",
        sacct="101|COMPLETED\n102|FAILED\n",
    )
    states, user = query_job_states(["100", "101", "102", "103"], runner)
    assert states == {"100": "RUNNING", "101": "COMPLETED", "102": "FAILED"}
    assert user == 2  # the full user queue, not just own jobs
    assert "103" not in states  # unresolved: counts as nonterminal later


def test_own_nonterminal_count_counts_unknown_and_uncertain() -> None:
    ids = ["1", "2", "3", "4"]
    states = {"1": "COMPLETED", "2": "FAILED", "3": "RUNNING"}
    assert own_nonterminal_count(ids, states, 0) == 2  # RUNNING + unresolved 4
    assert own_nonterminal_count(ids, states, 2) == 6  # + 2 uncertain reservations x 2
    assert own_nonterminal_count([], {}, 0) == 0


def test_per_fit_admission_boundaries() -> None:
    per_fit_admission(78, 349)  # exactly at the 80-job cap, below the user stop
    with pytest.raises(AdmissionError, match="no lane headroom"):
        per_fit_admission(79, 349)
    with pytest.raises(AdmissionError, match="stop threshold"):
        per_fit_admission(0, 350)
    per_fit_admission(77, 349, leg_jobs=3)
    with pytest.raises(AdmissionError, match="no lane headroom"):
        per_fit_admission(78, 349, leg_jobs=3)


def test_validate_wave_size() -> None:
    fits = [{"key": str(index)} for index in range(45)]
    with pytest.raises(AdmissionError):
        validate_wave_size(fits, 0)
    validate_wave_size(fits, 40)
    with pytest.raises(AdmissionError, match="fit lane allocation"):
        validate_wave_size(fits, 41)
    heavy = [{"key": "a", "leg_jobs": 60}, {"key": "b", "leg_jobs": 30}]
    with pytest.raises(AdmissionError, match="job lane allocation"):
        validate_wave_size(heavy, 2)


def test_lane_headroom() -> None:
    assert lane_headroom(0) == 40
    assert lane_headroom(78) == 1
    assert lane_headroom(79) == 0
    assert lane_headroom(80) == 0


def test_run_wave_reconciles_per_fit_and_stops_fail_closed() -> None:
    calls: list[str] = []
    reconciled = {"count": 0}

    def reconcile():
        reconciled["count"] += 1
        return 0, 10

    def submit(fit):
        calls.append(fit["key"])
        return {"key": fit["key"], "status": "submitted", "job_ids": {"train": "1", "best_eval": "2"}}

    processed = run_wave([{"key": "a"}, {"key": "b"}], 5, reconcile=reconcile, submit=submit)
    assert processed == 2
    assert calls == ["a", "b"]
    assert reconciled["count"] == 2  # re-checked before every fit

    def refusing_reconcile():
        return 79, 10

    refused = run_wave([{"key": "c"}], 5, reconcile=refusing_reconcile, submit=submit)
    assert refused == 0

    def uncertain_submit(fit):
        return {"key": fit["key"], "status": "uncertain", "job_ids": {}}

    processed = run_wave([{"key": "d"}, {"key": "e"}], 5, reconcile=reconcile, submit=uncertain_submit)
    assert processed == 1  # stops on the first non-submitted fit


def test_submit_fit_parses_success_and_preserves_partial() -> None:
    fit = {
        "key": "k",
        "run_name": "r",
        "config": "configs/x.yaml",
        "fold": 0,
        "seed": 7,
        "modality": "audio_only",
        "dataset": "daic",
        "deployment_id": "feat-qwen3-window15-20261008-abc",
    }
    success = _completed(
        0,
        '"attempt_id": "20261008T000000Z-run-abc-1234"\nsubmitted jobs: {\'train\': \'4701\', \'best_eval\': \'4702\'}\n',
    )
    record = submit_fit(fit, runner=lambda *args, **kwargs: success)
    assert record["status"] == "submitted"
    assert record["job_ids"] == {"train": "4701", "best_eval": "4702"}
    assert record["attempt_id"] == "20261008T000000Z-run-abc-1234"

    partial = _completed(1, '"attempt_id": "20261008T000000Z-run-abc-1234"\nsubmitted jobs: {\'train\': \'4701\'}\n', "boom")
    record = submit_fit(fit, runner=lambda *args, **kwargs: partial)
    assert record["status"] == "uncertain"
    assert record["job_ids"] == {"train": "4701"}  # partial delivery preserved
    assert "rc=1" in record["reason"]

    def timeout_runner(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    record = submit_fit(fit, runner=timeout_runner)
    assert record["status"] == "uncertain"
    assert record["reason"] == "timeout"


def test_validate_fit_requires_deployment_and_verified_env(tmp_path: Path) -> None:
    fit = {
        "key": "k",
        "run_name": "r",
        "modality": "audio_only",
        "deployment_id": "feat-qwen3-window15-20261008-abc",
    }
    with pytest.raises(AdmissionError, match="deployment record missing"):
        validate_fit(fit, deploy_root=tmp_path)
    record_dir = tmp_path / fit["deployment_id"]
    record_dir.mkdir(parents=True)
    (record_dir / "deployment.json").write_text("{}", encoding="utf-8")
    validate_fit(fit, deploy_root=tmp_path)
    bad_env = dict(fit, env_activate="/wrong/env/bin/activate")
    with pytest.raises(AdmissionError, match="does not match the verified environment"):
        validate_fit(bad_env, deploy_root=tmp_path)


def test_reconcile_evidence_payload(tmp_path: Path) -> None:
    ledger = tmp_path / "submissions.jsonl"
    ledger.write_text("", encoding="utf-8")
    exec_ledger = tmp_path / "state.json"
    exec_ledger.write_text(
        json.dumps(
            {
                "jobs": [
                    {"attempt_id": "a", "deployment_id": "feat-qwen3-window15-20261008-x", "slurm_job_id": "1"},
                    {"attempt_id": "b", "deployment_id": "feat-qwen3-window15-20261008-x", "slurm_job_id": "2"},
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = _FakeSsh(queue="10|RUNNING\n", sacct="1|COMPLETED\n2|RUNNING\n")
    evidence = reconcile_evidence(
        ledger,
        runner,
        exec_ledger,
        tmp_path / "exp_submit",
        tmp_path / "output_model",
    )
    assert evidence["own_job_count"] == 2
    assert evidence["own_nonterminal"] == 1  # job 2 RUNNING; job 1 terminal
    assert evidence["user_queue"] == 1
    assert evidence["lane_headroom_fits"] == 39


def test_build_matrix_marks_turkish_prebuilt(tmp_path: Path) -> None:
    out = tmp_path / "matrix.json"
    payload = build_matrix(out_path=out)
    assert payload["expected_fits"] == 126
    fits = payload["fits"]
    assert len(fits) == 126
    turkish = [fit for fit in fits if fit["dataset"] == "turkish"]
    others = [fit for fit in fits if fit["dataset"] != "turkish"]
    assert turkish and all(fit["manifest_policy"] == "prebuilt" for fit in turkish)
    assert others and all(fit["manifest_policy"] == "build" for fit in others)
    assert all(fit["modality"] in {"audio_only", "audio_text"} for fit in fits)
    assert all(fit["leg_jobs"] == 2 for fit in fits)
    assert out.is_file()


def test_extract_delivery_shapes() -> None:
    attempt, jobs = _extract_delivery(
        'noise\n"attempt_id": "abc"\nsubmitted jobs: {\'train\': \'1\', \'best_eval\': \'2\'}\n'
    )
    assert attempt == "abc"
    assert jobs == {"train": "1", "best_eval": "2"}
    attempt, jobs = _extract_delivery("no delivery here")
    assert attempt is None and jobs == {}


def test_exec_ledger_missing_or_malformed_fails_closed(tmp_path: Path) -> None:
    from tools.qwen3_window15_dispatch import exec_ledger_lane_jobs

    missing = tmp_path / "missing.json"
    with pytest.raises(AdmissionError, match="execution ledger missing"):
        exec_ledger_lane_jobs(missing)
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    with pytest.raises(AdmissionError, match="unreadable/malformed"):
        exec_ledger_lane_jobs(malformed)
    no_jobs = tmp_path / "no_jobs.json"
    no_jobs.write_text(json.dumps({"deployments": []}), encoding="utf-8")
    with pytest.raises(AdmissionError, match="no jobs section"):
        exec_ledger_lane_jobs(no_jobs)
    # own_job_ids must propagate the failure, never report zero ownership.
    with pytest.raises(AdmissionError):
        own_job_ids(tmp_path / "ledger.jsonl", tmp_path / "run_root", missing, tmp_path / "exp")


def test_lane_ledger_permitted_only_before_first_delivery(tmp_path: Path) -> None:
    ledger = tmp_path / "submissions.jsonl"
    exec_ledger = tmp_path / "state.json"
    exec_ledger.write_text(json.dumps({"jobs": []}), encoding="utf-8")
    exp_submit = tmp_path / "exp_submit"
    # No deliveries anywhere: a missing lane ledger is genuinely not-yet-created.
    ids, uncertain = own_job_ids(ledger, tmp_path / "run_root", exec_ledger, exp_submit)
    assert ids == [] and uncertain == 0
    # A delivery exists in the authoritative ledger: the missing lane ledger is
    # now an ownership gap and must fail closed.
    exec_ledger.write_text(
        json.dumps(
            {"jobs": [{"attempt_id": "a-q3w15_x", "deployment_id": "d", "slurm_job_id": "1"}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(AdmissionError, match="missing after deliveries exist"):
        own_job_ids(ledger, tmp_path / "run_root", exec_ledger, exp_submit)
    # Malformed lane ledger also fails closed.
    ledger.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(AdmissionError, match="malformed"):
        own_job_ids(ledger, tmp_path / "run_root", exec_ledger, exp_submit)


def test_partial_delivery_record_append_reconcile_round_trip(tmp_path: Path) -> None:
    from tools.qwen3_window15_dispatch import append_record

    ledger = tmp_path / "submissions.jsonl"
    exec_ledger = tmp_path / "state.json"
    exec_ledger.write_text(json.dumps({"jobs": []}), encoding="utf-8")
    fit = {
        "key": "smoke_x",
        "run_name": "r",
        "config": "configs/x.yaml",
        "fold": 0,
        "seed": 7,
        "modality": "audio_only",
        "dataset": "daic",
        "deployment_id": "feat-qwen3-window15-20261008-abc",
    }
    partial = _completed(
        1,
        '"attempt_id": "20261008T000000Z-run-abc-deadbeef"\n'
        "submitted jobs: {'train': '5001'}\n",
        "boom",
    )
    record = submit_fit(fit, runner=lambda *args, **kwargs: partial)
    assert record["status"] == "uncertain"
    append_record(record, ledger)
    evidence = reconcile_evidence(
        ledger,
        _FakeSsh(sacct="5001|FAILED\n"),
        exec_ledger,
        tmp_path / "exp_submit",
        tmp_path / "run_root",
    )
    # One uncertain record -> a two-job reservation on top of the FAILED job.
    assert evidence["uncertain_records"] == 1
    assert evidence["own_nonterminal"] == 2
    # The authoritative ledger later shows the complete graph: the reservation
    # reconciles automatically from evidence.
    exec_ledger.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "attempt_id": record["attempt_id"],
                        "deployment_id": "feat-qwen3-window15-20261008-abc",
                        "job_key": "train",
                        "slurm_job_id": "5001",
                    },
                    {
                        "attempt_id": record["attempt_id"],
                        "deployment_id": "feat-qwen3-window15-20261008-abc",
                        "job_key": "best_eval",
                        "slurm_job_id": "5002",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    evidence = reconcile_evidence(
        ledger,
        _FakeSsh(sacct="5001|FAILED\n5002|FAILED\n"),
        exec_ledger,
        tmp_path / "exp_submit",
        tmp_path / "run_root",
    )
    assert evidence["uncertain_records"] == 0
    assert evidence["own_nonterminal"] == 0


def test_seed_ledger_from_evidence_round_trip(tmp_path: Path) -> None:
    from tools.qwen3_window15_dispatch import seed_ledger_from_evidence

    ledger = tmp_path / "submissions.jsonl"
    exec_ledger = tmp_path / "state.json"
    exec_ledger.write_text(
        json.dumps(
            {
                "jobs": [
                    {"attempt_id": "attempt-1", "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "train", "slurm_job_id": "11"},
                    {"attempt_id": "attempt-1", "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "best_eval", "slurm_job_id": "12"},
                    {"attempt_id": "attempt-2", "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "train", "slurm_job_id": "13"},
                ]
            }
        ),
        encoding="utf-8",
    )
    exp_submit = tmp_path / "exp_submit"
    written = seed_ledger_from_evidence(ledger, exec_ledger, exp_submit)
    assert written == 2
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    by_attempt = {record["attempt_id"]: record for record in records}
    assert by_attempt["attempt-1"]["status"] == "submitted"
    assert by_attempt["attempt-2"]["status"] == "uncertain"  # incomplete graph
    assert all(record["source"] == "seed_from_execution_ledger" for record in records)
    ids, uncertain = own_job_ids(ledger, tmp_path / "run_root", exec_ledger, exp_submit)
    assert ids == ["11", "12", "13"]
    assert uncertain == 1  # attempt-2 keeps its reservation
    with pytest.raises(AdmissionError, match="already has records"):
        seed_ledger_from_evidence(ledger, exec_ledger, exp_submit)


def test_submit_fit_rejects_incomplete_or_duplicate_graphs() -> None:
    fit = {
        "key": "k",
        "run_name": "r",
        "config": "configs/x.yaml",
        "fold": 0,
        "seed": 7,
        "modality": "audio_only",
        "dataset": "daic",
        "deployment_id": "feat-qwen3-window15-20261008-abc",
    }
    base = '"attempt_id": "20261008T000000Z-run-abc-1234"\n'
    bad_outputs = (
        "submitted jobs: {'train': '4701'}\n",  # single ID
        "submitted jobs: {'train': '4701', 'best_eval': '4701'}\n",  # duplicate value
        "submitted jobs: {'train': 'abc', 'best_eval': '4702'}\n",  # non-numeric
        "submitted jobs: {'train': '4701', 'best_eval': '4702', 'extra': '4703'}\n",  # extra key
        "submitted jobs: {'best_eval': '4702'}\n",  # missing train
    )
    for output in bad_outputs:
        record = submit_fit(fit, runner=lambda *args, **kwargs: _completed(0, base + output))
        assert record["status"] == "uncertain", output
        assert "job graph" in record["reason"]
    good = submit_fit(
        fit,
        runner=lambda *args, **kwargs: _completed(
            0, base + "submitted jobs: {'train': '4701', 'best_eval': '4702'}\n"
        ),
    )
    assert good["status"] == "submitted"
    assert good["job_ids"] == {"train": "4701", "best_eval": "4702"}


def test_uncertain_reservation_requires_complete_distinct_ids(tmp_path: Path) -> None:
    from tools.qwen3_window15_dispatch import append_record

    ledger = tmp_path / "submissions.jsonl"
    exec_ledger = tmp_path / "state.json"
    fit = {
        "key": "k",
        "run_name": "r",
        "config": "configs/x.yaml",
        "fold": 0,
        "seed": 7,
        "modality": "audio_only",
        "dataset": "daic",
        "deployment_id": "feat-qwen3-window15-20261008-abc",
    }
    # rc0 single-ID output goes through the real submit path -> uncertain.
    record = submit_fit(
        fit,
        runner=lambda *args, **kwargs: _completed(
            0, '"attempt_id": "att-1"\nsubmitted jobs: {\'train\': \'5001\'}\n'
        ),
    )
    assert record["status"] == "uncertain"
    append_record(record, ledger)
    attempt = record["attempt_id"]
    # Key names present but no delivered IDs: the reservation is retained.
    exec_ledger.write_text(
        json.dumps(
            {
                "jobs": [
                    {"attempt_id": attempt, "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "train"},
                    {"attempt_id": attempt, "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "best_eval"},
                ]
            }
        ),
        encoding="utf-8",
    )
    _, uncertain = own_job_ids(ledger, tmp_path / "run_root", exec_ledger, tmp_path / "exp_submit")
    assert uncertain == 1
    # Duplicate delivered IDs never clear it.
    exec_ledger.write_text(
        json.dumps(
            {
                "jobs": [
                    {"attempt_id": attempt, "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "train", "slurm_job_id": "5001"},
                    {"attempt_id": attempt, "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "best_eval", "slurm_job_id": "5001"},
                ]
            }
        ),
        encoding="utf-8",
    )
    _, uncertain = own_job_ids(ledger, tmp_path / "run_root", exec_ledger, tmp_path / "exp_submit")
    assert uncertain == 1
    # Complete distinct IDs clear it.
    exec_ledger.write_text(
        json.dumps(
            {
                "jobs": [
                    {"attempt_id": attempt, "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "train", "slurm_job_id": "5001"},
                    {"attempt_id": attempt, "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "best_eval", "slurm_job_id": "5002"},
                ]
            }
        ),
        encoding="utf-8",
    )
    _, uncertain = own_job_ids(ledger, tmp_path / "run_root", exec_ledger, tmp_path / "exp_submit")
    assert uncertain == 0


def test_unresolved_delivery_reservations_requires_complete_ids(tmp_path: Path) -> None:
    ledger = tmp_path / "submissions.jsonl"
    ledger.write_text("", encoding="utf-8")
    exp_submit = tmp_path / "exp_submit"
    (exp_submit / "a").mkdir(parents=True)
    (exp_submit / "a" / "submit_output.log").write_text('"attempt_id": "att-1"\n', encoding="utf-8")
    exec_ledger = tmp_path / "state.json"
    graphs = (
        [{"job_key": "train", "slurm_job_id": "1"}],
        [{"job_key": "train", "slurm_job_id": "1"}, {"job_key": "best_eval"}],
        [{"job_key": "train", "slurm_job_id": "1"}, {"job_key": "best_eval", "slurm_job_id": "1"}],
    )
    for jobs in graphs:
        exec_ledger.write_text(
            json.dumps(
                {
                    "jobs": [
                        {"attempt_id": "att-1", "deployment_id": "feat-qwen3-window15-20261008-x", **job}
                        for job in jobs
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert unresolved_delivery_reservations(ledger, exec_ledger, exp_submit) == 1, jobs
    exec_ledger.write_text(
        json.dumps(
            {
                "jobs": [
                    {"attempt_id": "att-1", "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "train", "slurm_job_id": "1"},
                    {"attempt_id": "att-1", "deployment_id": "feat-qwen3-window15-20261008-x", "job_key": "best_eval", "slurm_job_id": "2"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert unresolved_delivery_reservations(ledger, exec_ledger, exp_submit) == 0


_BSC_QUOTA_SAMPLE = (
    "\x1b[0m\x1b[0m Printing quota for group etur92:\x1b[22m\x1b[0m\n\n"
    "\x1b[1;30m\x1b[0m    Filesystem   Type          Usage          Quota          Limit     In doubt     Grace  |       Files  In doubt  \x1b[0m\n"
    "\x1b[0m     gpfs_home    USR       31.88 GB       80.00 GB       84.00 GB\x1b[38;5;252m      0.00 GB      None  |      115940         0  \x1b[0m\n"
    "\x1b[0m gpfs_projects    GRP     1831.15 GB     4000.00 GB     4200.00 GB\x1b[38;5;252m      4.69 GB      None  |     1723914       469  \x1b[0m\n"
    "\x1b[0m  gpfs_scratch    GRP     1790.89 GB     2000.00 GB     2100.00 GB\x1b[38;5;252m      0.09 GB      None  |     1665976        39  \x1b[0m\n"
)


def _quota_output(rows: list[str], group: str = "etur92") -> str:
    header = f"\x1b[0m Printing quota for group {group}:\x1b[22m\x1b[0m\n\n"
    table = "    Filesystem   Type          Usage          Quota          Limit     In doubt     Grace  |       Files  In doubt\n"
    return header + table + "\n".join(rows) + "\n"


_GOOD_ROW = (
    " gpfs_projects    GRP     1831.15 GB     4000.00 GB     4200.00 GB      4.69 GB      None  |     1723914       469"
)


def test_parse_bsc_quota_projects_extracts_and_fails_closed() -> None:
    from tools.qwen3_window15_dispatch import parse_bsc_quota_projects

    parsed = parse_bsc_quota_projects(_BSC_QUOTA_SAMPLE)
    assert parsed == {"usage_gb": 1831.15, "quota_gb": 4000.0, "limit_gb": 4200.0, "in_doubt_gb": 4.69}

    refusals = (
        ("no rows here", "does not declare group"),
        (_quota_output([_GOOD_ROW], group="other"), "does not declare group"),
        (
            _quota_output([_GOOD_ROW.replace("GRP", "USR", 1)]),
            "type is 'USR'",
        ),
        (_quota_output([_GOOD_ROW, _GOOD_ROW]), "2 gpfs_projects rows"),
        (
            " Printing quota for group etur92:\n gpfs_projects    GRP     1.0 GB     2.0 GB     2.1 GB\n",
            "too short",
        ),
        (
            _quota_output([_GOOD_ROW.replace("1831.15 GB", "1831.15 KB", 1)]),
            "unit is 'KB'",
        ),
        (
            _quota_output([_GOOD_ROW.replace("1831.15", "abc", 1)]),
            "unparseable",
        ),
        (
            _quota_output([_GOOD_ROW.replace("1831.15", "nan", 1)]),
            "not finite",
        ),
        (
            _quota_output([_GOOD_ROW.replace("1831.15", "inf", 1)]),
            "not finite",
        ),
        (
            _quota_output([_GOOD_ROW.replace("1831.15", "-1.0", 1)]),
            "negative",
        ),
        (
            _quota_output([_GOOD_ROW.replace("4000.00", "0.00", 1)]),
            "soft quota must be positive",
        ),
        (
            _quota_output([_GOOD_ROW.replace("4200.00", "3999.00", 1)]),
            "hard limit is below the soft quota",
        ),
    )
    for output, message in refusals:
        with pytest.raises(AdmissionError, match=message):
            parse_bsc_quota_projects(output)


def test_storage_admission_gate_boundaries() -> None:
    from tools.qwen3_window15_dispatch import storage_admission

    evidence = storage_admission(
        _FakeSsh(quota=_BSC_QUOTA_SAMPLE), local_available_bytes=int(60 * 1024 ** 3)
    )
    assert evidence["admission"] == "admissible"
    assert evidence["gpfs_projects_remaining_gb"] == pytest.approx(4000.0 - 1831.15 - 4.69, abs=0.01)
    low = " gpfs_projects    GRP     3600.00 GB     4000.00 GB     4200.00 GB      4.69 GB      None"
    with pytest.raises(AdmissionError, match="below the 500 GB reserve"):
        storage_admission(_FakeSsh(quota=_quota_output([low])), local_available_bytes=int(60 * 1024 ** 3))
    with pytest.raises(AdmissionError, match="local available"):
        storage_admission(_FakeSsh(quota=_BSC_QUOTA_SAMPLE), local_available_bytes=int(10 * 1024 ** 3))
    with pytest.raises(AdmissionError, match="scheduler query failed"):
        storage_admission(_FakeSsh(returncode=255, stderr="ssh fail"), local_available_bytes=int(60 * 1024 ** 3))


def test_storage_admission_rejects_non_finite_inputs() -> None:
    from tools.qwen3_window15_dispatch import storage_admission

    with pytest.raises(AdmissionError, match="remaining_min_gb must be finite and non-negative"):
        storage_admission(
            _FakeSsh(quota=_BSC_QUOTA_SAMPLE),
            remaining_min_gb=float("nan"),
            local_available_bytes=int(60 * 1024 ** 3),
        )
    with pytest.raises(AdmissionError, match="local_min_gb must be finite and non-negative"):
        storage_admission(
            _FakeSsh(quota=_BSC_QUOTA_SAMPLE),
            local_min_gb=-1.0,
            local_available_bytes=int(60 * 1024 ** 3),
        )
    with pytest.raises(AdmissionError, match="local available bytes must be finite and non-negative"):
        storage_admission(_FakeSsh(quota=_BSC_QUOTA_SAMPLE), local_available_bytes=-5)


def test_run_wave_storage_check_stops_before_next_batch() -> None:
    calls: list[str] = []

    def reconcile():
        return 0, 10

    def submit(fit):
        calls.append(fit["key"])
        return {"key": fit["key"], "status": "submitted", "job_ids": {}}

    def storage_check():
        if len(calls) >= 5:
            raise AdmissionError("storage reserve reached")
        return {"admission": "admissible"}

    processed = run_wave(
        [{"key": str(index)} for index in range(10)],
        10,
        reconcile=reconcile,
        submit=submit,
        storage_check=storage_check,
        storage_every=5,
    )
    assert processed == 5
    assert calls == ["0", "1", "2", "3", "4"]
