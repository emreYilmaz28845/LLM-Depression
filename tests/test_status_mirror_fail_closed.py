"""Fail-closed status mirror regression.

When a collected fold's terminal evidence cannot be persisted (append
refused), ``exp.py status`` must return nonzero with a precise diagnostic
because the local job history is incomplete. A subsequent poll that can
persist the event must succeed, and a later no-op poll that finds the event
already present must also stay successful without duplicating or dropping
evidence.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.experiment_tracking import lifecycle  # noqa: E402
from src.experiment_tracking import monitor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import exp as exp_module  # noqa: E402

ATTEMPT = "20261002T204401Z-cmdc_audio_text_english_s7_f0-15b62075-ac4314dc"
SLURM_JOB_ID = "46953768"
BASE = datetime(2026, 10, 3, 5, 0, 0, tzinfo=timezone.utc)


class _FakeScheduler:
    def __init__(self, host=None):
        self.host = host

    def squeue(self, job_ids):
        return {jid: {"STATE": "RUNNING"} for jid in job_ids}

    def sacct(self, job_ids):
        return {jid: {"State": "FAILED", "ExitCode": "1:0"} for jid in job_ids}


def _build_lane(tmp_path: Path, monkeypatch) -> tuple[types.SimpleNamespace, Path]:
    monkeypatch.setattr(exp_module, "PROJECT_ROOT", tmp_path)
    fold = tmp_path / "output_model" / "camp" / "audio_text" / "cmdc" / "run" / "fold_0"
    fold.mkdir(parents=True)
    (fold / "metadata.json").write_text(
        json.dumps({"attempt_id": ATTEMPT, "fold": 0}), encoding="utf-8"
    )
    record = lifecycle.StatusRecord(ATTEMPT, 0, state="SUBMITTED")
    record.transition("RUNNING", reason="training job started")
    lifecycle.write_status(fold / "status.json", record)
    train_event = lifecycle.new_job_event(
        job_key="train", job_type="train", event_type="COMPLETED",
        attempt_id=ATTEMPT, fold=0, slurm_job_id="46953767", status="COMPLETED",
    )
    train_event["exit_code"] = "0:0"
    (fold / "jobs.jsonl").write_text(json.dumps(train_event) + "\n", encoding="utf-8")
    contract_path = tmp_path / "outputs" / "exp_submit" / ATTEMPT / "contract.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(
        json.dumps(
            {
                "attempt_id": ATTEMPT,
                "fold": 0,
                "job_type": "evaluation",
                "local_fold_rel": "output_model/camp/audio_text/cmdc/run/fold_0",
            }
        ),
        encoding="utf-8",
    )
    ledger_path = tmp_path / "state.json"
    ledger_path.write_text(
        json.dumps(
            {
                "deployments": [{"deployment_id": "dep1", "experiment_id": "test-exp"}],
                "jobs": [
                    {
                        "deployment_id": "dep1",
                        "attempt_id": ATTEMPT,
                        "job_key": "best_eval",
                        "job_type": "evaluation",
                        "event_type": "TERMINAL",
                        "slurm_job_id": SLURM_JOB_ID,
                        "status": "FAILED",
                        "fold": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp_module, "EXECUTION_LEDGER_PATH", ledger_path)
    monkeypatch.setattr(
        exp_module, "_resolve_lane_for_command", lambda slug: (tmp_path, {"experiment_id": "test-exp"})
    )
    monkeypatch.setattr(monitor, "SchedulerClient", _FakeScheduler)

    def fake_reconcile(record_dict, queue, accounting, artifacts_ok=None):
        return types.SimpleNamespace(
            slurm_job_id=str(record_dict["slurm_job_id"]),
            job_key=record_dict["job_key"],
            queue_state=None,
            account_state="FAILED",
            exit_code="1:0",
            classification="deterministic_code_config",
            terminal_failure=True,
        )

    monkeypatch.setattr(monitor, "reconcile_job", fake_reconcile)
    args = types.SimpleNamespace(slug="test-exp", scheduler_host="fake")
    return args, fold


def _best_eval_events(fold: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (fold / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("job_key") == "best_eval"
    ]


def test_status_fails_on_refused_mirror_then_repoll_succeeds(tmp_path, monkeypatch, capsys) -> None:
    args, fold = _build_lane(tmp_path, monkeypatch)
    state = {"refuse": True}
    original_append = lifecycle.append_job_event

    def flaky_append(path, event):
        if state["refuse"]:
            raise ValueError("simulated job history append refusal")
        return original_append(path, event)

    monkeypatch.setattr(lifecycle, "append_job_event", flaky_append)

    # 1. Refused append: nonzero, precise diagnostic, no evidence drop reported.
    rc = exp_module._cmd_status(args)
    captured = capsys.readouterr()
    assert rc == 1
    assert "could not persist terminal evidence" in captured.err
    assert "simulated job history append refusal" in captured.err
    assert str(fold / "jobs.jsonl") in captured.err
    assert _best_eval_events(fold) == []
    status = json.loads((fold / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "RUNNING"

    # 2. Re-poll with a working append persists the evidence and succeeds.
    state["refuse"] = False
    rc = exp_module._cmd_status(args)
    assert rc == 0
    events = _best_eval_events(fold)
    assert len(events) == 1
    assert events[0]["event_type"] == "FAILED"
    assert events[0]["status"] == "FAILED"
    assert events[0]["exit_code"] == "1:0"
    status = json.loads((fold / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "FAILED"

    # 3. No-op re-poll: already-present evidence stays successful, no duplicates.
    event_id = events[0]["event_id"]
    rc = exp_module._cmd_status(args)
    assert rc == 0
    events = _best_eval_events(fold)
    assert len(events) == 1
    assert events[0]["event_id"] == event_id
