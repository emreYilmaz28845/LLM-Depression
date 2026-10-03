"""Concurrency tests for the shared execution-ledger state tool.

The ledger is written by every lane's ``exp.py submit``/``status`` through
``tools/parallel_workflow_state.py``. Two failure modes previously existed:

* a shared fixed temp path (``state.json.tmp``) let concurrent writers
  interleave into a torn file;
* a read-modify-write transaction without a process lock could lose updates
  from a stale reader.

These tests exercise the fixed behavior with real concurrent CLI processes and
a continuous reader: every distinct job event must survive, no duplicate event
may appear, and the file must never be observed as malformed JSON.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/parallel_workflow_state.py"
RUNBOOK = ROOT / "configs/README.md"
WRITERS = 8


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args], capture_output=True, text=True, cwd=ROOT
    )


def test_concurrent_record_job_keeps_every_distinct_event(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    init = _run(
        "init",
        "--runbook",
        str(RUNBOOK),
        "--execution-id",
        "concurrency-test",
        "--output",
        str(state),
    )
    assert init.returncode == 0, init.stderr

    reader_errors: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            try:
                payload = json.loads(state.read_text(encoding="utf-8"))
                assert payload["schema_version"] == "audiollm.parallel_workflow_execution.v1"
                assert isinstance(payload["jobs"], list)
            except Exception as exc:  # noqa: BLE001
                reader_errors.append(repr(exc))
                return

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    procs = []
    for index in range(WRITERS):
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(TOOL),
                    "record-job",
                    "--state",
                    str(state),
                    "--attempt-id",
                    f"attempt-{index}",
                    "--job-key",
                    "train",
                    "--job-type",
                    "train",
                    "--event-type",
                    "SUBMITTED",
                    "--slurm-job-id",
                    str(1000 + index),
                    "--status",
                    "PENDING",
                    "--fold",
                    "0",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=ROOT,
            )
        )
    for proc in procs:
        _out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err
    stop.set()
    thread.join(timeout=10)

    assert reader_errors == []
    final = json.loads(state.read_text(encoding="utf-8"))
    job_ids = [job["slurm_job_id"] for job in final["jobs"]]
    assert sorted(job_ids) == [str(1000 + index) for index in range(WRITERS)]
    assert len(job_ids) == len(set(job_ids)), "a concurrent update was duplicated"
    assert list(tmp_path.glob("state.json.tmp*")) == [], "a temp file was left behind"


def test_state_lock_releases_after_use(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from tools.parallel_workflow_state import state_lock

    state = tmp_path / "state.json"
    with state_lock(state):
        pass
    # A leaked lock would make this acquisition time out.
    with state_lock(state, timeout=5):
        pass
