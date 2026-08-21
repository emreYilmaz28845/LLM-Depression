from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_parallel_workflow_implementation import (
    audit_state,
    check_clean_source,
    check_structured_records,
)

EXEC_STATE = Path(
    "/home/emre/Projects/AudioLLM/LLM-Depression/outputs/parallel_workflow_implementation/"
    "20260820T205735Z-parallel-workflow-2d995f4c/state.json"
)
INVALID_SNAPSHOT = EXEC_STATE.parent / "state.json.invalidated.20260821"
FINAL_AUDIT_OLD = EXEC_STATE.parent / "final_audit.json"


def _run_audit(state_path, *extra):
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools" / "audit_parallel_workflow_implementation.py"),
         "--state", str(state_path), *extra],
        capture_output=True, text=True,
    )
    return proc


def _tmp_state(tmp_path, source=EXEC_STATE, name="state.json"):
    dst = tmp_path / name
    shutil.copy(source, dst)
    return dst


def _all_passed_active_phase13(state):
    for i in range(13):
        state["phases"][str(i)]["status"] = "PASSED"
    state["phases"]["13"]["status"] = "IN_PROGRESS"
    state["current_phase"] = 13
    state["status"] = "ACTIVE"


# --- negative tests against the old false completion ---

@pytest.mark.skipif(not INVALID_SNAPSHOT.exists(), reason="invalidated snapshot missing")
def test_old_false_complete_fails_preterminal(tmp_path):
    """The old false COMPLETE ledger must fail a preterminal audit."""
    dst = _tmp_state(tmp_path, source=INVALID_SNAPSHOT)
    proc = _run_audit(dst, "--mode", "preterminal")
    assert proc.returncode != 0
    assert any("ACTIVE" in e or "PASSED" in e for e in proc.stderr.splitlines())


@pytest.mark.skipif(not FINAL_AUDIT_OLD.exists(), reason="old final audit missing")
def test_old_final_audit_is_not_a_preterminal_approval():
    old = json.loads(FINAL_AUDIT_OLD.read_text())
    assert old.get("mode") != "preterminal", "old keyword-based audit must not count as preterminal approval"


def test_prose_only_evidence_cannot_pass_substantive_gates(tmp_path):
    dst = _tmp_state(tmp_path)
    state = json.loads(dst.read_text())
    _all_passed_active_phase13(state)
    # structured records empty -> substantive gate failure even with prose evidence
    state["prs"] = []
    state["deployments"] = []
    state["attempts"] = []
    state["jobs"] = []
    for i in range(14):
        state["phases"][str(i)]["evidence"] = ["some prose claiming success"]
    dst.write_text(json.dumps(state))
    passed, messages, _ = audit_state(dst, mode="preterminal")
    assert not passed
    assert any("structured" in m.lower() for m in messages)


def test_missing_evidence_paths_fail(tmp_path):
    dst = _tmp_state(tmp_path)
    state = json.loads(dst.read_text())
    for i in range(3):
        state["phases"][str(i)]["evidence"] = ["/nonexistent/path/evidence.log"]
    dst.write_text(json.dumps(state))
    passed, messages, _ = audit_state(dst, allow_incomplete=True)
    assert not passed
    assert any("does not exist" in m for m in messages)


def test_empty_full_suite_log_fails(tmp_path):
    dst = _tmp_state(tmp_path)
    state = json.loads(dst.read_text())
    empty_log = tmp_path / "phase8_full_suite.log"
    empty_log.write_text("")
    for ph in ("11", "12"):
        state["phases"][ph]["evidence"].append(str(empty_log))
    dst.write_text(json.dumps(state))
    passed, messages, _ = audit_state(dst, mode="preterminal")
    assert any("full-suite log is empty" in m for m in messages)


def test_abbreviated_pr_sha_fails(tmp_path):
    dst = _tmp_state(tmp_path)
    state = json.loads(dst.read_text())
    for pr in state.get("prs", []):
        pr["merge_sha"] = pr["merge_sha"][:7]
    errors = check_structured_records(state, PROJECT_ROOT)
    assert any("not a full 40-char SHA" in e for e in errors)


def test_cancelled_eval_without_replacement_fails(tmp_path):
    dst = _tmp_state(tmp_path)
    state = json.loads(dst.read_text())
    jobs = state.get("jobs", [])
    # keep only the cancelled eval, drop completed replacements
    state["jobs"] = [
        j for j in jobs
        if not (j.get("job_key") == "best_eval" and j.get("status") == "COMPLETED")
    ]
    errors = check_structured_records(state, PROJECT_ROOT)
    assert any(
        ("lacks COMPLETED" in e) or ("without a completed replacement" in e)
        for e in errors
    )


def test_dirty_auditor_source_fails(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    auditor = repo / "tools" / "audit_parallel_workflow_implementation.py"
    auditor.write_text("# auditor\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-q", "-m", "x"],
                   cwd=repo, check=True, capture_output=True)
    auditor.write_text("# auditor tampered\n")  # dirty now
    errors = check_clean_source(repo)
    assert any("auditor/state-tool source is dirty" in e for e in errors)


def test_staged_unrelated_change_fails(tmp_path):
    repo = tmp_path / "repo2"
    (repo / "tools").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    (repo / "tools" / "audit_parallel_workflow_implementation.py").write_text("# ok\n")
    (repo / "unrelated.txt").write_text("user file\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-q", "-m", "x"],
                   cwd=repo, check=True, capture_output=True)
    (repo / "staged.txt").write_text("new\n")
    subprocess.run(["git", "add", "staged.txt"], cwd=repo, check=True, capture_output=True)
    errors = check_clean_source(repo)
    assert any("staged change present" in e for e in errors)


def test_terminal_requires_stored_preterminal_hash(tmp_path):
    dst = _tmp_state(tmp_path)
    state = json.loads(dst.read_text())
    for i in range(14):
        state["phases"][str(i)]["status"] = "PASSED"
    state["current_phase"] = 13
    state["status"] = "COMPLETE"
    dst.write_text(json.dumps(state))
    passed, messages, _ = audit_state(dst, mode="terminal")
    assert not passed
    assert any("lacks recorded preterminal audit path" in m for m in messages)


def test_tampered_preterminal_audit_detected(tmp_path):
    dst = _tmp_state(tmp_path)
    state = json.loads(dst.read_text())
    for i in range(14):
        state["phases"][str(i)]["status"] = "PASSED"
    state["current_phase"] = 13
    state["status"] = "COMPLETE"
    fake_audit = tmp_path / "preterminal_audit.json"
    payload = {"schema_version": "audiollm.parallel_workflow_audit.v1",
               "mode": "preterminal", "passed": True}
    fake_audit.write_text(json.dumps(payload))
    real_sha = hashlib.sha256(fake_audit.read_bytes()).hexdigest()
    state["completion"] = {
        "preterminal_audit_path": str(fake_audit),
        "preterminal_audit_sha256": real_sha,
    }
    dst.write_text(json.dumps(state))
    # Tamper with the audit after recording: hash mismatch must be caught.
    payload["tampered"] = True
    fake_audit.write_text(json.dumps(payload))
    passed2, messages2, _ = audit_state(dst, mode="terminal")
    assert not passed2
    assert any("hash does not match file" in m for m in messages2)


def test_state_tool_complete_consumes_preterminal_audit_atomically(tmp_path):
    state_path = tmp_path / "state.json"
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools" / "parallel_workflow_state.py"),
         "init", "--runbook", str(PROJECT_ROOT / "docs" / "PARALLEL_EXPERIMENT_WORKFLOW_PLAN.md"),
         "--execution-id", "test-exec-complete", "--output", str(state_path)],
        check=True, capture_output=True,
    )
    state = json.loads(state_path.read_text())
    for i in range(13):
        state["phases"][str(i)]["status"] = "PASSED"
    state["phases"]["13"]["status"] = "IN_PROGRESS"
    state["current_phase"] = 13
    state_path.write_text(json.dumps(state))

    tool = str(PROJECT_ROOT / "tools" / "parallel_workflow_state.py")

    # without an approved preterminal audit: refused
    r = subprocess.run([sys.executable, tool, "complete", "--state", str(state_path)],
                       capture_output=True, text=True)
    assert r.returncode != 0

    # with a failing audit: refused
    bad_audit = tmp_path / "bad_audit.json"
    bad_audit.write_text(json.dumps({"mode": "preterminal", "passed": False}))
    r = subprocess.run([sys.executable, tool, "complete", "--state", str(state_path),
                        "--audit", str(bad_audit)], capture_output=True, text=True)
    assert r.returncode != 0

    # with a passing preterminal audit: consumed and recorded
    good_audit = tmp_path / "good_audit.json"
    good_audit.write_text(json.dumps({"schema_version": "audiollm.parallel_workflow_audit.v1",
                                      "mode": "preterminal", "passed": True}))
    r = subprocess.run([sys.executable, tool, "complete", "--state", str(state_path),
                        "--audit", str(good_audit)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    final = json.loads(state_path.read_text())
    assert final["status"] == "COMPLETE"
    assert final["completion"]["preterminal_audit_sha256"] == hashlib.sha256(
        good_audit.read_bytes()).hexdigest()
    assert final["phases"]["13"]["status"] == "PASSED"

    # double completion refused
    r = subprocess.run([sys.executable, tool, "complete", "--state", str(state_path),
                        "--audit", str(good_audit)], capture_output=True, text=True)
    assert r.returncode != 0


def test_no_artifact_contains_its_own_hash(tmp_path):
    """The audit JSON we emit must never embed its own sha256."""
    dst = _tmp_state(tmp_path)
    out = tmp_path / "audit_out.json"
    proc = _run_audit(dst, "--allow-incomplete", "--output", str(out))
    if out.is_file():
        content = out.read_text()
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        assert digest not in content
