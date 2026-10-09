"""Tests for the candidate coordinator-owned global GPU gate (worker1 stop).

Rules under test (user directives, 2026-10-09):
- hard total cap 8 GPUs / 2 nodes across ALL own account jobs;
- at most ONE released GPU job at a time;
- fail closed on own running job, unheld pending GPU job, or unknown state;
- totals from AllocTRES/ReqTRES, never per-node values;
- never release a whole leg (single-job admission only);
- CPU heads require bounded billing and bounded wall time;
- root acceptance required before any release.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.check_worker1_gpu_gate import (  # noqa: E402
    Job,
    Tres,
    check_batch_release,
    check_release,
    parse_scontrol_dump,
    parse_tres,
)

APPROVAL = "root-approved-2026-10-09"


def audio_train(job_id="47077943", state="PENDING", reason="JobHeldUser"):
    return Job(
        job_id=job_id,
        name="sym-audi-cv-0-trai",
        state=state,
        reason=reason,
        req=Tres(gpus=8, nodes=2, billing=160),
        alloc=Tres(gpus=8, nodes=2, billing=320) if state == "RUNNING" else None,
        time_limit_minutes=4320,
    )


def audio_post(job_id="47077944", state="PENDING", reason="JobHeldUser"):
    return Job(
        job_id=job_id,
        name="sym-audi-cv-0-post",
        state=state,
        reason=reason,
        req=Tres(gpus=4, nodes=1, billing=80),
        time_limit_minutes=1440,
    )


def cpu_head(job_id="47077945", state="PENDING", reason="JobHeldUser", time_limit=30):
    return Job(
        job_id=job_id,
        name="sym-audi-cv-0-head",
        state=state,
        reason=reason,
        req=Tres(gpus=0, nodes=1, billing=20),
        time_limit_minutes=time_limit,
    )


def test_parse_tres_totals_only():
    t = parse_tres("billing=320,cpu=320,gres/gpu=8,mem=1000000M,node=2")
    assert (t.gpus, t.nodes, t.billing) == (8, 2, 320)


def test_allow_single_release_when_quiescent():
    cand = audio_post()
    dec = check_release(cand, [], APPROVAL)
    assert dec["allow"], dec["reasons"]


def test_deny_without_root_approval():
    dec = check_release(audio_post(), [], None)
    assert not dec["allow"]
    assert any("root_acceptance_required" in r for r in dec["reasons"])


def test_deny_when_any_own_job_running():
    running = audio_train(job_id="47102128", state="RUNNING", reason="None")
    dec = check_release(audio_post(), [running], APPROVAL)
    assert not dec["allow"]
    assert any("own_active_job" in r for r in dec["reasons"])


def test_deny_when_unheld_pending_gpu_job_exists():
    loose = audio_train(job_id="47999999", state="PENDING", reason="None")
    dec = check_release(audio_post(), [loose], APPROVAL)
    assert not dec["allow"]
    assert any("own_unheld_pending_gpu_job" in r for r in dec["reasons"])


def test_deny_on_unknown_state():
    weird = Job(job_id="47888888", state="NOT_A_STATE", req=Tres(gpus=1, nodes=1, billing=20))
    dec = check_release(audio_post(), [weird], APPROVAL)
    assert not dec["allow"]
    assert any("own_unknown_state" in r for r in dec["reasons"])


def test_held_pending_jobs_are_not_active():
    dec = check_release(audio_post(), [audio_train()], APPROVAL)
    assert dec["allow"], dec["reasons"]


def test_deny_batch_release_of_whole_leg():
    leg = [audio_train(f"4707794{i}") for i in range(5)] + [audio_post(f"4707795{i}") for i in range(5)]
    dec = check_batch_release(leg, [], APPROVAL)
    assert not dec["allow"]
    assert any("batch_release_forbidden" in r for r in dec["reasons"])


def test_deny_candidate_gpus_over_limit():
    cand = audio_train()
    cand.req = Tres(gpus=16, nodes=2, billing=160)
    dec = check_release(cand, [], APPROVAL)
    assert not dec["allow"]
    assert any("candidate_gpus_exceed_limit" in r for r in dec["reasons"])


def test_deny_candidate_nodes_over_limit():
    cand = audio_train()
    cand.req = Tres(gpus=8, nodes=3, billing=160)
    dec = check_release(cand, [], APPROVAL)
    assert not dec["allow"]
    assert any("candidate_nodes_exceed_limit" in r for r in dec["reasons"])


def test_deny_gpu_billing_rate_over_cap():
    cand = audio_train()
    cand.req = Tres(gpus=8, nodes=2, billing=640)
    dec = check_release(cand, [], APPROVAL)
    assert not dec["allow"]
    assert any("candidate_billing_exceeds_cap" in r for r in dec["reasons"])


def test_deny_cpu_head_unbounded_wall_time():
    cand = cpu_head(time_limit=None)
    dec = check_release(cand, [], APPROVAL)
    assert not dec["allow"]
    assert any("cpu_head_wall_time_unbounded_or_too_long" in r for r in dec["reasons"])


def test_deny_cpu_head_billing_over_cap():
    cand = cpu_head()
    cand.req = Tres(gpus=0, nodes=1, billing=160)
    dec = check_release(cand, [], APPROVAL)
    assert not dec["allow"]
    assert any("cpu_head_billing_exceeds_cap" in r for r in dec["reasons"])


def test_allow_head_at_cpu_cap_boundary():
    cand = cpu_head()
    dec = check_release(cand, [], APPROVAL)
    assert dec["allow"], dec["reasons"]


def test_account_total_defense_in_depth():
    # A released non-held pending job plus candidate would exceed 8 GPUs in sum.
    cand = audio_post()  # 4 GPU
    loose = audio_train(job_id="47000001", state="PENDING", reason="None")  # 8 GPU
    dec = check_release(cand, [loose], APPROVAL)
    assert not dec["allow"]
    assert any("own_unheld_pending_gpu_job" in r or "account_gpu_total_exceeds_limit" in r for r in dec["reasons"])


SCONTROL_SAMPLE = """
JobId=47077944 JobName=sym-audi-cv-0-post
   UserId=ozu647717(53836) GroupId=ozu(52230) MCS_label=N/A
   JobState=PENDING Reason=JobHeldUser Dependency=(null)
   Partition=acc AllocNode:Sid=alogin2:3056621
   NumNodes=1-1 NumCPUs=80 NumTasks=1 CPUs/Task=20 ReqB:S:C:T=0:0:*:1
   ReqTRES=cpu=80,mem=500000M,node=1,billing=80,gres/gpu=4
   AllocTRES=(null)
   TimeLimit=1-00:00:00
   
JobId=47077943 JobName=sym-audi-cv-0-trai
   UserId=ozu647717(53836) GroupId=ozu(52230) MCS_label=N/A
   JobState=COMPLETED Reason=None
   Partition=acc AllocNode:Sid=alogin2:3056621
   NumNodes=2 NumCPUs=160
   ReqTRES=cpu=160,mem=1000000M,node=2,billing=160,gres/gpu=8
   AllocTRES=billing=320,cpu=320,gres/gpu=8,mem=1000000M,node=2
   TimeLimit=3-00:00:00
"""


def test_parse_scontrol_dump_and_cli(tmp_path):
    jobs = parse_scontrol_dump(SCONTROL_SAMPLE)
    assert {j.job_id for j in jobs} == {"47077944", "47077943"}
    post = next(j for j in jobs if j.job_id == "47077944")
    assert post.held and post.req.gpus == 4 and post.req.nodes == 1
    dec = check_release(post, [j for j in jobs if j.job_id != "47077944"], APPROVAL)
    assert dec["allow"], dec["reasons"]

    dump = tmp_path / "scontrol.txt"
    dump.write_text(SCONTROL_SAMPLE, encoding="utf-8")
    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_worker1_gpu_gate.py"),
            "--scontrol-dump",
            str(dump),
            "--candidate-job",
            "47077944",
            "--root-approval",
            APPROVAL,
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert json.loads(out.stdout)["allow"] is True

    out2 = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_worker1_gpu_gate.py"),
            "--scontrol-dump",
            str(dump),
            "--candidate-job",
            "47077944",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out2.returncode == 1
    assert json.loads(out2.stdout)["allow"] is False


def test_parse_scontrol_dump_deny_when_candidate_missing(tmp_path):
    dump = tmp_path / "scontrol.txt"
    dump.write_text(SCONTROL_SAMPLE, encoding="utf-8")
    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_worker1_gpu_gate.py"),
            "--scontrol-dump",
            str(dump),
            "--candidate-job",
            "47999999",
            "--root-approval",
            APPROVAL,
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 1
    assert not json.loads(out.stdout)["allow"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
