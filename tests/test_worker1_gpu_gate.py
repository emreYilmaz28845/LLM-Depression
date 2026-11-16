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
    TresParseError,
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
    t, present = parse_tres("billing=320,cpu=320,gres/gpu=8,mem=1000000M,node=2")
    assert present
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


# ---------------------------------------------------------------------------
# Focused regression tests for the reviewed gate defects (2026-10-09).
# ---------------------------------------------------------------------------


def test_deny_candidate_pending_unheld():
    cand = audio_post(reason="None")
    dec = check_release(cand, [], APPROVAL)
    assert not dec["allow"]
    assert any("candidate_must_be_pending_with_exact_JobHeldUser" in r for r in dec["reasons"])


def test_deny_candidate_pending_other_reason():
    cand = audio_post(reason="Resources")
    dec = check_release(cand, [], APPROVAL)
    assert not dec["allow"]
    assert any("candidate_must_be_pending_with_exact_JobHeldUser" in r for r in dec["reasons"])


def test_deny_candidate_missing_req_tres():
    cand = audio_post()
    cand.req_present = False
    dec = check_release(cand, [], APPROVAL)
    assert not dec["allow"]
    assert any("candidate_req_tres_missing" in r for r in dec["reasons"])


def test_deny_candidate_nodes_zero():
    cand = audio_post()
    cand.req = Tres(gpus=4, nodes=0, billing=80)
    dec = check_release(cand, [], APPROVAL)
    assert not dec["allow"]
    assert any("candidate_nodes_missing_or_zero" in r for r in dec["reasons"])


def test_deny_candidate_billing_zero():
    cand = audio_post()
    cand.req = Tres(gpus=4, nodes=1, billing=0)
    dec = check_release(cand, [], APPROVAL)
    assert not dec["allow"]
    assert any("candidate_billing_missing_or_zero" in r for r in dec["reasons"])


def test_parse_tres_malformed_raises():
    with pytest.raises(TresParseError):
        parse_tres("gres/gpu=abc,node=1,billing=80")
    with pytest.raises(TresParseError):
        parse_tres("node=1,billing=,gres/gpu=4")
    with pytest.raises(TresParseError):
        parse_tres("not_a_keyvalue")


def test_parse_rejects_incomplete_record():
    incomplete = "JobId=47077999 JobName=x\n   JobState=PENDING Reason=JobHeldUser\n   NumNodes=1\n"
    with pytest.raises(TresParseError):
        parse_scontrol_dump(incomplete)


def test_parse_rejects_duplicate_job_records():
    dup = SCONTROL_SAMPLE + "\n" + SCONTROL_SAMPLE
    with pytest.raises(TresParseError):
        parse_scontrol_dump(dup)


def test_cli_denies_malformed_dump(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text(SCONTROL_SAMPLE.replace("gres/gpu=4", "gres/gpu=four"), encoding="utf-8")
    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_worker1_gpu_gate.py"),
            "--scontrol-dump",
            str(bad),
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
    assert out.returncode == 1
    assert "snapshot_unusable" in json.loads(out.stdout)["reasons"][0]


def test_cli_denies_duplicate_snapshot(tmp_path):
    dup = tmp_path / "dup.txt"
    dup.write_text(SCONTROL_SAMPLE + "\n" + SCONTROL_SAMPLE, encoding="utf-8")
    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_worker1_gpu_gate.py"),
            "--scontrol-dump",
            str(dup),
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
    assert out.returncode == 1
    assert "duplicate_job_records" in json.loads(out.stdout)["reasons"][0]


def test_allow_gpu_candidate_one_node_boundary():
    cand = audio_post()
    dec = check_release(cand, [], APPROVAL)
    assert dec["allow"], dec["reasons"]


# ---------------------------------------------------------------------------
# JobHeldAdmin peer handling + stricter nonterminal/duplicate checks (2026-10-09
# root review round 2).
# ---------------------------------------------------------------------------


def admin_held_peer(job_id="47100001", priority=0, gpus=8):
    return Job(
        job_id=job_id,
        name="peer-train",
        state="PENDING",
        reason="JobHeldAdmin",
        priority=priority,
        req=Tres(gpus=gpus, nodes=2, billing=160),
        time_limit_minutes=4320,
    )


def test_heldadmin_peer_with_priority_zero_accepted():
    dec = check_release(audio_post(), [admin_held_peer()], APPROVAL)
    assert dec["allow"], dec["reasons"]


def test_heldadmin_candidate_refused():
    cand = admin_held_peer(job_id="47077944")
    dec = check_release(cand, [], APPROVAL)
    assert not dec["allow"]
    assert any("candidate_must_be_pending_with_exact_JobHeldUser" in r for r in dec["reasons"])


def test_heldadmin_peer_without_priority_zero_refused():
    dec = check_release(audio_post(), [admin_held_peer(priority=None)], APPROVAL)
    assert not dec["allow"]
    assert any("own_admin_hold_without_priority_zero" in r for r in dec["reasons"])
    dec2 = check_release(audio_post(), [admin_held_peer(priority=5)], APPROVAL)
    assert not dec2["allow"]
    assert any("own_admin_hold_without_priority_zero" in r for r in dec2["reasons"])


def test_unknown_reason_peer_refused():
    peer = audio_train(job_id="47100002", state="PENDING", reason="Resources")
    dec = check_release(audio_post(), [peer], APPROVAL)
    assert not dec["allow"]
    assert any(
        ("own_unheld_pending_gpu_job" in r) or ("own_pending_unknown_reason" in r)
        for r in dec["reasons"]
    )
    cpu_peer = Job(
        job_id="47100003",
        state="PENDING",
        reason="Resources",
        req=Tres(gpus=0, nodes=1, billing=20),
        time_limit_minutes=30,
    )
    dec2 = check_release(audio_post(), [cpu_peer], APPROVAL)
    assert not dec2["allow"]
    assert any("own_pending_unknown_reason" in r for r in dec2["reasons"])


def test_other_nonterminal_missing_nodes_refused():
    peer = audio_train()
    peer.req = Tres(gpus=8, nodes=0, billing=160)
    dec = check_release(audio_post(), [peer], APPROVAL)
    assert not dec["allow"]
    assert any("other_nonterminal_invalid_tres" in r for r in dec["reasons"])


def test_other_nonterminal_missing_billing_refused():
    peer = audio_train()
    peer.req = Tres(gpus=8, nodes=2, billing=0)
    dec = check_release(audio_post(), [peer], APPROVAL)
    assert not dec["allow"]
    assert any("other_nonterminal_invalid_tres" in r for r in dec["reasons"])


def test_duplicate_tres_keys_fail():
    with pytest.raises(TresParseError):
        parse_tres("node=1,node=2,billing=80,gres/gpu=4")
    with pytest.raises(TresParseError):
        parse_tres("billing=80,billing=160,node=1,gres/gpu=4")


def test_duplicate_field_in_block_fail():
    block = (
        "JobId=47077944 JobName=x\n"
        "   JobState=PENDING Reason=JobHeldUser Priority=0 Priority=5\n"
        "   ReqTRES=cpu=80,node=1,billing=80,gres/gpu=4\n"
        "   TimeLimit=1-00:00:00\n"
    )
    with pytest.raises(TresParseError):
        parse_scontrol_dump(block)


HELDADMIN_SAMPLE = """
JobId=47100001 JobName=peer-train
   UserId=ozu647717(53836) GroupId=ozu(52230) MCS_label=N/A
   JobState=PENDING Reason=JobHeldAdmin Priority=0 Dependency=(null)
   Partition=acc AllocNode:Sid=alogin2:3056621
   NumNodes=2 NumCPUs=160
   ReqTRES=cpu=160,mem=1000000M,node=2,billing=160,gres/gpu=8
   AllocTRES=(null)
   TimeLimit=3-00:00:00
   
JobId=47077944 JobName=sym-audi-cv-0-post
   UserId=ozu647717(53836) GroupId=ozu(52230) MCS_label=N/A
   JobState=PENDING Reason=JobHeldUser Priority=0 Dependency=(null)
   Partition=acc AllocNode:Sid=alogin2:3056621
   NumNodes=1 NumCPUs=80
   ReqTRES=cpu=80,mem=500000M,node=1,billing=80,gres/gpu=4
   AllocTRES=(null)
   TimeLimit=1-00:00:00
"""


def test_cli_accepts_with_heldadmin_peer(tmp_path):
    dump = tmp_path / "queue.txt"
    dump.write_text(HELDADMIN_SAMPLE, encoding="utf-8")
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
