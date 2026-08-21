from __future__ import annotations

import json
import base64
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.submit import (
    SubmissionError,
    build_common_overrides,
    build_remote_submit_script,
    check_collisions,
    decode_overrides,
    encode_overrides,
    parse_submitted_job_ids,
    resolve_contract,
)
from src.experiment_tracking.identity import validate_attempt_id


def _deployment(commit="a" * 40):
    return {
        "schema_version": "audiollm.deployment.v1",
        "deployment_id": "exp-x-20260821T000000Z-abcdef01-12345678",
        "experiment_id": "exp-x",
        "git_commit": commit,
        "git_branch_at_deploy": "agent/exp-x",
        "git_dirty": False,
        "source_manifest_sha256": "b" * 64,
        "uncommitted_patch_sha256": "c" * 64,
        "deployed_code_path": "/gpfs/AudioLLM/deployments/exp-x-20260821T000000Z-abcdef01-12345678/code",
        "created_at_utc": "2026-08-21T00:00:00Z",
    }


def _config(**eval_kwargs):
    evaluation = {
        "sample_prediction_mode": "original_teacher_forced",
        "aggregation_level": "subject",
        "evaluation_view": "harmonized_all_windows_full_coverage",
    }
    evaluation.update(eval_kwargs)
    return {
        "dataset": "daic",
        "evaluation": evaluation,
        "output_dirs": {"run_root": "${PROJECT_ROOT}/output_model/harmonized_v1"},
    }


BASE_KW = dict(
    experiment_id="exp-x",
    config_path_remote="/gpfs/AudioLLM/deployments/d1/code/configs/main/x.yaml",
    fold=0,
    seed=1337,
    run_name="smoke_run",
    campaign="parallel_workflow_smoke_v2",
    modality="audio_text",
    dataset="daic",
    extra_overrides=["--set=training.num_train_epochs=1", "--set=split.smoke_subject_limit=6"],
)


def test_common_overrides_include_path_contract_and_extras():
    tokens = build_common_overrides(
        manifest_dir="/rt/manifests/daic",
        split_dir="/rt/splits/daic",
        run_root="/out/output_model/c/m/d",
        extra_overrides=["--set=training.num_train_epochs=1"],
    )
    assert "--set=output_dirs.manifest_dir=/rt/manifests/daic" in tokens
    assert "--set=output_dirs.split_dir=/rt/splits/daic" in tokens
    assert "--set=output_dirs.run_root=/out/output_model/c/m/d" in tokens
    assert "--set=training.num_train_epochs=1" in tokens


def test_override_roundtrip_is_lossless_with_spaces_and_shell_metacharacters():
    tricky = [
        "--set=transcripts.cache_path=/gpfs/some dir with spaces/file.jsonl",
        "--set=prompt.prefix=hello; rm -rf $HOME | `whoami` && echo done",
        "--set=nested.a.b=c=d=e",
        "--set=quoted='single \"double\" quotes'",
    ]
    decoded = decode_overrides(encode_overrides(tricky))
    assert decoded == tricky


def test_resolve_contract_paths_follow_writable_contract():
    contract = resolve_contract(deployment=_deployment(), config_dict=_config(), **BASE_KW)
    exp_id = BASE_KW["experiment_id"]
    assert contract["manifest_dir"] == f"/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime/{exp_id}/manifests/daic"
    assert contract["split_dir"] == f"/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime/{exp_id}/splits/daic"
    assert contract["run_root"] == "/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/output_model/parallel_workflow_smoke_v2/audio_text/daic"
    assert contract["fold_dir"] == contract["run_root"] + "/smoke_run/fold_0"
    assert contract["checkpoint_dir"] == contract["fold_dir"] + "/best_model"
    assert contract["standalone_eval_dir"] == contract["checkpoint_dir"] + "/standalone_eval"
    # identical override array for train and eval is enforced by construction
    assert contract["overrides"][0].startswith("--set=output_dirs.manifest_dir=")
    assert decode_overrides(contract["overrides_b64"]) == contract["overrides"]


def test_missing_evaluation_view_fails_closed():
    cfg = _config(evaluation_view=None)
    with pytest.raises(SubmissionError, match="evaluation_view"):
        resolve_contract(deployment=_deployment(), config_dict=cfg, **BASE_KW)


def test_dataset_mismatch_fails():
    with pytest.raises(SubmissionError, match="dataset"):
        resolve_contract(deployment=_deployment(), config_dict=_config(),
                         **{**BASE_KW, "dataset": "cmdc"})


def test_job_graph_preserves_gpu_shapes():
    contract = resolve_contract(deployment=_deployment(), config_dict=_config(), **BASE_KW)
    train, ev = contract["job_graph"]
    assert "4 H100" in train["shape"] and "DDP" in train["shape"]
    assert "1 H100" in ev["shape"]
    assert ev["depends_on"] == ["train"]
    assert ev["checkpoint_dir"].endswith("/best_model")
    assert ev["output_dir"].endswith("/best_model/standalone_eval")


def test_attempt_ids_are_never_reused():
    c1 = resolve_contract(deployment=_deployment(), config_dict=_config(), **BASE_KW)
    c2 = resolve_contract(deployment=_deployment(), config_dict=_config(), **BASE_KW)
    assert c1["attempt_id"] != c2["attempt_id"]
    assert validate_attempt_id(c1["attempt_id"])
    assert c1["context"]["source"]["deployed_source_sha256"] == "b" * 64


def test_supersedes_link_recorded():
    contract = resolve_contract(
        deployment=_deployment(), config_dict=_config(),
        supersedes_attempt_id="20260820T212938Z-old-run-abcdef01-00112233", **BASE_KW)
    assert contract["context"]["supersedes_attempt_id"] == "20260820T212938Z-old-run-abcdef01-00112233"


def test_collision_checks_fail_before_sbatch():
    contract = resolve_contract(deployment=_deployment(), config_dict=_config(), **BASE_KW)
    check_collisions(contract, lambda p: False)  # no collision passes
    with pytest.raises(SubmissionError, match="fold dir exists"):
        check_collisions(contract, lambda p: p.endswith("fold_0"))
    with pytest.raises(SubmissionError, match="attempt id reuse"):
        check_collisions(contract, lambda p: "contexts/" in p)
    with pytest.raises(SubmissionError, match="standalone eval dir exists"):
        check_collisions(contract, lambda p: p.endswith("standalone_eval"))


def test_remote_script_quotes_every_value():
    import shlex
    kw = {**BASE_KW, "extra_overrides": ['--set=weird.path=value with spaces;$(dangerous)`bt`']}
    contract = resolve_contract(deployment=_deployment(), config_dict=_config(), **kw)
    script = build_remote_submit_script(contract)
    assert "set -euo pipefail" in script.splitlines()[0]
    assert f"cd {shlex.quote(contract['deployed_code_path'])}" in script
    # the b64 payload is a single shell-safe token
    b64_line = [l for l in script.splitlines() if l.startswith("export OVERRIDES_JSON_B64=")][0]
    token = b64_line.split("=", 1)[1]
    assert all(c.isalnum() or c in "+/=_" for c in token.strip("'"))
    assert decode_overrides(token.strip("'")) == contract["overrides"]
    # the dangerous payload only appears inside the quoted EXTRA_*_ARGS value and the b64 blob
    stripped = script.replace(shlex.quote(" ".join(contract["overrides"])), "<OVERRIDES>")
    assert "rm -rf" not in stripped


def test_parse_submitted_job_ids():
    out = "Submitting workflow...\nSubmitted training job: 12345\nSubmitted best-checkpoint eval job: 12346\n"
    ids = parse_submitted_job_ids(out)
    assert ids == {"train": "12345", "best_eval": "12346"}
    assert parse_submitted_job_ids("nothing here") == {}


def test_state_tool_record_job_appends_atomically(tmp_path):
    state_path = tmp_path / "state.json"
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools" / "parallel_workflow_state.py"),
         "init", "--runbook", "docs/PARALLEL_EXPERIMENT_WORKFLOW_PLAN.md",
         "--execution-id", "test-exec", "--output", str(state_path)],
        check=True, capture_output=True,
    )
    cmd = [sys.executable, str(PROJECT_ROOT / "tools" / "parallel_workflow_state.py"),
           "record-job", "--state", str(state_path),
           "--attempt-id", "20260821T000000Z-run-abcdef01-12345678",
           "--job-key", "train", "--job-type", "train", "--event-type", "SUBMITTED",
           "--slurm-job-id", "999", "--status", "PENDING", "--fold", "0",
           "--deployment-id", "dep1", "--evaluation-view", "v1",
           "--backend", "original_teacher_forced", "--aggregation", "subject_level"]
    subprocess.run(cmd, check=True, capture_output=True)
    subprocess.run(cmd + ["--slurm-job-id", "1000", "--job-key", "best_eval",
                          "--job-type", "evaluation", "--dependency-job-ids", "999"], check=True, capture_output=True)
    state = json.loads(state_path.read_text())
    assert len(state["jobs"]) == 2
    assert state["jobs"][1]["dependency_job_ids"] == "999"
    assert state["jobs"][0]["event_type"] == "SUBMITTED"


def test_worker_scripts_decode_overrides_json_b64():
    for script in ("scripts/run_train_slurm.sh", "scripts/run_eval_slurm.sh"):
        text = (PROJECT_ROOT / script).read_text()
        assert "OVERRIDES_JSON_B64" in text
        assert "base64.b64decode" in text
        assert 'OVERRIDE_ARGS' in text
    wrapper = (PROJECT_ROOT / "scripts" / "submit_train_and_eval.sh").read_text()
    assert "OVERRIDES_JSON_B64" in wrapper
