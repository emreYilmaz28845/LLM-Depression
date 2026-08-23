from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.monitor_experiment import (
    build_collect_command,
    deployed_source_manifest,
    expected_compact_evidence,
    format_plan,
    parse_sacct_output,
    parse_sbatch_parsable,
    parse_squeue_output,
    plan_matrix,
    source_manifest_sha256,
    terminal_job_states,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_normalized_native_text_head_collection_path_does_not_mutate_contract():
    from tools.exp import _normalized_local_evidence_rel

    contract = {
        "local_evidence_rel": "output_model/output_model/campaign/text_only/d3tec/run/fold_2",
    }
    assert _normalized_local_evidence_rel(contract) == (
        "output_model/campaign/text_only/d3tec/run/fold_2"
    )
    assert contract["local_evidence_rel"].startswith("output_model/output_model/")

SQUEUE_FIXTURE = """JOBID STATE TIME NODELIST JOBNAME REASON
1843921 RUNNING 01:23:45 as01r2b25 llm-depression-train None
1843922 PENDING 00:00:00 (null) llm-depression-eval Dependency
"""

SACCT_FIXTURE = """JobIDRaw JobName State ExitCode Elapsed MaxRSS AllocCPUS NodeList
1843921 llm-depression-train COMPLETED 0:0 01:23:45 20.00G 80 as01r2b25
1843921.0 llm-depression-train COMPLETED 0:0 01:23:45 20.00G 80 as01r2b25
1843922 llm-depression-eval FAILED 1:0 00:01:00 1.00G 20 as01r2b24
1843923 llm-depression-eval CANCELLED 0:15 00:00:30 0.00G 20 as01r2b26
"""

SBATCH_PARSABLE_FIXTURE = "1843924;cluster.regular\n"


def test_parse_sbatch_parsable() -> None:
    assert parse_sbatch_parsable(SBATCH_PARSABLE_FIXTURE) == "1843924"
    with pytest.raises(ValueError):
        parse_sbatch_parsable("not-a-job-id\n")


def test_parse_squeue_output_fixture() -> None:
    rows = parse_squeue_output(SQUEUE_FIXTURE)
    assert len(rows) == 2
    assert rows[0]["job_id"] == "1843921"
    assert rows[0]["state"] == "RUNNING"
    assert rows[1]["reason"] == "Dependency"


def test_parse_sacct_output_fixture() -> None:
    rows = parse_sacct_output(SACCT_FIXTURE)
    assert len(rows) == 4
    assert rows[1]["job_id"] == "1843921.0"
    assert rows[2]["state"] == "FAILED"
    assert rows[2]["exit_code"] == "1:0"
    assert terminal_job_states(rows) == ["CANCELLED", "COMPLETED", "FAILED"]


def test_plan_matrix_counts_jobs() -> None:
    cells = plan_matrix(logical_run_names=["daic_rotary_k4_seed1337"], seeds=[7, 1337], folds=[0])
    assert len(cells) == 2
    assert sum(cell["job_count"] for cell in cells) == 4
    assert cells[0]["jobs"] == ["train", "evaluation"]


def test_format_plan_lists_endpoints_and_policies() -> None:
    plan = format_plan(
        plan_matrix(logical_run_names=["daic_rotary_k4_seed1337"], seeds=[7], folds=[0]),
        git_commit="1c2344f1d33e301978549748c5bf936319a43db6",
        git_dirty=False,
    )
    assert plan["total_jobs"] == 2
    assert plan["endpoint_split"]["scheduler"] == "ozu647717@alogin2.bsc.es"
    assert "no --delete" in plan["rsync_policy"]


def test_deployed_source_manifest_uses_git_ls_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)
    records = deployed_source_manifest(tmp_path)
    assert [record["path"] for record in records] == ["a.py", "b.txt"]
    assert records[0]["size_bytes"] == 6
    assert source_manifest_sha256(records) == source_manifest_sha256(list(reversed(records)))


def test_collect_command_is_dry_run_and_never_deletes(tmp_path: Path) -> None:
    command = build_collect_command(
        "/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/output_model/audio_text/daic/run/fold_0",
        str(tmp_path),
        dry_run=True,
    )
    assert "-n" in command
    assert "--delete" not in command
    assert "--exclude" in command
    assert "best_model/" in command
    assert any("transfer1" in token for token in command)


def test_expected_compact_evidence_lists(tmp_path: Path) -> None:
    fold_dir = tmp_path / "fold_0"
    (fold_dir / "logs").mkdir(parents=True)
    for name in ("run_config.yaml", "metadata.json", "status.json", "jobs.jsonl"):
        (fold_dir / name).write_text("{}", encoding="utf-8")
    (fold_dir / "logs" / "training_history.json").write_text("[]", encoding="utf-8")
    (fold_dir / "logs" / "selected_checkpoint_selection_metrics.json").write_text("{}", encoding="utf-8")
    (fold_dir / "logs" / "final_eval_truncation.jsonl").write_text("", encoding="utf-8")
    standalone = fold_dir / "best_model" / "standalone_eval"
    standalone.mkdir(parents=True)
    (standalone / "metrics_original_teacher_forced.json").write_text("{}", encoding="utf-8")
    (standalone / "predictions_subject_level.csv").write_text("a\n", encoding="utf-8")
    files = expected_compact_evidence(fold_dir)
    assert "run_config.yaml" in files
    assert "logs/training_history.json" in files
    assert "logs/final_eval_truncation.jsonl" in files
    assert "best_model/standalone_eval/metrics_original_teacher_forced.json" in files
    assert "best_model/standalone_eval/predictions_subject_level.csv" in files


def _run_script(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )


def test_plan_action_prints_matrix_and_endpoints() -> None:
    group_def = PROJECT_ROOT / "experiments" / "schemas" / "experiment_group.v1.schema.json"
    config = PROJECT_ROOT / "configs" / "main" / "daic_audio_text_harmonized_selmacrof1_tf.yaml"
    result = _run_script(
        "submit_experiment.sh",
        "plan",
        "--group",
        str(group_def),
        "--config",
        str(config),
        "--seeds",
        "7 1337",
        "--folds",
        "0",
        "--issue",
        "86",
        "--pr",
        "91",
    )
    assert result.returncode == 0, result.stderr
    assert "total_jobs: 4" in result.stdout
    assert "transfer=ozu647717@transfer1.bsc.es" in result.stdout
    assert "alogin2" in result.stdout
    assert "harmonized configs select best_model by inner_val_macro_f1" in result.stdout
    assert "attempt_ids: minted at deploy time" in result.stdout


def test_mutating_actions_refuse_without_authorization() -> None:
    for action in ("deploy", "submit", "collect"):
        result = _run_script("submit_experiment.sh", action, "--plan", "x.json")
        assert result.returncode != 0
        assert "refusing" in result.stderr or "requires" in result.stderr


def test_deploy_requires_plan_file() -> None:
    result = _run_script("submit_experiment.sh", "deploy", "--authorized")
    assert result.returncode != 0


def test_deprecated_mutating_action_never_reports_success_when_authorized() -> None:
    result = _run_script("submit_experiment.sh", "deploy", "--plan", "x.json", "--authorized")
    assert result.returncode != 0
    assert "never performs remote mutation" in result.stderr


def test_collect_dry_run_prints_compact_evidence_command() -> None:
    result = _run_script(
        "collect_experiment.sh",
        "--fold-dir",
        "/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/output_model/c/m/d/run/fold_0",
        "--output",
        "/tmp/opencode/collect",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert "rsync" in result.stdout
    assert "--delete" not in result.stdout
    assert "best_model/standalone_eval/**" in result.stdout
    assert "exclude=best_model/**" in result.stdout


def test_collect_refuses_placeholder_fold_paths(tmp_path=None) -> None:
    result = _run_script(
        "collect_experiment.sh",
        "--fold-dir",
        "/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/output_model/<modality>/<dataset>/run/fold_0",
        "--output",
        "/tmp/opencode/collect",
    )
    assert result.returncode != 0
    assert "placeholder" in result.stderr
