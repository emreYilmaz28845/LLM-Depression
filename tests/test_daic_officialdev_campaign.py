from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MATRIX_PATH = PROJECT_ROOT / "configs/experiments/daic_officialdev/matrix.yaml"
LAUNCHER = PROJECT_ROOT / "scripts/submit_daic_officialdev.sh"
RETRY_LAUNCHER = PROJECT_ROOT / "scripts/submit_daic_officialdev_retry.sh"
PREFLIGHT_SCRIPT = PROJECT_ROOT / "scripts/prepare_daic_officialdev_mn5.py"
RUN_AUDIT_SCRIPT = PROJECT_ROOT / "scripts/audit_daic_officialdev_run.py"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def test_matrix_declares_the_locked_campaign_scope() -> None:
    matrix = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
    assert matrix["seed"] == 1337
    assert matrix["folds"] == [0]
    assert matrix["fixed_heads"] == ["logreg_raw", "xgb_raw"]
    assert matrix["checkpoint_selection"] == "inner_val_macro_f1"
    assert matrix["separate_final_eval"] is True
    assert matrix["run_final_eval_in_train"] is False
    assert matrix["evaluation_view"] == "harmonized_all_windows_full_coverage"
    assert matrix["principal_jobs"] == 24
    assert matrix["max_train_lanes"] == 6
    assert matrix["max_aux_lanes"] == 12 or matrix["max_aux_lanes"] == 6
    assert matrix["expected_counts"]["audio"] == {
        "fit_rows": 1312, "fit_subjects": 86, "dev_rows": 603, "dev_subjects": 35,
    }
    assert matrix["expected_counts"]["text"] == {
        "fit_rows": 86, "fit_subjects": 86, "dev_rows": 35, "dev_subjects": 35,
    }
    experiments = matrix["experiments"]
    assert len(experiments) == 6
    assert {item["backbone"] for item in experiments} == {"qwen", "gemma4"}
    assert {item["modality"] for item in experiments} == {"audio_only", "audio_text", "text_only"}
    for item in experiments:
        assert (PROJECT_ROOT / item["config"]).is_file()
        config = yaml.safe_load((PROJECT_ROOT / item["config"]).read_text(encoding="utf-8"))
        assert config["split"]["final_eval_partition"] == "val"
        assert "selection_partition" not in config["split"]
        assert config["training"]["run_final_eval_in_train"] is False
        assert config["evaluation"]["evaluation_view"] == "harmonized_all_windows_full_coverage"


def test_launcher_dry_run_describes_exactly_24_jobs_without_mutation() -> None:
    run_id = "test_dry_run_000001"
    submissions = PROJECT_ROOT / "outputs/daic_officialdev_submissions" / run_id
    contexts = PROJECT_ROOT / "outputs/daic_officialdev_experiment_contexts" / run_id
    for path in (submissions, contexts):
        if path.exists():
            import shutil

            shutil.rmtree(path)
    env = {
        **__import__("os").environ,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "RUN_ID": run_id,
        "DRY_RUN": "1",
        "GITHUB_ISSUE": "88",
        "GITHUB_PR": "89",
    }
    result = _run(["bash", str(LAUNCHER)], env=env)
    assert result.returncode == 0, result.stderr
    dry_lines = [line for line in result.stderr.splitlines() if line.startswith("DRY_RUN")]
    assert len(dry_lines) == 24, f"expected 24 principal jobs in dry run, got {len(dry_lines)}"
    assert "cells=6" in result.stdout
    assert "max_gpus=" in result.stdout
    assert not submissions.exists(), "dry run must not create the submission registry"
    assert not contexts.exists(), "dry run must not create experiment contexts"
    # Production commands must never carry a smoke selection or subject limit.
    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    assert "subject-selection" not in launcher_text
    assert "smoke_subject_limit" not in launcher_text
    assert "smoke" not in launcher_text.lower()


def test_launcher_dry_run_prints_gpu_allocation_info() -> None:
    env = {
        **__import__("os").environ,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "RUN_ID": "test_allocation_000001",
        "DRY_RUN": "1",
        "GITHUB_ISSUE": "88",
        "GITHUB_PR": "89",
        "MAX_CONCURRENT_TRAINS": "6",
        "MAX_CONCURRENT_AUX": "18",
    }
    result = _run(["bash", str(LAUNCHER)], env=env)
    assert result.returncode == 0, result.stderr
    assert "currently_allocated=" in result.stdout
    assert "requested_by_campaign=42" in result.stdout


def test_launcher_allows_large_lane_shape() -> None:
    env = {
        **__import__("os").environ,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "RUN_ID": "test_lanes_000001",
        "DRY_RUN": "1",
        "GITHUB_ISSUE": "88",
        "GITHUB_PR": "89",
        "MAX_CONCURRENT_TRAINS": "20",
        "MAX_CONCURRENT_AUX": "12",
    }
    result = _run(["bash", str(LAUNCHER)], env=env)
    assert result.returncode == 0, result.stderr
    assert "max_gpus=92" in result.stdout


def test_workers_have_offline_flags_and_heads_has_no_gpu() -> None:
    for script in (
        "scripts/run_daic_officialdev_extract_slurm.sh",
        "scripts/run_daic_officialdev_heads_slurm.sh",
    ):
        result = _run(["bash", "-n", str(PROJECT_ROOT / script)])
        assert result.returncode == 0, f"{script}: {result.stderr}"
        text = (PROJECT_ROOT / script).read_text(encoding="utf-8")
        assert "HF_HUB_OFFLINE=1" in text
        assert "TRANSFORMERS_OFFLINE=1" in text
        assert "HF_DATASETS_OFFLINE=1" in text
        assert "TOKENIZERS_PARALLELISM=false" in text
        for forbidden in ("huggingface-cli", "pip ", "git clone", "wget ", "curl "):
            assert forbidden not in text, f"{script} contains {forbidden!r}"
    for script in ("scripts/submit_daic_officialdev.sh", "scripts/submit_daic_officialdev_retry.sh",
                   "scripts/run_daic_officialdev_extract_slurm.sh", "scripts/submit_daic_officialdev_preflight.sh",
                   "scripts/run_daic_officialdev_preflight_slurm.sh"):
        result = _run(["bash", "-n", str(PROJECT_ROOT / script)])
        assert result.returncode == 0, f"{script}: {result.stderr}"
    heads = (PROJECT_ROOT / "scripts/run_daic_officialdev_heads_slurm.sh").read_text(encoding="utf-8")
    assert "--gres=gpu" not in heads
    extract = (PROJECT_ROOT / "scripts/run_daic_officialdev_extract_slurm.sh").read_text(encoding="utf-8")
    assert "--gres=gpu:1" in extract
    assert "gemma4_12b_tf5_14_1" in extract or "ENV_ACTIVATE" in extract
    heads_text = heads
    assert "qwen_mn5_rebuilt" in heads_text
    assert ".deps/qwen_hidden" in heads_text


def test_workbook_tables_are_filled_and_self_consistent() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import importlib

    builder = importlib.import_module("build_clean_workbook")
    # The DAIC official-development tables carry exactly the locked campaign
    # scope: six teacher-forced cells and twelve fixed-head cells, all with
    # registry-derived provenance (attempt + evaluation ids).
    assert len(builder.DAIC_OFFICIALDEV_TEACHER_FORCED) == 6
    assert len(builder.DAIC_OFFICIALDEV_HEADS) == 12
    for value in builder.DAIC_OFFICIALDEV_TEACHER_FORCED.values():
        assert value["attempt_id"].startswith("20")
        assert value["evaluation_id"].startswith("eval-")
        assert value["support"] == 35 if "support" in value else True
    for value in builder.DAIC_OFFICIALDEV_HEADS.values():
        assert value["attempt_id"].startswith("20")
        assert value["evaluation_id"].startswith("eval-")
    assert builder.DAIC_OFFICIALDEV_CAMPAIGN["campaign_id"] is not None
    assert len(builder.DAIC_OFFICIALDEV_LITERATURE) == 5


def test_run_audit_rejects_missing_registry(tmp_path: Path) -> None:
    env = {
        **__import__("os").environ,
        "PROJECT_ROOT": str(tmp_path),
    }
    result = _run(
        [
            sys.executable, str(RUN_AUDIT_SCRIPT),
            "--run-id", "missing",
            "--use-live-sacct",
            "--submissions-root", str(tmp_path / "subs"),
        ],
        env=env,
    )
    assert result.returncode == 1
    assert "missing submission registry" in result.stderr


def test_preflight_script_writes_job_scope_24(tmp_path: Path) -> None:
    # Model-free against the real canonical manifest, rebuilt so the metadata
    # carries the officialdev build signature.
    run_id = "test_preflight_000001"
    output = tmp_path / "audit.json"
    env = {
        **__import__("os").environ,
        "DAIC_UNPROCESSED_ROOT": "/media/emre/Backup/AudioLLM/Datasets/DAIC-WOZ/unprocessed",
        "DAIC_LABEL_ROOT": "/media/emre/Backup/AudioLLM/Datasets/DAIC-WOZ/minimal_zips",
    }
    result = _run(
        [
            sys.executable, str(PREFLIGHT_SCRIPT),
            "--run-id", run_id,
            "--build",
            "--output", str(output),
        ],
        env=env,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record["job_scope"]["principal_jobs"] == 24
    assert len(record["configs"]) == 6
    split_audit = record["split_audit"]
    assert split_audit["inner_split_counts"] == {"train_inner": 86, "val_inner": 21}
    assert split_audit["row_counts"]["audio_only"]["fit_rows"] == 1312
    assert split_audit["row_counts"]["audio_only"]["dev_rows"] == 603
    assert split_audit["disjointness"]["official_test_absent"] is True


def test_retry_launcher_syntax_and_refuses_unknown_kind(tmp_path: Path) -> None:
    cells = tmp_path / "cells.tsv"
    cells.write_text("qwen\taudio_text\twat\t123\n", encoding="utf-8")
    env = {
        **__import__("os").environ,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "RUN_ID": "test_retry_000001",
        "CELLS": str(cells),
        "DRY_RUN": "1",
        "GITHUB_ISSUE": "88",
        "GITHUB_PR": "89",
    }
    result = _run(["bash", str(RETRY_LAUNCHER)], env=env)
    assert result.returncode == 2
    assert "kind must be train, eval, extract, or heads" in result.stderr
