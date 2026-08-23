from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml
import pytest

from scripts.prepare_harmonized_en_mn5 import (
    EN_RECIPE,
    equivalence_audit,
    recipe_and_scope_audit,
    validate_component,
)


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"
MATRIX = ROOT / "configs/experiments/harmonized/english_translation_matrix.yaml"

EN_CONFIGS = {
    "d3tec": {
        "audio_text": "d3tec_audio_text_harmonized_selmacrof1_tf_en.yaml",
        "text_only": "d3tec_text_only_harmonized_selmacrof1_tf_en.yaml",
    },
    "androids_interview": {
        "audio_text": "androids_audio_text_harmonized_selmacrof1_tf_en.yaml",
        "text_only": "androids_text_only_harmonized_selmacrof1_tf_en.yaml",
    },
    "cmdc": {
        "audio_text": "cmdc_audio_text_harmonized_selmacrof1_tf_en.yaml",
        "text_only": "cmdc_text_only_harmonized_selmacrof1_tf_en.yaml",
    },
    "turkish": {
        "audio_text": "turkish_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr_en.yaml",
        "text_only": "turkish_t17_text_only_harmonized_selmacrof1_tf_qwen3asr_en.yaml",
    },
}


def en_config_paths() -> list[Path]:
    return sorted(
        path
        for path in MAIN.glob("*harmonized_selmacrof1_tf*_en.yaml")
        if "turkish_negative_only" not in path.name
    )


def test_exactly_eight_english_configs_exist() -> None:
    paths = en_config_paths()
    assert len(paths) == 8
    expected = {
        name for modality in EN_CONFIGS.values() for name in modality.values()
    }
    assert {path.name for path in paths} == expected


def test_english_configs_are_derived_only_from_native_bases() -> None:
    for path in en_config_paths():
        native = path.name[: -len("_en.yaml")] + ".yaml"
        native_path = MAIN / native
        assert native_path.is_file(), f"no native base {native_path}"
        base = yaml.safe_load(native_path.read_text(encoding="utf-8"))
        en = yaml.safe_load(path.read_text(encoding="utf-8"))
        allowed = {"recipe_id", "transcripts", "output_dirs"}
        base_minus = {k: v for k, v in base.items() if k not in allowed}
        en_minus = {k: v for k, v in en.items() if k not in allowed}
        assert base_minus == en_minus, f"non-allowed field changed in {path}"
        assert en["recipe_id"] == EN_RECIPE
        assert en["prompt"] == base["prompt"]
        assert en["split"] == base["split"]
        assert en["data"] == base["data"]
        assert en["training"] == base["training"]
        assert en["evaluation"] == base["evaluation"]


def test_no_english_audio_only_or_daic_or_edaic_config() -> None:
    assert not list(MAIN.glob("*audio_only*_en.yaml"))
    assert not list(MAIN.glob("daic*_en.yaml"))
    assert not list(MAIN.glob("edaic*_en.yaml"))
    for path in en_config_paths():
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["dataset"] not in {"daic", "edaic"}
        data = config["data"]
        assert (data["use_audio"] and data["use_text"]) or (data["use_text"] and not data["use_audio"])


def test_english_recipe_invariants_and_transcripts_policy() -> None:
    for path in en_config_paths():
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["training"]["num_train_epochs"] == 20
        assert config["training"]["selection_metric"] == "inner_val_macro_f1"
        assert config["training"]["selection_metric_mode"] == "max"
        assert config["training"]["early_stopping"]["patience"] == 3
        assert config["evaluation"]["sample_prediction_mode"] == "original_teacher_forced"
        assert config["evaluation"]["headline_mode"] == "original_teacher_forced"
        audio_adapter = config.get("audio_adapter") or {}
        assert not audio_adapter.get("enabled")
        assert not audio_adapter.get("train_projector")
        assert "optuna" not in config
        assert config["quarantine_path"].endswith("configs/quarantines.yaml")
        transcripts = config["transcripts"]
        assert transcripts["variant"] == "english"
        assert transcripts["minimum_status"] == "automatic_low"
        assert transcripts["require_complete"] is True
        assert transcripts["include_failed"] is False
        assert transcripts["cache_path"].endswith("accepted.jsonl")
        assert "harmonized_en_complete_v1" in transcripts["cache_path"]
        assert "${TRANSLATION_ROOT" in transcripts["cache_path"]


def test_english_output_locations_cannot_collide() -> None:
    for path in en_config_paths():
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        run_root = config["output_dirs"]["run_root"]
        assert "/output_model/harmonized_v1_en/" in run_root
        assert "output_model_en" not in run_root
        assert "/harmonized_v1/" not in run_root
        assert "/manifests_harmonized_en/" in config["output_dirs"]["manifest_dir"]
        assert "/splits_harmonized_en/" in config["output_dirs"]["split_dir"]
        assert "/manifests_harmonized/" not in config["output_dirs"]["manifest_dir"]
        assert "/splits_harmonized/" not in config["output_dirs"]["split_dir"]


def test_english_matrix_shape_and_recipe() -> None:
    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    assert matrix["fixed_heads"] == ["logreg_raw", "xgb_raw"]
    assert matrix["max_epochs"] == 20
    assert matrix["checkpoint_selection"] == "inner_val_macro_f1"
    assert matrix["optuna"] is False
    assert matrix["recipe_id"] == EN_RECIPE
    assert len(matrix["experiments"]) == 8
    assert sum(len(item["folds"]) for item in matrix["experiments"]) == 40
    assert sum(len(item["folds"]) for item in matrix["experiments"] if item["separate_eval"]) == 20
    assert not any("audio_only" in item["config"] for item in matrix["experiments"])
    assert not any("daic" in item["config"] for item in matrix["experiments"])


def test_recipe_and_scope_audit_passes_without_dataset_access() -> None:
    result = recipe_and_scope_audit()
    assert result["failures"] == []
    assert result["train_folds"] == 40
    assert result["eval_folds"] == 20
    assert result["hidden_folds"] == 40
    assert result["total_jobs"] == 100


def test_english_launcher_dry_run_has_exactly_100_jobs() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_harmonized_en_standalone.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "PROJECT_ROOT": str(ROOT),
            "RUN_ID": "unit",
            "DRY_RUN": "1",
            "GITHUB_ISSUE": "20",
            "GITHUB_PR": "99",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    commands = [line for line in result.stderr.splitlines() if line.startswith("DRY_RUN sbatch")]
    assert len(commands) == 100
    assert sum("run_train_slurm.sh" in line for line in commands) == 40
    assert sum("run_eval_slurm.sh" in line for line in commands) == 20
    assert sum("run_qwen_hidden_extract_slurm.sh" in line for line in commands) == 40
    assert "max_gpus=200" in result.stdout
    assert "xgb_optuna" not in result.stdout + result.stderr
    assert "github_issue=20 github_pr=99" in result.stdout
    assert "audio_only" not in result.stdout + result.stderr


def test_english_launcher_allows_unlimited_parallelism() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_harmonized_en_standalone.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "PROJECT_ROOT": str(ROOT),
            "RUN_ID": "unit",
            "DRY_RUN": "1",
            "GITHUB_ISSUE": "20",
            "GITHUB_PR": "99",
            "MAX_CONCURRENT_TRAINS": "40",
            "MAX_CONCURRENT_AUX": "40",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "max_gpus=200" in result.stdout


def test_english_launcher_requires_campaign_provenance() -> None:
    env = {
        **os.environ,
        "PROJECT_ROOT": str(ROOT),
        "RUN_ID": "unit",
        "DRY_RUN": "1",
    }
    env.pop("GITHUB_ISSUE", None)
    env.pop("GITHUB_PR", None)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_harmonized_en_standalone.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "GITHUB_ISSUE" in result.stderr


def test_english_launcher_refuses_failed_preflight_audit(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "failed", "run_id": "unit"}), encoding="utf-8")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_harmonized_en_standalone.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "PROJECT_ROOT": str(ROOT),
            "RUN_ID": "unit",
            "DRY_RUN": "0",
            "GITHUB_ISSUE": "20",
            "GITHUB_PR": "99",
            "PREFLIGHT_AUDIT": str(audit),
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "Incompatible English MN5 preflight audit" in result.stderr


def test_experiment_context_payload_records_issue_pr_and_source(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "passed",
                "run_id": "campaign",
                "source_commit": "a" * 40,
                "source_branch": "main",
                "components": [
                    {
                        "dataset": "d3tec",
                        "manifest_file_sha256": "m" * 64,
                        "split_metadata_sha256": "s" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    context_path = tmp_path / "context.json"
    subprocess.run(
        [
            "python", "-",
            str(context_path), str(audit), "campaign", "d3tec", "audio_text", "0",
            "harmonized_v1_en_campaign_d3tec_audio_text", str(ROOT), "20", "99",
            "harmonized-v1-en", "harmonized_v1_en",
        ],
        cwd=ROOT,
        input="""import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[11])
from src.experiment_tracking.identity import new_attempt_id
context_path, audit_path, run_id, dataset, modality, fold, run_name, root, github_issue, github_pr, group_prefix, logical_prefix = sys.argv[1:]
audit = json.load(open(audit_path, encoding="utf-8"))
component = next(item for item in audit["components"] if item["dataset"] == dataset)
commit = str(audit.get("source_commit") or "")
logical = f"{logical_prefix}_{dataset}_{modality}_seed1337"
payload = {
    "schema_version": "audiollm.experiment_context.v1",
    "group_id": f"{group_prefix}-{run_id}",
    "logical_run_name": logical,
    "attempt_id": new_attempt_id(logical, commit),
    "fold": int(fold),
    "seed": 1337,
    "source": {"git_commit": commit, "git_branch": audit.get("source_branch"), "git_dirty": False},
    "research": {"github_issue": int(github_issue), "github_pr": int(github_pr)},
    "hashes": {"manifest_sha256": component["manifest_file_sha256"], "split_sha256": component["split_metadata_sha256"]},
    "slurm": {"train_job_id": None, "eval_job_ids": []},
}
Path(context_path).write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")
""",
        text=True,
        check=True,
    )
    context = json.loads(context_path.read_text(encoding="utf-8"))
    assert context["research"] == {"github_issue": 20, "github_pr": 99}
    assert context["source"]["git_commit"] == "a" * 40
    assert context["logical_run_name"] == "harmonized_v1_en_d3tec_audio_text_seed1337"
    assert context["group_id"] == "harmonized-v1-en-campaign"
    assert context["seed"] == 1337
    assert context["hashes"]["manifest_sha256"] == "m" * 64


def test_equivalence_audit_accepts_identical_and_rejects_changes(tmp_path: Path) -> None:
    def make_rows(transcript: str, language: str, variant: str, label: int = 0) -> list[dict]:
        return [
            {
                "dataset": "d3tec",
                "subject_id": "001",
                "sample_id": "001_p0_s0",
                "response_id": "001_p0",
                "audio_path": "/data/001.wav",
                "start_time": 0.0,
                "end_time": 2.5,
                "segment_duration": 2.5,
                "label": label,
                "transcript": transcript,
                "language": language,
                "transcript_variant": variant,
                "translation_sha256": "x" * 64 if variant == "english" else "",
            }
        ]

    def make_meta(fold_hash: str, folds_path: Path) -> dict:
        return {
            "manifest_hash": "n" * 64,
            "manifest_row_count": 1,
            "fold_hash": fold_hash,
            "folds_path": str(folds_path),
            "build_signature": {
                "split_options": {"mode": "cv", "outer_folds": 5, "seed": 1337}
            },
        }

    def write_folds(tmp: Path, fold_hash: str, assignment: dict | None = None) -> Path:
        path = tmp / f"folds_{fold_hash}.json"
        payload = assignment or {
            "0": {"outer_train_subject_ids": ["001"], "final_eval_subject_ids": []}
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    class FakeMeta:
        def __init__(self, fold_hash: str, folds_path: Path):
            self._data = make_meta(fold_hash, folds_path)

        def __getitem__(self, key):
            return self._data[key]

        def get(self, key, default=None):
            return self._data.get(key, default)

    native_rows = make_rows("hola mundo", "es", "")
    en_rows = make_rows("hello world", "en", "english", label=0)
    native_meta = FakeMeta("fold-a", write_folds(tmp_path, "fold-a"))
    en_meta = FakeMeta("fold-a", write_folds(tmp_path, "fold-a"))
    result = equivalence_audit(native_meta, native_rows, en_meta, en_rows)
    assert result["failures"] == []

    changed = equivalence_audit(native_meta, [dict(native_rows[0], label=1)], en_meta, en_rows)
    assert any("labels" in failure for failure in changed["failures"])

    changed = equivalence_audit(native_meta, [dict(native_rows[0], end_time=9.9)], en_meta, en_rows)
    assert any("identity" in failure for failure in changed["failures"])

    changed = equivalence_audit(
        native_meta,
        native_rows,
        FakeMeta("fold-b", write_folds(tmp_path, "fold-b", {"1": {"outer_train_subject_ids": ["001"], "final_eval_subject_ids": []}})),
        en_rows,
    )
    assert any("fold" in failure for failure in changed["failures"])

    fallback = equivalence_audit(native_meta, native_rows, en_meta, make_rows("hola mundo", "en", "english"))
    assert any("fallback" in failure for failure in fallback["failures"])


def test_synthetic_translation_cache_gate(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    units = [
        {
            "dataset": "cmdc", "unit_id": "U1", "field": "transcript",
            "scope": "response", "source_language": "zh", "target_language": "en",
            "source_text": "今天很开心。", "source_sha256": "a" * 64,
            "context_id": "", "context_text": "", "context_sha256": "",
            "part_index": 0, "part_count": 1,
        },
        {
            "dataset": "cmdc", "unit_id": "U2", "field": "transcript",
            "scope": "response", "source_language": "zh", "target_language": "en",
            "source_text": "昨天有点累。", "source_sha256": "b" * 64,
            "context_id": "", "context_text": "", "context_sha256": "",
            "part_index": 0, "part_count": 1,
        },
    ]
    import hashlib
    translations = {"U1": "I was very happy today.", "U2": "I was a bit tired yesterday."}
    candidates = []
    for unit in units:
        text = translations[unit["unit_id"]]
        candidates.append({
            "dataset": "cmdc", "unit_id": unit["unit_id"], "field": "transcript",
            "part_index": 0, "part_count": 1,
            "translation": text,
            "translation_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "model": "test-model", "model_revision": "UNKNOWN", "precision": "",
            "prompt_version": "test_v1", "source_sha256": unit["source_sha256"],
            "status": "translated",
        })
    for name, rows in (("units.jsonl", units), ("candidates.jsonl", candidates)):
        with (cache / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = subprocess.run(
        ["python", "-m", "src.translation.validate",
         "--units", str(cache / "units.jsonl"),
         "--candidates", str(cache / "candidates.jsonl"),
         "--accepted", str(cache / "accepted.jsonl"),
         "--rejected", str(cache / "rejected.jsonl"),
         "--audit", str(cache / "audit.json"),
         "--seed", "42"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    combined = result.stdout + result.stderr
    assert "accepted=2 rejected=0" in combined
    accepted = [json.loads(line) for line in (cache / "accepted.jsonl").read_text().splitlines()]
    assert len(accepted) == 2
    assert (cache / "rejected.jsonl").read_text().strip() == ""

    broken = list(candidates)
    broken[0] = dict(broken[0], source_sha256="c" * 64)
    with (cache / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in broken:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = subprocess.run(
        ["python", "-m", "src.translation.validate",
         "--units", str(cache / "units.jsonl"),
         "--candidates", str(cache / "candidates.jsonl"),
         "--accepted", str(cache / "accepted_broken.jsonl"),
         "--rejected", str(cache / "rejected_broken.jsonl"),
         "--audit", str(cache / "audit_broken.jsonl"),
         "--seed", "42"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    combined = result.stdout + result.stderr
    assert "accepted=1 rejected=1" in combined
    rejected = [json.loads(line) for line in (cache / "rejected_broken.jsonl").read_text().splitlines()]
    assert rejected[0]["reasons"] == ["source hash mismatch"]


def test_context_fit_assembly_dedupes_natural_units_like_runtime() -> None:
    from scripts.prepare_harmonized_en_mn5 import _join_full_subject_transcripts

    def android_row(sample_id, turn_id, text):
        return {
            "dataset": "androids_interview",
            "subject_id": "S1",
            "sample_id": sample_id,
            "response_id": f"r{turn_id}",
            "recording_id": "rec1",
            "turn_id": turn_id,
            "prompt_id": 1,
            "question_id": "1",
            "full_turn_transcript": text,
            "transcript": text[:10],
        }

    rows = [
        android_row("t1_w00", 1, "First turn text."),
        android_row("t1_w01", 1, "First turn text."),
        android_row("t2_w00", 2, "Second turn text."),
    ]
    joined = _join_full_subject_transcripts(rows)
    assert joined == {"S1": "First turn text.\nSecond turn text."}


def test_cv_smoke_split_guard_allows_per_partition_limits(tmp_path: Path) -> None:
    from src.features.extract_qwen_hidden import _validate_saved_split

    def make_meta(tmp: Path) -> Path:
        path = tmp / "folds.json"
        path.write_text(
            json.dumps(
                {
                    "0": {
                        "outer_train_subject_ids": [f"t{i}" for i in range(12)],
                        "final_eval_subject_ids": [f"h{i}" for i in range(6)],
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    meta = make_meta(tmp_path)
    saved = {"split_mode": "cv", "cv_protocol": "train_val_test", "split_metadata_path": str(meta)}
    config = {
        "split": {
            "smoke_subject_limit": 6,
            "train_partition": "train",
            "selection_partition": "val",
            "final_eval_partition": "test",
        }
    }
    partitions = {
        "outer_train": [f"t{i}" for i in range(12)],
        "final_eval": [f"h{i}" for i in range(6)],
    }
    assert _validate_saved_split(saved, config, partitions, 0, 2) == meta
    with pytest.raises(ValueError, match="exceeds split.smoke_subject_limit"):
        _validate_saved_split(saved, config, partitions, 0, 1)


def test_retry_script_english_compatibility_dry_run(tmp_path: Path) -> None:
    cells = tmp_path / "cells.tsv"
    cells.write_text("d3tec\taudio_text\t0\t0\t123456\t\t\tFAILED\n", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_harmonized_standalone_retry.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "PROJECT_ROOT": str(ROOT),
            "RUN_ID": "unit",
            "DRY_RUN": "1",
            "GITHUB_ISSUE": "20",
            "GITHUB_PR": "99",
            "MATRIX": str(MATRIX),
            "CELLS": str(cells),
            "PREFLIGHT_AUDIT": str(tmp_path / "audit.json"),
            "PREFLIGHT_COMPONENTS": "4",
            "PREFLIGHT_MERGED": "0",
            "SUBMISSIONS_ROOT": str(tmp_path / "submissions"),
            "CONTEXTS_ROOT": str(tmp_path / "contexts"),
            "FEATURES_ROOT": str(tmp_path / "features"),
            "CLASSIFIERS_ROOT": str(tmp_path / "classifiers"),
            "RUN_PREFIX": "harmonized_v1_en",
            "GROUP_PREFIX": "harmonized-v1-en",
            "LOGICAL_PREFIX": "harmonized_v1_en",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    commands = [line for line in result.stderr.splitlines() if line.startswith("DRY_RUN sbatch")]
    assert len(commands) == 3
    assert sum("run_train_slurm.sh" in line for line in commands) == 1
    assert sum("run_eval_slurm.sh" in line for line in commands) == 1
    assert sum("run_qwen_hidden_extract_slurm.sh" in line for line in commands) == 1
    assert "harmonized_v1_en" in result.stderr
