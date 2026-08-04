from __future__ import annotations

import copy
import os
import subprocess
from collections import Counter
from pathlib import Path

import yaml

from scripts.audit_daic_independent_modalities import audit
from src.data.runtime import build_examples
from src.utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/experiments/daic_chunking"
AUDIO_CONFIG = CONFIG_DIR / "daic_audio_only_independent_all_equalrow_selposf1_tf.yaml"
TEXT_CONFIG = CONFIG_DIR / "daic_text_only_independent_family_selposf1_tf.yaml"
MATRIX = ROOT / "configs/features/daic_independent_all_modalities_matrix.yaml"


def _without_output_root(config: dict) -> dict:
    result = copy.deepcopy(config)
    result["output_dirs"].pop("run_root")
    return result


def _row(subject: str, index: int, label: int, transcript: str) -> dict:
    return {
        "dataset": "daic",
        "subject_id": subject,
        "sample_id": f"{subject}_{index}",
        "chunk_id": str(index),
        "label": label,
        "label_text": "Depressed" if label else "Non-depressed",
        "transcript": transcript,
        "audio_path": f"/tmp/{subject}_{index}.wav",
    }


def test_audio_only_diff_is_limited_to_independent_construction() -> None:
    main = _without_output_root(load_yaml(ROOT / "configs/main/daic_audio_only_selposf1_tf.yaml"))
    experiment = _without_output_root(load_yaml(AUDIO_CONFIG))
    assert main["data"].pop("sample_mode") == "subject_audio"
    assert main["data"].pop("chunks_per_subject") == 4
    for key in (
        "sample_mode",
        "train_chunk_policy",
        "train_chunks_per_subject",
        "eval_chunk_policy",
        "eval_chunks_per_subject",
        "loss_weight_rescale",
        "equal_row_weight",
    ):
        experiment["data"].pop(key)
    assert experiment["training"].pop("match_joint_optimizer_updates") is False
    assert experiment == main


def test_text_only_is_the_main_subject_recipe_with_isolated_output() -> None:
    main = _without_output_root(load_yaml(ROOT / "configs/main/daic_text_only_selposf1_tf.yaml"))
    experiment = _without_output_root(load_yaml(TEXT_CONFIG))
    assert experiment == main
    assert load_yaml(TEXT_CONFIG)["data"]["sample_mode"] == "subject"


def test_audio_only_uses_all_independent_rows_without_transcript() -> None:
    rows = [
        *[_row("non", index, 0, "private full transcript") for index in range(10)],
        *[_row("dep", index, 1, "another full transcript") for index in range(15)],
    ]
    examples = build_examples(rows, load_yaml(AUDIO_CONFIG), "train")
    assert Counter(item["subject_id"] for item in examples) == {"non": 10, "dep": 15}
    assert all(len(item["audio_paths"]) == 1 for item in examples)
    assert all(item["transcript"] == "" for item in examples)
    assert all("private full transcript" not in item["prompt_text"] for item in examples)


def test_text_only_collapses_manifest_chunks_to_one_subject_example() -> None:
    rows = [
        *[_row("non", index, 0, "same non transcript") for index in range(10)],
        *[_row("dep", index, 1, "same dep transcript") for index in range(15)],
    ]
    examples = build_examples(rows, load_yaml(TEXT_CONFIG), "train")
    assert Counter(item["subject_id"] for item in examples) == {"non": 1, "dep": 1}
    assert all(not item["audio_paths"] for item in examples)
    assert {item["transcript"] for item in examples} == {
        "same non transcript",
        "same dep transcript",
    }


def test_matrix_declares_three_modalities_and_two_raw_heads() -> None:
    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    assert matrix["expected_jobs"] == 3
    assert {item["modality"] for item in matrix["experiments"]} == {
        "audio_text",
        "audio_only",
        "text_only",
    }
    assert matrix["variants"] == ["logreg_raw", "xgb_raw"]
    assert "xgb_pca32" not in matrix["variants"]


def test_matrix_submission_forwards_only_declared_heads(tmp_path: Path) -> None:
    payload = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    for item in payload["experiments"]:
        checkpoint = tmp_path / item["run_dir"] / "fold_0/best_model"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_model.safetensors").write_bytes(b"test")
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump(payload), encoding="utf-8")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_qwen_hidden_matrix.sh")],
        cwd=ROOT,
        env={**os.environ, "PROJECT_ROOT": str(tmp_path), "MATRIX": str(matrix), "DRY_RUN": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.count("DRY RUN:") == 3
    assert result.stdout.count("CLASSIFIER_VARIANTS=logreg_raw\\,xgb_raw") == 3
    assert "pca" not in result.stdout.lower()


def test_incomplete_matrix_artifacts_fail_acceptance_audit(tmp_path: Path) -> None:
    result = audit(MATRIX, tmp_path)
    assert result["passed"] is False
    assert result["expected_cells"] == 9
    assert result["audited_cells"] == 0
    assert len(result["failures"]) == 3
