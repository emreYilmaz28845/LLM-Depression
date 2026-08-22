from __future__ import annotations

import csv
import json
import os
import subprocess
import unicodedata
from pathlib import Path

import pytest
import yaml

from scripts.audit_turkish_negative_only_pipeline import _subject_scores_from_source
from src.data.turkish import build_turkish_manifest
from src.translation.units import unit_rows_for_dataset


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"
PREFIX = "turkish_negative_only_t17_"
NATIVE_RECIPE = "harmonized_full_transcript_single30_allwindows_selmacrof1_tf_v1"
EN_RECIPE = "harmonized_full_transcript_single30_allwindows_selmacrof1_tf_en_v1"


def _write_fixture(root: Path, *, wrong_label: bool = False) -> None:
    metadata_rows = []
    transcript_rows = []
    for index in range(8):
        subject = "aç1" if index == 0 else f"s{index}"
        score = 18 if index % 2 else 10
        label = int(score >= 17)
        if wrong_label and index == 0:
            label = 1 - label
        target = "depressed" if label else "non_depressed"
        basename = f"{subject}-2-1-depr.wav"
        (root / basename).write_bytes(b"RIFF")
        metadata_rows.append(
            {
                "file_name": basename,
                "patient_id": subject,
                "depresyon_skoru": score,
                "label_t17": label,
                "target_t17": target,
            }
        )
        transcript_rows.append(
            {
                "audio_path": str(root / basename),
                "transcript": f"Turkish response {index}",
                "language": "tr",
                "repair_status": "QWEN3ASR_RAW",
            }
        )
    with (root / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0]))
        writer.writeheader()
        writer.writerows(metadata_rows)
    with (root / "transcripts.jsonl").open("w", encoding="utf-8") as handle:
        for row in transcript_rows:
            handle.write(json.dumps(row) + "\n")
    (root / "excluded-2-1-depr.wav").write_bytes(b"RIFF")


def _config(root: Path, *, threshold: int = 17) -> dict:
    return {
        "dataset": "turkish",
        "dataset_variant": "negative_only_t17",
        "metadata_schema": "minimal_t17",
        "dataset_root": str(root),
        "audio_dir": ".",
        "metadata_csv": "metadata.csv",
        "transcript_file": "transcripts.jsonl",
        "threshold": threshold,
        "seed": 1337,
        "split": {"outer_folds": 2, "seed": 1337, "inner_val_ratio": 0.5},
    }


def test_minimal_t17_manifest_uses_only_scores_and_selected_rows(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    result = build_turkish_manifest(_config(tmp_path), {})
    rows = result["manifest_rows"]

    assert len(rows) == 8
    assert {row["dataset_variant"] for row in rows} == {"negative_only_t17"}
    assert all(unicodedata.normalize("NFD", row["subject_id"]) == row["subject_id"] for row in rows)
    assert {row["label"] for row in rows} == {0, 1}
    assert all(row["comorbid"] is None for row in rows)
    assert all(row["anxiety_score"] is None for row in rows)
    assert all(row["w2v2_predicted_score"] is None for row in rows)
    assert len(result["subject_rows"]) == 8
    assert len(result["folds"]) == 2
    assert [row["reason"] for row in result["extra_file_audit"]] == ["no_label_row"]

    units = unit_rows_for_dataset(rows, "turkish")
    assert len(units) == 8
    assert all(unit["scope"] == "audio_chunk" for unit in units)
    assert all(unit["source_language"] == "tr" for unit in units)
    forbidden = {"label", "label_t17", "target_t17", "score", "depresyon_skoru", "fold"}
    assert all(not forbidden.intersection(unit) for unit in units)


def test_minimal_t17_manifest_rejects_wrong_threshold_and_label(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="requires threshold=17"):
        build_turkish_manifest(_config(tmp_path, threshold=25), {})

    other = tmp_path / "wrong"
    other.mkdir()
    _write_fixture(other, wrong_label=True)
    with pytest.raises(ValueError, match="source label mismatch"):
        build_turkish_manifest(_config(other), {})


def test_reference_score_audit_normalizes_turkish_subject_ids(tmp_path: Path) -> None:
    composed = tmp_path / "composed.csv"
    decomposed = tmp_path / "decomposed.csv"
    for path, subject in ((composed, "aç1"), (decomposed, "ac\u03271")):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["patient_id", "depresyon_skoru"])
            writer.writeheader()
            writer.writerow({"patient_id": subject, "depresyon_skoru": 23})

    assert _subject_scores_from_source(composed) == _subject_scores_from_source(decomposed)


def _variant_configs() -> tuple[list[Path], list[Path]]:
    native = sorted(
        path
        for path in MAIN.glob(f"{PREFIX}*harmonized_selmacrof1_tf_qwen3asr.yaml")
        if not path.name.endswith("_en.yaml")
    )
    english = sorted(MAIN.glob(f"{PREFIX}*harmonized_selmacrof1_tf_qwen3asr_en.yaml"))
    return native, english


def test_negative_only_configs_are_isolated_and_recipe_matched() -> None:
    native_paths, english_paths = _variant_configs()
    assert len(native_paths) == 3
    assert len(english_paths) == 2

    native_modes = set()
    for path in native_paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        native_modes.add((config["data"]["use_audio"], config["data"]["use_text"]))
        assert config["dataset"] == "turkish"
        assert config["dataset_variant"] == "negative_only_t17"
        assert config["metadata_schema"] == "minimal_t17"
        assert config["audio_dir"] == "."
        assert config["threshold"] == 17
        assert config["recipe_id"] == NATIVE_RECIPE
        assert "Turkish_Negative_Only" in config["dataset_root"]
        assert "turkish_negative_only_t17_qwen3asr" in config["output_dirs"]["manifest_dir"]
        assert config["evaluation"]["evaluation_view"] == "harmonized_all_windows_full_coverage"
    assert native_modes == {(True, False), (True, True), (False, True)}

    for path in english_paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["recipe_id"] == EN_RECIPE
        assert config["data"]["use_text"] is True
        assert "manifests_harmonized_en/turkish_negative_only" in config["output_dirs"]["manifest_dir"]
        transcripts = config["transcripts"]
        assert transcripts == {
            "variant": "english",
            "cache_path": "${TRANSLATION_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/translations}/harmonized_en_complete_v1/turkish_negative_only_t17/accepted.jsonl",
            "minimum_status": "automatic_low",
            "require_complete": True,
            "include_failed": False,
        }


def test_translation_runner_rejects_unsafe_or_unisolated_smoke_roots(tmp_path: Path) -> None:
    script = ROOT / "scripts/run_translation_slurm.sh"
    base_env = {
        **os.environ,
        "DATASET": "turkish",
        "MANIFEST_CONFIG": "unused.yaml",
        "TRANSLATION_ROOT": str(tmp_path / "translations"),
    }
    unsafe = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env={**base_env, "TRANSLATION_RUN_ROOT": str(tmp_path / "outside")},
        text=True,
        capture_output=True,
    )
    assert unsafe.returncode != 0
    assert "absolute child of TRANSLATION_ROOT" in unsafe.stderr

    root_itself = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env={**base_env, "TRANSLATION_RUN_ROOT": base_env["TRANSLATION_ROOT"]},
        text=True,
        capture_output=True,
    )
    assert root_itself.returncode != 0
    assert "absolute child of TRANSLATION_ROOT" in root_itself.stderr

    unisolated = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env={**base_env, "UNIT_LIMIT": "4"},
        text=True,
        capture_output=True,
    )
    assert unisolated.returncode != 0
    assert "requires an explicit isolated TRANSLATION_RUN_ROOT" in unisolated.stderr


@pytest.mark.parametrize(
    ("stage", "expected_root", "expected_count", "expected_limit"),
    [
        ("smoke", "/smokes/test-run", "4", "4"),
        ("production", "/harmonized_en_complete_v1/turkish_negative_only_t17", "1170", "0"),
    ],
)
def test_translation_submitter_dry_run_is_isolated(
    tmp_path: Path,
    stage: str,
    expected_root: str,
    expected_count: str,
    expected_limit: str,
) -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_turkish_negative_only_translation.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "PROJECT_ROOT": str(ROOT),
            "TRANSLATION_ROOT": str(tmp_path / "translations"),
            "STAGE": stage,
            "RUN_ID": "test-run",
            "DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert f"run_root={tmp_path / 'translations'}{expected_root}" in result.stdout
    assert f"expected_units={expected_count}" in result.stdout
    assert f"UNIT_LIMIT={expected_limit}" in result.stdout
    assert "REQUIRE_COMPLETE=1" in result.stdout
