from __future__ import annotations

import csv
import json
import unicodedata
import wave
from pathlib import Path

import pytest
import yaml

from src.data.turkish_combined import build_turkish_combined_manifest
from src.data.build_manifest import build_for_config
from src.data.build_manifest import manifest_build_signature
from src.data.runtime import build_examples


def test_combined_configs_keep_four_sources_and_isolated_outputs() -> None:
    main = Path(__file__).resolve().parents[1] / "configs/main"
    configs = sorted(main.glob("turkish_all_geriatri_t17_*_qwen3asr.yaml"))
    assert len(configs) == 3
    expected_sources = {
        "original_positive", "original_negative", "geriatri_positive", "geriatri_negative"
    }
    modalities = set()
    signatures = []
    for path in configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["dataset_variant"] == "all_geriatri_t17"
        assert config["score_authority"] == "metadata_csv"
        assert config["threshold"] == 17
        assert {source["id"] for source in config["sources"]} == expected_sources
        assert config["evaluation"]["sample_prediction_mode"] == "likelihood"
        assert config["training"]["selection_metric"] == "inner_val_macro_f1"
        assert "turkish_all_geriatri_t17" in config["output_dirs"]["run_root"]
        modalities.add((config["data"]["use_audio"], config["data"]["use_text"]))
        signatures.append(manifest_build_signature(config))
    assert modalities == {(True, False), (True, True), (False, True)}
    assert signatures[0] == signatures[1] == signatures[2]


def _source(root: Path, cohort: str, condition: str, *, geriatri: bool) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    audio_dir = root / condition
    audio_dir.mkdir()
    metadata = root / f"{condition}.csv"
    transcript = root / f"{condition}.jsonl"
    metadata_rows = []
    transcript_rows = []
    for index in range(8):
        patient = "aç1" if index == 0 else f"s{index}"
        score = (18 if index % 2 else 10) + (2 if geriatri else 0)
        question = 1 if condition == "pos_only_t17" else 2
        basename = f"{patient}-{question}-001.wav"
        if geriatri and condition == "negative_only_t17" and index == 0:
            basename = unicodedata.normalize("NFD", basename)
        disk_basename = unicodedata.normalize("NFD", basename) if geriatri else basename
        audio_path = audio_dir / disk_basename
        with wave.open(str(audio_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00" * 3200)
        row = {"file_name": basename, "depresyon_skoru": score}
        if geriatri:
            row.update(label=0, anksiyete_skoru=99)
            if condition == "negative_only_t17":
                row["patient_id"] = patient
        else:
            label = int(score >= 17)
            row.update(
                patient_id=patient,
                label_t17=label,
                target_t17="depressed" if label else "non_depressed",
            )
        metadata_rows.append(row)
        transcript_rows.append(
            {"audio_path": str(audio_path), "transcript": "Kısa yanıt", "language": "tr"}
        )
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0]))
        writer.writeheader()
        writer.writerows(metadata_rows)
    transcript.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in transcript_rows),
        encoding="utf-8",
    )
    return {
        "id": f"{cohort}_{condition}",
        "cohort": cohort,
        "question_condition": condition,
        "dataset_root": str(root),
        "metadata_csv": metadata.name,
        "metadata_schema": "geriatri_bdo_t17" if geriatri else "minimal_t17",
        "audio_dir": condition,
        "transcript_file": transcript.name,
    }


def _config(tmp_path: Path) -> dict:
    sources = [
        _source(tmp_path / cohort, cohort, condition, geriatri=cohort == "geriatri")
        for cohort in ("original", "geriatri")
        for condition in ("pos_only_t17", "negative_only_t17")
    ]
    return {
        "dataset": "turkish",
        "dataset_variant": "all_geriatri_t17",
        "score_authority": "metadata_csv",
        "sources": sources,
        "threshold": 17,
        "seed": 1337,
        "split": {"outer_folds": 2, "seed": 1337, "inner_val_ratio": 0.5},
    }


def test_combined_geriatri_keeps_cohorts_and_question_sets_grouped(tmp_path: Path) -> None:
    result = build_turkish_combined_manifest(_config(tmp_path), {})
    rows = result["manifest_rows"]
    assert len(rows) == 32
    assert len(result["subject_rows"]) == 16
    assert len({row["sample_id"] for row in rows}) == 32
    assert {row["dataset_variant"] for row in rows} == {
        "pos_only_t17", "negative_only_t17"
    }
    assert {row["source_cohort"] for row in rows} == {"original", "geriatri"}
    assert all(Path(row["audio_path"]).is_file() for row in rows)
    assert all(row["subject_id"].startswith("geriatri:") for row in rows if row["source_cohort"] == "geriatri")
    assert all(row["anxiety_score"] is None and row["comorbid"] is None for row in rows if row["source_cohort"] == "geriatri")
    assert {row["label"] for row in rows if row["source_cohort"] == "geriatri"} == {0, 1}
    for fold in result["folds"].values():
        train = set(fold["outer_train_subject_ids"])
        holdout = set(fold["final_eval_subject_ids"])
        assert not train & holdout
        for row in rows:
            assert (row["subject_id"] in train) != (row["subject_id"] in holdout)


def test_combined_geriatri_rejects_conflicting_bdo_scores(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = config["sources"][3]
    path = Path(source["dataset_root"]) / source["metadata_csv"]
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["depresyon_skoru"] = "11"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="mixed labels|conflicting BDO"):
        build_turkish_combined_manifest(config, {})


def test_combined_manifest_entrypoint_writes_split_provenance(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["output_dirs"] = {
        "manifest_dir": str(tmp_path / "manifests"),
        "split_dir": str(tmp_path / "splits"),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    build_for_config(config_path)
    metadata = json.loads((tmp_path / "splits/turkish_manifest_metadata.json").read_text())
    assert metadata["manifest_row_count"] == 32
    assert metadata["manifest_subject_count"] == 16
    assert set(metadata["source_hashes"]) == {source["id"] for source in config["sources"]}


def test_combined_text_examples_keep_one_full_transcript_per_patient(tmp_path: Path) -> None:
    result = build_turkish_combined_manifest(_config(tmp_path), {})
    main = Path(__file__).resolve().parents[1] / "configs/main"
    path = next(main.glob("turkish_all_geriatri_t17_text_only*_qwen3asr.yaml"))
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    examples = build_examples(result["manifest_rows"], config, "train")
    assert len(examples) == 16
    assert len({example["subject_id"] for example in examples}) == 16
    assert all(example["transcript"].count("Kısa yanıt") == 2 for example in examples)


def test_combined_audio_examples_keep_both_question_sets(tmp_path: Path) -> None:
    result = build_turkish_combined_manifest(_config(tmp_path), {})
    main = Path(__file__).resolve().parents[1] / "configs/main"
    path = next(main.glob("turkish_all_geriatri_t17_audio_text*_qwen3asr.yaml"))
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    examples = build_examples(result["manifest_rows"], config, "train")
    assert len(examples) == 32
    assert len({example["response_id"] for example in examples}) == 32
    assert {example["question_condition"] for example in examples} == {
        "pos_only_t17", "negative_only_t17"
    }
    assert all(example["transcript"].count("Kısa yanıt") == 2 for example in examples)
