from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
import wave
from pathlib import Path

import pytest
import yaml

from src.data.build_manifest import build_for_config, manifest_build_signature
from src.data.runtime import build_examples
from src.data.turkish_combined import build_turkish_combined_manifest

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"

# The two arms of the four-source comparison. The baseline pooled configs are the
# recipe authority; the four-source configs only change the data contract and the
# runtime shape of the Qwen3-Omni cells (two nodes, accumulation 16).
BASELINE_TREATMENT_PAIRS = (
    (
        "turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml",
        "turkish_all_geriatri_t17_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml",
        4,
    ),
    (
        "turkish_pooled_t17_audio_only_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml",
        "turkish_all_geriatri_t17_audio_only_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml",
        8,
    ),
    (
        "turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml",
        "turkish_all_geriatri_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml",
        8,
    ),
)
# Accumulation as recorded by the submitted baseline runs: the pooled YAML text
# predates the submission override, so the resolved value is the authority
# (qwen38_pc_turkish_f0_20260922 and qwen3omni_turkish_pooled_*_prod_20260923
# run_config.yaml, training_strategy.gradient_accumulation_steps).
BASELINE_RESOLVED_ACCUMULATION = {
    "turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml": 32,
    "turkish_pooled_t17_audio_only_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml": 16,
    "turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml": 16,
}

FROZEN_TOP_LEVEL = ("seed", "threshold", "labels", "prompt")
FROZEN_TRAINING = (
    "dist_timeout_minutes",
    "num_train_epochs",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "logging_steps",
    "bf16",
    "gradient_checkpointing",
    "run_final_eval_in_train",
    "dataloader_num_workers",
    "max_grad_norm",
    "class_balance",
    "selection_metric",
    "selection_metric_mode",
    "early_stopping",
    "strategy",
    "activation_offload",
)
FROZEN_LORA = ("rank", "alpha", "dropout", "bias", "target_modules")
FROZEN_DATA = (
    "sample_mode",
    "segment_seconds",
    "segment_partition",
    "train_chunk_policy",
    "eval_chunk_policy",
    "transcript_max_chars",
    "allow_empty_transcript",
)
FROZEN_EVALUATION = (
    "sample_prediction_mode",
    "headline_mode",
    "aggregation_level",
    "hierarchical_score_aggregation",
    "subject_score_aggregation",
    "evaluation_view",
    "inference_dtype",
    "generation_max_new_tokens",
    "num_beams",
    "do_sample",
    "evaluate_last_checkpoint",
)
FROZEN_SPLIT = ("mode", "cv_protocol", "outer_folds", "inner_val_ratio", "seed")


def test_treatment_configs_keep_four_sources_and_isolated_fold_lock() -> None:
    configs = sorted(MAIN.glob("turkish_all_geriatri_t17_*.yaml"))
    assert len(configs) == 3
    expected_sources = {
        "original_positive", "original_negative", "geriatri_positive", "geriatri_negative"
    }
    modalities = set()
    signatures = []
    for path in configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["dataset"] == "turkish"
        assert config["dataset_variant"] == "all_geriatri_t17"
        assert config["score_authority"] == "metadata_csv"
        assert config["threshold"] == 17
        assert {source["id"] for source in config["sources"]} == expected_sources
        assert config["evaluation"]["sample_prediction_mode"] == "likelihood"
        assert config["training"]["selection_metric"] == "inner_val_macro_f1"
        assert config["split"]["locked_original_folds_path"]
        assert len(config["split"]["locked_original_folds_sha256"]) == 64
        assert config["output_dirs"]["run_root"].endswith("/turkish")
        modalities.add((config["data"]["use_audio"], config["data"]["use_text"]))
        signatures.append(manifest_build_signature(config))
    assert modalities == {(True, False), (True, True), (False, True)}
    assert signatures[0] == signatures[1] == signatures[2]


def test_treatment_arms_keep_the_baseline_recipe_fields() -> None:
    for baseline_name, treatment_name, world_size in BASELINE_TREATMENT_PAIRS:
        baseline = yaml.safe_load((MAIN / baseline_name).read_text(encoding="utf-8"))
        treatment = yaml.safe_load((MAIN / treatment_name).read_text(encoding="utf-8"))
        for key in FROZEN_TOP_LEVEL:
            assert treatment[key] == baseline[key], (baseline_name, key)
        for key in FROZEN_LORA:
            assert treatment["lora"][key] == baseline["lora"][key], (baseline_name, key)
        for key in FROZEN_DATA:
            if key not in baseline["data"] and key not in treatment["data"]:
                continue
            assert treatment["data"][key] == baseline["data"][key], (baseline_name, key)
        for key in FROZEN_EVALUATION:
            if key not in baseline["evaluation"] and key not in treatment["evaluation"]:
                continue
            assert treatment["evaluation"][key] == baseline["evaluation"][key], (baseline_name, key)
        for key in FROZEN_SPLIT:
            assert treatment["split"][key] == baseline["split"][key], (baseline_name, key)
        for key in FROZEN_TRAINING:
            assert treatment["training"][key] == baseline["training"][key], (baseline_name, key)
        assert treatment["data"]["use_audio"] == baseline["data"]["use_audio"]
        assert treatment["data"]["use_text"] == baseline["data"]["use_text"]
        assert treatment["data"].get("audio_text_transcript_scope") == baseline["data"].get(
            "audio_text_transcript_scope"
        )
        assert treatment.get("audio_adapter") == baseline.get("audio_adapter")
        assert treatment.get("resources") == baseline.get("resources")
        assert treatment["prompt"]["user_template"] == baseline["prompt"]["user_template"]
        assert "{question_context}" in treatment["prompt"]["user_template"]
        # Effective global batch stays 128 in both arms at the submitted shapes.
        baseline_accumulation = BASELINE_RESOLVED_ACCUMULATION[baseline_name]
        treatment_accumulation = treatment["training"]["gradient_accumulation_steps"]
        assert baseline_accumulation * world_size == 128
        assert treatment_accumulation * world_size == 128


def _source(root: Path, cohort: str, condition: str, *, geriatri: bool) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    audio_dir = root / condition
    audio_dir.mkdir(exist_ok=True)
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


# The baseline fold assignment the four-source build has to reproduce exactly.
LOCKED_ORIGINAL_FOLDS = {
    "aç1": 0,
    "s1": 0,
    "s2": 0,
    "s3": 0,
    "s4": 1,
    "s5": 1,
    "s6": 1,
    "s7": 1,
}


def _locked_subject_id(patient: str) -> str:
    return unicodedata.normalize("NFD", patient)


def _write_locked_folds(tmp_path: Path, assignment: dict[str, int]) -> tuple[Path, str]:
    folds = {
        str(fold): {
            "final_eval_subject_ids": sorted(
                _locked_subject_id(patient)
                for patient, value in assignment.items()
                if value == fold
            )
        }
        for fold in sorted(set(assignment.values()))
    }
    path = tmp_path / "locked_folds" / "turkish_folds.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(folds, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path) -> dict:
    sources = [
        _source(tmp_path / cohort, cohort, condition, geriatri=cohort == "geriatri")
        for cohort in ("original", "geriatri")
        for condition in ("pos_only_t17", "negative_only_t17")
    ]
    locked_path, locked_sha256 = _write_locked_folds(tmp_path, LOCKED_ORIGINAL_FOLDS)
    return {
        "dataset": "turkish",
        "dataset_variant": "all_geriatri_t17",
        "score_authority": "metadata_csv",
        "sources": sources,
        "threshold": 17,
        "seed": 1337,
        "split": {
            "outer_folds": 2,
            "seed": 1337,
            "inner_val_ratio": 0.5,
            "locked_original_folds_path": str(locked_path),
            "locked_original_folds_sha256": locked_sha256,
        },
    }


def test_combined_geriatri_locks_original_folds_and_places_new_subjects(tmp_path: Path) -> None:
    result = build_turkish_combined_manifest(_config(tmp_path), {})
    rows = result["manifest_rows"]
    assert len(rows) == 32
    assert len(result["subject_rows"]) == 16
    assert len({row["sample_id"] for row in rows}) == 32
    assert {row["dataset_variant"] for row in rows} == {"pos_only_t17", "negative_only_t17"}
    assert {row["source_cohort"] for row in rows} == {"original", "geriatri"}
    assert all(Path(row["audio_path"]).is_file() for row in rows)
    assert all(
        row["subject_id"].startswith("geriatri:")
        for row in rows
        if row["source_cohort"] == "geriatri"
    )
    assert all(
        row["anxiety_score"] is None and row["comorbid"] is None
        for row in rows
        if row["source_cohort"] == "geriatri"
    )
    assert {row["label"] for row in rows if row["source_cohort"] == "geriatri"} == {0, 1}

    fold_of_subject = {
        subject: fold
        for fold, payload in result["folds"].items()
        for subject in payload["final_eval_subject_ids"]
    }
    for patient, locked_fold in LOCKED_ORIGINAL_FOLDS.items():
        assert fold_of_subject[_locked_subject_id(patient)] == locked_fold
    geriatri_subjects = [
        subject for subject in fold_of_subject if subject.startswith("geriatri:")
    ]
    assert len(geriatri_subjects) == 8

    labels = {row["subject_id"]: row["label"] for row in rows}
    for label in (0, 1):
        counts = [
            sum(1 for subject in payload["final_eval_subject_ids"] if labels[subject] == label)
            for payload in result["folds"].values()
        ]
        assert max(counts) - min(counts) <= 1, (label, counts)
    for fold, payload in result["folds"].items():
        train = set(payload["outer_train_subject_ids"])
        heldout = set(payload["final_eval_subject_ids"])
        assert not train & heldout
        for row in rows:
            assert (row["subject_id"] in train) != (row["subject_id"] in heldout)

    lock = result["fold_lock"]
    assert lock["original_subject_count"] == 8
    assert lock["new_subject_count"] == 8
    assert lock["locked_original_folds"]["subject_count"] == 8
    assert lock["combined_mapping_canonical_sha256"] != lock["new_assignment_canonical_sha256"]
    assert result["fold_hash"] == lock["combined_mapping_canonical_sha256"]
    repeated = build_turkish_combined_manifest(_config(tmp_path), {})
    assert (
        repeated["fold_lock"]["new_assignment_canonical_sha256"]
        == lock["new_assignment_canonical_sha256"]
    )


def test_combined_geriatri_rejects_locked_folds_hash_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["split"]["locked_original_folds_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="locked baseline folds hash mismatch"):
        build_turkish_combined_manifest(config, {})


def test_combined_geriatri_requires_the_locked_folds_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)
    del config["split"]["locked_original_folds_path"]
    with pytest.raises(ValueError, match="locked_original_folds_path"):
        build_turkish_combined_manifest(config, {})


def test_combined_geriatri_rejects_original_cohort_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    drifted = dict(LOCKED_ORIGINAL_FOLDS)
    drifted.pop("s7")
    drifted["s8"] = 1
    locked_path, locked_sha256 = _write_locked_folds(tmp_path, drifted)
    config["split"]["locked_original_folds_path"] = str(locked_path)
    config["split"]["locked_original_folds_sha256"] = locked_sha256
    with pytest.raises(ValueError, match="original cohort does not match the locked baseline folds"):
        build_turkish_combined_manifest(config, {})


def test_combined_geriatri_rejects_fold_count_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["split"]["outer_folds"] = 3
    with pytest.raises(ValueError, match="locked baseline folds use folds"):
        build_turkish_combined_manifest(config, {})


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


def test_combined_geriatri_prefers_the_csv_score_over_the_workbook(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = build_turkish_combined_manifest(config, {})
    geriatri = [row for row in result["manifest_rows"] if row["source_cohort"] == "geriatri"]
    assert {row["source_id"] for row in geriatri} == {
        "geriatri_pos_only_t17", "geriatri_negative_only_t17"
    }
    for row in geriatri:
        assert row["threshold"] == 17.0
        assert row["label"] == int(float(row["score"]) >= 17.0)


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
    assert metadata["split_source"] == "locked_original_folds_plus_stratified_new_cohort"
    assert metadata["fold_lock"]["original_subject_count"] == 8
    assert metadata["fold_lock"]["new_subject_count"] == 8
    assert metadata["fold_hash"] == metadata["fold_lock"]["combined_mapping_canonical_sha256"]
    audit = json.loads((tmp_path / "splits/turkish_fold_lock_audit.json").read_text())
    assert audit["locked_original_folds"]["file_sha256"] == config["split"][
        "locked_original_folds_sha256"
    ]


def test_combined_text_examples_pair_each_question_set(tmp_path: Path) -> None:
    result = build_turkish_combined_manifest(_config(tmp_path), {})
    config = yaml.safe_load(
        (
            MAIN
            / "turkish_all_geriatri_t17_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml"
        ).read_text(encoding="utf-8")
    )
    examples = build_examples(result["manifest_rows"], config, "train")
    assert len(examples) == 32
    assert len({(example["subject_id"], example["question_condition"]) for example in examples}) == 32
    assert {example["question_condition"] for example in examples} == {
        "pos_only_t17", "negative_only_t17"
    }
    assert all(example["transcript"].count("Kısa yanıt") == 1 for example in examples)


def test_combined_audio_examples_keep_both_question_sets(tmp_path: Path) -> None:
    result = build_turkish_combined_manifest(_config(tmp_path), {})
    config = yaml.safe_load(
        (
            MAIN
            / "turkish_all_geriatri_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml"
        ).read_text(encoding="utf-8")
    )
    examples = build_examples(result["manifest_rows"], config, "train")
    assert len(examples) == 32
    assert len({example["response_id"] for example in examples}) == 32
    assert {example["question_condition"] for example in examples} == {
        "pos_only_t17", "negative_only_t17"
    }
    assert all(example["transcript"].count("Kısa yanıt") == 2 for example in examples)
