from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.daic_chunking import (
    JOINT_PACKED30_MODE,
    build_joint_epoch_schedule,
    canonical_sha256,
)
from src.data.daic import PACKED30_PROTOCOL_ID
from src.data.runtime import (
    AUDIO_PLACEHOLDER,
    JOINT_PACKED30_RECIPE_ID,
    build_examples,
    load_span_group_audio_arrays,
    uses_audio_spans,
)
from src.features.extract_qwen_hidden import _selected_epoch_fit_examples
from src.utils import save_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKED30_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "manifests_daic_participant_packed30"
    / "daic_participant_speech_packed30_manifest.jsonl"
)

EXPECTED_TRAIN_SUBJECTS = 107
EXPECTED_EPOCH_BUNDLES = 107
EXPECTED_EPOCH_SLOTS = 427
EXPECTED_VAL_BUNDLES = 445
EXPECTED_VAL_SLOTS = 1780
EXPECTED_TEST_BUNDLES = 617
EXPECTED_TEST_SLOTS = 2468


def joint_config(**data_overrides) -> dict:
    config = {
        "dataset": "daic",
        "seed": 1337,
        "protocol_id": PACKED30_PROTOCOL_ID,
        "prompt": {
            "system": "You are a psychologist analyzing speech audio for depression screening.",
            "user_template": "{audio_context_block}\n{transcript_block}Based on the {decision_basis}, determine whether the subject is {label_descriptor}.\n{label_instruction}",
        },
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "data": {
            "use_audio": True,
            "use_text": False,
            "recipe_id": JOINT_PACKED30_RECIPE_ID,
            "sample_mode": JOINT_PACKED30_MODE,
            "participant_chunk_samples": 480000,
            "inter_span_silence_samples": 0,
            "train_chunk_policy": "joint_random_k",
            "train_chunks_per_subject": 4,
            "eval_chunk_policy": "balanced_joint_cover",
            "eval_chunks_per_subject": 4,
            "loss_weight_rescale": "mean_one",
            "equal_row_weight": False,
            "transcript_max_chars": 4000,
        },
        "training": {"num_train_epochs": 3},
        "evaluation": {
            "sample_prediction_mode": "original_teacher_forced",
            "aggregation_level": "subject",
            "subject_score_aggregation": "mean_score",
        },
    }
    config["data"].update(data_overrides)
    return config


def packed30_row(
    subject_id: str,
    chunk_index: int,
    num_chunks: int,
    label: int = 0,
    transcript: str = "full transcript",
    wav_path: str = "/tmp/x.wav",
    sample_count: int = 480000,
) -> dict:
    return {
        "schema_version": "daic_participant_speech_packed30_manifest.v1",
        "protocol_id": PACKED30_PROTOCOL_ID,
        "dataset": "daic",
        "subject_id": subject_id,
        "sample_id": f"{subject_id}_participant_p30_{chunk_index:03d}",
        "audio_path": wav_path,
        "audio_spans": [
            {
                "start_frame": chunk_index * 1000,
                "end_frame": chunk_index * 1000 + 1000,
                "source_row_index": chunk_index,
                "source_start_time": 1.0,
                "source_stop_time": 2.0,
            }
        ],
        "participant_sample_count": sample_count,
        "chunk_index": chunk_index,
        "num_chunks": num_chunks,
        "chunk_transcript": "row text",
        "full_participant_transcript": transcript,
        "full_participant_transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
        "transcript": transcript,
        "label": label,
        "label_text": "Depressed" if label else "Non-depressed",
        "split_original": "train",
    }


def synthetic_rows(subjects: dict[str, tuple[int, int]]) -> list[dict]:
    rows = []
    for subject_id, (num_chunks, label) in subjects.items():
        for index in range(num_chunks):
            rows.append(packed30_row(subject_id, index, num_chunks, label=label))
    return rows


def test_span_groups_loaded_in_declared_order(tmp_path: Path) -> None:
    import soundfile as sf

    wav = tmp_path / "s.wav"
    sr = 16000
    # One second per span; each span has a distinct amplitude level (<= 1.0 so
    # PCM_16 writes do not clip).
    total = 4 * sr
    audio = np.zeros(total, dtype=np.float32)
    for index in range(4):
        audio[index * sr : (index + 1) * sr] = 0.1 * float(index + 1)
    sf.write(wav, audio, sr, subtype="PCM_16")
    rows = [
        {
            **packed30_row("300", index, 4, wav_path=str(wav), sample_count=sr),
            "audio_spans": [
                {
                    "start_frame": index * sr,
                    "end_frame": index * sr + sr,
                    "source_row_index": index,
                }
            ],
        }
        for index in range(4)
    ]
    examples = build_examples(rows, joint_config(), "train")
    assert len(examples) == 1
    assert len(examples[0]["subject_chunk_span_groups"]) == 4
    arrays = load_span_group_audio_arrays(examples[0], sr, False)
    assert len(arrays) == 4
    assert [float(array[0]) for array in arrays] == pytest.approx([0.1, 0.2, 0.3, 0.4], abs=1e-3)
    assert [array.shape[0] for array in arrays] == [sr] * 4


def test_span_group_exact_sample_preservation(tmp_path: Path) -> None:
    import soundfile as sf

    wav = tmp_path / "s.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
    rows = [
        {
            **packed30_row("300", index, 2, wav_path=str(wav)),
            "participant_sample_count": 8000,
            "audio_spans": [
                {"start_frame": index * 8000, "end_frame": index * 8000 + 8000, "source_row_index": index}
            ],
        }
        for index in range(2)
    ]
    examples = build_examples(rows, joint_config(), "train")
    arrays = load_span_group_audio_arrays(examples[0], 16000, False)
    assert [array.shape[0] for array in arrays] == [8000, 8000]
    bad = dict(examples[0])
    bad["subject_chunk_span_groups"][0]["participant_sample_count"] = 9000
    bad["audio_span_groups"][0]["participant_sample_count"] = 9000
    with pytest.raises(ValueError, match="expected 9000"):
        load_span_group_audio_arrays(bad, 16000, False)


def test_placeholder_group_waveform_cardinality() -> None:
    rows = synthetic_rows({"300": (10, 0), "385": (3, 1)})
    examples = build_examples(rows, joint_config(), "val")
    for example in examples:
        placeholders = example["prompt_text"].count(AUDIO_PLACEHOLDER)
        assert placeholders == len(example["audio_span_groups"])
        assert placeholders == example["effective_k"]
    mismatch = dict(examples[0])
    mismatch["audio_span_groups"] = mismatch["audio_span_groups"][:3]
    with pytest.raises(ValueError, match="placeholder/group mismatch"):
        load_span_group_audio_arrays(mismatch, 16000, False)


def test_joint_random_k4_deterministic_and_epoch_varying() -> None:
    rows = synthetic_rows({"300": (10, 0), "301": (15, 1)})
    examples = build_examples(rows, joint_config(), "train")
    schedules, audit = build_joint_epoch_schedule(
        examples, policy="joint_random_k", k=4, seed=1337, epochs=3, loss_weight_rescale="mean_one"
    )
    again, again_audit = build_joint_epoch_schedule(
        examples, policy="joint_random_k", k=4, seed=1337, epochs=3, loss_weight_rescale="mean_one"
    )
    for first, second in zip(schedules, again):
        assert [row["bundle_chunk_ids"] for row in first] == [row["bundle_chunk_ids"] for row in second]
    assert audit["schedule_sha256"] == again_audit["schedule_sha256"]
    memberships = {
        subject: {
            epoch: tuple(row["bundle_chunk_ids"])
            for epoch, epoch_rows in enumerate(schedules)
            for row in epoch_rows
            if row["subject_id"] == subject
        }
        for subject in ("300", "301")
    }
    assert len({memberships["300"][epoch] for epoch in range(3)}) == 3
    assert len({memberships["301"][epoch] for epoch in range(3)}) == 3


def test_no_replacement_for_n_ge_4() -> None:
    rows = synthetic_rows({"300": (10, 0)})
    examples = build_examples(rows, joint_config(), "train")
    schedules, audit = build_joint_epoch_schedule(
        examples, policy="joint_random_k", k=4, seed=1337, epochs=5, loss_weight_rescale="mean_one"
    )
    for epoch in schedules:
        members = epoch[0]["bundle_chunk_ids"]
        assert len(members) == len(set(members)) == 4
    exposure = audit["exposure_counts_by_subject"]["300"]
    assert sum(exposure.values()) == 5 * 4
    assert all(1 <= count <= 5 for count in exposure.values())
    assert set(exposure) == {f"{index:03d}" for index in range(10)}


def test_effective_k_min_for_n3_subject() -> None:
    rows = synthetic_rows({"385": (3, 1)})
    examples = build_examples(rows, joint_config(), "train")
    schedules, audit = build_joint_epoch_schedule(
        examples, policy="joint_random_k", k=4, seed=1337, epochs=4, loss_weight_rescale="mean_one"
    )
    for epoch in schedules:
        row = epoch[0]
        assert row["effective_k"] == 3
        assert len(row["audio_span_groups"]) == 3
        assert sorted(row["bundle_chunk_ids"]) == ["000", "001", "002"]
    assert audit["effective_k_by_epoch"][0]["385"] == 3
    assert audit["epoch_mean_effective_weights"] == [1.0] * 4


def test_balanced_cover_complete_and_equal() -> None:
    rows = synthetic_rows({"300": (10, 0), "301": (15, 1), "385": (3, 0)})
    examples = build_examples(rows, joint_config(), "val")
    by_subject = {subject: [e for e in examples if e["subject_id"] == subject] for subject in ("300", "301", "385")}
    assert len(by_subject["300"]) == 5
    assert len(by_subject["301"]) == 15
    assert len(by_subject["385"]) == 1
    for subject, bundles in by_subject.items():
        coverage = sorted(e["bundle_coverage_count"] for e in bundles)
        assert coverage == coverage  # present on every bundle
        seen = []
        for bundle in bundles:
            seen.extend(bundle["bundle_chunk_ids"])
        counts = {chunk: seen.count(chunk) for chunk in set(seen)}
        assert len(set(counts.values())) == 1
        expected_n = {"300": 10, "301": 15, "385": 3}[subject]
        assert set(seen) == {f"{index:03d}" for index in range(expected_n)}


def test_locked_cardinalities_from_current_manifest() -> None:
    if not PACKED30_MANIFEST.is_file():
        raise FileNotFoundError(
            "Locked cardinality test requires the packed30 manifest at "
            f"{PACKED30_MANIFEST}; build it with src/data/build_manifest.py first."
        )
    config = joint_config()
    rows = [json.loads(line) for line in PACKED30_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_rows = [row for row in rows if row["split_original"] == "train"]
    val_rows = [row for row in rows if row["split_original"] == "val"]
    test_rows = [row for row in rows if row["split_original"] == "test"]

    train_examples = build_examples(train_rows, config, "train")
    assert len(train_examples) == EXPECTED_TRAIN_SUBJECTS
    schedules, audit = build_joint_epoch_schedule(
        train_examples, policy="joint_random_k", k=4, seed=1337, epochs=2, loss_weight_rescale="mean_one"
    )
    for epoch in schedules:
        assert len(epoch) == EXPECTED_EPOCH_BUNDLES
        assert sum(len(row["bundle_chunk_ids"]) for row in epoch) == EXPECTED_EPOCH_SLOTS
        assert audit["effective_k_by_epoch"][0]["385"] == 3
        assert audit["epoch_mean_effective_weights"] == [1.0] * 2

    val_examples = build_examples(val_rows, config, "val")
    test_examples = build_examples(test_rows, config, "test")
    assert len(val_examples) == EXPECTED_VAL_BUNDLES
    assert sum(len(e["audio_span_groups"]) for e in val_examples) == EXPECTED_VAL_SLOTS
    assert len(test_examples) == EXPECTED_TEST_BUNDLES
    assert sum(len(e["audio_span_groups"]) for e in test_examples) == EXPECTED_TEST_SLOTS
    assert {str(e["subject_id"]) for e in test_examples} == {str(r["subject_id"]) for r in test_rows}
    for example in test_examples:
        assert example["prompt_text"].count(AUDIO_PLACEHOLDER) == len(example["audio_span_groups"])


def test_transcript_rendered_once_per_joint_prompt() -> None:
    rows = synthetic_rows({"300": (10, 0)}, )
    for row in rows:
        row["full_participant_transcript"] = "line one\nline two"
        row["full_participant_transcript_sha256"] = hashlib.sha256(b"line one\nline two").hexdigest()
    examples = build_examples(rows, joint_config(use_audio=True, use_text=True), "train")
    prompt = examples[0]["prompt_text"]
    assert prompt.count("The transcript of the subject's speech is:") == 1
    assert prompt.count(AUDIO_PLACEHOLDER) == 4
    assert "line one" in prompt


def test_no_transcript_in_audio_only_prompts() -> None:
    rows = synthetic_rows({"300": (10, 0)})
    examples = build_examples(rows, joint_config(use_audio=True, use_text=False), "train")
    prompt = examples[0]["prompt_text"]
    assert "transcript" not in prompt.lower()
    assert "line" not in prompt


def test_total_chunk_count_absent_from_prompts() -> None:
    rows = synthetic_rows({"300": (15, 0)})
    examples = build_examples(rows, joint_config(use_audio=True, use_text=True), "val")
    for example in examples:
        prompt = example["prompt_text"]
        assert "15 segments" not in prompt
        assert "num_chunks" not in prompt
        assert prompt.count("segment") == 1


def _fake_checkpoint_dir(tmp_path: Path, config: dict, train_rows: list[dict], selected_epoch: int, epochs: int, tamper_hash: bool = False) -> Path:
    checkpoint = tmp_path / "run" / "fold_0" / "best_model"
    logs = checkpoint.parent / "logs"
    logs.mkdir(parents=True)
    train_examples = build_examples(train_rows, config, "train")
    schedules, audit = build_joint_epoch_schedule(
        train_examples,
        policy="joint_random_k",
        k=4,
        seed=int(config["seed"]),
        epochs=epochs,
        loss_weight_rescale="mean_one",
    )
    if tamper_hash:
        audit = dict(audit)
        audit["schedule_sha256"] = "0" * 64
        audit["bundle_membership_sha256"] = "0" * 64
    save_json(audit, logs / "daic_chunk_schedule_audit.json")
    save_json(
        {"selected_epoch": selected_epoch, "selection_metric": "inner_val_positive_f1", "selection_metric_mode": "max"},
        logs / "selected_checkpoint_selection_metrics.json",
    )
    return checkpoint


def test_selected_epoch_fit_reproduces_schedule_memberships(tmp_path: Path) -> None:
    rows = synthetic_rows({"300": (10, 0), "301": (15, 1), "385": (3, 0)})
    config = joint_config()
    config["training"]["num_train_epochs"] = 4
    checkpoint = _fake_checkpoint_dir(tmp_path, config, rows, selected_epoch=3, epochs=4)
    fit_examples, provenance = _selected_epoch_fit_examples(checkpoint, config, rows, 0)
    assert provenance["head_fit_view"] == "selected_checkpoint_training_epoch"
    assert provenance["selected_epoch"] == 3
    assert provenance["schedule_epoch_index"] == 2
    assert len(fit_examples) == 3
    assert {e["subject_id"] for e in fit_examples} == {"300", "301", "385"}
    for example in fit_examples:
        assert example["prompt_text"].count(AUDIO_PLACEHOLDER) == len(example["audio_span_groups"])
    assert "385" in provenance["schedule_sha256"] or provenance["schedule_sha256"]


def test_selected_epoch_fit_refuses_missing_or_inconsistent(tmp_path: Path) -> None:
    rows = synthetic_rows({"300": (10, 0)})
    config = joint_config()
    config["training"]["num_train_epochs"] = 2
    checkpoint = tmp_path / "best_model"
    checkpoint.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="selected_checkpoint_selection_metrics"):
        _selected_epoch_fit_examples(checkpoint, config, rows, 0)
    tampered = _fake_checkpoint_dir(tmp_path, config, rows, selected_epoch=1, epochs=2, tamper_hash=True)
    with pytest.raises(ValueError, match="does not match the saved"):
        _selected_epoch_fit_examples(tampered, config, rows, 0)


def test_production_head_cache_cardinality() -> None:
    if not PACKED30_MANIFEST.is_file():
        raise FileNotFoundError(f"Requires {PACKED30_MANIFEST}; build the packed30 manifest first.")
    config = joint_config()
    config["training"]["num_train_epochs"] = 20
    rows = [json.loads(line) for line in PACKED30_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_rows = [row for row in rows if row["split_original"] == "train"]
    test_rows = [row for row in rows if row["split_original"] == "test"]
    fit = build_examples(train_rows, config, "train")
    schedules, _ = build_joint_epoch_schedule(
        fit, policy="joint_random_k", k=4, seed=1337, epochs=20, loss_weight_rescale="mean_one"
    )
    assert len(schedules[0]) == 107
    assert len({e["subject_id"] for e in schedules[0]}) == 107
    test_examples = build_examples(test_rows, config, "test")
    assert len(test_examples) == 617
    assert len({e["subject_id"] for e in test_examples}) == 47


def test_packed30_single_chunk_behavior_unchanged() -> None:
    rows = synthetic_rows({"300": (3, 0)})
    config = joint_config()
    config["data"]["sample_mode"] = "participant_speech_packed30"
    config["data"]["train_chunk_policy"] = "all_chunks_subject_normalized"
    config["data"]["eval_chunk_policy"] = "all_chunks_mean_score"
    examples = build_examples(rows, config, "train")
    assert len(examples) == 3
    assert all(example["prompt_text"].count(AUDIO_PLACEHOLDER) == 1 for example in examples)
    assert all(uses_audio_spans(example) for example in examples)
    assert all(example["audio_paths"] == [] for example in examples)
    assert all("audio_span_groups" not in example for example in examples)


def test_canonical_subject_audio_joint_behavior_unchanged() -> None:
    config = {
        "dataset": "daic",
        "seed": 1337,
        "prompt": {"system": "system", "user_template": "{audio_context_block} {label_instruction}"},
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "data": {
            "use_audio": True,
            "use_text": False,
            "sample_mode": "subject_audio",
            "chunks_per_subject": 4,
            "train_chunk_policy": "joint_random_k",
            "train_chunks_per_subject": 4,
            "eval_chunk_policy": "balanced_joint_cover",
            "eval_chunks_per_subject": 4,
            "max_audio_seconds_per_chunk": 30.0,
        },
        "evaluation": {"subject_score_aggregation": "mean_score"},
    }
    rows = []
    for index in range(10):
        rows.append(
            {
                "dataset": "daic",
                "subject_id": "300",
                "sample_id": f"300_{index}",
                "chunk_id": f"{index:03d}",
                "label": 0,
                "label_text": "Non-depressed",
                "transcript": "",
                "audio_path": f"/tmp/300_{index}.wav",
            }
        )
    path_examples = build_examples(rows, config, "train")
    assert len(path_examples) == 1
    assert "subject_chunk_paths" in path_examples[0]
    schedules, audit = build_joint_epoch_schedule(
        path_examples, policy="joint_random_k", k=4, seed=1337, epochs=3, loss_weight_rescale="mean_one"
    )
    path_memberships = [[row["bundle_chunk_ids"] for row in epoch] for epoch in schedules]

    span_rows = synthetic_rows({"300": (10, 0)})
    span_examples = build_examples(span_rows, joint_config(), "train")
    span_schedules, _ = build_joint_epoch_schedule(
        span_examples, policy="joint_random_k", k=4, seed=1337, epochs=3, loss_weight_rescale="mean_one"
    )
    span_memberships = [[row["bundle_chunk_ids"] for row in epoch] for epoch in span_schedules]
    assert path_memberships == span_memberships
    assert audit["epoch_mean_effective_weights"] == [1.0] * 3


def _run_dry_run(script: str, run_id: str) -> subprocess.CompletedProcess:
    env = dict(__import__("os").environ)
    env.update(
        {
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "RUN_ID": run_id,
            "DRY_RUN": "1",
        }
    )
    return subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_submit_production_dry_run_two_chains() -> None:
    result = _run_dry_run("submit_daic_participant_packed30_jointk4.sh", "dry_prod_jointk4")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("  train   : sbatch --gres=gpu:4") == 2
    assert "DRY RUN complete: no jobs submitted" in result.stdout


def test_submit_smoke_dry_run_two_chains() -> None:
    result = _run_dry_run("submit_daic_participant_packed30_jointk4_smoke.sh", "dry_smoke_jointk4")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("  smoke-train : sbatch --gres=gpu:1") == 2
    assert "DRY RUN complete: no jobs submitted" in result.stdout


def test_eval_determinism_comparison_accepts_identical_passes(tmp_path: Path) -> None:
    from scripts.compare_eval_determinism import compare_determinism

    pass1 = tmp_path / "pass1"
    pass2 = tmp_path / "pass2"
    pass1.mkdir()
    pass2.mkdir()
    sample_rows = [
        {"subject_id": "300", "sample_id": "300__bundle_000", "dep_score": -0.5, "non_score": -0.2},
        {"subject_id": "301", "sample_id": "301__bundle_000", "dep_score": 0.1, "non_score": -0.3},
    ]
    (pass1 / "predictions_sample_level.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in sample_rows), encoding="utf-8"
    )
    (pass2 / "predictions_sample_level.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in sample_rows), encoding="utf-8"
    )
    (pass1 / "predictions_subject_level.csv").write_text(
        "subject_id,label,prediction\n300,0,0\n301,1,1\n", encoding="utf-8"
    )
    (pass2 / "predictions_subject_level.csv").write_text(
        "label,prediction,subject_id\n0,0,300\n1,1,301\n", encoding="utf-8"
    )
    metrics = {"positive_f1": 0.5, "confusion_matrix": [[2, 1], [0, 1]]}
    (pass1 / "metrics_original_teacher_forced.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (pass2 / "metrics_original_teacher_forced.json").write_text(
        json.dumps(metrics, sort_keys=True), encoding="utf-8"
    )
    (pass1 / "final_and_best_validation_metrics.json").write_text("{}", encoding="utf-8")
    (pass2 / "final_and_best_validation_metrics.json").write_text("{}", encoding="utf-8")
    assert compare_determinism(pass1, pass2) == []


def test_eval_determinism_comparison_rejects_score_differences(tmp_path: Path) -> None:
    from scripts.compare_eval_determinism import compare_determinism

    pass1 = tmp_path / "pass1"
    pass2 = tmp_path / "pass2"
    pass1.mkdir()
    pass2.mkdir()
    (pass1 / "predictions_sample_level.jsonl").write_text(
        '{"subject_id": "300", "dep_score": -0.5, "non_score": -0.2}\n', encoding="utf-8"
    )
    (pass2 / "predictions_sample_level.jsonl").write_text(
        '{"subject_id": "300", "dep_score": -0.4, "non_score": -0.2}\n', encoding="utf-8"
    )
    for name in ("predictions_subject_level.csv", "metrics_original_teacher_forced.json", "final_and_best_validation_metrics.json"):
        (pass1 / name).write_text("{}" if name.endswith(".json") else "subject_id,label\n300,0\n", encoding="utf-8")
        (pass2 / name).write_text("{}" if name.endswith(".json") else "subject_id,label\n300,0\n", encoding="utf-8")
    mismatches = compare_determinism(pass1, pass2)
    assert any("predictions_sample_level.jsonl" in message for message in mismatches)


def _repo_short_commit() -> str:
    provenance = PROJECT_ROOT / ".provenance" / "git_commit.txt"
    if provenance.is_file():
        return provenance.read_text(encoding="utf-8").strip()[:8]
    return "deadbeef"


def test_auditor_smoke_mode_includes_smoke_runs(tmp_path: Path) -> None:
    from scripts.audit_daic_participant_packed30_jointk4 import JointK4Auditor

    short = _repo_short_commit()
    for modality in ("audio_only", "audio_text"):
        run_dir = tmp_path / modality / f"smoke_p30_jointk4_{modality}_s1337_{short}" / "fold_0"
        run_dir.mkdir(parents=True)
    production = JointK4Auditor(tmp_path, tmp_path, tmp_path, smoke=False)
    assert set(production._runs_by_modality()) == set()
    smoke = JointK4Auditor(tmp_path, tmp_path, tmp_path, smoke=True)
    assert set(smoke._runs_by_modality()) == {"audio_only", "audio_text"}


def test_auditor_smoke_mode_ignores_stale_commit_runs(tmp_path: Path) -> None:
    from scripts.audit_daic_participant_packed30_jointk4 import JointK4Auditor

    short = _repo_short_commit()
    for modality in ("audio_only", "audio_text"):
        fresh = tmp_path / modality / f"smoke_p30_jointk4_{modality}_s1337_{short}" / "fold_0"
        fresh.mkdir(parents=True)
        stale = tmp_path / modality / f"smoke_p30_jointk4_{modality}_s1337_00000000" / "fold_0"
        stale.mkdir(parents=True)
    smoke = JointK4Auditor(tmp_path, tmp_path, tmp_path, smoke=True)
    selected = smoke._runs_by_modality()
    assert set(selected) == {"audio_only", "audio_text"}
    for modality, fold_dirs in selected.items():
        assert all(short in str(fold_dir) for fold_dir in fold_dirs)
    production = JointK4Auditor(tmp_path, tmp_path, tmp_path, smoke=False)
    assert set(production._runs_by_modality()) == set()
