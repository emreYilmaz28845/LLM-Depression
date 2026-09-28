"""Tests for the smoke acceptance gate.

The fixture builds the evidence, sidecar and log layout the campaign produces,
so the gate's checks and its fail-closed behaviour are exercised without any real
run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tools import verify_qwen3_daic_label_smoke as gate
from tools.qwen3_daic_label_vocab_matrix import SMOKE_CAMPAIGN, SOURCES, run_name

TEST_SUBJECTS = 47
ARMS = ("ab", "01", "truefalse", "yesno", "en")
MANIFEST_HASH = "72e2dd204b915ccba3ebf922f030531fe5678b3ea8c9c52b81b41242fe9dda17"
SPLIT_HASH = "441333e0c88845eeacba9ea5355a8920cdd1f70e8cf7a7c15b9547b46da51473"


def _write_fixture(tmp_path: Path, *, break_chain: tuple[str, str] | None = None) -> tuple[Path, Path, Path]:
    evidence_root = tmp_path / "output_model"
    logs_root = tmp_path / "logs" / SMOKE_CAMPAIGN
    (logs_root / "slurm_train" / "daic").mkdir(parents=True, exist_ok=True)
    (logs_root / "slurm_eval" / "daic").mkdir(parents=True, exist_ok=True)
    split_metadata = logs_root / "daic_subject_partitions.json"
    split_metadata.write_text(
        json.dumps(
            [
                {"subject_id": str(300 + index), "partition": "train" if index < 100 else "test", "label": index % 2, "label_text": "Depressed"}
                for index in range(100 + TEST_SUBJECTS)
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    index = 0
    for modality in SOURCES:
        for arm in ARMS:
            index += 1
            run = run_name(smoke=True, modality=modality, tag=arm, seed=1337)
            fold = evidence_root / SMOKE_CAMPAIGN / modality / "daic" / run / "fold_0"
            standalone = fold / "best_model" / "standalone_eval"
            standalone.mkdir(parents=True, exist_ok=True)
            broken_here = break_chain == (modality, arm)
            train_job, eval_job = str(46758470 + index), str(46758570 + index)

            (fold / "run_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "config": {
                            "dataset": "daic",
                            "seed": 1337,
                            "split": {"seed": 1337},
                            "labels": {"label_vocab_version": "short_internal_ab_labels"},
                            "output_dirs": {"run_root": f"/permanent/output_model/{SMOKE_CAMPAIGN}/{modality}/daic"},
                            "evaluation": {
                                "sample_prediction_mode": "likelihood",
                                "evaluation_view": "harmonized_all_windows_full_coverage",
                            },
                        },
                        "training_strategy": {"strategy": "fsdp", "world_size": 4},
                        "manifest_hash": MANIFEST_HASH,
                        "split_metadata_hash": SPLIT_HASH,
                    }
                ),
                encoding="utf-8",
            )
            (fold / "metadata.json").write_text("{}\n", encoding="utf-8")
            (fold / "status.json").write_text(
                json.dumps({"state": "LOCALLY_VALIDATED" if broken_here else "REPORTABLE", "attempt_id": f"att-{modality}-{arm}"}),
                encoding="utf-8",
            )
            events = [
                {"job_key": "train", "event_type": "SUBMITTED", "slurm_job_id": train_job},
                {"job_key": "best_eval", "event_type": "SUBMITTED", "slurm_job_id": eval_job, "dependency_job_ids": [train_job]},
                {"job_key": "train", "event_type": "COMPLETED", "exit_code": "0:0", "slurm_job_id": train_job},
                {"job_key": "best_eval", "event_type": "COMPLETED", "exit_code": "0:0", "slurm_job_id": eval_job},
            ]
            (fold / "jobs.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
            (fold / "artifacts.json").write_text("{}\n", encoding="utf-8")
            (fold / "evaluations.json").write_text("{}\n", encoding="utf-8")

            (standalone / "metrics_likelihood.json").write_text(
                json.dumps(
                    {
                        "aggregation_level": "subject",
                        "evaluation_view": "harmonized_all_windows_full_coverage",
                        "checkpoint_name": "best_model",
                        "num_subjects": TEST_SUBJECTS,
                        "binary_strict_macro_f1": 0.5,
                        "binary_strict_positive_f1": 0.4,
                        "binary_strict_uar": 0.55,
                    }
                ),
                encoding="utf-8",
            )
            rows = ["subject_id,label,prediction_text"]
            rows += [f"{300 + index},{index % 2},{'Depressed' if index % 2 else 'Non-depressed'}" for index in range(TEST_SUBJECTS)]
            (standalone / "predictions_subject_level.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

            (logs_root / "slurm_train" / "daic" / f"train-{train_job}-2026-09-28_18:28:51.log").write_text(
                f"Run Name: {run}\n"
                "Qwen3.8 LoRA audit | matched_modules=256 lora_trainable_params=79691776\n"
                "trainable params: 79,691,776 || all params: 27,436,420,336 || trainable%: 0.2905\n"
                "Training strategy | strategy=fsdp world_size=4 per_device_train_batch_size=1\n"
                "Audio encoder frozen: whisper_tower requires_grad=False\n"
                "2026-09-28 18:36:11,209 | INFO | epoch=1 step=25 loss=1.234\n",
                encoding="utf-8",
            )
            (logs_root / "slurm_train" / "daic" / f"eval-{eval_job}-2026-09-28_18:43:56.log").write_text(
                f"Run Name: {run}\n"
                "Checkpoint Dir: /permanent/output_model/fold_0/best_model\n"
                "'num_subjects': 47, 'aggregation_level': 'subject', \"checkpoint_name': 'best_model'\"\n"
                "FINAL EVALUATION RESULT | split=test backend=likelihood aggregation=subject\n",
                encoding="utf-8",
            )
    return evidence_root, logs_root, split_metadata


def _run(tmp_path: Path, **kwargs):
    evidence_root, logs_root, split_metadata = _write_fixture(tmp_path, break_chain=kwargs.get("break_chain"))
    output = tmp_path / "gate.json"
    code = gate.main(
        [
            "--evidence-root", str(evidence_root),
            "--logs-root", str(logs_root),
            "--split-metadata", str(split_metadata),
            "--output", str(output),
        ]
        + (kwargs.get("extra_args") or [])
    )
    return code, json.loads(output.read_text(encoding="utf-8"))


def test_expected_test_subjects_reads_the_split(tmp_path: Path) -> None:
    _, _, split_metadata = _write_fixture(tmp_path)
    assert gate.expected_test_subjects(split_metadata) == TEST_SUBJECTS


def test_gate_passes_on_a_complete_smoke_set(tmp_path: Path) -> None:
    code, report = _run(tmp_path)
    assert code == 0, report["failures"]
    assert report["passed"] is True
    assert len(report["chains"]) == 15
    assert all(chain["subject_rows"] == TEST_SUBJECTS for chain in report["chains"])
    first = report["chains"][0]
    assert first["train_log"]["lora_targets"]["matched"] is True
    assert first["train_log"]["audio_encoder_frozen"]["matched"] is True
    assert first["train_log"]["distributed_shape"]["matched"] is True
    assert first["train_losses"]["last"] == pytest.approx(1.234)
    assert first["config_qualifiers"]["manifest_hash"] == MANIFEST_HASH
    assert report["manifest_hashes"] == [MANIFEST_HASH]


def test_gate_checks_the_expected_provenance_hashes(tmp_path: Path) -> None:
    code, report = _run(
        tmp_path,
        extra_args=[
            "--expected-manifest-hash", MANIFEST_HASH,
            "--expected-split-metadata-hash", SPLIT_HASH,
        ],
    )
    assert code == 0, report["failures"]
    wrong, wrong_report = _run(tmp_path, extra_args=["--expected-manifest-hash", "0" * 64])
    assert wrong == 1
    assert any("differs from the expected" in message for message in wrong_report["failures"])


def test_gate_fails_closed_when_a_chain_is_not_reportable(tmp_path: Path) -> None:
    code, report = _run(tmp_path, break_chain=("audio_only", "yesno"))
    assert code == 1
    assert any("lifecycle state" in message and "yesno" in message for message in report["failures"])


def test_gate_fails_on_missing_logs(tmp_path: Path) -> None:
    evidence_root, logs_root, split_metadata = _write_fixture(tmp_path)
    run = run_name(smoke=True, modality="text_only", tag="en", seed=1337)
    (logs_root / "slurm_train" / "daic" / "train-46758471-2026-09-28_18:28:51.log").unlink()
    # The first chain (text_only/ab) is index 1, so its train job id is 46758471.
    output = tmp_path / "gate.json"
    code = gate.main(
        [
            "--evidence-root", str(evidence_root),
            "--logs-root", str(logs_root),
            "--split-metadata", str(split_metadata),
            "--output", str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 1
    assert any("missing training log" in message for message in report["failures"])
    assert run  # the run name is only used to document which chain lost its log


def test_gate_fails_on_non_finite_metric(tmp_path: Path) -> None:
    evidence_root, logs_root, split_metadata = _write_fixture(tmp_path)
    run = run_name(smoke=True, modality="audio_text", tag="ab", seed=1337)
    metrics_path = evidence_root / SMOKE_CAMPAIGN / "audio_text" / "daic" / run / "fold_0" / "best_model" / "standalone_eval" / "metrics_likelihood.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["binary_strict_macro_f1"] = None
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "gate.json"
    code = gate.main(
        [
            "--evidence-root", str(evidence_root),
            "--logs-root", str(logs_root),
            "--split-metadata", str(split_metadata),
            "--output", str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 1
    assert any("binary_strict_macro_f1 is not finite" in message for message in report["failures"])


def test_gate_fails_when_subject_coverage_is_short(tmp_path: Path) -> None:
    evidence_root, logs_root, split_metadata = _write_fixture(tmp_path)
    run = run_name(smoke=True, modality="text_only", tag="ab", seed=1337)
    predictions = evidence_root / SMOKE_CAMPAIGN / "text_only" / "daic" / run / "fold_0" / "best_model" / "standalone_eval" / "predictions_subject_level.csv"
    rows = predictions.read_text(encoding="utf-8").splitlines()
    predictions.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    output = tmp_path / "gate.json"
    code = gate.main(
        [
            "--evidence-root", str(evidence_root),
            "--logs-root", str(logs_root),
            "--split-metadata", str(split_metadata),
            "--output", str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 1
    assert any("test subjects" in message for message in report["failures"])


def test_gate_requires_the_token_audit_configs_to_be_present() -> None:
    """The gate reads the real generated configs, so a missing one breaks it."""
    from scripts.build_qwen3_daic_label_configs import generated_name

    assert generated_name("text_only", "ab").endswith("_ab_v1.yaml")
    assert (gate.ROOT / "configs/labels" / generated_name("audio_only", "en")).is_file()
