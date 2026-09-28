"""Tests for the smoke acceptance gate.

The fixture builds the evidence, sidecar and log layout the campaign produces, so
the gate's checks and its fail-closed behaviour are exercised without any real
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


def _write_fixture(tmp_path: Path, *, break_chain: tuple[str, str] | None = None) -> tuple[Path, Path, Path]:
    evidence_root = tmp_path / "output_model"
    logs_root = tmp_path / "logs" / SMOKE_CAMPAIGN
    (logs_root / "slurm_train" / "daic").mkdir(parents=True, exist_ok=True)
    (logs_root / "slurm_eval" / "daic").mkdir(parents=True, exist_ok=True)
    split_metadata = logs_root / "daic_subject_partitions.json"
    split_metadata.write_text(
        json.dumps(
            [{"subject_id": str(300 + index), "partition": "train" if index < 100 else "test", "label": index % 2, "label_text": "Depressed"} for index in range(100 + TEST_SUBJECTS)]
        )
        + "\n",
        encoding="utf-8",
    )

    for modality in SOURCES:
        for arm in ARMS:
            run = run_name(smoke=True, modality=modality, tag=arm, seed=1337)
            fold = evidence_root / SMOKE_CAMPAIGN / modality / "daic" / run / "fold_0"
            standalone = fold / "best_model" / "standalone_eval"
            standalone.mkdir(parents=True, exist_ok=True)
            broken_here = break_chain == (modality, arm)
            (fold / "run_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "dataset": "daic",
                        "seed": 1337,
                        "split": {"seed": 1337},
                        "labels": {"label_vocab_version": "short_internal_ab_labels"},
                        "output_dirs": {"run_root": f"/permanent/output_model/{SMOKE_CAMPAIGN}/{modality}/daic"},
                        "evaluation": {
                            "sample_prediction_mode": "likelihood",
                            "evaluation_view": "harmonized_all_windows_full_coverage",
                        },
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
                {"job_key": key, "event_type": "COMPLETED", "exit_code": "0:0"}
                for key in ("train", "best_eval")
            ]
            (fold / "jobs.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
            (fold / "artifacts.json").write_text("{}\n", encoding="utf-8")
            (fold / "evaluations.json").write_text("{}\n", encoding="utf-8")

            (standalone / "metrics_likelihood.json").write_text(
                json.dumps(
                    {
                        "aggregation_level": "subject",
                        "headline_metrics": {
                            "binary_strict_macro_f1": 0.5,
                            "binary_strict_positive_f1": 0.4,
                            "binary_strict_uar": 0.55,
                        },
                    }
                ),
                encoding="utf-8",
            )
            rows = ["subject_id,label,prediction_text"]
            rows += [f"{300 + index},{index % 2},{'Depressed' if index % 2 else 'Non-depressed'}" for index in range(TEST_SUBJECTS)]
            (standalone / "predictions_subject_level.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

            (logs_root / "slurm_train" / "daic" / f"train-1234-{run}.log").write_text(
                f"Run Name: {run}\n"
                "Qwen3.8 LoRA audit | matched_modules=256 lora_trainable_params=79691776\n"
                "trainable params: 79,691,776 || all params: 27,436,420,336 || trainable%: 0.2905\n"
                "Training strategy | strategy=fsdp world_size=4 per_device_train_batch_size=1\n"
                "Audio encoder frozen: whisper_tower requires_grad=False\n"
                "{'loss': 1.234, 'epoch': 0.5}\n",
                encoding="utf-8",
            )
            (logs_root / "slurm_eval" / "daic" / f"eval-5678-{run}.log").write_text(
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


def test_gate_fails_closed_when_a_chain_is_not_reportable(tmp_path: Path) -> None:
    code, report = _run(tmp_path, break_chain=("audio_only", "yesno"))
    assert code == 1
    assert any("lifecycle state" in message and "yesno" in message for message in report["failures"])


def test_gate_fails_on_missing_logs(tmp_path: Path) -> None:
    evidence_root, logs_root, split_metadata = _write_fixture(tmp_path)
    run = run_name(smoke=True, modality="text_only", tag="en", seed=1337)
    (logs_root / "slurm_train" / "daic" / f"train-1234-{run}.log").unlink()
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


def test_gate_fails_on_non_finite_metric(tmp_path: Path) -> None:
    evidence_root, logs_root, split_metadata = _write_fixture(tmp_path)
    run = run_name(smoke=True, modality="audio_text", tag="ab", seed=1337)
    metrics_path = evidence_root / SMOKE_CAMPAIGN / "audio_text" / "daic" / run / "fold_0" / "best_model" / "standalone_eval" / "metrics_likelihood.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["headline_metrics"]["binary_strict_macro_f1"] = None
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


def test_gate_requires_the_token_audit_configs_to_be_present(tmp_path: Path) -> None:
    """The gate reads the real generated configs, so a missing one breaks it."""
    from scripts.build_qwen3_daic_label_configs import generated_name

    assert generated_name("text_only", "ab").endswith("_ab_v1.yaml")
    assert (gate.ROOT / "configs/labels" / generated_name("audio_only", "en")).is_file()
