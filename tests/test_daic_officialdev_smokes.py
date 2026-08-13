from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SMOKE_SUBMITTER = PROJECT_ROOT / "scripts/submit_daic_officialdev_smokes.sh"
SELECTION_BUILDER = PROJECT_ROOT / "scripts/build_daic_officialdev_smoke_selection.py"
SMOKE_AUDIT = PROJECT_ROOT / "scripts/audit_daic_officialdev_smoke.py"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _make_smoke_parent(tmp_path: Path, split_metadata_path: Path) -> Path:
    """A smoke-trained parent fold: limited official-train subjects, both
    classes, plus a partition file and run_config."""
    fold = tmp_path / "parent" / "fold_0"
    (fold / "logs").mkdir(parents=True)
    (fold / "best_model").mkdir()
    run_config = {"split_metadata_path": str(split_metadata_path)}
    (fold / "run_config.yaml").write_text(json.dumps(run_config), encoding="utf-8")
    (fold / "logs" / "split_used.json").write_text(
        json.dumps(
            {
                "train_subject_ids": ["t000", "t001", "t002", "t003", "t004", "t005"],
                "selection_subject_ids": ["t086", "t087", "t088", "t089"],
                "train_inner_subject_ids": ["t000", "t001", "t002", "t003", "t004", "t005"],
                "val_inner_subject_ids": ["t086", "t087", "t088", "t089"],
            }
        ),
        encoding="utf-8",
    )
    return fold


def _write_partitions(tmp_path: Path) -> Path:
    path = tmp_path / "partitions.json"
    rows = []
    # Official train 107: t000..t106 (t000-t005 depressed for the smoke pick).
    for index in range(107):
        rows.append(
            {
                "subject_id": f"t{index:03d}",
                "partition": "train",
                "label": 1 if index in (0, 1, 86, 87) else 0,
            }
        )
    for index in range(35):
        rows.append({"subject_id": f"v{index:02d}", "partition": "val", "label": index % 2})
    for index in range(47):
        rows.append({"subject_id": f"e{index:02d}", "partition": "test", "label": index % 2})
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_selection_builder_picks_both_classes_from_official_train(tmp_path: Path) -> None:
    partitions = _write_partitions(tmp_path)
    parent = _make_smoke_parent(tmp_path, partitions)
    output = tmp_path / "selection.json"
    result = _run(
        [
            sys.executable, str(SELECTION_BUILDER),
            "--parent-fold-dir", str(parent),
            "--output", str(output),
        ]
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["outer_train"]) == 4
    assert len(payload["final_eval"]) == 4
    assert set(payload["outer_train"]).isdisjoint(payload["final_eval"])
    partition_rows = json.loads(partitions.read_text(encoding="utf-8"))
    labels = {row["subject_id"]: row["label"] for row in partition_rows}
    assert {labels[s] for s in payload["outer_train"]} == {0, 1}
    assert {labels[s] for s in payload["final_eval"]} == {0, 1}
    # No official dev/test subjects anywhere.
    official_dev = {row["subject_id"] for row in partition_rows if row["partition"] == "val"}
    official_test = {row["subject_id"] for row in partition_rows if row["partition"] == "test"}
    assert not (set(payload["outer_train"]) | set(payload["final_eval"])) & official_dev
    assert not (set(payload["outer_train"]) | set(payload["final_eval"])) & official_test
    # Deterministic.
    output2 = tmp_path / "selection2.json"
    _run([sys.executable, str(SELECTION_BUILDER), "--parent-fold-dir", str(parent), "--output", str(output2)])
    assert output.read_bytes() == output2.read_bytes()


def test_smoke_audit_accepts_and_rejects(tmp_path: Path) -> None:
    partitions = _write_partitions(tmp_path)
    parent = _make_smoke_parent(tmp_path, partitions)
    selection = json.loads(
        (tmp_path / "sel.json").read_text()
        if (tmp_path / "sel.json").is_file()
        else "{}"
    )
    result = _run(
        [
            sys.executable, str(SELECTION_BUILDER),
            "--parent-fold-dir", str(parent),
            "--output", str(tmp_path / "sel.json"),
        ]
    )
    assert result.returncode == 0, result.stderr
    selection = json.loads((tmp_path / "sel.json").read_text(encoding="utf-8"))
    fit = selection["outer_train"]
    eval_ = selection["final_eval"]

    attempt = tmp_path / "attempt"
    features = attempt / "hidden_features"
    features.mkdir(parents=True)
    extraction_metadata = {
        "cache_config": {"subject_selection_sha256": "a" * 64},
        "evaluation_provenance": {"evaluation_protocol": "daic_official_train_inner_split_dev_evaluation"},
        "partitions": {
            "outer_train": {"rows": 8, "subjects": 4},
            "final_eval": {"rows": 8, "subjects": 4},
        },
    }
    (features / "extraction_metadata.json").write_text(json.dumps(extraction_metadata), encoding="utf-8")
    partition_rows = json.loads(partitions.read_text(encoding="utf-8"))
    labels = {row["subject_id"]: row["label"] for row in partition_rows}

    def write_rows(name: str, subjects: list[str]) -> None:
        rows = []
        for subject_id in subjects:
            for chunk in range(2):
                rows.append(
                    {
                        "sample_id": f"{subject_id}_{chunk}",
                        "subject_id": subject_id,
                        "label": labels[subject_id],
                    }
                )
        (features / f"{name}_rows.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )

    write_rows("outer_train", fit)
    write_rows("final_eval", eval_)
    classifiers = attempt / "hidden_classifiers"
    for variant in ("logreg_raw", "xgb_raw"):
        vdir = classifiers / variant
        vdir.mkdir(parents=True)
        (vdir / "metrics.json").write_text(json.dumps({"macro_f1": 0.8}), encoding="utf-8")
        (vdir / "classifier_metadata.json").write_text(
            json.dumps({"prediction_backend": f"qwen_hidden_{variant}"}), encoding="utf-8"
        )
        (vdir / "predictions_subject_level.jsonl").write_text(
            "\n".join(
                json.dumps({"subject_id": subject_id, "prediction": 1})
                for subject_id in eval_
            )
            + "\n",
            encoding="utf-8",
        )
    audit_out = tmp_path / "audit.json"
    result = _run(
        [
            sys.executable, str(SMOKE_AUDIT),
            "--attempt-dir", str(attempt),
            "--parent-fold-dir", str(parent),
            "--backbone", "qwen",
            "--modality", "audio_text",
            "--output", str(audit_out),
        ]
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(audit_out.read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record["official_train_only"] is True
    assert record["official_dev_absent"] is True

    # A dev subject in the eval rows must fail the audit.
    extraction_metadata["partitions"]["final_eval"] = {"rows": 4, "subjects": 2}
    (features / "extraction_metadata.json").write_text(json.dumps(extraction_metadata), encoding="utf-8")
    write_rows("final_eval", [eval_[0], "v00"])
    result = _run(
        [
            sys.executable, str(SMOKE_AUDIT),
            "--attempt-dir", str(attempt),
            "--parent-fold-dir", str(parent),
            "--backbone", "qwen",
            "--modality", "audio_text",
            "--output", str(tmp_path / "audit2.json"),
        ]
    )
    assert result.returncode == 1
    assert "official-development subjects entered the smoke" in result.stderr


def test_smoke_submitter_dry_run_has_20_jobs_and_no_mutation() -> None:
    smoke_id = "test_smoke_000001"
    smoke_root = PROJECT_ROOT / "outputs/daic_officialdev_smokes" / smoke_id
    if smoke_root.exists():
        import shutil

        shutil.rmtree(smoke_root)
    env = {
        **os.environ,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "SMOKE_ID": smoke_id,
        "DRY_RUN": "1",
    }
    result = _run(["bash", str(SMOKE_SUBMITTER)], env=env)
    assert result.returncode == 0, result.stderr
    dry_lines = [line for line in result.stderr.splitlines() if line.startswith("DRY_RUN")]
    assert len(dry_lines) == 20, f"expected 20 smoke jobs, got {len(dry_lines)}"
    assert "contracts=6" in result.stdout
    assert "trains=2" in result.stdout
    assert "extract=6" in result.stdout
    assert "heads=6" in result.stdout
    assert not smoke_root.exists(), "dry run must not create the smoke root"


def test_smoke_workers_offline_and_no_gpu_for_heads() -> None:
    for script in (
        "scripts/run_daic_officialdev_contract_slurm.sh",
        "scripts/run_daic_officialdev_smoke_train_slurm.sh",
        "scripts/run_daic_officialdev_smoke_extract_slurm.sh",
        "scripts/run_daic_officialdev_smoke_heads_slurm.sh",
        "scripts/submit_daic_officialdev_smokes.sh",
    ):
        result = _run(["bash", "-n", str(PROJECT_ROOT / script)])
        assert result.returncode == 0, f"{script}: {result.stderr}"
        text = (PROJECT_ROOT / script).read_text(encoding="utf-8")
        for forbidden in ("huggingface-cli", "pip ", "git clone", "wget ", "curl "):
            assert forbidden not in text, f"{script} contains {forbidden!r}"
    for script in (
        "scripts/run_daic_officialdev_contract_slurm.sh",
        "scripts/run_daic_officialdev_smoke_train_slurm.sh",
        "scripts/run_daic_officialdev_smoke_extract_slurm.sh",
        "scripts/run_daic_officialdev_smoke_heads_slurm.sh",
    ):
        text = (PROJECT_ROOT / script).read_text(encoding="utf-8")
        assert "HF_HUB_OFFLINE=1" in text, script
        assert "TRANSFORMERS_OFFLINE=1" in text, script
    heads = (PROJECT_ROOT / "scripts/run_daic_officialdev_smoke_heads_slurm.sh").read_text(encoding="utf-8")
    assert "--gres=gpu" not in heads
    extract = (PROJECT_ROOT / "scripts/run_daic_officialdev_smoke_extract_slurm.sh").read_text(encoding="utf-8")
    assert "--gres=gpu:1" in extract
    assert "SUBJECT_SELECTION" in extract
