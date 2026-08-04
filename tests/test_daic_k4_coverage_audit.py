from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from scripts.evaluate_daic_comprehensive_views import view_construction_rows
from scripts.run_daic_k4_coverage_audit import audit_and_report, materialize_runtime_config


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_view_construction_rows_are_path_free_and_record_fixed_selection() -> None:
    rows = view_construction_rows(
        [
            {
                "subject_id": "s",
                "sample_id": "s",
                "label": 0,
                "subject_chunk_paths": [f"/private/s/{i}.wav" for i in range(10)],
                "subject_chunk_ids": [str(i) for i in range(10)],
                "audio_paths": [f"/private/s/{i}.wav" for i in (0, 3, 6, 9)],
            }
        ],
        "fixed4",
    )
    assert rows[0]["selected_chunk_ids"] == ["0", "3", "6", "9"]
    assert "/private" not in json.dumps(rows)


def test_runtime_config_adds_audit_controls_without_mutating_source(tmp_path: Path) -> None:
    source = tmp_path / "main.yaml"
    source.write_text(
        yaml.safe_dump({"dataset": "daic", "evaluation": {"aggregation_level": "subject"}}),
        encoding="utf-8",
    )
    before = source.read_text(encoding="utf-8")
    destination = materialize_runtime_config(source, tmp_path / "runtime.yaml")
    resolved = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert source.read_text(encoding="utf-8") == before
    assert resolved["evaluation"] == {
        "aggregation_level": "subject",
        "inference_dtype": "fp32",
        "candidate_batching": "sequential",
        "subject_score_aggregation": "mean_score",
        "reuse_derived_views": True,
    }


def test_complete_coverage_audit_and_report_pass(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "fold_0" / "best_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint.parent / "run_config.yaml").write_text(
        yaml.safe_dump(
            {
                "fold": 0,
                "config": {
                    "dataset": "daic",
                    "data": {
                        "sample_mode": "subject_audio",
                        "chunks_per_subject": 4,
                        "use_audio": True,
                        "use_text": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    _write_json(
        checkpoint.parent / "logs" / "split_used.json",
        {
            "train_subject_ids": ["train"],
            "selection_subject_ids": ["val"],
            "final_eval_subject_ids": ["n", "p"],
        },
    )
    subjects = [
        {"subject_id": "n", "label": 0, "prediction": 0, "score_margin": -1.0},
        {"subject_id": "p", "label": 1, "prediction": 1, "score_margin": 1.0},
    ]
    metrics = {
        "accuracy": 1.0,
        "positive_f1": 1.0,
        "macro_f1": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "binary_strict_confusion_matrix": [[1, 0], [0, 1]],
    }
    for view in ("fixed4", "mincover4", "fixed15"):
        _write_json(tmp_path / "out" / view / "metrics_original_teacher_forced.json", metrics)
        _write_csv(tmp_path / "out" / view / "predictions_subject_level.csv", subjects)
    fixed = [
        {
            "subject_id": sid,
            "sample_id": sid,
            "label": label,
            "num_chunks_available": n,
            "chunks_per_model_input": 4,
            "selected_chunk_ids": [str(i) for i in range(4)],
        }
        for sid, label, n in (("n", 0, 10), ("p", 1, 15))
    ]
    _write_jsonl(tmp_path / "out" / "fixed4" / "view_construction.jsonl", fixed)
    cover = []
    for sid, label, n in (("n", 0, 10), ("p", 1, 15)):
        bundles = n // __import__("math").gcd(n, 4)
        for bundle in range(bundles):
            cover.append(
                {
                    "subject_id": sid,
                    "sample_id": f"{sid}_{bundle}",
                    "label": label,
                    "num_chunks_available": n,
                    "chunks_per_model_input": 4,
                    "selected_chunk_ids": [str((bundle * 4 + offset) % n) for offset in range(4)],
                }
            )
    _write_jsonl(tmp_path / "out" / "mincover4" / "view_construction.jsonl", cover)
    _write_jsonl(
        tmp_path / "out" / "fixed15" / "predictions_sample_level.jsonl",
        [{"subject_id": sid, "label": label} for sid, label in (("n", 0), ("p", 1)) for _ in range(15)],
    )
    _write_json(
        tmp_path / "out" / "fixed15" / "derivation_metadata.json",
        {"actual_model_forwards": 0},
    )

    audit = audit_and_report(tmp_path / "out", checkpoint, expected_subjects=2)
    assert audit["passed"], audit["failures"]
    assert audit["changed_subjects"] == 0
    assert (tmp_path / "out" / "comparison.csv").is_file()
    assert (tmp_path / "out" / "results.md").is_file()
