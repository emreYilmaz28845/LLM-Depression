from __future__ import annotations

from collections import defaultdict

from src.merged.protocol import (
    DATASETS,
    audit_protocol_splits,
    build_dataset_aware_schedule,
    build_final_partitions,
    build_grouped_inner_folds,
    build_merged_manifest,
    build_protocol_splits,
    compute_hierarchical_example_weights,
)
from src.merged.runtime import fold_subject_ids


def _records():
    records = []
    for dataset in DATASETS:
        rows = []
        for subject_index in range(12):
            subject = f"{dataset[:3]}-{subject_index:02d}"
            label = subject_index % 2
            if dataset == "daic" and subject_index >= 10:
                split = "test"
            else:
                split = "train"
            for response in range(2):
                for window in range(response + 1):
                    rows.append(
                        {
                            "dataset": dataset,
                            "subject_id": subject,
                            "sample_id": f"{subject}-r{response}-w{window}",
                            "response_id": f"{subject}-r{response}",
                            "num_segments": response + 1,
                            "label": label,
                            "label_text": "Depressed" if label else "Non-depressed",
                            "transcript": "synthetic",
                            "audio_path": "synthetic.wav",
                            "split_original": split,
                        }
                    )
        records.append(
            {
                "dataset": dataset,
                "config_path": f"configs/{dataset}.yaml",
                "config": {"dataset": dataset},
                "manifest_hash": f"manifest-{dataset}",
                "rows": rows,
                "labels": {f"{dataset[:3]}-{i:02d}": i % 2 for i in range(12)},
                "folds": {},
                "official_test_subject_ids": [f"{dataset[:3]}-10", f"{dataset[:3]}-11"] if dataset == "daic" else [],
            }
        )
    return records


def test_manifest_namespaces_subject_and_sample_identities() -> None:
    rows, metadata = build_merged_manifest(_records())
    assert metadata["dataset_subject_counts"]["daic"] == 12
    assert all("::" in str(row["subject_id"]) for row in rows)
    assert len({row["sample_id"] for row in rows}) == len(rows)
    daic_test = [row for row in rows if row["dataset"] == "daic" and row["component_subject_id"] == "dai-10"]
    assert daic_test and all(row["official_test_subject"] for row in daic_test)


def test_protocol_has_exact_outer_coverage_and_disjoint_inner_partitions() -> None:
    protocol = build_protocol_splits(_records(), seed=1337, inner_val_ratio=0.2)
    assert audit_protocol_splits(protocol)["status"] == "passed"
    for dataset in DATASETS:
        holdouts = []
        for fold in range(5):
            payload = protocol["components"][dataset]["folds"][str(fold)]
            assert set(payload["qwen_train_subject_ids"]).isdisjoint(payload["inner_val_subject_ids"])
            assert set(payload["outer_train_subject_ids"]).isdisjoint(payload["outer_holdout_subject_ids"])
            holdouts.extend(payload["outer_holdout_subject_ids"])
        assert len(holdouts) == len(set(holdouts))
        assert not any(subject.startswith("daic::") for subject in holdouts if dataset == "daic" and subject.endswith("10"))


def test_final_partitions_keep_only_daic_official_test_outside_training() -> None:
    final = build_final_partitions(_records())
    assert len(final["daic_official_test_subject_ids"]) == 2
    assert not set(final["train_subject_ids"]) & set(final["daic_official_test_subject_ids"])
    assert final["by_dataset"]["cmdc"]["official_test_count"] == 0


def test_runtime_reads_folds_from_saved_protocol_artifact_shape() -> None:
    protocol = build_protocol_splits(_records(), seed=1337, inner_val_ratio=0.2)
    subjects = fold_subject_ids({"protocol": protocol}, 0, "qwen_train")
    assert set(subjects) == set(DATASETS)
    assert all(value.startswith(f"{dataset}::") for dataset, values in subjects.items() for value in values)


def test_hierarchical_weights_equalize_dataset_subject_response_and_windows() -> None:
    rows, _ = build_merged_manifest(_records())
    weighted, audit = compute_hierarchical_example_weights(rows)
    assert abs(sum(row["loss_weight"] for row in weighted) / len(weighted) - 1.0) < 1e-12
    assert len({round(value, 10) for value in audit["dataset_weight_totals"].values()}) == 1
    subject_totals = defaultdict(float)
    response_totals = defaultdict(float)
    for row in weighted:
        subject_totals[row["subject_id"]] += row["loss_weight"]
        response_totals[row["response_id"]] += row["loss_weight"]
    assert len({round(value, 10) for value in subject_totals.values()}) == 1
    assert len({round(value, 10) for value in response_totals.values()}) == 1


def test_schedule_and_head_inner_folds_are_deterministic_and_one_time() -> None:
    rows, _ = build_merged_manifest(_records())
    weighted, _ = compute_hierarchical_example_weights(rows)
    first = build_dataset_aware_schedule(weighted, seed=1337, epoch=1, accumulation_steps=8)
    second = build_dataset_aware_schedule(weighted, seed=1337, epoch=1, accumulation_steps=8)
    assert first["indices"] == second["indices"]
    assert sorted(first["indices"]) == list(range(len(weighted)))
    assert set(first["audit"]["sample_occurrence_counts"].values()) == {1}
    folds = build_grouped_inner_folds(weighted, inner_folds=3, seed=1337)
    validation = [index for fold in folds["folds"] for index in fold["validation_row_indices"]]
    assert sorted(validation) == list(range(len(weighted)))
    for fold in folds["folds"]:
        assert set(fold["train_subject_ids"]).isdisjoint(fold["validation_subject_ids"])
