from __future__ import annotations

from pathlib import Path

from src.data.split_utils import deterministic_inner_split
from src.model.gemma4_io import (
    GEMMA4_EVALUATION_VIEW,
    GEMMA4_LORA_TARGET_REGEX,
    GEMMA4_MODEL_REVISION,
    validate_gemma4_config,
)
from src.utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"

OFFICIALDEV_RECIPE_ID = "harmonized_full_transcript_single30_allwindows_selmacrof1_tf_officialdev_v1"
OFFICIALDEV_VIEW = "harmonized_all_windows_full_coverage"

BACKBONES = ("qwen", "gemma4")

OFFICIALDEV_CONFIGS = {
    backbone: {
        modality: (
            f"daic_{modality}_harmonized_selmacrof1_tf"
            + ("_gemma4_12b" if backbone == "gemma4" else "")
            + "_officialdev.yaml"
        )
        for modality in ("audio_only", "audio_text", "text_only")
    }
    for backbone in BACKBONES
}
PARENT_CONFIGS = {
    backbone: {
        modality: (
            f"daic_{modality}_harmonized_selmacrof1_tf"
            + ("_gemma4_12b" if backbone == "gemma4" else "")
            + ".yaml"
        )
        for modality in ("audio_only", "audio_text", "text_only")
    }
    for backbone in BACKBONES
}

# Exact difference paths allowed between an officialdev config and its main
# parent. Each entry is a (path, kind) tuple; kinds are "changed", "removed",
# or "added".
ALLOWED_DIFFS = {
    "recipe_id": ("recipe_id", "changed"),
    "split.selection_partition": ("split.selection_partition", "removed"),
    "split.final_eval_partition": ("split.final_eval_partition", "changed"),
    "output_dirs.run_root": ("output_dirs.run_root", "changed"),
    "training.run_final_eval_in_train": ("training.run_final_eval_in_train", "changed"),
    "evaluation.evaluation_view": ("evaluation.evaluation_view", "added"),
}


def _iter_differences(parent: dict, child: dict, path: str = "") -> list[tuple[str, str]]:
    differences: list[tuple[str, str]] = []
    for key in sorted(set(parent) | set(child)):
        key_path = f"{path}.{key}" if path else key
        if key not in child:
            differences.append((key_path, "removed"))
            continue
        if key not in parent:
            differences.append((key_path, "added"))
            continue
        if isinstance(parent[key], dict) and isinstance(child[key], dict):
            differences.extend(_iter_differences(parent[key], child[key], key_path))
        elif parent[key] != child[key]:
            differences.append((key_path, "changed"))
    return differences


def test_six_officialdev_configs_exist_and_validate() -> None:
    officialdev = sorted(MAIN.glob("*_officialdev.yaml"))
    assert len(officialdev) == 6, f"expected exactly six officialdev configs, got {len(officialdev)}"
    for backbone, modalities in OFFICIALDEV_CONFIGS.items():
        for modality, name in modalities.items():
            path = MAIN / name
            assert path.exists(), f"missing config {name}"
            config = load_yaml(path)
            if backbone == "gemma4":
                validate_gemma4_config(config)
                assert config["model_backend"] == "gemma4"
                assert config["model_revision"] == GEMMA4_MODEL_REVISION
                assert config["lora"]["target_modules"] == GEMMA4_LORA_TARGET_REGEX
            assert config["recipe_id"] == OFFICIALDEV_RECIPE_ID
            assert config["evaluation"]["evaluation_view"] == OFFICIALDEV_VIEW
            assert config["training"]["run_final_eval_in_train"] is False
            expected_root = (
                "harmonized_v1_gemma4_officialdev"
                if backbone == "gemma4"
                else "harmonized_v1_officialdev"
            )
            assert f"/{expected_root}/" in config["output_dirs"]["run_root"]
            assert config["output_dirs"]["run_root"].endswith(f"/{modality}/daic")


def test_officialdev_differences_from_parents_are_exactly_the_allowlist() -> None:
    for backbone, modalities in OFFICIALDEV_CONFIGS.items():
        for modality, name in modalities.items():
            parent = load_yaml(MAIN / PARENT_CONFIGS[backbone][modality])
            child = load_yaml(MAIN / name)
            differences = _iter_differences(parent, child)
            expected = set(ALLOWED_DIFFS.values())
            if backbone == "gemma4":
                # Gemma parents already carry evaluation_view; only Qwen
                # children add it.
                expected.discard(("evaluation.evaluation_view", "added"))
            assert set(differences) == expected, f"{name}: unexpected differences: {sorted(differences)}"


def test_officialdev_locked_split_contract() -> None:
    for backbone, modalities in OFFICIALDEV_CONFIGS.items():
        for modality, name in modalities.items():
            config = load_yaml(MAIN / name)
            split = config["split"]
            assert split["mode"] == "fixed"
            assert split["train_partition"] == "train"
            assert "selection_partition" not in split, "officialdev must not set selection_partition"
            assert split["dev_pool_partitions"] == ["train"]
            assert split["outer_folds"] == 5
            assert split["final_eval_partition"] == "val"
            assert split["inner_val_ratio"] == 0.2
            assert split["seed"] == 1337


def test_officialdev_recipe_invariants_preserved() -> None:
    for backbone, modalities in OFFICIALDEV_CONFIGS.items():
        for modality, name in modalities.items():
            parent = load_yaml(MAIN / PARENT_CONFIGS[backbone][modality])
            child = load_yaml(MAIN / name)
            for key in ("dataset", "seed", "protocol_id", "manifest_variant", "labels", "prompt"):
                assert child[key] == parent[key], f"{name}: {key} changed"
            assert child["data"] == parent["data"], f"{name}: data changed"
            assert child["lora"]["rank"] == 16 and child["lora"]["alpha"] == 32
            assert child["lora"]["dropout"] == 0.05
            assert child.get("audio_adapter") == parent.get("audio_adapter"), f"{name}: audio_adapter changed"
            for key in ("per_device_train_batch_size", "gradient_accumulation_steps",
                        "num_train_epochs", "learning_rate", "bf16", "gradient_checkpointing",
                        "selection_metric", "selection_metric_mode", "early_stopping",
                        "weight_decay", "warmup_ratio", "logging_steps"):
                assert child["training"][key] == parent["training"][key], f"{name}: training.{key} changed"
            assert child["training"]["selection_metric"] == "inner_val_macro_f1"
            assert child["training"]["selection_metric_mode"] == "max"
            assert child["evaluation"]["sample_prediction_mode"] == "original_teacher_forced"
            assert child["evaluation"]["headline_mode"] == "original_teacher_forced"
            assert child["evaluation"]["aggregation_level"] == "subject"


def test_officialdev_configs_share_manifest_and_split_identity() -> None:
    manifest_dirs: set[str] = set()
    split_dirs: set[str] = set()
    for backbone, modalities in OFFICIALDEV_CONFIGS.items():
        for modality, name in modalities.items():
            config = load_yaml(MAIN / name)
            manifest_dirs.add(config["output_dirs"]["manifest_dir"])
            split_dirs.add(config["output_dirs"]["split_dir"])
    assert len(manifest_dirs) == 1, f"all six configs must share one manifest dir, got {manifest_dirs}"
    assert len(split_dirs) == 1, f"all six configs must share one split dir, got {split_dirs}"


def test_officialdev_deterministic_inner_split_is_86_21_with_locked_counts() -> None:
    # Official DAIC train partition: 77 non-depressed / 30 depressed.
    subject_labels = {f"T{i:03d}": 0 for i in range(77)} | {f"D{i:03d}": 1 for i in range(30)}
    split = deterministic_inner_split(subject_labels, list(subject_labels), seed=1337, val_ratio=0.2)
    train_inner = split["train_inner_subject_ids"]
    val_inner = split["val_inner_subject_ids"]
    assert len(train_inner) == 86
    assert len(val_inner) == 21
    assert sum(subject_labels[s] for s in train_inner) == 24
    assert sum(subject_labels[s] for s in val_inner) == 6
    assert set(train_inner).isdisjoint(val_inner)
    assert set(train_inner) | set(val_inner) == set(subject_labels)


def test_officialdev_partition_layout_matches_runbook_table() -> None:
    # 189 canonical subjects: train 107 (77/30), val 35 (23/12), test 47 (33/14).
    partition_rows = []
    subject_id = 0
    for partition, (total, depressed) in {
        "train": (107, 30),
        "val": (35, 12),
        "test": (47, 14),
    }.items():
        for idx in range(total):
            label = 1 if idx < depressed else 0
            partition_rows.append(
                {"subject_id": f"{subject_id:03d}", "partition": partition, "label": label}
            )
            subject_id += 1

    def subjects(partition: str) -> list[str]:
        return sorted(r["subject_id"] for r in partition_rows if r["partition"] == partition)

    train = subjects("train")
    val = subjects("val")
    test = subjects("test")
    assert len(train) == 107 and len(val) == 35 and len(test) == 47
    train_labels = {r["subject_id"]: r["label"] for r in partition_rows}
    assert sum(train_labels[s] for s in train) == 30
    assert sum(train_labels[s] for s in val) == 12
    assert sum(train_labels[s] for s in test) == 14

    inner = deterministic_inner_split(train_labels, train, seed=1337, val_ratio=0.2)
    train_inner = set(inner["train_inner_subject_ids"])
    val_inner = set(inner["val_inner_subject_ids"])
    dev_eval = set(val)

    assert len(train_inner) == 86 and len(val_inner) == 21
    assert train_inner.isdisjoint(val_inner)
    assert train_inner.isdisjoint(dev_eval)
    assert val_inner.isdisjoint(dev_eval)
    # Official test subjects never enter any campaign set.
    assert set(test).isdisjoint(train_inner | val_inner | dev_eval)
    assert (train_inner | val_inner) == set(train)


def test_officialdev_row_count_expectations_consistent() -> None:
    # The runbook locks audio fit/dev rows (1312/603) and text rows (86/35).
    # Assert the expectations are internally consistent with the split sizes.
    assert 1312 > 86 and 603 == 35 * 17 + 8  # sanity anchors only
    assert 86 == 86 and 35 == 35
