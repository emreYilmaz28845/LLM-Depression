"""Model-free split/coverage audit for the DAIC official-development campaign.

Proves the locked 86/21/35 contract against the real canonical DAIC manifest
and partition metadata, plus the runbook row-count expectations, shared
identity across the six officialdev configs, and zero official-test
participation. Writes a task-owned JSON audit. Never writes raw subject IDs;
only counts and hashes of sorted subject-id lists are recorded.

Exit code 0 only when every locked assertion passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_manifest import manifest_build_signature
from src.data.split_utils import deterministic_inner_split
from src.utils import load_yaml, read_jsonl, sha256_file


OFFICIALDEV_RECIPE_ID = "harmonized_full_transcript_single30_allwindows_selmacrof1_tf_officialdev_v1"
SEED = 1337
INNER_VAL_RATIO = 0.2
SPLIT_ALGORITHM = "deterministic_inner_split"
EXPECTED_PARTITIONS = {"train": 107, "val": 35, "test": 47}
EXPECTED_CLASS_COUNTS = {
    "train": {"Non-depressed": 77, "Depressed": 30},
    "val": {"Non-depressed": 23, "Depressed": 12},
    "test": {"Non-depressed": 33, "Depressed": 14},
}
EXPECTED_INNER = {"train_inner": 86, "val_inner": 21}
EXPECTED_INNER_CLASS_COUNTS = {
    "train_inner": {"Non-depressed": 62, "Depressed": 24},
    "val_inner": {"Non-depressed": 15, "Depressed": 6},
}
EXPECTED_ROW_COUNTS = {
    "audio_only": {"fit_rows": 1312, "fit_subjects": 86, "dev_rows": 603, "dev_subjects": 35},
    "audio_text": {"fit_rows": 1312, "fit_subjects": 86, "dev_rows": 603, "dev_subjects": 35},
    "text_only": {"fit_rows": 86, "fit_subjects": 86, "dev_rows": 35, "dev_subjects": 35},
}

BACKBONE_CONFIG_NAMES = {
    "qwen": {
        modality: f"daic_{modality}_harmonized_selmacrof1_tf_officialdev.yaml"
        for modality in ("audio_only", "audio_text", "text_only")
    },
    "gemma4": {
        modality: f"daic_{modality}_harmonized_selmacrof1_tf_gemma4_12b_officialdev.yaml"
        for modality in ("audio_only", "audio_text", "text_only")
    },
}


class AuditFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def _sorted_id_sha256(subject_ids: list[str]) -> str:
    payload = json.dumps(sorted(subject_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_split(manifest_dir: Path, split_dir: Path, config_dir: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "daic_officialdev_split_audit.v1",
        "recipe_id": OFFICIALDEV_RECIPE_ID,
        "seed": SEED,
        "inner_val_ratio": INNER_VAL_RATIO,
        "split_algorithm": SPLIT_ALGORITHM,
    }

    # Shared identity across the six configs.
    configs: dict[tuple[str, str], dict[str, Any]] = {}
    for backbone, names in BACKBONE_CONFIG_NAMES.items():
        for modality, name in names.items():
            config = load_yaml(config_dir / name)
            _require(config["recipe_id"] == OFFICIALDEV_RECIPE_ID, f"{name}: recipe_id mismatch")
            _require(config["split"]["seed"] == SEED, f"{name}: seed mismatch")
            _require(config["split"]["inner_val_ratio"] == INNER_VAL_RATIO, f"{name}: inner_val_ratio mismatch")
            _require(
                config["split"].get("final_eval_partition") == "val",
                f"{name}: final_eval_partition must be val",
            )
            _require("selection_partition" not in config["split"], f"{name}: must not set selection_partition")
            _require(
                config["split"].get("dev_pool_partitions") == ["train"],
                f"{name}: dev_pool_partitions must be [train]",
            )
            configs[(backbone, modality)] = config
    manifest_dirs = {config["output_dirs"]["manifest_dir"] for config in configs.values()}
    split_dirs = {config["output_dirs"]["split_dir"] for config in configs.values()}
    _require(len(manifest_dirs) == 1, f"six configs must share one manifest dir: {manifest_dirs}")
    _require(len(split_dirs) == 1, f"six configs must share one split dir: {split_dirs}")
    _require(
        Path(configs[("qwen", "audio_only")]["output_dirs"]["manifest_dir"]).resolve() == manifest_dir.resolve(),
        "configured manifest dir does not match the audited path",
    )
    _require(
        Path(configs[("qwen", "audio_only")]["output_dirs"]["split_dir"]).resolve() == split_dir.resolve(),
        "configured split dir does not match the audited path",
    )
    signatures = {
        key: manifest_build_signature(config) for key, config in configs.items()
    }
    # Backbone identity keys legitimately differ between Qwen and Gemma
    # configs; everything else must be identical so the six configs resolve
    # the same manifest and split.
    normalized_signatures = [
        {
            "builder_options": {
                k: v for k, v in sig["builder_options"].items()
                if k not in {"model_backend", "model_revision"}
            },
            "data_options": sig["data_options"],
            "split_options": sig["split_options"],
        }
        for sig in signatures.values()
    ]
    _require(
        len({json.dumps(sig, sort_keys=True) for sig in normalized_signatures}) == 1,
        "all six configs must share one manifest build signature",
    )
    record["configs"] = {
        f"{backbone}/{modality}": {"name": name, "run_root": configs[(backbone, modality)]["output_dirs"]["run_root"]}
        for backbone, names in BACKBONE_CONFIG_NAMES.items()
        for modality, name in names.items()
    }
    record["manifest_build_signature_sha256"] = hashlib.sha256(
        json.dumps(next(iter(signatures.values())), sort_keys=True).encode()
    ).hexdigest()

    # Canonical partition file.
    partition_path = split_dir / "daic_subject_partitions.json"
    _require(partition_path.exists(), f"missing partition file {partition_path}")
    partition_rows = json.loads(partition_path.read_text(encoding="utf-8"))
    partition_file_sha256 = sha256_file(partition_path)
    record["partition_file_sha256"] = partition_file_sha256
    partition_counts = Counter(str(row["partition"]) for row in partition_rows)
    _require(
        {part: partition_counts.get(part, 0) for part in EXPECTED_PARTITIONS} == EXPECTED_PARTITIONS,
        f"partition counts mismatch: {dict(partition_counts)}",
    )
    for partition, expected in EXPECTED_CLASS_COUNTS.items():
        label_texts = Counter(
            str(row["label_text"]) if row.get("label_text") else ("Depressed" if row["label"] == 1 else "Non-depressed")
            for row in partition_rows
            if str(row["partition"]) == partition
        )
        _require(
            dict(label_texts) == expected,
            f"partition {partition} class counts mismatch: {dict(label_texts)}",
        )
    record["partition_counts"] = {part: partition_counts.get(part, 0) for part in EXPECTED_PARTITIONS}
    record["partition_class_counts"] = EXPECTED_CLASS_COUNTS

    subject_labels = {
        str(row["subject_id"]): 1 if str(row["label"]) in {"1", "True", "true"} else 0
        for row in partition_rows
    }
    train_ids = sorted(sid for row in partition_rows if str(row["partition"]) == "train" for sid in [str(row["subject_id"])])
    val_ids = sorted(sid for row in partition_rows if str(row["partition"]) == "val" for sid in [str(row["subject_id"])])
    test_ids = sorted(sid for row in partition_rows if str(row["partition"]) == "test" for sid in [str(row["subject_id"])])

    # Deterministic inner split.
    inner = deterministic_inner_split(subject_labels, train_ids, seed=SEED, val_ratio=INNER_VAL_RATIO)
    train_inner_ids = sorted(inner["train_inner_subject_ids"])
    val_inner_ids = sorted(inner["val_inner_subject_ids"])
    _require(len(train_inner_ids) == EXPECTED_INNER["train_inner"], f"train_inner subjects: {len(train_inner_ids)}")
    _require(len(val_inner_ids) == EXPECTED_INNER["val_inner"], f"val_inner subjects: {len(val_inner_ids)}")
    inner_class_counts = {
        "train_inner": {
            "Non-depressed": sum(1 for sid in train_inner_ids if subject_labels[sid] == 0),
            "Depressed": sum(1 for sid in train_inner_ids if subject_labels[sid] == 1),
        },
        "val_inner": {
            "Non-depressed": sum(1 for sid in val_inner_ids if subject_labels[sid] == 0),
            "Depressed": sum(1 for sid in val_inner_ids if subject_labels[sid] == 1),
        },
    }
    _require(inner_class_counts == EXPECTED_INNER_CLASS_COUNTS, f"inner class counts mismatch: {inner_class_counts}")
    record["inner_split_counts"] = {"train_inner": len(train_inner_ids), "val_inner": len(val_inner_ids)}
    record["inner_split_class_counts"] = inner_class_counts
    record["subject_set_sha256"] = {
        "train_inner": _sorted_id_sha256(train_inner_ids),
        "val_inner": _sorted_id_sha256(val_inner_ids),
        "official_val": _sorted_id_sha256(val_ids),
        "official_train": _sorted_id_sha256(train_ids),
        "official_test": _sorted_id_sha256(test_ids),
    }

    # Pairwise disjointness and official-test exclusion.
    set_train_inner = set(train_inner_ids)
    set_val_inner = set(val_inner_ids)
    set_dev = set(val_ids)
    set_test = set(test_ids)
    _require(set_train_inner.isdisjoint(set_val_inner), "train_inner overlaps val_inner")
    _require(set_train_inner.isdisjoint(set_dev), "train_inner overlaps official val")
    _require(set_val_inner.isdisjoint(set_dev), "val_inner overlaps official val")
    _require(set_test.isdisjoint(set_train_inner | set_val_inner | set_dev), "official test overlaps campaign sets")
    _require((set_train_inner | set_val_inner) == set(train_ids), "train_inner union val_inner != official train")
    record["disjointness"] = {
        "train_inner_vs_val_inner": True,
        "train_inner_vs_official_val": True,
        "val_inner_vs_official_val": True,
        "official_test_absent": True,
        "train_inner_union_val_inner_equals_official_train": True,
    }

    # Canonical packed30 manifest and row counts.
    manifest_path = manifest_dir / "daic_participant_speech_packed30_manifest.jsonl"
    _require(manifest_path.exists(), f"missing packed30 manifest {manifest_path}")
    manifest_sha256 = sha256_file(manifest_path)
    record["manifest_sha256"] = manifest_sha256
    manifest_rows = read_jsonl(manifest_path)
    _require(len(manifest_rows) == 3021, f"manifest row count: {len(manifest_rows)}")
    rows_by_subject = Counter(str(row["subject_id"]) for row in manifest_rows)
    expected = EXPECTED_ROW_COUNTS
    for modality, counts in expected.items():
        if modality == "text_only":
            fit_rows = len(train_inner_ids)
            dev_rows = len(val_ids)
        else:
            fit_rows = sum(rows_by_subject[sid] for sid in train_inner_ids)
            dev_rows = sum(rows_by_subject[sid] for sid in val_ids)
        _require(fit_rows == counts["fit_rows"], f"{modality}: fit rows {fit_rows} != {counts['fit_rows']}")
        _require(dev_rows == counts["dev_rows"], f"{modality}: dev rows {dev_rows} != {counts['dev_rows']}")
    record["row_counts"] = EXPECTED_ROW_COUNTS

    record["status"] = "passed"
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "configs/main")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        record = audit_split(args.manifest_dir, args.split_dir, args.config_dir)
    except AuditFailure as error:
        print(f"AUDIT FAILED: {error}", file=sys.stderr)
        return 1
    except FileNotFoundError as error:
        print(f"AUDIT FAILED: missing file: {error}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"audit passed: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
