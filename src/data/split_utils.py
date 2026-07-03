from __future__ import annotations

import itertools
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from src.utils import read_json, save_json


XML_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
SPLIT_MODE_FIXED = "fixed"
SPLIT_MODE_CV = "cv"
SPLIT_MODE_FULL_TRAIN = "full_train"
CV_PROTOCOL_TRAIN_VAL = "train_val"
CV_PROTOCOL_TRAIN_VAL_TEST = "train_val_test"
SUPPORTED_CV_PROTOCOLS = (
    CV_PROTOCOL_TRAIN_VAL,
    CV_PROTOCOL_TRAIN_VAL_TEST,
)
FIXED_PROTOCOL_TRAIN_VAL = "train_val"
FIXED_PROTOCOL_TRAIN_VAL_TEST = "train_val_test"
SUPPORTED_FIXED_PROTOCOLS = (
    FIXED_PROTOCOL_TRAIN_VAL,
    FIXED_PROTOCOL_TRAIN_VAL_TEST,
)
SUPPORTED_SPLIT_MODES = (
    SPLIT_MODE_FIXED,
    SPLIT_MODE_CV,
    SPLIT_MODE_FULL_TRAIN,
)


def expand_integer_ranges(spec: str) -> list[int]:
    result: list[int] = []
    for part in [chunk.strip() for chunk in spec.split("&") if chunk.strip()]:
        if "-" in part:
            start_text, end_text = [item.strip() for item in part.split("-", 1)]
            start = int(start_text)
            end = int(end_text)
            result.extend(list(range(start, end + 1)))
        else:
            result.append(int(part))
    return result


def _read_cmdc_sheet_rows(xlsx_path: str | Path) -> list[list[str]]:
    xlsx_path = Path(xlsx_path)
    with ZipFile(xlsx_path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for shared_item in root.findall("a:si", XML_NS):
                text = "".join((node.text or "") for node in shared_item.iterfind(".//a:t", XML_NS))
                shared_strings.append(text)

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        first_sheet = workbook.find("a:sheets/a:sheet", XML_NS)
        if first_sheet is None:
            raise ValueError("Could not find any sheets in CMDC fold workbook.")
        rel_id = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        target = "xl/" + rel_map[rel_id]
        sheet_root = ET.fromstring(archive.read(target))

        rows: list[list[str]] = []
        for row in sheet_root.findall(".//a:sheetData/a:row", XML_NS):
            values: list[str] = []
            for cell in row.findall("a:c", XML_NS):
                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", XML_NS)
                value = value_node.text if value_node is not None else ""
                if cell_type == "s" and value:
                    value = shared_strings[int(value)]
                values.append(value)
            rows.append(values)
        return rows


def build_cmdc_official_folds(
    subject_ids: list[str],
    xlsx_path: str | Path,
    strict: bool = True,
) -> dict[int, dict[str, list[str]]]:
    rows = _read_cmdc_sheet_rows(xlsx_path)
    folds: dict[int, dict[str, list[str]]] = {}
    expected_subjects = set(subject_ids)
    heldout_coverage: set[str] = set()
    fold_block_rows: list[list[str]] = []
    seen_first_fold = False
    for row in rows:
        if row and row[0].startswith("Fold "):
            seen_first_fold = True
            fold_block_rows.append(row)
            continue
        if seen_first_fold:
            break

    for row in fold_block_rows:
        fold_idx = int(row[0].split()[-1])
        test_mdd_numbers = expand_integer_ranges(row[3])
        test_hc_numbers = expand_integer_ranges(row[4])
        test_subjects = [f"MDD{number:02d}" for number in test_mdd_numbers] + [f"HC{number:02d}" for number in test_hc_numbers]
        train_subjects = sorted(expected_subjects - set(test_subjects))
        folds[fold_idx] = {
            "outer_train_subject_ids": train_subjects,
            "final_eval_subject_ids": sorted(test_subjects),
            "final_eval_mdd_subject_ids": [f"MDD{number:02d}" for number in test_mdd_numbers],
            "final_eval_hc_subject_ids": [f"HC{number:02d}" for number in test_hc_numbers],
        }
        overlap = heldout_coverage.intersection(test_subjects)
        if overlap and strict:
            raise ValueError(f"CMDC held-out fold overlap detected: {sorted(overlap)}")
        heldout_coverage.update(test_subjects)

    if heldout_coverage != expected_subjects and strict:
        missing = sorted(expected_subjects - heldout_coverage)
        extra = sorted(heldout_coverage - expected_subjects)
        raise ValueError(f"CMDC held-out fold coverage mismatch | missing={missing} extra={extra}")
    return folds


def cmdc_fold_report(
    folds: dict[int, dict[str, list[str]]],
    subject_labels: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    report: list[dict[str, object]] = []
    for fold_idx, payload in sorted(folds.items()):
        heldout_ids = payload["final_eval_subject_ids"]
        mdd_ids = payload.get("final_eval_mdd_subject_ids") or [subject_id for subject_id in heldout_ids if subject_id.startswith("MDD")]
        hc_ids = payload.get("final_eval_hc_subject_ids") or [subject_id for subject_id in heldout_ids if subject_id.startswith("HC")]
        train_ids = payload["outer_train_subject_ids"]
        if subject_labels is not None:
            outer_train_mdd_count = sum(subject_labels[subject_id] for subject_id in train_ids)
            outer_train_hc_count = len(train_ids) - outer_train_mdd_count
        else:
            outer_train_mdd_count = sum(1 for subject_id in train_ids if subject_id.startswith("MDD"))
            outer_train_hc_count = sum(1 for subject_id in train_ids if subject_id.startswith("HC"))
        report.append(
            {
                "fold": fold_idx,
                "heldout_subject_ids": heldout_ids,
                "heldout_mdd_subject_ids": mdd_ids,
                "heldout_hc_subject_ids": hc_ids,
                "heldout_mdd_count": len(mdd_ids),
                "heldout_hc_count": len(hc_ids),
                "heldout_class_counts": {
                    "Depressed": len(mdd_ids),
                    "Non-depressed": len(hc_ids),
                },
                "outer_train_count": len(train_ids),
                "outer_train_mdd_subject_ids": [subject_id for subject_id in train_ids if subject_id.startswith("MDD")],
                "outer_train_hc_subject_ids": [subject_id for subject_id in train_ids if subject_id.startswith("HC")],
                "outer_train_class_counts": {
                    "Depressed": outer_train_mdd_count,
                    "Non-depressed": outer_train_hc_count,
                },
            }
        )
    return report


def validate_non_overlapping_folds(
    folds: dict[int, dict[str, list[str]]],
    expected_subjects: list[str],
) -> None:
    heldout_coverage: set[str] = set()
    for fold_idx in sorted(folds):
        test_subjects = folds[fold_idx]["final_eval_subject_ids"]
        overlap = sorted(heldout_coverage.intersection(test_subjects))
        if overlap:
            raise ValueError(f"CMDC held-out fold overlap detected: {overlap}")
        heldout_coverage.update(test_subjects)
    expected_subject_set = set(expected_subjects)
    if heldout_coverage != expected_subject_set:
        missing = sorted(expected_subject_set - heldout_coverage)
        extra = sorted(heldout_coverage - expected_subject_set)
        raise ValueError(f"CMDC held-out fold coverage mismatch | missing={missing} extra={extra}")


def resolve_requested_split_mode(config: dict[str, Any]) -> str:
    raw_mode = str(config.get("split", {}).get("mode", "")).strip().lower()
    if not raw_mode:
        return ""
    if raw_mode not in SUPPORTED_SPLIT_MODES:
        raise ValueError(
            f"Unsupported split.mode={raw_mode!r}. Expected one of {', '.join(SUPPORTED_SPLIT_MODES)}."
        )
    return raw_mode


def resolve_split_mode(config: dict[str, Any], metadata: dict[str, Any] | None = None) -> str:
    requested_mode = resolve_requested_split_mode(config)
    if requested_mode:
        return requested_mode
    if metadata and metadata.get("subject_partition_path"):
        return SPLIT_MODE_FIXED
    if metadata and metadata.get("folds_path"):
        return SPLIT_MODE_CV
    split_cfg = config.get("split", {})
    if any(key in split_cfg for key in ("train_partition", "train_partitions", "selection_partition", "dev_pool_partitions")):
        return SPLIT_MODE_FIXED
    if "outer_folds" in split_cfg:
        return SPLIT_MODE_CV
    return SPLIT_MODE_FIXED


def resolve_cv_protocol(config: dict[str, Any]) -> str:
    raw_protocol = str(config.get("split", {}).get("cv_protocol", "")).strip().lower()
    if not raw_protocol:
        return CV_PROTOCOL_TRAIN_VAL if str(config.get("dataset", "")).lower() == "turkish" else CV_PROTOCOL_TRAIN_VAL_TEST
    if raw_protocol not in SUPPORTED_CV_PROTOCOLS:
        raise ValueError(
            f"Unsupported split.cv_protocol={raw_protocol!r}. "
            f"Expected one of {', '.join(SUPPORTED_CV_PROTOCOLS)}."
        )
    return raw_protocol


def resolve_fixed_protocol(config: dict[str, Any]) -> str:
    raw_protocol = str(config.get("split", {}).get("fixed_protocol", "")).strip().lower()
    if not raw_protocol:
        return FIXED_PROTOCOL_TRAIN_VAL_TEST
    if raw_protocol not in SUPPORTED_FIXED_PROTOCOLS:
        raise ValueError(
            f"Unsupported split.fixed_protocol={raw_protocol!r}. "
            f"Expected one of {', '.join(SUPPORTED_FIXED_PROTOCOLS)}."
        )
    return raw_protocol


def resolve_dev_pool_partitions(config: dict[str, Any]) -> list[str]:
    split_cfg = config.get("split", {})
    explicit = split_cfg.get("dev_pool_partitions")
    if explicit:
        partitions = [str(item).strip() for item in explicit if str(item).strip()]
        if partitions:
            return partitions
    train_partitions = split_cfg.get("train_partitions")
    if train_partitions:
        partitions = [str(item).strip() for item in train_partitions if str(item).strip()]
        if partitions:
            return partitions
    partitions: list[str] = []
    train_partition = str(split_cfg.get("train_partition", "")).strip()
    selection_partition = str(split_cfg.get("selection_partition", "")).strip()
    if train_partition:
        partitions.append(train_partition)
    if selection_partition and selection_partition not in partitions:
        partitions.append(selection_partition)
    if not partitions:
        raise ValueError("Could not resolve development-pool partitions from the split config.")
    return partitions


def resolve_outer_fold_count(config: dict[str, Any], default: int = 5) -> int:
    return int(config.get("split", {}).get("outer_folds", default))


def read_fold_payload(metadata: dict[str, Any], fold: int) -> dict[str, Any]:
    if not metadata.get("folds_path"):
        raise ValueError("Split metadata does not include folds_path.")
    folds = read_json(metadata["folds_path"])
    return folds[str(fold)] if str(fold) in folds else folds[fold]


def subject_ids_for_partitions(
    partition_rows: list[dict[str, Any]],
    partitions: list[str] | set[str] | tuple[str, ...],
) -> list[str]:
    partition_set = {str(item) for item in partitions}
    return sorted([row["subject_id"] for row in partition_rows if str(row["partition"]) in partition_set])


def subject_fold_report(
    folds: dict[int, dict[str, list[str]]],
    subject_labels: dict[str, int],
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for fold_idx, payload in sorted(folds.items()):
        heldout_ids = sorted(payload["final_eval_subject_ids"])
        train_ids = sorted(payload["outer_train_subject_ids"])
        heldout_depressed_ids = [subject_id for subject_id in heldout_ids if int(subject_labels[subject_id]) == 1]
        heldout_non_depressed_ids = [subject_id for subject_id in heldout_ids if int(subject_labels[subject_id]) == 0]
        outer_train_depressed_ids = [subject_id for subject_id in train_ids if int(subject_labels[subject_id]) == 1]
        outer_train_non_depressed_ids = [subject_id for subject_id in train_ids if int(subject_labels[subject_id]) == 0]
        report.append(
            {
                "fold": int(fold_idx),
                "heldout_subject_ids": heldout_ids,
                "heldout_depressed_subject_ids": heldout_depressed_ids,
                "heldout_non_depressed_subject_ids": heldout_non_depressed_ids,
                "heldout_depressed_count": len(heldout_depressed_ids),
                "heldout_non_depressed_count": len(heldout_non_depressed_ids),
                "heldout_class_counts": {
                    "Depressed": len(heldout_depressed_ids),
                    "Non-depressed": len(heldout_non_depressed_ids),
                },
                "outer_train_subject_ids": train_ids,
                "outer_train_count": len(train_ids),
                "outer_train_depressed_subject_ids": outer_train_depressed_ids,
                "outer_train_non_depressed_subject_ids": outer_train_non_depressed_ids,
                "outer_train_class_counts": {
                    "Depressed": len(outer_train_depressed_ids),
                    "Non-depressed": len(outer_train_non_depressed_ids),
                },
            }
        )
    return report


def build_partition_scoped_stratified_folds(
    *,
    partition_rows: list[dict[str, Any]],
    subject_labels: dict[str, int],
    dev_pool_partitions: list[str],
    final_eval_partition: str,
    n_splits: int,
    seed: int,
) -> dict[int, dict[str, list[str]]]:
    dev_subject_ids = subject_ids_for_partitions(partition_rows, dev_pool_partitions)
    final_eval_subject_ids = subject_ids_for_partitions(partition_rows, [final_eval_partition])
    overlap = sorted(set(dev_subject_ids).intersection(final_eval_subject_ids))
    if overlap:
        raise ValueError(
            f"Development-pool partitions overlap with final eval partition {final_eval_partition!r}: {overlap[:10]}"
        )
    dev_subject_labels = {subject_id: int(subject_labels[subject_id]) for subject_id in dev_subject_ids}
    folds = assign_stratified_group_folds(dev_subject_labels, n_splits=n_splits, seed=seed)
    validate_non_overlapping_folds(folds, dev_subject_ids)
    for fold_idx, payload in sorted(folds.items()):
        heldout_overlap = sorted(set(payload["final_eval_subject_ids"]).intersection(final_eval_subject_ids))
        if heldout_overlap:
            raise ValueError(f"Fold {fold_idx} unexpectedly includes final-eval subjects: {heldout_overlap[:10]}")
    return folds


def assign_stratified_group_folds(subject_labels: dict[str, int], n_splits: int, seed: int) -> dict[int, dict[str, list[str]]]:
    rng = random.Random(seed)
    grouped = {
        0: sorted([subject_id for subject_id, label in subject_labels.items() if label == 0]),
        1: sorted([subject_id for subject_id, label in subject_labels.items() if label == 1]),
    }
    for label_subjects in grouped.values():
        rng.shuffle(label_subjects)

    fold_subjects: dict[int, list[str]] = {fold_idx: [] for fold_idx in range(n_splits)}
    for label in [1, 0]:
        for item_idx, subject_id in enumerate(grouped[label]):
            fold_idx = item_idx % n_splits
            fold_subjects[fold_idx].append(subject_id)

    folds: dict[int, dict[str, list[str]]] = {}
    all_subjects = set(subject_labels)
    for fold_idx in range(n_splits):
        test_subjects = sorted(fold_subjects[fold_idx])
        train_subjects = sorted(all_subjects - set(test_subjects))
        folds[fold_idx] = {
            "outer_train_subject_ids": train_subjects,
            "final_eval_subject_ids": test_subjects,
        }
    return folds


def deterministic_inner_split(
    subject_labels: dict[str, int],
    subject_ids: list[str],
    seed: int,
    val_ratio: float,
    max_attempts: int = 50,
) -> dict[str, list[str]]:
    subjects = sorted(subject_ids)
    labels = [subject_labels[subject_id] for subject_id in subjects]
    if len(set(labels)) < 2:
        raise ValueError("Inner split requires both classes to be present.")
    class_subjects = {
        0: [subject_id for subject_id in subjects if subject_labels[subject_id] == 0],
        1: [subject_id for subject_id in subjects if subject_labels[subject_id] == 1],
    }
    attempts = itertools.count(seed)
    for _ in range(max_attempts):
        rng = random.Random(next(attempts))
        val_subjects: list[str] = []
        train_subjects: list[str] = []
        for label, label_subjects in class_subjects.items():
            items = label_subjects[:]
            rng.shuffle(items)
            val_count = max(1, int(round(len(items) * val_ratio)))
            val_count = min(val_count, len(items) - 1)
            label_val = items[:val_count]
            label_train = items[val_count:]
            val_subjects.extend(label_val)
            train_subjects.extend(label_train)
        train_labels = {subject_labels[subject_id] for subject_id in train_subjects}
        val_labels = {subject_labels[subject_id] for subject_id in val_subjects}
        if train_labels == {0, 1} and val_labels == {0, 1}:
            return {
                "train_inner_subject_ids": sorted(train_subjects),
                "val_inner_subject_ids": sorted(val_subjects),
            }
    raise ValueError("Could not find a deterministic inner split with both classes in train and val.")


def partition_class_counts(subject_ids: list[str], subject_labels: dict[str, int]) -> dict[str, int]:
    counter = Counter(subject_labels[subject_id] for subject_id in subject_ids)
    return {
        "depressed_subjects": int(counter[1]),
        "non_depressed_subjects": int(counter[0]),
        "total_subjects": len(subject_ids),
    }


def save_split_file(data: dict, path: str | Path) -> None:
    save_json(data, path)
