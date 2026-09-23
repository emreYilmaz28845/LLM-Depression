#!/usr/bin/env python3
"""Model-free preflight audit for the Qwen3-Omni standalone prompt-context campaign.

For every dataset family it verifies that the manifest and fold files in the
attempt runtime reproduce the reference campaign's recorded identity, and audits
the dataset counts, subject labels, leakage unit and fold invariants. Nothing
here loads model weights and nothing is written outside ``--output``.

Expected values come from the reference runs' own ``fold_0/run_config.yaml``
(``manifest_hash`` over the manifest rows and ``split_metadata_hash`` over the
folds file) plus the canonical manifest counts:

* d3tec              harmonized_v1_..._d3tec_audio_only                (62 subjects)
* androids_interview harmonized_v1_..._androids_interview_audio_only_r1 (116 subjects)
* cmdc               harmonized_v1_..._cmdc_audio_only_r1               (78 subjects)
* turkish (pooled)   tpq_prod_v1_qwen_native_audio_only_s1337_f0_216bcec8 (120 subjects)

The audit fails closed: any missing file, hash mismatch, count mismatch or fold
invariant violation is recorded as a failed check and the exit code is non-zero.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import read_jsonl, sha256_file, sha256_jsonl_rows  # noqa: E402

SCHEMA_VERSION = "audiollm.qwen3omni_campaign_preflight.v1"
POOLED_CONDITIONS = {"pos_only_t17": 1051, "negative_only_t17": 1170}

# dataset -> expectations (hashes, row/subject counts, subject-level label counts)
EXPECTED: dict[str, dict[str, Any]] = {
    "d3tec": {
        "manifest_sha256": "67a62eb73b4ab7e0cd810b81af5e424f6bf9deea9cfdbc322fb32057a6e6f799",
        "folds_sha256": "a672e309fb193d7fd76e7283f5f42828c33713fe770ccbbf90f1ed72bf3fc15c",
        "rows": 2003,
        "subjects": 62,
        "subject_labels": {0: 33, 1: 29},
        "reference_run": "harmonized_v1_..._d3tec_audio_only",
        "cv_protocol": "train_val_test",
    },
    "androids_interview": {
        "manifest_sha256": "01a351f7277e4763a8bb9e4983bba190b265becafafca6d7ee04bdcfc948cbed",
        "folds_sha256": "f75dd2ba7bb324af26de8c5ae3497d2108e6b50815c0ef6cbcade7de70992518",
        "rows": 1302,
        "subjects": 116,
        "subject_labels": {0: 52, 1: 64},
        "reference_run": "harmonized_v1_..._androids_interview_audio_only_r1",
        "cv_protocol": "train_val_test",
    },
    "cmdc": {
        "manifest_sha256": "d9984856e243b6e32c087170652a9202c128f55270c0d8443bb45f7dea794d0f",
        "folds_sha256": "404342f7594833379eb8c15e6f4f5641c0ccec415374cb95df18f3723b9b54a0",
        "rows": 923,
        "subjects": 78,
        "subject_labels": {0: 52, 1: 26},
        "reference_run": "harmonized_v1_..._cmdc_audio_only_r1",
        "cv_protocol": "train_val",
    },
    "turkish": {
        "manifest_sha256": "37e991526986d9693c9620682719a6b54c7d30ec86b53152ae23dde167271b70",
        "folds_sha256": "3262a009db52c6d049e223947a9be6ce119a31e816b5c3072ce84b3ad92ecd58",
        "rows": 2221,
        "subjects": 120,
        "subject_labels": {0: 37, 1: 83},
        "reference_run": "tpq_prod_v1_qwen_native_audio_only_s1337_f0_216bcec8",
        "cv_protocol": "train_val",
        "conditions": POOLED_CONDITIONS,
    },
}


def _check(record: dict[str, Any], name: str, ok: bool, detail: Any) -> None:
    record["checks"][name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        record["failed_checks"].append(name)


def audit_dataset(name: str, manifest_dir: Path, split_dir: Path) -> dict[str, Any]:
    expected = EXPECTED[name]
    manifest_path = manifest_dir / f"{name}_manifest.jsonl"
    manifest_csv = manifest_dir / f"{name}_manifest.csv"
    folds_path = split_dir / f"{name}_folds.json"
    metadata_path = split_dir / f"{name}_manifest_metadata.json"
    record: dict[str, Any] = {
        "dataset": name,
        "manifest_dir": str(manifest_dir),
        "split_dir": str(split_dir),
        "expected": expected,
        "checks": {},
        "failed_checks": [],
    }
    for path in (manifest_path, manifest_csv, folds_path, metadata_path):
        _check(record, f"exists:{path.name}", path.is_file(), str(path))
    if record["failed_checks"]:
        return record

    rows = read_jsonl(manifest_path)
    record["rows"] = len(rows)
    record["manifest_sha256"] = sha256_jsonl_rows(rows)
    record["folds_sha256"] = sha256_file(folds_path)
    _check(record, "manifest_sha256", record["manifest_sha256"] == expected["manifest_sha256"], record["manifest_sha256"])
    _check(record, "folds_sha256", record["folds_sha256"] == expected["folds_sha256"], record["folds_sha256"])
    _check(record, "row_count", len(rows) == expected["rows"], len(rows))

    subject_labels: dict[str, set[int]] = collections.defaultdict(set)
    for row in rows:
        subject_labels[str(row["subject_id"])].add(int(row["label"]))
    inconsistent = sorted(subject for subject, labels in subject_labels.items() if len(labels) != 1)
    label_counts = collections.Counter(next(iter(labels)) for labels in subject_labels.values())
    record["subjects"] = len(subject_labels)
    record["subject_label_counts"] = {str(key): value for key, value in sorted(label_counts.items())}
    _check(record, "subject_count", len(subject_labels) == expected["subjects"], len(subject_labels))
    _check(record, "one_label_per_subject", not inconsistent, inconsistent)
    _check(
        record,
        "subject_label_counts",
        dict(label_counts) == expected["subject_labels"],
        record["subject_label_counts"],
    )

    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    fold_ids = sorted(str(key) for key in folds)
    eval_sets = {str(key): set(map(str, value.get("final_eval_subject_ids") or [])) for key, value in folds.items()}
    train_sets = {str(key): set(map(str, value.get("outer_train_subject_ids") or [])) for key, value in folds.items()}
    union = set().union(*eval_sets.values()) if eval_sets else set()
    record["folds"] = {
        "fold_ids": fold_ids,
        "final_eval_sizes": {key: len(value) for key, value in sorted(eval_sets.items())},
    }
    _check(record, "five_folds", fold_ids == ["0", "1", "2", "3", "4"], fold_ids)
    _check(record, "eval_covers_every_subject_once", union == set(subject_labels), len(union))
    _check(
        record,
        "eval_folds_disjoint",
        sum(len(value) for value in eval_sets.values()) == len(union),
        sum(len(value) for value in eval_sets.values()),
    )
    _check(
        record,
        "train_eval_disjoint",
        all(not (train_sets[key] & eval_sets[key]) for key in eval_sets),
        "outer_train ∩ final_eval = ∅",
    )
    _check(
        record,
        "train_union_covers_subjects",
        all(union - eval_sets[key] == train_sets[key] for key in eval_sets if train_sets[key]),
        "outer_train = subjects - final_eval",
    )

    if "conditions" in expected:
        condition_counts = collections.Counter(str(row.get("dataset_variant")) for row in rows)
        record["condition_counts"] = dict(sorted(condition_counts.items()))
        _check(
            record,
            "condition_row_counts",
            dict(condition_counts) == expected["conditions"],
            record["condition_counts"],
        )
        conditions_per_subject: dict[str, set[str]] = collections.defaultdict(set)
        for row in rows:
            conditions_per_subject[str(row["subject_id"])].add(str(row.get("dataset_variant")))
        missing_condition = sorted(s for s, c in conditions_per_subject.items() if c != set(POOLED_CONDITIONS))
        _check(record, "both_conditions_per_subject", not missing_condition, missing_condition)
        _check(
            record,
            "condition_subject_identity",
            set(conditions_per_subject) == set(subject_labels),
            len(conditions_per_subject),
        )

    record["status"] = "passed" if not record["failed_checks"] else "failed"
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, help="experiment runtime directory")
    parser.add_argument("--datasets", nargs="+", default=sorted(EXPECTED))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    runtime_root = Path(args.runtime_root)
    unknown = sorted(set(args.datasets) - set(EXPECTED))
    if unknown:
        print(f"unknown datasets: {unknown}", file=sys.stderr)
        return 2

    records = []
    for name in args.datasets:
        records.append(
            audit_dataset(name, runtime_root / "manifests" / name, runtime_root / "splits" / name)
        )
    failed = [record["dataset"] for record in records if record["status"] != "passed"]
    audit = {
        "schema_version": SCHEMA_VERSION,
        "runtime_root": str(runtime_root),
        "datasets": records,
        "status": "passed" if not failed else "failed",
        "failed_datasets": failed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for record in records:
        print(
            f"{record['dataset']:<20} status={record['status']:<7} "
            f"rows={record.get('rows', '-')} subjects={record.get('subjects', '-')} "
            f"failed={record['failed_checks']}"
        )
    print(f"wrote {args.output}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
