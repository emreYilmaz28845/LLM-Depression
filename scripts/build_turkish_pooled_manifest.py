#!/usr/bin/env python3
"""Concatenate audited Turkish pos_only + negative_only manifests into a pooled manifest.

Locked plan: docs/TURKISH_POOLED_TRAINING_PLAN.md Step 2.
- Stack native manifests (1,051 + 1,170 = 2,221 rows) preserving per-row
  ``dataset_variant``, transcripts, and audio paths. Repeat for the EN pair.
- Rows are byte-identical to sources: no re-transcription, no re-cutting.
- Transcript asymmetry is inherited deliberately (pos_only unreviewed,
  negative_only reviewed); recorded as a known limitation, never upgraded here.
- Splits: byte-compare the two existing 5-fold files. On mismatch ABORT
  (non-zero exit) - no fallback preserves baseline pairing.
- Emit pooled manifest + fresh manifest_hash + adopted split + audit into new
  turkish_pooled_t17_qwen3asr (+ _en) dirs. Old manifests untouched.
- Consumed with SKIP_MANIFEST_BUILD=1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_manifest import manifest_build_signature
from src.utils import (
    ensure_dir,
    load_yaml_with_overrides,
    read_jsonl,
    save_json,
    sha256_file,
    sha256_jsonl_rows,
    write_jsonl,
)

POS_VARIANT = "pos_only_t17"
NEG_VARIANT = "negative_only_t17"
KNOWN_CONDITIONS = (POS_VARIANT, NEG_VARIANT)


def _canonical_json_sha(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_folds(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def concat_and_audit(
    pos_rows: list[dict],
    neg_rows: list[dict],
    *,
    expected_pos: int = 1051,
    expected_neg: int = 1170,
    check_audio_exists: bool = True,
    require_translation_sha: bool = False,
) -> tuple[list[dict], dict]:
    """Stack pos + neg rows. Raises ValueError on any fail-closed violation."""
    if len(pos_rows) != expected_pos:
        raise ValueError(f"pos_only row count {len(pos_rows)} != expected {expected_pos}")
    if len(neg_rows) != expected_neg:
        raise ValueError(f"negative_only row count {len(neg_rows)} != expected {expected_neg}")

    for row in pos_rows:
        if str(row.get("dataset_variant", "")).strip() != POS_VARIANT:
            raise ValueError(
                f"pos_only manifest has unexpected dataset_variant "
                f"{row.get('dataset_variant')!r} for sample_id={row.get('sample_id')!r}"
            )
    for row in neg_rows:
        if str(row.get("dataset_variant", "")).strip() != NEG_VARIANT:
            raise ValueError(
                f"negative_only manifest has unexpected dataset_variant "
                f"{row.get('dataset_variant')!r} for sample_id={row.get('sample_id')!r}"
            )

    seen: dict[str, dict] = {}
    for row in pos_rows + neg_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError("manifest row with empty sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id across pooled sources: {sample_id!r}")
        seen[sample_id] = row

    labels_by_subject: dict[str, set[int]] = {}
    for row in pos_rows + neg_rows:
        labels_by_subject.setdefault(str(row["subject_id"]), set()).add(int(row["label"]))
    conflicts = sorted(
        subject for subject, labels in labels_by_subject.items() if len(labels) != 1
    )
    if conflicts:
        raise ValueError(f"subjects with conflicting labels across conditions: {conflicts[:10]}")

    if require_translation_sha:
        missing = sorted(
            str(row.get("sample_id"))
            for row in pos_rows + neg_rows
            if not str(row.get("translation_sha256", "")).strip()
        )
        if missing:
            raise ValueError(f"EN rows missing translation_sha256: {missing[:10]}")

    if check_audio_exists:
        missing_audio = sorted(
            str(row.get("sample_id"))
            for row in pos_rows + neg_rows
            if not row.get("audio_path") or not Path(str(row["audio_path"])).is_file()
        )
        if missing_audio:
            raise ValueError(f"audio_path missing on disk for: {missing_audio[:10]}")

    pooled = sorted(pos_rows + neg_rows, key=lambda row: str(row["sample_id"]))
    if len(pooled) != expected_pos + expected_neg:
        raise ValueError("pooled row count mismatch after concat")

    audit = {
        "pos_row_count": len(pos_rows),
        "neg_row_count": len(neg_rows),
        "pooled_row_count": len(pooled),
        "pos_subject_count": len({str(r["subject_id"]) for r in pos_rows}),
        "neg_subject_count": len({str(r["subject_id"]) for r in neg_rows}),
        "pooled_subject_count": len({str(r["subject_id"]) for r in pooled}),
        "pos_variant": POS_VARIANT,
        "neg_variant": NEG_VARIANT,
        "transcript_asymmetry": (
            "inherited deliberately: pos_only rows use whisper_transcripts_qwen3_asr.jsonl "
            "(unreviewed); negative_only rows use whisper_transcripts_qwen3_asr_reviewed.jsonl "
            "(reviewed); byte-identical to baseline campaigns"
        ),
        "manifest_hash": sha256_jsonl_rows(pooled),
    }
    return pooled, audit


def verify_splits_identical(pos_folds_path: Path, neg_folds_path: Path) -> dict:
    """Byte-compare the two 5-fold files. Raises ValueError on mismatch (abort)."""
    pos_bytes = pos_folds_path.read_bytes()
    neg_bytes = neg_folds_path.read_bytes()
    pos_sha = hashlib.sha256(pos_bytes).hexdigest()
    neg_sha = hashlib.sha256(neg_bytes).hexdigest()
    if pos_bytes != neg_bytes:
        raise ValueError(
            "pooled split ABORT: pos_only and negative_only 5-fold files differ "
            f"(pos sha={pos_sha[:12]}, neg sha={neg_sha[:12]}). No fallback preserves "
            "baseline pairing; joint re-splitting is out of scope."
        )
    pos_folds = json.loads(pos_bytes.decode("utf-8"))
    neg_folds = json.loads(neg_bytes.decode("utf-8"))
    if _canonical_json_sha(pos_folds) != _canonical_json_sha(neg_folds):
        raise ValueError("pooled split ABORT: folds differ after canonical JSON round-trip.")
    return {
        "pos_folds_sha256": pos_sha,
        "neg_folds_sha256": neg_sha,
        "folds_match": True,
        "fold_count": len(pos_folds),
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos-manifest", required=True, help="pos_only turkish_manifest.jsonl")
    parser.add_argument("--neg-manifest", required=True, help="negative_only turkish_manifest.jsonl")
    parser.add_argument("--pos-folds", required=True, help="pos_only turkish_folds.json")
    parser.add_argument("--neg-folds", required=True, help="negative_only turkish_folds.json")
    parser.add_argument("--out-manifest-dir", required=True)
    parser.add_argument("--out-split-dir", required=True)
    parser.add_argument("--pooled-config", required=True, help="one pooled config for build_signature")
    parser.add_argument("--pos-subjects", default=None)
    parser.add_argument("--neg-subjects", default=None)
    parser.add_argument("--expected-pos", type=int, default=1051)
    parser.add_argument("--expected-neg", type=int, default=1170)
    parser.add_argument("--require-translation-sha", action="store_true")
    parser.add_argument("--skip-audio-check", action="store_true")
    args = parser.parse_args()

    pos_manifest = Path(args.pos_manifest)
    neg_manifest = Path(args.neg_manifest)
    pos_folds_path = Path(args.pos_folds)
    neg_folds_path = Path(args.neg_folds)
    for path in (pos_manifest, neg_manifest, pos_folds_path, neg_folds_path):
        if not path.is_file():
            raise FileNotFoundError(f"input not found: {path}")

    pos_rows = read_jsonl(pos_manifest)
    neg_rows = read_jsonl(neg_manifest)
    pooled_rows, audit = concat_and_audit(
        pos_rows,
        neg_rows,
        expected_pos=args.expected_pos,
        expected_neg=args.expected_neg,
        check_audio_exists=not args.skip_audio_check,
        require_translation_sha=args.require_translation_sha,
    )
    split_audit = verify_splits_identical(pos_folds_path, neg_folds_path)
    audit.update(split_audit)
    audit.update(
        {
            "pos_manifest": str(pos_manifest),
            "neg_manifest": str(neg_manifest),
            "pos_manifest_sha256": sha256_file(pos_manifest),
            "neg_manifest_sha256": sha256_file(neg_manifest),
            "pos_folds": str(pos_folds_path),
            "neg_folds": str(neg_folds_path),
            "require_translation_sha": bool(args.require_translation_sha),
        }
    )

    out_manifest_dir = ensure_dir(Path(args.out_manifest_dir))
    out_split_dir = ensure_dir(Path(args.out_split_dir))
    manifest_jsonl = out_manifest_dir / "turkish_manifest.jsonl"
    manifest_csv = out_manifest_dir / "turkish_manifest.csv"
    write_jsonl(pooled_rows, manifest_jsonl)
    _write_csv(pooled_rows, manifest_csv)

    folds = _load_folds(pos_folds_path)
    folds_path = out_split_dir / "turkish_folds.json"
    save_json(folds, folds_path)

    subjects: list | None = None
    if args.pos_subjects and args.neg_subjects:
        pos_subjects = json.loads(Path(args.pos_subjects).read_text(encoding="utf-8"))
        neg_subjects = json.loads(Path(args.neg_subjects).read_text(encoding="utf-8"))
        by_id: dict[str, object] = {}
        for entry in (pos_subjects if isinstance(pos_subjects, list) else []) + (
            neg_subjects if isinstance(neg_subjects, list) else []
        ):
            if isinstance(entry, dict) and "subject_id" in entry:
                by_id[str(entry["subject_id"])] = entry
        subjects = [by_id[key] for key in sorted(by_id)]
        save_json(subjects, out_split_dir / "turkish_subjects.json")
        audit["pooled_subject_records"] = len(subjects)

    pooled_config = load_yaml_with_overrides(args.pooled_config, [])
    metadata = {
        "dataset": "turkish",
        "dataset_variant": "pooled_t17",
        "manifest_path": str(manifest_jsonl),
        "manifest_hash": sha256_jsonl_rows(pooled_rows),
        "build_signature": manifest_build_signature(pooled_config),
        "manifest_row_count": len(pooled_rows),
        "manifest_subject_count": len({str(r["subject_id"]) for r in pooled_rows}),
        "folds_path": str(folds_path),
        "split_source": "adopted_identical_pos_neg_5fold",
        "split_source_notes": (
            "pooled split adopts the byte-identical pos_only/negative_only 5-fold files; "
            "same 120 subjects, seed 1337"
        ),
        "source_manifest_hashes": {
            "pos_only": sha256_file(pos_manifest),
            "negative_only": sha256_file(neg_manifest),
        },
        "source_folds_hashes": {
            "pos_only": split_audit["pos_folds_sha256"],
            "negative_only": split_audit["neg_folds_sha256"],
        },
        "pooled_config": str(args.pooled_config),
    }
    if subjects is not None:
        metadata["subject_rows_path"] = str(out_split_dir / "turkish_subjects.json")

    audit_path = out_split_dir / "turkish_pooled_audit.json"
    save_json(audit, audit_path)
    metadata["pooled_audit_path"] = str(audit_path)
    metadata["artifact_hashes"] = {
        "manifest_path_sha256": sha256_file(manifest_jsonl),
        "folds_path_sha256": sha256_file(folds_path),
        "pooled_audit_path_sha256": sha256_file(audit_path),
    }
    save_json(metadata, out_split_dir / "turkish_manifest_metadata.json")
    print(f"pooled rows={len(pooled_rows)} manifest={manifest_jsonl} folds={folds_path}")


if __name__ == "__main__":
    main()
