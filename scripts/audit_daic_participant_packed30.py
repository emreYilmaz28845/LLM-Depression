#!/usr/bin/env python
"""Final artifact auditor for the daic_participant_speech_packed30_v1 protocol.

Implements Section 11.2 (manifest acceptance) and, when run roots are given,
Section 11.4 (result acceptance) of
docs/DAIC_ANDROIDS_STYLE_RUNTIME_CHUNKING_PLAN.md. Exits nonzero unless every
locked assertion holds. Rebuild determinism (--rebuild) requires the local
corpus roots and a config path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf

from src.data.daic import (
    PACKED30_CHUNK_SAMPLES,
    PACKED30_CORPUS_AUDIT_JSON,
    PACKED30_INVALID_ROW_ALLOWLIST,
    PACKED30_JOIN_AUDIT_JSONL,
    PACKED30_MANIFEST_JSONL,
    PACKED30_MANIFEST_VARIANT,
    PACKED30_METADATA_JSON,
    PACKED30_PROTOCOL_ID,
    PACKED30_SAMPLE_RATE,
    PACKED30_SCHEMA_VERSION,
    PACKED30_SUBJECTS_JSONL,
)
from src.utils import read_json, read_jsonl


CHUNK_ID_RE = re.compile(r"^\d+_participant_p30_\d{3}$")
EXPECTED_SUBJECTS = {"train": 107, "val": 35, "test": 47}
EXPECTED_CHUNKS_BY_SPLIT_LABEL = {
    "train": {0: 1148, 1: 450},
    "val": {0: 363, 1: 240},
    "test": {0: 595, 1: 225},
}
EXPECTED_TOTALS = {
    "nonblank_rows": 47400,
    "retained_rows": 32373,
    "chunks": 3021,
    "retained_frames": 1406614000,
    "chunks_per_subject_range": (3, 43),
    "final_chunk_samples_range": (1440, 478480),
    "blank_lines": 14,
}


class Auditor:
    def __init__(self, manifest_dir: Path, split_dir: Path):
        self.manifest_dir = Path(manifest_dir)
        self.split_dir = Path(split_dir)
        self.failures: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.failures.append(message)
            print(f"FAIL: {message}", flush=True)

    def check_manifest_acceptance(self) -> None:
        manifest_path = self.manifest_dir / PACKED30_MANIFEST_JSONL
        subjects_path = self.manifest_dir / PACKED30_SUBJECTS_JSONL
        join_audit_path = self.manifest_dir / PACKED30_JOIN_AUDIT_JSONL
        corpus_audit_path = self.manifest_dir / PACKED30_CORPUS_AUDIT_JSON
        metadata_path = self.manifest_dir / PACKED30_METADATA_JSON
        for path in (manifest_path, subjects_path, join_audit_path, corpus_audit_path, metadata_path):
            self.require(path.is_file(), f"Missing packed30 artifact: {path}")
        if not all(p.is_file() for p in (manifest_path, subjects_path, join_audit_path, corpus_audit_path, metadata_path)):
            return

        rows = read_jsonl(manifest_path)
        subjects = read_jsonl(subjects_path)
        join_audit = read_jsonl(join_audit_path)
        corpus_audit = read_json(corpus_audit_path)
        metadata = read_json(metadata_path)

        self.require(
            all(str(row.get("protocol_id")) == PACKED30_PROTOCOL_ID for row in rows),
            "Protocol ID does not match v1 on every manifest row.",
        )
        self.require(
            all(str(row.get("schema_version")) == PACKED30_SCHEMA_VERSION for row in rows),
            "Schema version does not match v1 on every manifest row.",
        )
        self.require(
            str(metadata.get("protocol_id")) == PACKED30_PROTOCOL_ID
            and str(metadata.get("manifest_variant")) == PACKED30_MANIFEST_VARIANT
            and str(metadata.get("schema_version")) == PACKED30_SCHEMA_VERSION,
            "Metadata protocol/schema/variant mismatch.",
        )

        split_counts = Counter(str(row["split_original"]) for row in subjects)
        self.require(
            {name: split_counts.get(name, 0) for name in ("train", "val", "test")}
            == EXPECTED_SUBJECTS
            and len(subjects) == 189,
            f"Subject split counts mismatch: {dict(split_counts)}",
        )
        label_counts = Counter(int(row["label"]) for row in subjects)
        self.require(
            label_counts[0] == 133 and label_counts[1] == 56 and len(subjects) == 189,
            f"Label coverage mismatch: {dict(label_counts)}",
        )

        statuses = Counter(str(row["status"]) for row in join_audit)
        nonblank = (
            statuses["retained"]
            + statuses["excluded_non_participant"]
            + statuses["excluded_empty_participant_text"]
            + statuses["invalid_allowlisted_row"]
        )
        self.require(nonblank == EXPECTED_TOTALS["nonblank_rows"], f"Nonblank row total {nonblank} != 47400")
        self.require(
            statuses["retained"] == EXPECTED_TOTALS["retained_rows"],
            f"Retained Participant rows {statuses['retained']} != 32373",
        )
        invalid_rows = [row for row in join_audit if row["status"] == "invalid_allowlisted_row"]
        allowlist_keys = {
            (
                str(item["subject_id"]),
                round(float(item["start_time"]), 3),
                round(float(item["stop_time"]), 3),
                str(item["speaker"]),
            )
            for item in PACKED30_INVALID_ROW_ALLOWLIST
        }
        observed_keys = {
            (
                str(row["subject_id"]),
                round(float(row["start_time"]), 3),
                round(float(row["stop_time"]), 3),
                str(row["speaker"]),
            )
            for row in invalid_rows
        }
        self.require(len(invalid_rows) == 4 and observed_keys == allowlist_keys, "Invalid-row allowlist mismatch.")
        unexpected_invalid = [
            row for row in join_audit if row.get("invalid_reason") and row["status"] != "invalid_allowlisted_row"
        ]
        self.require(not unexpected_invalid, f"Unexpected invalid rows: {unexpected_invalid[:3]}")

        chunks_by_split_label: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            chunks_by_split_label[str(row["split_original"])][int(row["label"])] += 1
        self.require(
            {name: dict(chunks_by_split_label[name]) for name in ("train", "val", "test")}
            == EXPECTED_CHUNKS_BY_SPLIT_LABEL,
            f"Chunk split/label totals mismatch: {dict(chunks_by_split_label)}",
        )
        self.require(len(rows) == EXPECTED_TOTALS["chunks"], f"Chunk total {len(rows)} != 3021")

        sample_ids = [str(row["sample_id"]) for row in rows]
        self.require(len(sample_ids) == len(set(sample_ids)), "Duplicate sample IDs in manifest.")
        self.require(all(CHUNK_ID_RE.fullmatch(sample_id) for sample_id in sample_ids), "Chunk ID format mismatch.")
        subject_chunks: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            subject_chunks[str(row["subject_id"])].append(int(row["chunk_index"]))
        self.require(
            all(chunks == list(range(len(chunks))) for chunks in subject_chunks.values()),
            "Chunk indices are not consecutive from zero for every subject.",
        )
        per_subject_chunk_range = (min(len(v) for v in subject_chunks.values()), max(len(v) for v in subject_chunks.values()))
        self.require(
            per_subject_chunk_range == EXPECTED_TOTALS["chunks_per_subject_range"],
            f"Per-subject chunk range {per_subject_chunk_range} != (3, 43)",
        )
        retained_by_subject = {str(row["subject_id"]): int(row["retained_frames"]) for row in subjects}
        coverage_by_subject: dict[str, int] = defaultdict(int)
        final_samples: list[int] = []
        non_final_bad = []
        for row in rows:
            subject_id = str(row["subject_id"])
            coverage_by_subject[subject_id] += int(row["participant_sample_count"])
            if int(row["chunk_index"]) == int(row["num_chunks"]) - 1:
                final_samples.append(int(row["participant_sample_count"]))
            elif int(row["participant_sample_count"]) != PACKED30_CHUNK_SAMPLES:
                non_final_bad.append(row["sample_id"])
        self.require(not non_final_bad, f"Non-final chunks without 480000 samples: {non_final_bad[:5]}")
        self.require(all(1 <= value <= PACKED30_CHUNK_SAMPLES for value in final_samples), "Final chunk sample count out of range.")
        self.require(
            (min(final_samples), max(final_samples)) == EXPECTED_TOTALS["final_chunk_samples_range"],
            f"Final-chunk range {(min(final_samples), max(final_samples))} != (1440, 478480)",
        )
        self.require(
            all(coverage_by_subject[subject_id] == retained_by_subject.get(subject_id, -1) for subject_id in coverage_by_subject),
            "sum(participant_sample_count) != retained frames for some subject.",
        )
        self.require(
            sum(coverage_by_subject.values()) == EXPECTED_TOTALS["retained_frames"],
            f"Total covered frames {sum(coverage_by_subject.values())} != 1406614000",
        )

        retained_status_rows = {str(row["source_row_index"]): row for row in join_audit if row["status"] == "retained"}
        wav_paths = sorted({str(row["audio_path"]) for row in rows})
        self.require(len(wav_paths) == 189, f"Expected 189 source WAVs, found {len(wav_paths)}.")
        for wav_path in wav_paths:
            info = sf.info(wav_path)
            self.require(
                int(info.samplerate) == PACKED30_SAMPLE_RATE
                and int(info.channels) == 1
                and str(info.subtype) == "PCM_16",
                f"WAV contract violation: {wav_path} ({info.samplerate} Hz, {info.channels} ch, {info.subtype})",
            )
        for row in rows:
            for span in row["audio_spans"]:
                source = retained_status_rows.get(str(span["source_row_index"]))
                self.require(
                    source is not None and source["speaker"] == "Participant" and source["status"] == "retained",
                    f"Span not backed by a retained Participant row: {row['sample_id']} span {span['source_row_index']}",
                )

        for subject_id, subject_rows in sorted(subject_chunks.items()):
            spans: list[tuple[int, int]] = []
            for row in rows:
                if str(row["subject_id"]) != subject_id:
                    continue
                spans.extend((int(span["start_frame"]), int(span["end_frame"])) for span in row["audio_spans"])
            spans.sort()
            overlap = any(
                previous_end > start for (_, previous_end), (start, _) in zip(spans, spans[1:])
            )
            self.require(not overlap, f"Span overlap detected for subject {subject_id}.")

        totals = corpus_audit.get("totals", {})
        for key, expected in (
            ("chunks", EXPECTED_TOTALS["chunks"]),
            ("retained_rows", EXPECTED_TOTALS["retained_rows"]),
            ("retained_frames", EXPECTED_TOTALS["retained_frames"]),
            ("nonblank_rows", EXPECTED_TOTALS["nonblank_rows"]),
            ("blank_lines", EXPECTED_TOTALS["blank_lines"]),
        ):
            self.require(int(totals.get(key, -1)) == expected, f"Corpus audit {key} mismatch: {totals.get(key)} != {expected}")
        diagnostics = corpus_audit.get("diagnostics", {})
        self.require(
            all(feature in diagnostics for feature in (
                "full_interview_duration_seconds",
                "participant_speech_seconds",
                "retained_participant_row_count",
                "chunk_count",
            ))
            and all(cohort in diagnostics[feature] for feature in diagnostics for cohort in ("train", "val", "test", "all")),
            "Corpus audit diagnostics are incomplete.",
        )
        self.require(bool(metadata.get("build_signature")), "Metadata lacks the config build signature.")
        self.require(bool(metadata.get("manifest_sha256")), "Metadata lacks the canonical manifest hash.")

    def check_result_acceptance(self, run_root: Path) -> None:
        run_root = Path(run_root)
        runs_by_modality: dict[str, list[Path]] = defaultdict(list)
        for fold_dir in run_root.glob("*/*/fold_0"):
            modality = str(fold_dir.parent.parent.name)
            run_name = str(fold_dir.parent.name)
            if run_name.startswith("smoke_"):
                continue
            runs_by_modality[modality].append(fold_dir)
        for modality, fold_dirs in sorted(runs_by_modality.items()):
            complete_run: str | None = None
            issues: list[str] = []
            for fold_dir in fold_dirs:
                run_name = str(fold_dir.parent.name)
                run_ok, run_issues = self._check_one_run(modality, run_name, fold_dir)
                issues.extend(run_issues)
                if run_ok and complete_run is None:
                    complete_run = run_name
            self.require(
                complete_run is not None,
                f"modality {modality}: no complete production run among "
                f"{[str(d.parent.name) for d in fold_dirs]}; issues: {issues[:6]}",
            )
            if complete_run is not None:
                print(f"  audited modality={modality} complete_run={complete_run}", flush=True)
        self.require(
            set(runs_by_modality) == {"audio_only", "audio_text", "text_only"},
            f"Expected result runs for all three modalities, found {sorted(runs_by_modality)}",
        )

    def _check_one_run(self, modality: str, run_name: str, fold_dir: Path) -> tuple[bool, list[str]]:
        issues: list[str] = []

        def require(condition: bool, message: str) -> None:
            if not condition:
                issues.append(message)

        run_config_path = fold_dir / "run_config.yaml"
        best_dir = fold_dir / "best_model"
        require(run_config_path.is_file(), f"{modality}/{run_name}: missing run_config.yaml")
        if run_config_path.is_file():
            import yaml

            saved = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
            resolved_config = (saved or {}).get("config") or {}
            require(
                str(resolved_config.get("protocol_id")) == PACKED30_PROTOCOL_ID,
                f"{modality}/{run_name}: run_config.yaml protocol_id mismatch",
            )
            require(
                (fold_dir / "logs" / "split_used.json").is_file(),
                f"{modality}/{run_name}: missing logs/split_used.json",
            )
        require(
            (best_dir / "adapter_model.safetensors").is_file()
            and (best_dir / "adapter_config.json").is_file(),
            f"{modality}/{run_name}: best_model is missing",
        )
        eval_metrics = best_dir / "standalone_eval" / "metrics_original_teacher_forced.json"
        require(eval_metrics.is_file(), f"{modality}/{run_name}: missing official-test Qwen metrics")
        subject_rows_path = best_dir / "standalone_eval" / "predictions_subject_level.csv"
        if subject_rows_path.is_file():
            import csv

            with subject_rows_path.open(encoding="utf-8") as handle:
                subject_rows = list(csv.DictReader(handle))
            require(
                len(subject_rows) == 47,
                f"{modality}/{run_name}: expected 47 official-test subject rows, found {len(subject_rows)}",
            )
        cache_dir = fold_dir.parent / "hidden_features" / modality
        for name in (
            "outer_train.npz",
            "outer_train_rows.jsonl",
            "final_eval.npz",
            "final_eval_rows.jsonl",
            "extraction_metadata.json",
        ):
            require((cache_dir / name).is_file(), f"{modality}/{run_name}: missing hidden cache artifact {name}")
        if (cache_dir / "final_eval_rows.jsonl").is_file():
            heldout_rows = read_jsonl(cache_dir / "final_eval_rows.jsonl")
            heldout_subjects = {str(row["subject_id"]) for row in heldout_rows}
            require(len(heldout_subjects) == 47, f"{modality}/{run_name}: hidden cache heldout subjects != 47")
            row_protocol_ok = all(
                str(row.get("protocol_id", "")) == PACKED30_PROTOCOL_ID for row in heldout_rows
            )
            if not row_protocol_ok and modality == "text_only":
                extraction_metadata = read_json(cache_dir / "extraction_metadata.json")
                row_protocol_ok = str(extraction_metadata.get("protocol_id", "")) == PACKED30_PROTOCOL_ID
            require(
                row_protocol_ok,
                f"{modality}/{run_name}: hidden cache rows lack the v1 protocol id",
            )
        for variant in ("logreg_raw", "xgb_raw"):
            variant_dir = fold_dir.parent / "hidden_classifiers" / modality / variant
            for name in ("metrics.json", "classifier_metadata.json", "pipeline.joblib"):
                require((variant_dir / name).is_file(), f"{modality}/{run_name}/{variant}: missing {name}")
            if (variant_dir / "predictions_subject_level.jsonl").is_file():
                variant_rows = read_jsonl(variant_dir / "predictions_subject_level.jsonl")
                require(
                    len(variant_rows) == 47,
                    f"{modality}/{run_name}/{variant}: expected 47 subject rows, found {len(variant_rows)}",
                )
        return not issues, issues

    def rebuild_determinism(self, config_path: Path, unprocessed_root: Path, label_root: Path) -> None:
        import os

        os.environ["DAIC_UNPROCESSED_ROOT"] = str(unprocessed_root)
        os.environ["DAIC_LABEL_ROOT"] = str(label_root)
        from src.data.build_manifest import build_for_config

        def snapshot() -> dict[str, str]:
            build_for_config(config_path, [])
            return {
                name: hashlib.sha256((self.manifest_dir / name).read_bytes()).hexdigest()
                for name in (
                    PACKED30_MANIFEST_JSONL,
                    PACKED30_SUBJECTS_JSONL,
                    PACKED30_JOIN_AUDIT_JSONL,
                    PACKED30_CORPUS_AUDIT_JSON,
                    PACKED30_METADATA_JSON,
                )
            }

        first = snapshot()
        second = snapshot()
        self.require(first == second, "Rebuild is not byte-identical: " + json.dumps({k: (first[k], second[k]) for k in first if first[k] != second[k]}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit daic_participant_speech_packed30_v1 artifacts.")
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--run-root", type=Path, default=None, help="output_model/experiments/daic_participant_packed30")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the manifest twice and compare hashes (requires corpus).")
    parser.add_argument("--config", type=Path, default=None, help="Config path for --rebuild.")
    parser.add_argument("--unprocessed-root", type=Path, default=None)
    parser.add_argument("--label-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auditor = Auditor(args.manifest_dir, args.split_dir)
    auditor.check_manifest_acceptance()
    if args.run_root is not None:
        auditor.check_result_acceptance(args.run_root)
    if args.rebuild:
        if args.config is None or args.unprocessed_root is None or args.label_root is None:
            print("--rebuild requires --config, --unprocessed-root, and --label-root.", file=sys.stderr)
            sys.exit(2)
        auditor.rebuild_determinism(args.config, args.unprocessed_root, args.label_root)
    if auditor.failures:
        print(f"packed30 audit FAILED with {len(auditor.failures)} assertion(s).", file=sys.stderr)
        sys.exit(1)
    print("packed30 audit PASSED: protocol, corpus contract, manifest schema, and results are consistent.")


if __name__ == "__main__":
    main()
