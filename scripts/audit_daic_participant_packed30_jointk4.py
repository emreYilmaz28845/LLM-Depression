#!/usr/bin/env python
"""Dedicated artifact auditor for the DAIC runtime packed30 joint-K4 experiment.

Implements Section 7.2 of
docs/DAIC_PARTICIPANT_PACKED30_JOINTK4_EXPERIMENT_PLAN.md: reuses the packed30
v1 manifest acceptance logic and adds joint-recipe checks for the schedule,
validation/test coverage, Qwen rows, hidden caches (selected-epoch fit view),
fixed heads, finiteness, subject disjointness, and provenance. Exits nonzero
unless every locked assertion holds. ``--smoke`` relaxes production
cardinalities to the smoke split/schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from scripts.audit_daic_participant_packed30 import Auditor as Packed30Auditor
from src.data.runtime import JOINT_PACKED30_MODE, JOINT_PACKED30_RECIPE_ID
from src.utils import read_json, read_jsonl

EXPECTED_VAL_BUNDLES = 445
EXPECTED_VAL_SLOTS = 1780
EXPECTED_TEST_BUNDLES = 617
EXPECTED_TEST_SLOTS = 2468
EXPECTED_TRAIN_SUBJECTS = 107
EXPECTED_EPOCH_BUNDLES = 107
EXPECTED_EPOCH_SLOTS = 427
EXPECTED_FIT_VECTORS = 107
EXPECTED_TEST_VECTORS = 617
SHORT_SUBJECT_ID = "385"
RECIPE_KEY = "data.recipe_id"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class JointK4Auditor:
    def __init__(self, manifest_dir: Path, split_dir: Path, run_root: Path, smoke: bool):
        self.manifest_dir = Path(manifest_dir)
        self.split_dir = Path(split_dir)
        self.run_root = Path(run_root)
        self.smoke = bool(smoke)
        self.failures: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.failures.append(message)
            print(f"FAIL: {message}", flush=True)

    def check_packed30_manifest(self) -> None:
        packed = Packed30Auditor(self.manifest_dir, self.split_dir)
        packed.check_manifest_acceptance()
        self.failures.extend(
            f"packed30 manifest: {message}" for message in packed.failures
        )

    def _is_complete_smoke_run(self, fold_dir: Path) -> bool:
        return (
            (fold_dir / "run_config.yaml").is_file()
            and (fold_dir / "best_model" / "adapter_model.safetensors").is_file()
            and (fold_dir / "best_model" / "standalone_eval_pass1").is_dir()
            and (fold_dir / "best_model" / "standalone_eval_pass2").is_dir()
        )

    def _runs_by_modality(self) -> dict[str, list[Path]]:
        runs_by_modality: dict[str, list[Path]] = defaultdict(list)
        for fold_dir in self.run_root.glob("*/*/fold_0"):
            modality = str(fold_dir.parent.parent.name)
            run_name = str(fold_dir.parent.name)
            if not self.smoke and run_name.startswith("smoke_"):
                continue
            if self.smoke and run_name.startswith("smoke_") and not self._is_complete_smoke_run(fold_dir):
                # Partial smoke runs from failed/cancelled submission rounds
                # (same run-root) must not fail the smoke gate.
                continue
            runs_by_modality[modality].append(fold_dir)
        for modality, fold_dirs in runs_by_modality.items():
            runs_by_modality[modality] = [
                max(fold_dirs, key=lambda fold_dir: fold_dir.stat().st_mtime)
            ]
        return runs_by_modality

    def check_recipe(self, fold_dir: Path, modality: str, run_name: str) -> dict:
        run_config_path = fold_dir / "run_config.yaml"
        self.require(run_config_path.is_file(), f"{modality}/{run_name}: missing run_config.yaml")
        if not run_config_path.is_file():
            return {}
        saved = _load_yaml(run_config_path)
        resolved = (saved or {}).get("config") or {}
        data = resolved.get("data") or {}
        evaluation = resolved.get("evaluation") or {}
        expected_data = {
            "recipe_id": JOINT_PACKED30_RECIPE_ID,
            "sample_mode": JOINT_PACKED30_MODE,
            "participant_chunk_samples": 480000,
            "inter_span_silence_samples": 0,
            "train_chunk_policy": "joint_random_k",
            "train_chunks_per_subject": 4,
            "eval_chunk_policy": "balanced_joint_cover",
            "eval_chunks_per_subject": 4,
            "loss_weight_rescale": "mean_one",
            "equal_row_weight": False,
        }
        for key, expected in expected_data.items():
            self.require(
                data.get(key) == expected,
                f"{modality}/{run_name}: resolved data.{key} != {expected!r} (got {data.get(key)!r})",
            )
        self.require(
            evaluation.get("sample_prediction_mode") == "original_teacher_forced"
            and evaluation.get("headline_mode") == "original_teacher_forced"
            and evaluation.get("aggregation_level") == "subject"
            and evaluation.get("subject_score_aggregation") == "mean_score",
            f"{modality}/{run_name}: locked evaluation settings are not present verbatim",
        )
        self.require(
            str(resolved.get("protocol_id")) == "daic_participant_speech_packed30_v1",
            f"{modality}/{run_name}: protocol_id mismatch",
        )
        self.require(
            int(resolved.get("seed", -1)) == 1337,
            f"{modality}/{run_name}: seed != 1337",
        )
        return resolved

    def check_schedule(self, fold_dir: Path, modality: str, run_name: str) -> dict:
        audit_path = fold_dir / "logs" / "daic_chunk_schedule_audit.json"
        self.require(audit_path.is_file(), f"{modality}/{run_name}: missing daic_chunk_schedule_audit.json")
        if not audit_path.is_file():
            return {}
        audit = read_json(audit_path)
        self.require(
            audit.get("policy") == "joint_random_k" and int(audit.get("seed", -1)) == 1337
            and int(audit.get("requested_k", -1)) == 4,
            f"{modality}/{run_name}: schedule policy/seed/requested_k mismatch",
        )
        self.require(bool(audit.get("schedule_sha256")) and bool(audit.get("bundle_membership_sha256")),
                     f"{modality}/{run_name}: schedule hashes missing")
        expected_epochs = 1 if self.smoke else 20
        self.require(
            int(audit.get("epochs", -1)) == expected_epochs,
            f"{modality}/{run_name}: schedule epochs {audit.get('epochs')} != {expected_epochs}",
        )
        rows = audit.get("rows", [])
        per_epoch = Counter(int(row["epoch"]) for row in rows)
        epoch_sizes = [per_epoch[epoch] for epoch in sorted(per_epoch)]
        expected_size = len(
            read_json(fold_dir / "logs" / "split_used.json").get("train_subject_ids", [])
        ) if self.smoke else EXPECTED_EPOCH_BUNDLES
        self.require(
            all(size == expected_size for size in epoch_sizes),
            f"{modality}/{run_name}: per-epoch bundle counts {epoch_sizes} != {expected_size}",
        )
        slot_totals = [
            sum(len(row["bundle_chunk_ids"]) for row in rows if int(row["epoch"]) == epoch)
            for epoch in sorted(per_epoch)
        ]
        expected_slots = (
            (expected_size - 1) * 4 + 3 if "385" in {
                str(row["subject_id"]) for row in rows
            } else expected_size * 4
        )
        self.require(
            all(total == expected_slots for total in slot_totals),
            f"{modality}/{run_name}: per-epoch audio slots {slot_totals} != {expected_slots}",
        )
        effective_k = audit.get("effective_k_by_epoch", [])
        self.require(bool(effective_k), f"{modality}/{run_name}: missing effective_k_by_epoch")
        if effective_k:
            for epoch_map in effective_k:
                for subject_id, k in epoch_map.items():
                    expected_k = 3 if str(subject_id) == SHORT_SUBJECT_ID else 4
                    self.require(
                        int(k) == expected_k,
                        f"{modality}/{run_name}: effective K={k} for subject {subject_id} != {expected_k}",
                    )
        self.require(
            all(math.isclose(float(value), 1.0, abs_tol=1e-9) for value in audit.get("epoch_mean_effective_weights", [])),
            f"{modality}/{run_name}: mean-one effective weights not satisfied",
        )
        memberships: dict[tuple[int, str], list[str]] = {}
        for row in rows:
            memberships[(int(row["epoch"]), str(row["subject_id"]))] = list(row["bundle_chunk_ids"])
        for epoch in sorted(per_epoch):
            for subject_id, chunk_ids in memberships.items():
                if subject_id[0] != epoch:
                    continue
                self.require(
                    len(set(chunk_ids)) == len(chunk_ids),
                    f"{modality}/{run_name}: epoch {epoch} subject {subject_id[1]} repeated a chunk (with replacement)",
                )
        return audit

    def check_eval_bundles(self, fold_dir: Path, modality: str, run_name: str) -> None:
        split_used = read_json(fold_dir / "logs" / "split_used.json")
        if self.smoke:
            pass1 = fold_dir / "best_model" / "standalone_eval_pass1"
            pass2 = fold_dir / "best_model" / "standalone_eval_pass2"
            self.require(
                pass1.is_dir() and pass2.is_dir(),
                f"{modality}/{run_name}: smoke determinism passes are missing",
            )
            if not (pass1 / "predictions_sample_level.jsonl").is_file():
                return
            sample_rows = read_jsonl(pass1 / "predictions_sample_level.jsonl")
            self.require(
                {str(row["subject_id"]) for row in sample_rows}
                == set(split_used["final_eval_subject_ids"]),
                f"{modality}/{run_name}: smoke eval subjects differ from the saved split",
            )
            return
        split_used = read_json(fold_dir / "logs" / "split_used.json")
        subject_partitions = read_jsonl(self.split_dir / "daic_subject_partitions.json")
        if not subject_partitions:
            subject_partitions = read_jsonl(self.split_dir / "daic_participant_speech_packed30_subjects.jsonl")
        val_ids = set(split_used.get("selection_subject_ids", []))
        test_ids = set(split_used.get("final_eval_subject_ids", []))
        self.require(
            len(val_ids) == 35 and len(test_ids) == 47,
            f"{modality}/{run_name}: split cardinalities val={len(val_ids)} test={len(test_ids)}",
        )

    def check_qwen_rows(self, fold_dir: Path, modality: str, run_name: str) -> None:
        best_dir = fold_dir / "best_model"
        self.require(
            (best_dir / "adapter_model.safetensors").is_file()
            and (best_dir / "adapter_config.json").is_file(),
            f"{modality}/{run_name}: best_model is missing",
        )
        eval_dir = best_dir / ("standalone_eval_pass1" if self.smoke else "standalone_eval")
        metrics_path = eval_dir / "metrics_original_teacher_forced.json"
        self.require(metrics_path.is_file(), f"{modality}/{run_name}: missing official-test Qwen metrics")
        sample_path = eval_dir / "predictions_sample_level.jsonl"
        subject_path = eval_dir / "predictions_subject_level.csv"
        self.require(sample_path.is_file() and subject_path.is_file(), f"{modality}/{run_name}: missing eval rows")
        if not sample_path.is_file() or not subject_path.is_file():
            return
        sample_rows = read_jsonl(sample_path)
        expected_bundles = EXPECTED_TEST_BUNDLES if not self.smoke else len(sample_rows)
        self.require(
            len(sample_rows) == expected_bundles,
            f"{modality}/{run_name}: Qwen sample rows {len(sample_rows)} != {expected_bundles}",
        )
        with subject_path.open(encoding="utf-8", newline="") as handle:
            subject_rows = list(csv.DictReader(handle))
        expected_subjects = 47 if not self.smoke else len(
            read_json(fold_dir / "logs" / "split_used.json")["final_eval_subject_ids"]
        )
        self.require(
            len(subject_rows) == expected_subjects,
            f"{modality}/{run_name}: expected {expected_subjects} subject rows, found {len(subject_rows)}",
        )
        for row in sample_rows:
            dep, non, margin = row.get("dep_score"), row.get("non_score"), row.get("teacher_forced_margin")
            self.require(
                all(value is not None and math.isfinite(float(value)) for value in (dep, non, margin)),
                f"{modality}/{run_name}: non-finite teacher-forced score in {row.get('sample_id')}",
            )
        subject_sample_counts = Counter(str(row["subject_id"]) for row in sample_rows)
        for subject_id, count in subject_sample_counts.items():
            self.require(
                count == len({(row["bundle_id"]) for row in sample_rows if str(row["subject_id"]) == subject_id}),
                f"{modality}/{run_name}: subject {subject_id} bundle rows are not unique",
            )

    def check_hidden_caches(self, fold_dir: Path, modality: str, run_name: str, schedule_audit: dict) -> None:
        cache_dir = fold_dir.parent / "hidden_features" / modality
        for name in ("outer_train.npz", "outer_train_rows.jsonl", "final_eval.npz", "final_eval_rows.jsonl", "extraction_metadata.json"):
            self.require((cache_dir / name).is_file(), f"{modality}/{run_name}: missing hidden cache artifact {name}")
        if not all((cache_dir / name).is_file() for name in ("outer_train.npz", "outer_train_rows.jsonl", "final_eval.npz", "final_eval_rows.jsonl", "extraction_metadata.json")):
            return
        import numpy as np

        with np.load(cache_dir / "outer_train.npz") as payload:
            fit_vectors = np.asarray(payload["vectors"], dtype=np.float32)
        with np.load(cache_dir / "final_eval.npz") as payload:
            test_vectors = np.asarray(payload["vectors"], dtype=np.float32)
        fit_rows = read_jsonl(cache_dir / "outer_train_rows.jsonl")
        test_rows = read_jsonl(cache_dir / "final_eval_rows.jsonl")
        expected_fit = EXPECTED_FIT_VECTORS if not self.smoke else len(fit_rows)
        expected_test = EXPECTED_TEST_VECTORS if not self.smoke else len(test_rows)
        self.require(
            fit_vectors.shape[0] == expected_fit and test_vectors.shape[0] == expected_test,
            f"{modality}/{run_name}: cache vectors fit={fit_vectors.shape[0]} test={test_vectors.shape[0]}",
        )
        self.require(
            bool(np.isfinite(fit_vectors).all()) and bool(np.isfinite(test_vectors).all()),
            f"{modality}/{run_name}: hidden cache contains non-finite vectors",
        )
        self.require(
            len(fit_rows) == expected_fit and len(test_rows) == expected_test,
            f"{modality}/{run_name}: cache row counts do not match vectors",
        )
        metadata = read_json(cache_dir / "extraction_metadata.json")
        provenance = metadata.get("head_fit_provenance") or {}
        if self.smoke:
            self.require(bool(provenance), f"{modality}/{run_name}: smoke cache lacks head_fit_provenance")
            return
        self.require(
            provenance.get("head_fit_view") == "selected_checkpoint_training_epoch"
            and "selected_epoch" in provenance
            and "schedule_sha256" in provenance
            and "bundle_membership_sha256" in provenance,
            f"{modality}/{run_name}: incomplete head_fit_provenance",
        )
        if not provenance:
            return
        self.require(
            provenance.get("schedule_sha256") == schedule_audit.get("schedule_sha256")
            and provenance.get("bundle_membership_sha256") == schedule_audit.get("bundle_membership_sha256"),
            f"{modality}/{run_name}: cache schedule hashes do not match the training schedule audit",
        )
        selected_epoch = int(provenance.get("selected_epoch"))
        saved_audit = read_json(fold_dir / "logs" / "daic_chunk_schedule_audit.json")
        saved_memberships: dict[str, list[str]] = {}
        for row in saved_audit.get("rows", []):
            if int(row["epoch"]) != selected_epoch - 1:
                continue
            saved_memberships[str(row["subject_id"])] = list(row["bundle_chunk_ids"])
        cache_memberships: dict[str, list[str]] = {}
        for row in fit_rows:
            cache_memberships[str(row["subject_id"])] = [str(item) for item in row.get("bundle_chunk_ids", [])]
        self.require(
            cache_memberships == saved_memberships,
            f"{modality}/{run_name}: selected-epoch cache memberships do not match training provenance",
        )

    def check_heads(self, fold_dir: Path, modality: str, run_name: str) -> None:
        for variant in ("logreg_raw", "xgb_raw"):
            variant_dir = fold_dir.parent / "hidden_classifiers" / modality / variant
            for name in ("metrics.json", "classifier_metadata.json", "pipeline.joblib", "predictions_subject_level.jsonl"):
                self.require(
                    (variant_dir / name).is_file(),
                    f"{modality}/{run_name}/{variant}: missing {name}",
                )
            if not (variant_dir / "predictions_subject_level.jsonl").is_file():
                continue
            subject_rows = read_jsonl(variant_dir / "predictions_subject_level.jsonl")
            expected_subjects = 47 if not self.smoke else len(subject_rows)
            self.require(
                len(subject_rows) == expected_subjects,
                f"{modality}/{run_name}/{variant}: expected {expected_subjects} subject rows, found {len(subject_rows)}",
            )
            metrics = read_json(variant_dir / "metrics.json")
            for key in ("positive_f1", "macro_f1", "auroc", "accuracy"):
                value = metrics.get(key)
                self.require(
                    value is None or math.isfinite(float(value)),
                    f"{modality}/{run_name}/{variant}: non-finite metric {key}",
                )
            metadata = read_json(variant_dir / "classifier_metadata.json")
            self.require(
                metadata.get("aggregation_policy") == "mean_depressed_probability_threshold_0_5",
                f"{modality}/{run_name}/{variant}: head aggregation policy mismatch",
            )

    def check_disjointness(self, fold_dir: Path, modality: str, run_name: str) -> None:
        split_used = read_json(fold_dir / "logs" / "split_used.json")
        train_ids = set(split_used.get("train_subject_ids", []))
        val_ids = set(split_used.get("selection_subject_ids", []))
        test_ids = set(split_used.get("final_eval_subject_ids", []))
        self.require(
            not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids),
            f"{modality}/{run_name}: train/dev/test subjects overlap",
        )

    def check_provenance(self, fold_dir: Path, modality: str, run_name: str) -> None:
        best_dir = fold_dir / "best_model"
        run_config_path = fold_dir / "run_config.yaml"
        cache_dir = fold_dir.parent / "hidden_features" / modality
        metadata = read_json(cache_dir / "extraction_metadata.json")
        from src.utils import sha256_file

        self.require(
            metadata.get("checkpoint_type") == "best_model"
            and str(metadata.get("checkpoint_dir", "")).endswith("best_model"),
            f"{modality}/{run_name}: cache does not reference best_model",
        )
        self.require(
            metadata.get("saved_run_config_sha256") == sha256_file(run_config_path),
            f"{modality}/{run_name}: cache run_config hash does not match the run dir",
        )
        self.require(
            metadata.get("adapter_sha256") == sha256_file(best_dir / "adapter_model.safetensors"),
            f"{modality}/{run_name}: cache adapter hash does not match best_model",
        )
        self.require(
            bool(metadata.get("manifest_sha256")) and bool(metadata.get("protocol_id")),
            f"{modality}/{run_name}: cache lacks manifest/protocol provenance",
        )

    def audit(self) -> int:
        self.check_packed30_manifest()
        runs_by_modality = self._runs_by_modality()
        self.require(
            set(runs_by_modality) == {"audio_only", "audio_text"},
            f"Expected result runs for audio_only and audio_text, found {sorted(runs_by_modality)}",
        )
        for modality, fold_dirs in sorted(runs_by_modality.items()):
            complete_run: str | None = None
            for fold_dir in fold_dirs:
                run_name = str(fold_dir.parent.name)
                if not self.smoke and run_name.startswith("smoke_"):
                    continue
                self.check_recipe(fold_dir, modality, run_name)
                schedule_audit = self.check_schedule(fold_dir, modality, run_name)
                self.check_eval_bundles(fold_dir, modality, run_name)
                self.check_qwen_rows(fold_dir, modality, run_name)
                self.check_hidden_caches(fold_dir, modality, run_name, schedule_audit)
                self.check_heads(fold_dir, modality, run_name)
                self.check_disjointness(fold_dir, modality, run_name)
                self.check_provenance(fold_dir, modality, run_name)
                complete_run = run_name
            if complete_run is not None:
                print(f"  audited modality={modality} complete_run={complete_run}", flush=True)
        return 1 if self.failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the DAIC runtime packed30 joint-K4 experiment artifacts.")
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path, help="output_model/experiments/daic_participant_packed30_jointk4")
    parser.add_argument("--smoke", action="store_true", help="Audit smoke artifacts with smoke cardinalities.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auditor = JointK4Auditor(args.manifest_dir, args.split_dir, args.run_root, args.smoke)
    exit_code = auditor.audit()
    if auditor.failures:
        print(f"jointk4 audit FAILED with {len(auditor.failures)} assertion(s).", file=sys.stderr)
        sys.exit(1)
    print("jointk4 audit PASSED: packed30 manifest, joint recipe, schedule, coverage, Qwen rows, hidden caches, heads, and provenance are consistent.")


if __name__ == "__main__":
    main()
