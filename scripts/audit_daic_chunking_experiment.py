from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import classification_metrics
from src.utils import read_json, read_jsonl, save_json


CONDITIONS = ("c1", "c2", "c3", "c4")
HEADS = ("qwen", "logreg_raw", "xgb_raw")
TRAINING_LAYOUT = {
    "joint": ("joint_random_k4", "joint"),
    "rotary": ("independent_rotary_k4", "rotary"),
    "all": ("independent_all", "all"),
}


def _qwen_dir(root: Path, condition: str) -> Path:
    return root / "qwen" / condition


def _head_dir(root: Path, condition: str, head: str) -> Path:
    return root / "classical" / condition / head


def _load_cell(root: Path, condition: str, head: str) -> tuple[dict, list[dict], list[dict]]:
    directory = _qwen_dir(root, condition) if head == "qwen" else _head_dir(root, condition, head)
    metrics_name = "metrics_original_teacher_forced.json" if head == "qwen" else "metrics.json"
    metrics = read_json(directory / metrics_name)
    samples = read_jsonl(directory / "predictions_sample_level.jsonl")
    if head == "qwen":
        with (directory / "predictions_subject_level.csv").open(newline="", encoding="utf-8") as handle:
            subjects = list(csv.DictReader(handle))
    else:
        subjects = read_jsonl(directory / "predictions_subject_level.jsonl")
    return metrics, samples, subjects


def _prediction(row: dict[str, Any]) -> int:
    return int(row.get("prediction", row.get("aggregated_prediction")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_balanced_bundles(samples: list[dict], cell: str, failures: list[str]) -> dict[str, Any]:
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in samples:
        by_subject[str(row["subject_id"])].append(row)
    summary: dict[str, Any] = {}
    for subject_id, rows in sorted(by_subject.items()):
        coverage = Counter(str(chunk) for row in rows for chunk in row.get("bundle_chunk_ids", []))
        n_chunks = len(coverage)
        if not n_chunks:
            failures.append(f"{cell}: missing C2 bundle membership")
            continue
        expected_bundles = n_chunks // math.gcd(n_chunks, 4)
        expected_occurrences = 4 // math.gcd(n_chunks, 4)
        bundle_ids = {str(row.get("bundle_id")) for row in rows}
        if len(rows) != expected_bundles or len(bundle_ids) != expected_bundles:
            failures.append(
                f"{cell}/{subject_id}: expected {expected_bundles} unique bundles, got {len(rows)} rows/{len(bundle_ids)} IDs"
            )
        if set(coverage.values()) != {expected_occurrences}:
            failures.append(
                f"{cell}/{subject_id}: unequal bundle coverage {sorted(set(coverage.values()))}, expected {expected_occurrences}"
            )
        declared = {int(row.get("bundle_coverage_count", -1)) for row in rows}
        if declared != {expected_occurrences}:
            failures.append(f"{cell}/{subject_id}: declared coverage {sorted(declared)} is invalid")
        summary[subject_id] = {
            "chunks": n_chunks,
            "bundles": len(rows),
            "occurrences_per_chunk": expected_occurrences,
        }
    return summary


def _training_dirs(root: Path) -> dict[str, Path]:
    project_root = root.parents[2]
    run_id = root.name
    return {
        strategy: project_root / "output_model" / "daic_chunking" / parent / f"{run_id}_{suffix}" / "fold_0"
        for strategy, (parent, suffix) in TRAINING_LAYOUT.items()
    }


def _audit_training(root: Path, failures: list[str]) -> dict[str, Any]:
    directories = _training_dirs(root)
    splits: dict[str, dict] = {}
    configs: dict[str, dict] = {}
    summary: dict[str, Any] = {}
    for strategy, directory in directories.items():
        try:
            run_record = yaml.safe_load((directory / "run_config.yaml").read_text(encoding="utf-8"))
            config = run_record["config"]
            configs[strategy] = config
            split = read_json(directory / "logs" / "split_used.json")
            splits[strategy] = split
            train_ids = set(map(str, split["train_subject_ids"]))
            val_ids = set(map(str, split["selection_subject_ids"]))
            test_ids = set(map(str, split["final_eval_subject_ids"]))
            if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
                failures.append(f"{strategy}: train/validation/test subjects overlap")
            if split.get("split_names") != {"train": "train", "selection": "val", "final_eval": "test"}:
                failures.append(f"{strategy}: not the official train/val/test protocol")
            history = read_json(directory / "logs" / "training_history.json")
            peak = read_json(directory / "logs" / "peak_gpu_memory.json")
            audio = read_json(directory / "logs" / "audio_budget_audit_train.json")
            if strategy != "joint" and float(audio["audio_seconds_per_example"]["max"]) > 30.0 + 1e-8:
                failures.append(f"{strategy}: independent audio exceeds 30 seconds")
            summary[strategy] = {
                "directory": str(directory),
                "configured_epochs": int(config["training"]["num_train_epochs"]),
                "epochs_completed": len(history),
                "selected_epoch": max(
                    history,
                    key=lambda row: float(row.get(config["training"]["selection_metric"], float("-inf"))),
                )["epoch"],
                "train_subjects": len(train_ids),
                "validation_subjects": len(val_ids),
                "test_subjects": len(test_ids),
                "peak_gpu_allocated_gib": peak.get("max_allocated_gib"),
                "peak_gpu_reserved_gib": peak.get("max_reserved_gib"),
                "max_audio_seconds_per_example": audio["audio_seconds_per_example"]["max"],
                "manifest_hash": run_record.get("manifest_hash"),
                "split_metadata_hash": run_record.get("split_metadata_hash"),
            }
            if strategy != "joint":
                schedule = read_json(directory / "logs" / "daic_chunk_schedule_audit.json")
                epochs = int(config["training"]["num_train_epochs"])
                if int(schedule.get("epochs", -1)) != epochs:
                    failures.append(f"{strategy}: schedule epoch count does not match resolved config")
                if not schedule.get("equal_total_subject_weight"):
                    failures.append(f"{strategy}: unequal total subject loss weights")
                for subject_id, totals in schedule.get("subject_epoch_weight_totals", {}).items():
                    if len(totals) != epochs or any(not math.isclose(float(value), 1.0, abs_tol=1e-8) for value in totals):
                        failures.append(f"{strategy}/{subject_id}: per-epoch subject weights are not all 1.0")
                for subject_id, counts in schedule.get("exposure_counts_by_subject", {}).items():
                    values = list(map(int, counts.values()))
                    if strategy == "all" and (not values or set(values) != {epochs}):
                        failures.append(f"{strategy}/{subject_id}: K=all did not include every chunk every epoch")
                    if strategy == "rotary":
                        if sum(values) != epochs * 4 or max(values) - min(values) > 1:
                            failures.append(f"{strategy}/{subject_id}: rotary exposure is not balanced over {epochs} epochs")
                summary[strategy]["schedule"] = {
                    "policy": schedule.get("policy"),
                    "epoch_example_counts": schedule.get("epoch_example_counts"),
                    "equal_total_subject_weight": schedule.get("equal_total_subject_weight"),
                    "gradient_accumulation": schedule.get("gradient_accumulation"),
                }
        except Exception as exc:
            failures.append(f"{strategy} training audit: {exc}")
    if len(splits) == 3:
        canonical = next(iter(splits.values()))
        for strategy, split in splits.items():
            for key in ("train_subject_ids", "selection_subject_ids", "final_eval_subject_ids"):
                if list(map(str, split[key])) != list(map(str, canonical[key])):
                    failures.append(f"{strategy}: {key} differs across checkpoints")
    if len(configs) == 3:
        try:
            joint_counts = read_json(directories["joint"] / "logs" / "sample_partition_counts.json")
            joint_examples = int(joint_counts["train"]["total_samples"])
            world_size = 4
            joint_accum = int(configs["joint"]["training"]["gradient_accumulation_steps"])
            reference_updates = math.ceil(joint_examples / (world_size * joint_accum))
            summary["joint"]["optimizer_updates_per_epoch"] = reference_updates
            for strategy in ("rotary", "all"):
                schedule = summary[strategy]["schedule"]
                examples = int(schedule["epoch_example_counts"][0])
                accum = int(schedule["gradient_accumulation"]["resolved_gradient_accumulation_steps"])
                updates = math.ceil(examples / (world_size * accum))
                summary[strategy]["optimizer_updates_per_epoch"] = updates
                if updates != reference_updates:
                    failures.append(
                        f"{strategy}: {updates} optimizer updates/epoch does not match joint reference {reference_updates}"
                    )
        except Exception as exc:
            failures.append(f"optimizer-update audit: {exc}")
    return summary


def _paired_bootstrap(
    baseline: list[dict], comparison: list[dict], *, iterations: int = 10000, seed: int = 1337
) -> dict[str, Any]:
    base = {str(row["subject_id"]): row for row in baseline}
    comp = {str(row["subject_id"]): row for row in comparison}
    ids = sorted(base)
    if ids != sorted(comp):
        raise ValueError("Paired comparison subject sets differ.")
    rng = random.Random(seed)
    deltas = {"positive_f1": [], "macro_f1": []}
    for _ in range(iterations):
        sampled = [rng.choice(ids) for _ in ids]
        y = [int(base[sid]["label"]) for sid in sampled]
        base_metrics = classification_metrics(y, [_prediction(base[sid]) for sid in sampled])
        comp_metrics = classification_metrics(y, [_prediction(comp[sid]) for sid in sampled])
        for metric in deltas:
            deltas[metric].append(comp_metrics[metric] - base_metrics[metric])
    output = {}
    for metric, values in deltas.items():
        values.sort()
        output[metric] = {
            "low": values[int(0.025 * (len(values) - 1))],
            "high": values[int(0.975 * (len(values) - 1))],
        }
    return output


def _mcnemar(baseline: list[dict], comparison: list[dict]) -> dict[str, Any]:
    base = {str(row["subject_id"]): row for row in baseline}
    comp = {str(row["subject_id"]): row for row in comparison}
    if set(base) != set(comp):
        raise ValueError("McNemar subject sets differ.")
    b = c = 0
    for subject_id in base:
        gold = int(base[subject_id]["label"])
        base_correct = _prediction(base[subject_id]) == gold
        comp_correct = _prediction(comp[subject_id]) == gold
        b += int(base_correct and not comp_correct)
        c += int(not base_correct and comp_correct)
    n = b + c
    tail = sum(math.comb(n, index) for index in range(0, min(b, c) + 1)) / (2**n) if n else 1.0
    return {"baseline_only_correct": b, "comparison_only_correct": c, "exact_two_sided_p": min(1.0, 2 * tail)}


def audit(root: Path, sacct_path: Path | None) -> dict[str, Any]:
    failures: list[str] = []
    cells: list[dict[str, Any]] = []
    subject_rows: dict[tuple[str, str], list[dict]] = {}
    sample_rows: dict[tuple[str, str], list[dict]] = {}
    coverage_audits: dict[str, Any] = {}
    for condition in CONDITIONS:
        for head in HEADS:
            try:
                metrics, samples, subjects = _load_cell(root, condition, head)
                if len(subjects) != len({str(row["subject_id"]) for row in subjects}):
                    failures.append(f"{condition}/{head}: not exactly one prediction per subject")
                required = {"accuracy", "positive_f1", "macro_f1", "precision", "recall", "confusion_matrix", "auroc", "predicted_positive_rate"}
                missing = sorted(required - set(metrics))
                if missing:
                    failures.append(f"{condition}/{head}: missing metrics {missing}")
                subject_rows[(condition, head)] = subjects
                sample_rows[(condition, head)] = samples
                cells.append(
                    {
                        "condition": condition,
                        "head": head,
                        **{key: metrics.get(key) for key in sorted(required)},
                        "num_samples": len(samples),
                        "num_subjects": len(subjects),
                    }
                )
                if condition in {"c3", "c4"}:
                    sample_ids = [str(row["sample_id"]) for row in samples]
                    if len(sample_ids) != len(set(sample_ids)):
                        failures.append(f"{condition}/{head}: held-out chunk duplicated")
                    if any("chunk_id" not in row for row in samples):
                        failures.append(f"{condition}/{head}: independent prediction is missing chunk_id")
                if condition == "c2":
                    coverage_audits[f"{condition}/{head}"] = _audit_balanced_bundles(
                        samples, f"{condition}/{head}", failures
                    )
                expected_aggregation = (
                    "mean_teacher_forced_score_margin"
                    if head == "qwen"
                    else "mean_depressed_probability_threshold_0.5"
                )
                if metrics.get("aggregation_method") != expected_aggregation:
                    failures.append(f"{condition}/{head}: wrong aggregation method")
            except Exception as exc:
                failures.append(f"{condition}/{head}: {exc}")

    if subject_rows:
        reference_key = ("c1", "qwen")
        reference = {
            str(row["subject_id"]): int(row["label"]) for row in subject_rows.get(reference_key, [])
        }
        for key, rows in subject_rows.items():
            observed = {str(row["subject_id"]): int(row["label"]) for row in rows}
            if observed != reference:
                failures.append(f"{key[0]}/{key[1]}: subject IDs or labels differ from C1/Qwen")
    for condition in ("c3", "c4"):
        reference_chunks = {
            (str(row["subject_id"]), str(row["chunk_id"])) for row in sample_rows.get((condition, "qwen"), [])
        }
        for head in ("logreg_raw", "xgb_raw"):
            observed = {
                (str(row["subject_id"]), str(row["chunk_id"]))
                for row in sample_rows.get((condition, head), [])
            }
            if observed != reference_chunks:
                failures.append(f"{condition}/{head}: evaluated chunk set differs from Qwen")

    try:
        c1_meta = read_json(root / "hidden/joint/c1_fixed/extraction_metadata.json")
        c2_meta = read_json(root / "hidden/joint/c2_balanced/extraction_metadata.json")
        for key in ("adapter_sha256", "adapter_config_sha256", "saved_split_sha256", "manifest_sha256"):
            if c1_meta.get(key) != c2_meta.get(key):
                failures.append(f"C1/C2 hidden metadata mismatch: {key}")
        for variant in ("logreg_raw", "xgb_raw"):
            c1_cfg = read_json(root / f"classical/c1/{variant}/result_config.json")
            c2_cfg = read_json(root / f"classical/c2/{variant}/result_config.json")
            c2_head = read_json(root / f"classical/c2/{variant}/classifier_metadata.json")
            if c1_cfg["fit_cache_identity"] != c2_cfg["fit_cache_identity"]:
                failures.append(f"C1/C2 {variant} fit cache was not reused")
            if c2_head.get("fit_action") != "reused":
                failures.append(f"C2 {variant} fitted model was not reused")
            if _sha256(root / f"classical/c1/{variant}/pipeline.joblib") != _sha256(
                root / f"classical/c2/{variant}/pipeline.joblib"
            ):
                failures.append(f"C1/C2 {variant} pipeline bytes differ")
    except Exception as exc:
        failures.append(f"C1/C2 reuse audit: {exc}")

    training_audit = _audit_training(root, failures)
    sacct = None
    if sacct_path:
        sacct = sacct_path.read_text(encoding="utf-8")
        bad = [
            state
            for state in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED")
            if state in sacct
        ]
        if bad:
            failures.append(f"Slurm accounting contains terminal failures: {bad}")

    comparisons = []
    for head in HEADS:
        baseline = subject_rows.get(("c1", head))
        if not baseline:
            continue
        for condition in ("c2", "c3", "c4"):
            comparison = subject_rows.get((condition, head))
            if not comparison:
                continue
            baseline_cell = next(row for row in cells if row["condition"] == "c1" and row["head"] == head)
            comparison_cell = next(row for row in cells if row["condition"] == condition and row["head"] == head)
            comparisons.append(
                {
                    "head": head,
                    "condition": condition,
                    "delta_positive_f1": comparison_cell["positive_f1"] - baseline_cell["positive_f1"],
                    "delta_macro_f1": comparison_cell["macro_f1"] - baseline_cell["macro_f1"],
                    "paired_bootstrap_95ci": _paired_bootstrap(baseline, comparison),
                    "mcnemar": _mcnemar(baseline, comparison),
                }
            )
    payload = {
        "schema_version": "daic_chunking_experiment_audit.v1",
        "root": str(root),
        "passed": not failures and len(cells) == 12,
        "failures": failures,
        "cells": cells,
        "comparisons_against_c1": comparisons,
        "coverage_audits": coverage_audits,
        "training_audit": training_audit,
        "slurm_accounting_path": str(sacct_path) if sacct_path else None,
        "slurm_accounting_captured": sacct is not None,
    }
    save_json(payload, root / "audit.json")
    with (root / "comparison_12_cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]) if cells else ["condition", "head"])
        writer.writeheader()
        writer.writerows(cells)
    _write_report(payload, root)
    return payload


def _write_report(payload: dict[str, Any], root: Path) -> None:
    notes_path = root / "experiment_notes.json"
    notes = read_json(notes_path) if notes_path.exists() else {}
    lines = [
        "# DAIC-WOZ Chunking K Experiment Report",
        "",
        f"- Run: `{root.name}`",
        f"- Audit passed: **{payload['passed']}**",
        f"- Result cells: {len(payload['cells'])}/12",
        "- Headline metric: positive-class F1; primary secondary metric: macro-F1.",
        "",
        "## Twelve-cell results",
        "",
        "| Condition | Head | Accuracy | Positive F1 | Macro-F1 | Precision | Recall | AUROC | Positive rate | Subjects | Samples |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in payload["cells"]:
        lines.append(
            "| {condition} | {head} | {accuracy:.4f} | {positive_f1:.4f} | {macro_f1:.4f} | "
            "{precision:.4f} | {recall:.4f} | {auroc:.4f} | {predicted_positive_rate:.4f} | "
            "{num_subjects} | {num_samples} |".format(**cell)
        )
    lines += [
        "",
        "## Deltas and paired tests against C1",
        "",
        "| Head | Condition | Δ positive F1 | 95% bootstrap CI | Δ macro-F1 | 95% bootstrap CI | McNemar p |",
        "|---|---|---:|---|---:|---|---:|",
    ]
    for row in payload["comparisons_against_c1"]:
        pos = row["paired_bootstrap_95ci"]["positive_f1"]
        macro = row["paired_bootstrap_95ci"]["macro_f1"]
        lines.append(
            f"| {row['head']} | {row['condition']} | {row['delta_positive_f1']:.4f} | "
            f"[{pos['low']:.4f}, {pos['high']:.4f}] | {row['delta_macro_f1']:.4f} | "
            f"[{macro['low']:.4f}, {macro['high']:.4f}] | {row['mcnemar']['exact_two_sided_p']:.4g} |"
        )
    lines += ["", "## Training and provenance", ""]
    for strategy, item in payload.get("training_audit", {}).items():
        if not isinstance(item, dict) or "directory" not in item:
            continue
        lines.append(
            f"- {strategy}: {item['epochs_completed']}/{item['configured_epochs']} epochs executed; "
            f"selected epoch {item['selected_epoch']}; {item.get('optimizer_updates_per_epoch', '?')} optimizer "
            f"updates/epoch; peak allocated GPU memory {item.get('peak_gpu_allocated_gib', '?')} GiB; "
            f"manifest `{item.get('manifest_hash')}`; split `{item.get('split_metadata_hash')}`."
        )
    lines += [
        "",
        "## Coverage and statistical scope",
        "",
        "- C2 uses minimum cyclic K=4 bundles with equal per-chunk coverage; C3/C4 evaluate every held-out chunk once.",
        "- Confidence intervals are subject-paired nonparametric bootstrap intervals (10,000 resamples, seed 1337).",
        "- McNemar tests are exact two-sided tests on paired subject-level hard predictions.",
        "- The locked test set was used only for final evaluation, never for configuration or hyperparameter selection.",
    ]
    for title in ("failures_repairs", "exclusions", "limitations"):
        values = notes.get(title, [])
        if values:
            lines += ["", f"## {title.replace('_', ' ').title()}", ""]
            lines.extend(f"- {value}" for value in values)
    if payload["failures"]:
        lines += ["", "## Audit failures", ""]
        lines.extend(f"- {failure}" for failure in payload["failures"])
    (root / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--sacct", type=Path)
    args = parser.parse_args()
    payload = audit(args.root, args.sacct)
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
