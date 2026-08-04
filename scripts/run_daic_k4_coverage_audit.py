from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_yaml, read_json, read_jsonl, save_json, save_yaml
from src.aggregate import aggregate_predictions
from src.utils import AGGREGATION_LEVEL_SUBJECT, PREDICTION_MODE_ORIGINAL_TEACHER_FORCED


VIEWS = ("fixed4", "mincover4", "fixed15")
METRICS = ("accuracy", "positive_f1", "macro_f1", "precision", "recall")


def materialize_runtime_config(source: Path, destination: Path) -> Path:
    """Add audit-only evaluation controls without mutating the canonical YAML."""
    config = load_yaml(source)
    evaluation = config.setdefault("evaluation", {})
    evaluation.update(
        {
            "inference_dtype": "fp32",
            "candidate_batching": "sequential",
            "subject_score_aggregation": "mean_score",
            "reuse_derived_views": True,
        }
    )
    save_yaml(config, destination)
    return destination


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _prediction(row: dict[str, Any]) -> int:
    return int(row.get("prediction", row.get("teacher_forced_prediction", -1)))


def _metric(metrics: dict[str, Any], name: str) -> float:
    return float(metrics.get(f"binary_strict_{name}", metrics[name]))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def audit_and_report(
    output_root: Path,
    checkpoint_dir: Path,
    *,
    expected_subjects: int = 47,
    allow_omitted_checkpoint_adapter: bool = False,
    expected_modality: str = "audio_text",
) -> dict[str, Any]:
    if expected_modality not in {"audio_text", "audio_only"}:
        raise ValueError(f"Unsupported expected_modality={expected_modality!r}")
    failures: list[str] = []
    metrics_by_view: dict[str, dict[str, Any]] = {}
    subjects_by_view: dict[str, dict[str, dict[str, str]]] = {}
    construction_by_view: dict[str, list[dict[str, Any]]] = {}

    try:
        if checkpoint_dir.name != "best_model":
            failures.append("checkpoint: expected the authoritative best_model directory")
        if (
            not allow_omitted_checkpoint_adapter
            and not (checkpoint_dir / "adapter_model.safetensors").is_file()
        ):
            failures.append("checkpoint: missing adapter_model.safetensors")
        run_config = read_json(checkpoint_dir.parent / "run_config.json") if (
            checkpoint_dir.parent / "run_config.json"
        ).is_file() else None
        if run_config is None:
            import yaml

            run_config = yaml.safe_load((checkpoint_dir.parent / "run_config.yaml").read_text(encoding="utf-8"))
        resolved = run_config.get("config", run_config)
        data = resolved.get("data", {})
        if str(resolved.get("dataset", "")).lower() != "daic":
            failures.append("checkpoint: resolved dataset is not DAIC")
        if data.get("sample_mode") != "subject_audio" or int(data.get("chunks_per_subject", 0)) != 4:
            failures.append("checkpoint: training construction was not subject_audio K=4")
        expected_use_text = expected_modality == "audio_text"
        if not bool(data.get("use_audio")) or bool(data.get("use_text")) != expected_use_text:
            failures.append(f"checkpoint: expected the {expected_modality} control")
        if int(run_config.get("fold", -1)) != 0:
            failures.append("checkpoint: expected fold 0")
        split_used = read_json(checkpoint_dir.parent / "logs" / "split_used.json")
        train_ids = set(map(str, split_used["train_subject_ids"]))
        selection_ids = set(map(str, split_used["selection_subject_ids"]))
        final_ids = set(map(str, split_used["final_eval_subject_ids"]))
        if train_ids & selection_ids or train_ids & final_ids or selection_ids & final_ids:
            failures.append("checkpoint: train/selection/test subject contamination")
        if len(final_ids) != expected_subjects:
            failures.append(f"checkpoint: expected {expected_subjects} final-eval subjects, found {len(final_ids)}")
    except Exception as exc:
        failures.append(f"checkpoint provenance audit failed: {exc}")
        final_ids = set()

    for view in VIEWS:
        view_dir = output_root / view
        try:
            metrics_by_view[view] = read_json(view_dir / "metrics_original_teacher_forced.json")
            subject_rows = _read_csv(view_dir / "predictions_subject_level.csv")
            subjects_by_view[view] = {str(row["subject_id"]): row for row in subject_rows}
            if len(subject_rows) != len(subjects_by_view[view]):
                failures.append(f"{view}: duplicate subject predictions")
            if len(subject_rows) != expected_subjects:
                failures.append(f"{view}: expected {expected_subjects} subjects, found {len(subject_rows)}")
            if view != "fixed15":
                construction_by_view[view] = read_jsonl(view_dir / "view_construction.jsonl")
        except Exception as exc:
            failures.append(f"{view}: incomplete artifacts: {exc}")

    if subjects_by_view:
        subject_sets = {view: set(rows) for view, rows in subjects_by_view.items()}
        canonical = next(iter(subject_sets.values()))
        for view, values in subject_sets.items():
            if values != canonical:
                failures.append(f"{view}: subject set differs from the other views")
        if final_ids and canonical != final_ids:
            failures.append("evaluation subjects differ from checkpoint final_eval_subject_ids")
        for subject_id in sorted(canonical):
            labels = {int(rows[subject_id]["label"]) for rows in subjects_by_view.values() if subject_id in rows}
            if len(labels) != 1:
                failures.append(f"{subject_id}: inconsistent labels across views")

    fixed_rows = construction_by_view.get("fixed4", [])
    fixed_counts = Counter(str(row["subject_id"]) for row in fixed_rows)
    for row in fixed_rows:
        if int(row["chunks_per_model_input"]) != 4 or len(row["selected_chunk_ids"]) != 4:
            failures.append(f"fixed4/{row['subject_id']}: model input is not K=4")
    if fixed_rows and set(fixed_counts.values()) != {1}:
        failures.append("fixed4: expected exactly one model input per subject")

    cover_rows = construction_by_view.get("mincover4", [])
    cover_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cover_rows:
        cover_by_subject[str(row["subject_id"])].append(row)
    coverage_summary: dict[str, Any] = {}
    for subject_id, rows in sorted(cover_by_subject.items()):
        available = {int(row["num_chunks_available"]) for row in rows}
        if len(available) != 1:
            failures.append(f"mincover4/{subject_id}: inconsistent available chunk count")
            continue
        n = next(iter(available))
        expected_bundles = n // math.gcd(n, 4)
        expected_occurrences = 4 // math.gcd(n, 4)
        coverage = Counter(str(chunk) for row in rows for chunk in row["selected_chunk_ids"])
        if len(rows) != expected_bundles:
            failures.append(f"mincover4/{subject_id}: expected {expected_bundles} bundles, found {len(rows)}")
        if any(int(row["chunks_per_model_input"]) != 4 for row in rows):
            failures.append(f"mincover4/{subject_id}: model input is not K=4")
        if len(coverage) != n or set(coverage.values()) != {expected_occurrences}:
            failures.append(f"mincover4/{subject_id}: incomplete or unequal chunk coverage")
        coverage_summary[subject_id] = {
            "label": int(rows[0]["label"]),
            "available_chunks": n,
            "bundles": len(rows),
            "occurrences_per_chunk": expected_occurrences,
        }

    fixed15_samples_path = output_root / "fixed15" / "predictions_sample_level.jsonl"
    if fixed15_samples_path.exists():
        fixed15_counts = Counter(str(row["subject_id"]) for row in read_jsonl(fixed15_samples_path))
        if fixed15_counts and set(fixed15_counts.values()) != {15}:
            failures.append("fixed15: expected exactly 15 derived rows per subject")
        derivation = read_json(output_root / "fixed15" / "derivation_metadata.json")
        if int(derivation.get("actual_model_forwards", -1)) != 0:
            failures.append("fixed15: derived view performed model inference")

    if all(view in subjects_by_view for view in ("mincover4", "fixed15")):
        for subject_id, row in subjects_by_view["mincover4"].items():
            if _prediction(row) != _prediction(subjects_by_view["fixed15"][subject_id]):
                failures.append(f"fixed15/{subject_id}: prediction differs from mincover4")
        if all(view in metrics_by_view for view in ("mincover4", "fixed15")):
            for metric in METRICS:
                if not math.isclose(
                    _metric(metrics_by_view["mincover4"], metric),
                    _metric(metrics_by_view["fixed15"], metric),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    failures.append(f"fixed15: {metric} differs from mincover4")

    historical_dir = checkpoint_dir / "standalone_eval"
    historical_comparison: dict[str, Any] = {"available": historical_dir.is_dir()}
    if historical_dir.is_dir() and "fixed4" in subjects_by_view:
        try:
            fixed4_samples = read_jsonl(output_root / "fixed4" / "predictions_sample_level.jsonl")
            replay_samples = []
            for source in fixed4_samples:
                row = copy.deepcopy(source)
                row.pop("subject_score_aggregation", None)
                replay_samples.append(row)
            replay_subjects, replay_metrics, _, _ = aggregate_predictions(
                replay_samples,
                mode=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
                aggregation_level=AGGREGATION_LEVEL_SUBJECT,
            )
            replay_dir = output_root / "fixed4_historical_replay"
            _write_csv(replay_dir / "predictions_subject_level.csv", replay_subjects)
            save_json(replay_metrics, replay_dir / "metrics_original_teacher_forced.json")
            historical_metrics = read_json(historical_dir / "metrics_original_teacher_forced.json")
            historical_subjects = {
                str(row["subject_id"]): row
                for row in _read_csv(historical_dir / "predictions_subject_level.csv")
            }
            replay_by_subject = {str(row["subject_id"]): row for row in replay_subjects}
            mismatches = sorted(
                subject_id
                for subject_id, row in replay_by_subject.items()
                if subject_id not in historical_subjects
                or _prediction(row) != _prediction(historical_subjects[subject_id])
            )
            metric_deltas = {
                metric: _metric(replay_metrics, metric) - _metric(historical_metrics, metric)
                for metric in METRICS
            }
            mean_score_changed_subjects = sorted(
                subject_id
                for subject_id, row in subjects_by_view["fixed4"].items()
                if _prediction(row) != _prediction(replay_by_subject[subject_id])
            )
            historical_comparison.update(
                {
                    "replay_method": "teacher_forced_hard_label_from_cached_fixed4_samples",
                    "prediction_mismatches": mismatches,
                    "metric_deltas": metric_deltas,
                    "mean_score_changed_subjects": mean_score_changed_subjects,
                    "replay_metrics": {metric: _metric(replay_metrics, metric) for metric in METRICS},
                }
            )
            if mismatches or any(abs(value) > 1e-12 for value in metric_deltas.values()):
                failures.append("fixed4 historical replay did not reproduce the checkpoint's standalone evaluation")
        except Exception as exc:
            failures.append(f"fixed4 historical comparison failed: {exc}")

    comparison_rows: list[dict[str, Any]] = []
    if "fixed4" in metrics_by_view:
        for view in VIEWS:
            if view not in metrics_by_view:
                continue
            row: dict[str, Any] = {"view": view}
            for metric in METRICS:
                value = _metric(metrics_by_view[view], metric)
                row[metric] = value
                row[f"delta_{metric}_vs_fixed4"] = value - _metric(metrics_by_view["fixed4"], metric)
            row["confusion_matrix"] = json.dumps(
                metrics_by_view[view].get("binary_strict_confusion_matrix"), separators=(",", ":")
            )
            comparison_rows.append(row)
    _write_csv(output_root / "comparison.csv", comparison_rows)

    subject_changes: list[dict[str, Any]] = []
    if all(view in subjects_by_view for view in ("fixed4", "mincover4")):
        for subject_id in sorted(subjects_by_view["fixed4"]):
            base = subjects_by_view["fixed4"][subject_id]
            cover = subjects_by_view["mincover4"][subject_id]
            subject_changes.append(
                {
                    "subject_id": subject_id,
                    "label": int(base["label"]),
                    "fixed4_prediction": _prediction(base),
                    "mincover4_prediction": _prediction(cover),
                    "prediction_changed": _prediction(base) != _prediction(cover),
                    "fixed4_score_margin": base.get("score_margin", ""),
                    "mincover4_score_margin": cover.get("score_margin", ""),
                }
            )
    _write_csv(output_root / "subject_changes.csv", subject_changes)

    audit = {
        "schema_version": "daic_k4_coverage_audit.v1",
        "passed": not failures,
        "failures": failures,
        "expected_subjects": expected_subjects,
        "checkpoint_dir": str(checkpoint_dir),
        "modality": expected_modality,
        "allow_omitted_checkpoint_adapter": allow_omitted_checkpoint_adapter,
        "checkpoint_provenance_passed": not any(value.startswith("checkpoint") for value in failures),
        "coverage_by_subject": coverage_summary,
        "historical_fixed4_comparison": historical_comparison,
        "changed_subjects": sum(bool(row["prediction_changed"]) for row in subject_changes),
        "comparison": comparison_rows,
    }
    save_json(audit, output_root / "coverage_audit.json")
    lines = [
        f"# DAIC K=4 Coverage Audit Results — {expected_modality}",
        "",
        f"Audit: **{'PASS' if audit['passed'] else 'FAIL'}**",
        "",
        "| View | Accuracy | Positive F1 | Macro F1 | Precision | Recall | Confusion |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['view']} | {row['accuracy']:.3f} | {row['positive_f1']:.3f} | "
            f"{row['macro_f1']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"`{row['confusion_matrix']}` |"
        )
    lines.extend(
        [
            "",
            f"Subjects whose prediction changed from fixed4 to mincover4: **{audit['changed_subjects']}**.",
            "",
            "The coverage table uses mean teacher-forced score margins for both fixed4 and mincover4. "
            "Historical fixed4 reproduction is audited separately from the same cached sample outputs using the original hard-label aggregation.",
            "",
            "This is a retrospective coverage sensitivity analysis. It must not be used to select a checkpoint or the best-looking test protocol.",
        ]
    )
    if failures:
        lines.extend(["", "## Audit failures", ""] + [f"- {failure}" for failure in failures])
    (output_root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and audit complete K=4 DAIC coverage.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--expected-subjects", type=int, default=47)
    parser.add_argument("--allow-omitted-checkpoint-adapter", action="store_true")
    parser.add_argument("--expected-modality", choices=("audio_text", "audio_only"), default="audio_text")
    args = parser.parse_args()
    output_root = args.output_root / args.run_id
    if not args.report_only:
        if output_root.exists() and any(output_root.iterdir()) and not args.resume:
            raise SystemExit(f"Collision: output exists: {output_root}")
        runtime_config = materialize_runtime_config(
            args.config, output_root / "audit_runtime_config.yaml"
        )
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/evaluate_daic_comprehensive_views.py"),
            "--config", str(runtime_config),
            "--checkpoint-dir", str(args.checkpoint_dir),
            "--fold", str(args.fold),
            "--output-root", str(output_root),
            "--views", ",".join(VIEWS),
            "--overrides-json", "{}",
        ]
        if args.resume:
            command.append("--resume")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    audit = audit_and_report(
        output_root,
        args.checkpoint_dir,
        expected_subjects=args.expected_subjects,
        allow_omitted_checkpoint_adapter=args.allow_omitted_checkpoint_adapter,
        expected_modality=args.expected_modality,
    )
    if not audit["passed"]:
        raise SystemExit("Coverage audit failed: " + "; ".join(audit["failures"]))
    print(output_root / "results.md")


if __name__ == "__main__":
    main()
