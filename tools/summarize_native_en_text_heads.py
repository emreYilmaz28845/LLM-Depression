#!/usr/bin/env python3
"""Deterministic aggregator for the native-versus-English text-only study.

Consumes an explicit, validated study manifest (no metric values inside,
only identities and attempt paths), applies the locked Section 18
aggregation hierarchy, and writes machine-readable JSON plus human-readable
Markdown. Refuses missing, duplicate, non-reportable, or mismatched cells;
never invents values; contains no hand-typed metrics.

Aggregation contract (locked):
- standalone d3tec/androids_interview: pool five outer folds at subject
  level per seed (each subject must appear exactly once);
- standalone cmdc/turkish: unweighted mean of the five fold scores;
- merged CV: per fold the unweighted mean of the five dataset metrics,
  then the mean of those fold means per seed;
- final DAIC: the canonical full-coverage teacher-forced/head evaluation
  per seed as recorded;
- across seeds: arithmetic mean, sample standard deviation, paired
  English-minus-native deltas within matched seeds.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST_SCHEMA = "audiollm.native_en_study_manifest.v1"
REPORT_SCHEMA = "audiollm.native_en_study_report.v1"
STUDY_SEEDS = (7, 1337, 2024)
CONDITIONS = ("native", "english")
BACKBONES = ("qwen", "gemma4")
HEADS = ("logreg", "xgb_optuna100")
STANDALONE_DATASETS = ("d3tec", "androids_interview", "cmdc", "turkish")
ENDPOINTS = ("standalone", "merged_cv", "final_daic")
METRIC_NAMES = ("macro_f1", "positive_f1")
POOL_DATASETS = {"d3tec", "androids_interview"}


def _fail(message: str) -> None:
    raise ValueError(message)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        _fail(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _status_state(attempt_dir: Path) -> str:
    status = _read_json(attempt_dir / "status.json")
    state = str(status.get("state") or "")
    if state != "REPORTABLE":
        _fail(f"{attempt_dir}: state is {state!r}, expected REPORTABLE")
    return state


def _metric_values(evaluations_doc: dict[str, Any], dataset: str | None) -> dict[str, float]:
    matches = [
        entry
        for entry in evaluations_doc.get("evaluations", [])
        if dataset is None or str(entry.get("dataset")) == dataset
    ]
    if len(matches) != 1:
        _fail(f"expected exactly one evaluation record, found {len(matches)}")
    entry = matches[0]
    values: dict[str, float] = {}
    for metric in entry.get("metrics", []):
        name = str(metric.get("name"))
        if name in METRIC_NAMES:
            values[name] = float(metric.get("value"))
    missing = [name for name in METRIC_NAMES if name not in values]
    if missing:
        _fail(f"evaluation record lacks metrics: {missing}")
    return values


def _strict_metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        _fail("no prediction rows to pool")
    y_true = [int(row["label"]) for row in rows]
    y_pred = [int(row["prediction"]) for row in rows]
    from src.metrics import classification_metrics

    metrics = classification_metrics(y_true, y_pred)
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "positive_f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
    }


def _pooled_seed_metrics(fold_dirs: list[Path]) -> dict[str, float]:
    """Pool five outer folds at subject level; each subject exactly once."""

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for attempt_dir in fold_dirs:
        _status_state(attempt_dir)
        predictions_path = attempt_dir / "predictions_subject_level.jsonl"
        if not predictions_path.is_file():
            _fail(f"missing predictions for pooling: {predictions_path}")
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = f"{row.get('dataset')}::{row['subject_id']}"
            if key in seen:
                _fail(f"subject appears in more than one held-out fold while pooling: {key}")
            seen.add(key)
            rows.append(row)
    return _strict_metrics_from_rows(rows)


def _fold_metrics(attempt_dir: Path, *, dataset: str | None) -> dict[str, float]:
    """Stored evaluation metrics for one fold attempt."""

    _status_state(attempt_dir)
    evaluations = _read_json(attempt_dir / "evaluations.json")
    return _metric_values(evaluations, dataset)


def _merged_fold_mean(attempt_dir: Path) -> dict[str, float]:
    """Unweighted mean of the five per-dataset metrics inside one fold."""

    _status_state(attempt_dir)
    evaluations = _read_json(attempt_dir / "evaluations.json")
    expected_datasets = {"daic", "cmdc", "turkish", "d3tec", "androids_interview"}
    per_dataset: dict[str, dict[str, float]] = {}
    for entry in evaluations.get("evaluations", []):
        ds = str(entry.get("dataset"))
        values = {
            str(metric.get("name")): metric.get("value")
            for metric in entry.get("metrics", [])
        }
        clean = {m: float(values[m]) for m in METRIC_NAMES if values.get(m) is not None}
        if len(clean) != len(METRIC_NAMES):
            continue
        per_dataset[ds] = clean
    if set(per_dataset) != expected_datasets:
        _fail(
            f"{attempt_dir}: merged fold datasets {sorted(per_dataset)} != "
            f"{sorted(expected_datasets)}"
        )
    return {
        name: sum(per_dataset[d][name] for d in sorted(expected_datasets)) / len(expected_datasets)
        for name in METRIC_NAMES
    }


def load_manifest(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        doc = _read_json(path)
    else:
        import yaml

        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if str(doc.get("schema_version")) != MANIFEST_SCHEMA:
        _fail(f"manifest schema_version must be {MANIFEST_SCHEMA}")
    group_id = str(doc.get("group_id") or "")
    if not group_id:
        _fail("manifest group_id missing")
    cells = doc.get("cells") or []
    if not cells:
        _fail("manifest has no cells")
    return doc


def _validate_cell_identity(cell: dict[str, Any]) -> tuple[str, ...]:
    endpoint = str(cell.get("endpoint"))
    condition = str(cell.get("condition"))
    backbone = str(cell.get("backbone"))
    head = str(cell.get("head"))
    dataset = str(cell.get("dataset") or "")
    if endpoint not in ENDPOINTS:
        _fail(f"unknown endpoint {endpoint!r}")
    if condition not in CONDITIONS:
        _fail(f"unknown condition {condition!r}")
    if backbone not in BACKBONES:
        _fail(f"unknown backbone {backbone!r}")
    if head not in HEADS:
        _fail(f"unknown head {head!r}")
    if endpoint == "standalone":
        if dataset not in STANDALONE_DATASETS:
            _fail(f"standalone endpoint requires dataset in {STANDALONE_DATASETS}")
        declared_aggregation = str(cell.get("aggregation"))
        expected = (
            "pooled_subject_level"
            if dataset in POOL_DATASETS
            else "fold_mean_unweighted"
        )
        if declared_aggregation != expected:
            _fail(
                f"{endpoint}/{dataset}: aggregation must be {expected!r}, got "
                f"{declared_aggregation!r}"
            )
    else:
        if dataset not in {"", "merged", "daic"}:
            _fail(f"{endpoint}: unexpected dataset {dataset!r}")
        if endpoint == "merged_cv" and str(cell.get("aggregation")) != "mean_dataset_fold_mean_unweighted":
            _fail("merged_cv aggregation must be mean_dataset_fold_mean_unweighted")
        if endpoint == "final_daic" and str(cell.get("aggregation")) != "subject_level":
            _fail("final_daic aggregation must be subject_level")
    seeds = cell.get("seeds") or []
    present = sorted(int(item.get("seed")) for item in seeds)
    if present != sorted(STUDY_SEEDS):
        _fail(f"cell {endpoint}/{dataset}/{condition}/{backbone}/{head} must list seeds {list(STUDY_SEEDS)}, got {present}")
    return (endpoint, dataset, condition, backbone, head)


def summarize(manifest_path: Path) -> dict[str, Any]:
    doc = load_manifest(manifest_path)
    comparisons: dict[tuple[str, ...], dict[str, Any]] = {}
    seen_ids: set[tuple[str, ...]] = set()
    evidence: list[dict[str, Any]] = []
    for cell in doc["cells"]:
        identity = _validate_cell_identity(cell)
        if identity in seen_ids:
            _fail(f"duplicate cell identity: {identity}")
        seen_ids.add(identity)
        endpoint, dataset, condition, backbone, head = identity
        pooled = endpoint == "standalone" and dataset in POOL_DATASETS
        seed_scores: dict[int, dict[str, float]] = {}
        for seed_entry in cell["seeds"]:
            seed = int(seed_entry["seed"])
            folds = seed_entry.get("folds") or []
            if not folds:
                _fail(f"seed {seed} of {identity} lists no folds")
            if endpoint in {"standalone", "merged_cv"} and len(folds) != 5:
                _fail(f"{identity} seed {seed}: expected five folds, got {len(folds)}")
            if endpoint == "final_daic" and len(folds) != 1:
                _fail(f"final_daic seed {seed}: exactly one fold entry required")
            fold_dirs = [Path(str(fe["attempt_dir"])) for fe in folds]
            for fold_entry, attempt_dir in zip(folds, fold_dirs):
                evidence.append(
                    {
                        "endpoint": endpoint,
                        "dataset": dataset,
                        "condition": condition,
                        "backbone": backbone,
                        "head": head,
                        "seed": seed,
                        "fold": int(fold_entry.get("fold", -1)),
                        "attempt_dir": str(attempt_dir),
                    }
                )
            if pooled:
                # Locked rule: pool the five folds at subject level.
                seed_scores[seed] = _pooled_seed_metrics(fold_dirs)
            else:
                if endpoint == "standalone":
                    fold_metrics = [_fold_metrics(d, dataset=None) for d in fold_dirs]
                elif endpoint == "merged_cv":
                    fold_metrics = [_merged_fold_mean(d) for d in fold_dirs]
                else:
                    fold_metrics = [_fold_metrics(fold_dirs[0], dataset="daic")]
                seed_scores[seed] = {
                    name: float(sum(m[name] for m in fold_metrics) / len(fold_metrics))
                    for name in METRIC_NAMES
                }

        key = (endpoint, dataset, backbone, head)

        def _stats(values: list[float]) -> dict[str, float]:
            return {
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            }

        entry = comparisons.setdefault(key, {"endpoint": endpoint, "dataset": dataset, "backbone": backbone, "head": head})
        entry[condition] = {
            name: {
                **_stats([seed_scores[s][name] for s in STUDY_SEEDS]),
                "per_seed": {str(s): seed_scores[s][name] for s in STUDY_SEEDS},
            }
            for name in METRIC_NAMES
        }

    # Paired deltas require both conditions present.
    incomplete = []
    for key, entry in comparisons.items():
        if "native" not in entry or "english" not in entry:
            incomplete.append(key)
    if incomplete:
        _fail(f"comparisons missing a condition (paired deltas impossible): {incomplete}")

    for key, entry in comparisons.items():
        for name in METRIC_NAMES:
            deltas = [
                entry["english"][name]["per_seed"][str(s)]
                - entry["native"][name]["per_seed"][str(s)]
                for s in STUDY_SEEDS
            ]
            entry.setdefault("paired_delta", {})[name] = {
                **_stats(deltas),
                "per_seed": {str(s): d for s, d in zip(STUDY_SEEDS, deltas)},
            }

    return {
        "schema_version": REPORT_SCHEMA,
        "group_id": doc["group_id"],
        "seeds": list(STUDY_SEEDS),
        "comparisons": [
            {
                "endpoint": k[0],
                "dataset": k[1],
                "backbone": k[2],
                "head": k[3],
                **v,
            }
            for k, v in sorted(comparisons.items())
        ],
        "evidence_index": evidence,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Native vs English — text-only head study",
        "",
        f"group_id: `{report['group_id']}` · seeds: {report['seeds']}",
        "",
        "| endpoint | dataset | backbone | head | metric | native mean ± sd | english mean ± sd | Δ (EN − native) mean ± sd |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for comp in report["comparisons"]:
        for name in METRIC_NAMES:
            n = comp["native"][name]
            e = comp["english"][name]
            d = comp["paired_delta"][name]
            lines.append(
                f"| {comp['endpoint']} | {comp['dataset'] or '—'} | {comp['backbone']} | {comp['head']} "
                f"| {name} | {n['mean']:.3f} ± {n['sample_sd']:.3f} "
                f"| {e['mean']:.3f} ± {e['sample_sd']:.3f} "
                f"| {d['mean']:+.3f} ± {d['sample_sd']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = summarize(args.manifest)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"comparisons": len(report["comparisons"]), "evidence": len(report["evidence_index"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
