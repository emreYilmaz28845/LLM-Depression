"""Paired significance tests for the deck's within-corpus comparisons.

Reads a frozen retrospective comparison-family file (default:
``experiments/definitions/significance_family.yaml``), resolves both sides of
every comparison to subject-level predictions, and runs:

* paired prediction-swap permutation test (primary p-value),
* stratified paired bootstrap (delta and 95% CI),
* exact McNemar test (binary disagreement view),

applying Holm correction inside every family block. The tool is read-only with
respect to evidence: it writes only its own report under the output directory.

Sides are resolved from symbolic references so the family file stays portable:

* ``record``: a cell of the presentation evidence (``records`` section);
* ``merged_cv`` / ``merged_final``: symmetric-merged campaign artifacts;
* ``joint_k4``: a record of the joint-K evidence file.

CV sides are pooled out-of-fold: every per-fold ``predictions_subject_level``
file is concatenated and each subject must appear exactly once. The reported
delta therefore describes the pooled subject view; the decks' headline cells are
unweighted fold means, which is a different (and stated) aggregation.

``--mcnemar-table`` exports the exact McNemar view on its own, with no
family-wise correction: one test per comparison and seed. Multi-seed sides are
tested once per seed, because several seeds do not define one final subject
correctness decision. The table is written for the slide-format comparison the
regression script used; it never replaces the corrected report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import classification_metrics  # noqa: E402
from src.daic_statistics import (  # noqa: E402
    exact_mcnemar,
    exact_paired_prediction_swap,
    holm_adjust,
    paired_prediction_swap_permutation,
    paired_prediction_swap_permutation_many,
    stratified_paired_bootstrap,
    stratified_paired_bootstrap_many,
)

DEFAULT_FAMILY = PROJECT_ROOT / "experiments/definitions/significance_family.yaml"
PRIMARY_METRICS = ("macro_f1", "positive_f1", "macro_recall")  # macro_recall is UAR
METRIC_LABELS = {"macro_f1": "Macro-F1", "macro_recall": "UAR", "positive_f1": "Positive-F1"}


class SignificanceError(RuntimeError):
    """Raised when the family or its evidence cannot be resolved."""


def expand_generated_families(family: dict[str, Any], native_en_report: dict[str, Any] | None) -> None:
    """Expand compact, reviewable matrix declarations into comparison rows."""
    for declaration in family.get("generated_families", []):
        kind = declaration["kind"]
        comparisons: list[dict[str, Any]] = []
        if kind == "backbone_same_route":
            for dataset in declaration["datasets"]:
                for modality in declaration["modalities"]:
                    for route in declaration["routes"]:
                        comparisons.append({
                            "id": f"{dataset}|{modality}|{route}|Qwen vs Gemma 4",
                            "dataset": dataset,
                            "description": "Same route and input; backbone is the only displayed factor changed.",
                            "baseline": {"kind": "record", "dataset": dataset, "modality": modality,
                                         "condition": declaration.get("condition", "native"), "model": "qwen", "route": route},
                            "comparison": {"kind": "record", "dataset": dataset, "modality": modality,
                                           "condition": declaration.get("condition", "native"), "model": "gemma4", "route": route},
                        })
        elif kind == "native_en_hidden_heads":
            if native_en_report is None:
                raise SignificanceError("generated native/English hidden-head family requires --native-en-head-report")
            for row in native_en_report["summary"]:
                # The deck's aggregate "five datasets" row mixes different
                # subjects and corpora, so it is coverage-only, not pairable.
                if row["dataset"] == "merged":
                    continue
                base = {"kind": "native_en_head", "endpoint": row["endpoint"], "dataset": row["dataset"],
                        "model": row["backbone"], "head": row["head"]}
                comparisons.append({
                    "id": f"{row['endpoint']}|{row['dataset']}|{row['backbone']}|{row['head']}|native vs English",
                    "dataset": row["dataset"],
                    "description": "Three-seed hidden-head native versus English comparison.",
                    "baseline": {**base, "condition": "native"},
                    "comparison": {**base, "condition": "english"},
                })
        else:
            raise SignificanceError(f"unknown generated family kind: {kind!r}")
        family["families"].append({
            "id": declaration["id"], "description": declaration.get("description"),
            "comparisons": comparisons,
        })


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path, dataset: str | None = None) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise SignificanceError(f"unsupported predictions file: {path}")
    if dataset is not None and rows and "dataset" in rows[0]:
        rows = [row for row in rows if row.get("dataset") == dataset]
    out = []
    for row in rows:
        out.append(
            {
                "subject_id": str(row["subject_id"]),
                "label": int(row["label"]),
                "prediction": int(row["prediction"]),
                "seed": int(row.get("seed", 0) or 0),
            }
        )
    return out


def pool(entries: list[tuple]) -> list[dict[str, Any]]:
    """Pool per-fold predictions; each subject/seed key must appear once."""
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in entries:
        path, dataset = entry[0], entry[1]
        forced_seed = int(entry[2]) if len(entry) > 2 else None
        rows = read_rows(path, dataset)
        if not rows:
            raise SignificanceError(f"no subject rows in {path} for dataset {dataset!r}")
        for row in rows:
            if forced_seed is not None:
                row["seed"] = forced_seed
            key = (row["subject_id"], row["seed"])
            if key in merged:
                raise SignificanceError(f"subject/seed {key!r} appears in more than one fold file")
            merged[key] = row
    if not merged:
        raise SignificanceError("no subject predictions found")
    return list(merged.values())


def _campaign(root: Path) -> Path:
    for candidate in sorted(root.glob("*")):
        if "preflight" not in candidate.name and "smoke" not in candidate.name and (candidate / "cv").exists():
            return candidate
    raise SignificanceError(f"no merged campaign under {root}")


def resolve_merged_cv(model: str, route: str, modality: str, dataset: str) -> list[tuple[Path, str | None]]:
    root = (PROJECT_ROOT / "outputs/symmetric_merged/harmonized_v1" if model == "qwen"
            else PROJECT_ROOT / "outputs/symmetric_merged/gemma4/harmonized_v1") / modality
    campaign = _campaign(root)
    entries: list[tuple[Path, str | None]] = []
    for fold in sorted((campaign / "cv").glob("fold_*")):
        if route == "teacher_forced":
            candidate = fold / model / dataset / "predictions_subject_level.csv"
            entries.append((candidate, None))
        else:
            candidate = fold / "heads" / route / "predictions_subject_level.csv"
            entries.append((candidate, dataset))
        if not candidate.is_file():
            raise SignificanceError(f"missing merged CV predictions: {candidate}")
    return entries


def resolve_merged_final(model: str, route: str, modality: str) -> list[tuple[Path, str | None]]:
    root = (PROJECT_ROOT / "outputs/symmetric_merged/harmonized_v1" if model == "qwen"
            else PROJECT_ROOT / "outputs/symmetric_merged/gemma4/harmonized_v1") / modality
    campaign = _campaign(root)
    fold = campaign / "final" / "fold_0"
    if route == "teacher_forced":
        candidate, dataset = fold / model / "daic" / "predictions_subject_level.csv", None
    else:
        candidate, dataset = fold / "heads" / route / "predictions_subject_level.csv", "daic"
    if not candidate.is_file():
        raise SignificanceError(f"missing merged final predictions: {candidate}")
    return [(candidate, dataset)]


def resolve_record(evidence: dict[str, Any], ref: dict[str, Any]) -> list[tuple[Path, str | None]]:
    match = next(
        (row for row in evidence["records"]
         if (row["dataset"], row["modality"], row["condition"], row["model"], row["route"])
         == (ref["dataset"], ref["modality"], ref.get("condition", "native"), ref["model"], ref["route"])),
        None,
    )
    if match is None:
        raise SignificanceError(f"record not found for {ref!r}")
    entries: list[tuple[Path, str | None]] = []
    for item in match["fold_evidence"]:
        metrics_path = Path(item.get("metrics_path") or item["evaluation_metrics_artifact"]["path"])
        candidate = metrics_path.parent / "predictions_subject_level.csv"
        if not candidate.is_file():
            candidate = metrics_path.parent / "predictions_subject_level.jsonl"
        if not candidate.is_file():
            raise SignificanceError(f"missing predictions beside {metrics_path}")
        entries.append((candidate, None))
    return entries


def resolve_joint(joint: dict[str, Any], ref: dict[str, Any]) -> list[tuple[Path, str | None]]:
    row = next((item for item in joint["records"]
                if item["key"] == ref["key"] and item["modality"] == ref["modality"]
                and item["route"] == ref["route"]), None)
    if row is None:
        raise SignificanceError(f"joint-K record not found for {ref!r}")
    path = Path(row["predictions_path"])
    candidates = [path, path.parent / "predictions_subject_level.csv", path.parent / "predictions_subject_level.jsonl"]
    for candidate in candidates:
        if candidate.is_file():
            return [(candidate, None)]
    raise SignificanceError(f"missing joint-K predictions for {ref!r}")


def resolve_side(spec: dict[str, Any], evidence: dict[str, Any],
                 joint: dict[str, Any] | None) -> list[tuple[Path, str | None]]:
    kind = spec.get("kind")
    if kind == "record":
        return resolve_record(evidence, spec)
    if kind == "merged_cv":
        return resolve_merged_cv(spec["model"], spec["route"], spec.get("modality", "audio_text"), spec["dataset"])
    if kind == "merged_final":
        return resolve_merged_final(spec["model"], spec["route"], spec.get("modality", "audio_text"))
    if kind == "joint_k4":
        if joint is None:
            raise SignificanceError("joint-K evidence file is required for this family")
        return resolve_joint(joint, spec)
    if kind == "path":
        path = Path(spec["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise SignificanceError(f"missing predictions file {path}")
        return [(path, spec.get("dataset"))]
    if kind == "native_en_head":
        report = evidence.get("_native_en_head_report")
        if report is None:
            raise SignificanceError("native/English hidden-head report is required")
        condition = spec["condition"]
        key = (spec["endpoint"], spec["dataset"], spec["model"], spec["head"])
        matches = [item for item in report["seed_details"]
                   if (item["endpoint"], item["dataset"], item["backbone"], item["head"]) == key]
        entries = []
        provenance_key = f"{condition}_provenance"
        for item in matches:
            for fold in item[provenance_key]:
                for artifact in fold.get("metrics_artifacts", []):
                    path = Path(artifact["prediction_path"])
                    if not path.is_file():
                        raise SignificanceError(f"missing hidden-head predictions: {path}")
                    entries.append((path, spec.get("dataset"), int(item["seed"])))
        if not entries:
            raise SignificanceError(f"no hidden-head predictions for {spec!r}")
        return entries
    raise SignificanceError(f"unknown side kind: {kind!r}")


def side_provenance(spec: dict[str, Any], evidence: dict[str, Any], joint: dict[str, Any] | None) -> dict[str, Any]:
    """Return the available source-evidence chain without inventing missing fields."""
    kind = spec.get("kind")
    if kind == "record":
        row = next(
            (item for item in evidence["records"]
             if (item["dataset"], item["modality"], item["condition"], item["model"], item["route"])
             == (spec["dataset"], spec["modality"], spec.get("condition", "native"), spec["model"], spec["route"])),
            None,
        )
        if row is None:
            return {"kind": kind, "status": "missing"}
        return {
            "kind": kind,
            "status": "locally_verified" if row.get("fold_evidence") else "legacy_incomplete",
            "cell": row.get("cell"),
            "aggregation": row.get("aggregation"),
            "fold_evidence": row.get("fold_evidence", []),
        }
    if kind in {"merged_cv", "merged_final"}:
        endpoint = "cv" if kind == "merged_cv" else "final"
        matches = [item for item in evidence.get("merged_comparisons", [])
                   if item.get("dataset") == spec.get("dataset") and item.get("model") == spec.get("model")
                   and item.get("route") == spec.get("route")]
        return {
            "kind": kind, "endpoint": endpoint,
            "status": "locally_verified" if matches else "legacy_incomplete",
            "evidence": matches,
        }
    if kind == "joint_k4" and joint is not None:
        row = next((item for item in joint.get("records", [])
                    if item.get("key") == spec.get("key") and item.get("modality") == spec.get("modality")
                    and item.get("route") == spec.get("route")), None)
        return {"kind": kind, "status": "locally_verified" if row else "missing", "evidence": row}
    if kind == "native_en_head":
        report = evidence.get("_native_en_head_report") or {}
        key = (spec["endpoint"], spec["dataset"], spec["model"], spec["head"])
        matches = [item for item in report.get("seed_details", [])
                   if (item["endpoint"], item["dataset"], item["backbone"], item["head"]) == key]
        return {"kind": kind, "status": "reportable_local_evidence" if matches else "missing",
                "condition": spec["condition"], "seed_evidence": matches}
    return {"kind": kind, "status": "file_hash_only"}


def normalize_subjects(rows: list[dict[str, Any]], dataset: str | None) -> list[dict[str, Any]]:
    """Strip a leading ``<dataset>::`` namespace from subject ids.

    Merged campaign artifacts namespace subject ids by dataset while standalone
    runs use the bare id; both name the same people within one corpus.
    """
    out: dict[tuple[str, int], dict[str, Any]] = {}
    prefix = f"{dataset}::" if dataset else None
    for row in rows:
        subject_id = row["subject_id"]
        if prefix and subject_id.startswith(prefix):
            subject_id = subject_id[len(prefix):]
        key = (subject_id, int(row.get("seed", 0)))
        if key in out:
            raise SignificanceError(f"duplicate subject/seed after normalization: {key!r}")
        out[key] = {**row, "subject_id": subject_id}
    return list(out.values())


def correction_family_id(block_id: str, comparison_id: str) -> str:
    """Map one comparison to the user-defined local scientific question."""
    parts = comparison_id.split("|")
    if block_id == "model_qwen_vs_gemma4_teacher_forced":
        return f"F1|backbone|dataset={parts[0]}|condition=native|route=teacher-forced"
    if block_id == "daic_official_development_backbone":
        return "F1|backbone|dataset=DAIC-official-development|condition=native|route=teacher-forced"
    if block_id == "model_qwen_vs_gemma4_hidden_routes":
        return f"F1|backbone|dataset={parts[0]}|condition=native|route={parts[2]}"
    if block_id == "route_pairs_native":
        return f"F3|route|dataset={parts[0]}|modality={parts[1]}|backbone={parts[2]}|condition=native"
    if block_id == "native_vs_english_transcript":
        return f"F2|translation|dataset={parts[0]}|backbone={parts[2]}|route={parts[3]}"
    if block_id == "standalone_vs_merged_audio_text":
        return f"F4|training-regime|dataset={parts[0]}|modality=A+T|backbone={parts[1]}|route={parts[2]}"
    if block_id == "joint_k4_v1_vs_runtime":
        return f"F5|recipe|dataset=DAIC|modality={parts[1]}|backbone=Qwen|method={parts[2]}"
    if block_id == "native_vs_english_hidden_heads_three_seed":
        return (f"F2|translation|dataset={parts[1]}|backbone={parts[2]}|route={parts[3]}"
                f"|endpoint={parts[0]}|modality=T")
    return block_id


def run_family(family: dict[str, Any], evidence: dict[str, Any], joint: dict[str, Any] | None,
               *, iterations: int, bootstrap_iterations: int, seed: int,
               metrics: list[str]) -> dict[str, Any]:
    blocks_out: list[dict[str, Any]] = []

    for block in family["families"]:
        rows_out: list[dict[str, Any]] = []
        for comparison in block["comparisons"]:
            left_files = resolve_side(comparison["baseline"], evidence, joint)
            right_files = resolve_side(comparison["comparison"], evidence, joint)
            left_rows = normalize_subjects(pool(left_files), comparison.get("dataset"))
            right_rows = normalize_subjects(pool(right_files), comparison.get("dataset"))
            left_keys = {(row["subject_id"], row.get("seed", 0)) for row in left_rows}
            right_keys = {(row["subject_id"], row.get("seed", 0)) for row in right_rows}
            if left_keys != right_keys:
                raise SignificanceError(
                    f"{comparison['id']}: subject sets differ ({len(left_keys)} vs {len(right_keys)})"
                )
            entry: dict[str, Any] = {
                "id": comparison["id"],
                "description": comparison.get("description"),
                "dataset": comparison.get("dataset"),
                "n_subjects": len({key[0] for key in left_keys}),
                "n_seeds": len({key[1] for key in left_keys}),
                "correction_family": correction_family_id(block["id"], comparison["id"]),
                "baseline_files": [str(item[0].resolve()) for item in left_files],
                "comparison_files": [str(item[0].resolve()) for item in right_files],
                "baseline_file_sha256": {str(item[0].resolve()): sha256_file(item[0]) for item in left_files},
                "comparison_file_sha256": {str(item[0].resolve()): sha256_file(item[0]) for item in right_files},
                "baseline_provenance": side_provenance(comparison["baseline"], evidence, joint),
                "comparison_provenance": side_provenance(comparison["comparison"], evidence, joint),
                "metrics": {},
            }
            if entry["n_seeds"] == 1:
                entry["mcnemar"] = {**exact_mcnemar(left_rows, right_rows), "status": "tested"}
                permutation = exact_paired_prediction_swap(left_rows, right_rows, metrics=metrics)
            else:
                entry["mcnemar"] = {
                    "status": "not_identifiable",
                    "reason": "multiple seeds do not define one final subject correctness decision",
                }
                permutation = paired_prediction_swap_permutation_many(
                    left_rows, right_rows, metrics=metrics, iterations=iterations, seed=seed
                )
            bootstrap = stratified_paired_bootstrap_many(
                left_rows, right_rows, metrics=metrics, iterations=bootstrap_iterations, seed=seed
            )
            for metric in metrics:
                entry["metrics"][metric] = {
                    "permutation": permutation[metric],
                    "bootstrap": bootstrap[metric],
                }
            rows_out.append(entry)
        blocks_out.append({
            "id": block["id"],
            "description": block.get("description"),
            "comparisons": rows_out,
        })

    # Conservative sensitivity: all comparison x metric p-values share one
    # Holm family inside each broad block.
    for block in blocks_out:
        joint_refs = [
            (row, metric)
            for row in block["comparisons"]
            for metric in metrics
        ]
        joint_adjusted = holm_adjust([
            row["metrics"][metric]["permutation"]["p_value"]
            for row, metric in joint_refs
        ])
        for (row, metric), value in zip(joint_refs, joint_adjusted):
            row["metrics"][metric]["permutation"]["p_value_holm_joint_block"] = value
        for metric in metrics:
            p_values = [row["metrics"][metric]["permutation"]["p_value"] for row in block["comparisons"]]
            adjusted = holm_adjust(p_values)
            for row, value in zip(block["comparisons"], adjusted):
                row["metrics"][metric]["permutation"]["p_value_holm_metric_block"] = value
        mcnemar_rows = [row for row in block["comparisons"] if row["mcnemar"]["status"] == "tested"]
        mcnemar_p = [row["mcnemar"]["p_value"] for row in mcnemar_rows]
        for row, value in zip(mcnemar_rows, holm_adjust(mcnemar_p)):
            row["mcnemar"]["p_value_holm_block"] = value

    # Each metric has its own Holm correction inside the same local scientific
    # family. Macro-F1 alone drives the primary decision; the other two metrics
    # are secondary and never share its correction.
    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks_out:
        for row in block["comparisons"]:
            family_groups[row["correction_family"]].append(row)
    for family_id, rows in family_groups.items():
        for metric in metrics:
            adjusted = holm_adjust([row["metrics"][metric]["permutation"]["p_value"] for row in rows])
            for row, value in zip(rows, adjusted):
                permutation = row["metrics"][metric]["permutation"]
                permutation["p_value_holm_family"] = value
                if metric == "macro_f1":
                    permutation["p_value_holm_primary_family"] = value
                    permutation["primary_significant"] = value <= family.get("alpha", 0.05)
        tested = [row for row in rows if row["mcnemar"]["status"] == "tested"]
        adjusted_mc = holm_adjust([row["mcnemar"]["p_value"] for row in tested])
        for row, value in zip(tested, adjusted_mc):
            row["mcnemar"]["p_value_holm_primary_family"] = value
            row["mcnemar"]["primary_significant"] = value <= family.get("alpha", 0.05)

    metric_refs = [
        (row, metric)
        for block in blocks_out for row in block["comparisons"] for metric in metrics
    ]
    for (row, metric), value in zip(metric_refs, holm_adjust([
        row["metrics"][metric]["permutation"]["p_value"] for row, metric in metric_refs
    ])):
        row["metrics"][metric]["permutation"]["p_value_holm_global"] = value
    mcnemar_rows = [
        row for block in blocks_out for row in block["comparisons"]
        if row["mcnemar"]["status"] == "tested"
    ]
    for row, value in zip(mcnemar_rows, holm_adjust([row["mcnemar"]["p_value"] for row in mcnemar_rows])):
        row["mcnemar"]["p_value_holm_global"] = value
    return {"blocks": blocks_out}


def format_markdown(payload: dict[str, Any], family: dict[str, Any]) -> str:
    lines = [
        "# Paired significance report",
        "",
        f"Family file: `{payload['family_path']}` (sha256 `{payload['family_sha256'][:16]}…`)",
        f"Evidence: `{payload['evidence_path']}` (sha256 `{payload['evidence_sha256'][:16]}…`)",
        "Retrospective exploratory analysis; the results were inspected before this final analysis specification.",
        f"Single-seed comparisons use exact paired swaps. Multi-seed comparisons use up to "
        f"{payload['iterations']} subject-clustered permutations, seed {payload['seed']}.",
        f"Bootstrap: {payload['bootstrap_iterations']} subject-clustered, label-stratified resamples.",
        "Primary decision: Macro-F1 with Holm correction inside each pre-specified scientific contrast family.",
        "Positive-F1 and UAR are supporting effect-size metrics, each corrected separately with the same family membership.",
        "Broad joint-block Holm remains a conservative sensitivity view.",
        "McNemar is a separate correctness view corrected inside the same contrast families.",
        "Displayed deltas are exact observed differences; intervals are unadjusted bootstrap 95% CIs.",
        "Pooled out-of-fold subject predictions; the decks' headline cells are unweighted fold means.",
        "",
    ]
    for block in payload["results"]["blocks"]:
        lines.append(f"## {block['id']}")
        if block.get("description"):
            lines.append(block["description"])
        lines += ["", "| Comparison | Family | n | Metric | Observed Δ [unadjusted 95% CI] | p | primary Holm | joint-block sensitivity | McNemar b/c | McNemar p | McNemar family Holm |",
                  "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in block["comparisons"]:
            mc = row["mcnemar"]
            mc_pair = (f"{mc['baseline_only_correct']}/{mc['comparison_only_correct']}"
                       if mc["status"] == "tested" else "not identifiable")
            mc_p = f"{mc['p_value']:.6g}" if mc["status"] == "tested" else "—"
            mc_holm = f"{mc['p_value_holm_primary_family']:.6g}" if mc["status"] == "tested" else "—"
            for metric in payload["metrics"]:
                result = row["metrics"][metric]
                perm, boot = result["permutation"], result["bootstrap"]
                lines.append(
                    f"| {row['id']} | {row['correction_family']} | {row['n_subjects']} | {METRIC_LABELS[metric]} | "
                    f"{perm['observed_delta']:+.4f} [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] | "
                    f"{perm['p_value']:.6g} | {perm.get('p_value_holm_primary_family', '—')} | "
                    f"{perm['p_value_holm_joint_block']:.6g} | {mc_pair} | {mc_p} | {mc_holm} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def flat_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """One display row represents one comparison × metric result."""
    rows: list[dict[str, Any]] = []
    alpha = float(payload["alpha"])
    for block in payload["results"]["blocks"]:
        for comparison in block["comparisons"]:
            mc = comparison["mcnemar"]
            for metric in payload["metrics"]:
                result = comparison["metrics"][metric]
                perm, boot = result["permutation"], result["bootstrap"]
                rows.append({
                    "block": block["id"],
                    "comparison_id": comparison["id"],
                    "correction_family": comparison["correction_family"],
                    "dataset": comparison.get("dataset"),
                    "subjects": comparison["n_subjects"],
                    "seeds": comparison["n_seeds"],
                    "metric": metric,
                    "observed_delta": perm["observed_delta"],
                    "bootstrap_ci_low_unadjusted": boot["ci_low"],
                    "bootstrap_ci_high_unadjusted": boot["ci_high"],
                    "permutation_method": perm["method"],
                    "permutation_p": perm["p_value"],
                    "permutation_holm_family": perm["p_value_holm_family"],
                    "permutation_holm_primary_family": perm.get("p_value_holm_primary_family"),
                    "permutation_holm_joint_block": perm["p_value_holm_joint_block"],
                    "permutation_holm_metric_block": perm["p_value_holm_metric_block"],
                    "permutation_holm_global": perm["p_value_holm_global"],
                    "metric_primary_significant": (
                        perm.get("primary_significant") if metric == "macro_f1" else None
                    ),
                    "mcnemar_status": mc["status"],
                    "mcnemar_baseline_only_correct": mc.get("baseline_only_correct"),
                    "mcnemar_comparison_only_correct": mc.get("comparison_only_correct"),
                    "mcnemar_p": mc.get("p_value"),
                    "mcnemar_holm_block": mc.get("p_value_holm_block"),
                    "mcnemar_holm_primary_family": mc.get("p_value_holm_primary_family"),
                    "mcnemar_holm_global": mc.get("p_value_holm_global"),
                    "mcnemar_primary_significant": (
                        mc.get("primary_significant") if mc["status"] == "tested" else None
                    ),
                    "baseline_files": " | ".join(comparison["baseline_files"]),
                    "comparison_files": " | ".join(comparison["comparison_files"]),
                })
    return rows


def _rows_of_seed(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return [row for row in rows if int(row.get("seed", 0)) == seed]


def mcnemar_rows(
    family: dict[str, Any], evidence: dict[str, Any], joint: dict[str, Any] | None, *, alpha: float
) -> list[dict[str, Any]]:
    """Uncorrected exact McNemar rows, one per comparison and seed.

    Multi-seed sides are tested once per seed: the seeds stay separate because
    they do not define one final subject correctness decision.
    """
    rows: list[dict[str, Any]] = []
    for block in family["families"]:
        for comparison in block["comparisons"]:
            left = normalize_subjects(
                pool(resolve_side(comparison["baseline"], evidence, joint)), comparison.get("dataset")
            )
            right = normalize_subjects(
                pool(resolve_side(comparison["comparison"], evidence, joint)), comparison.get("dataset")
            )
            left_seeds = sorted({int(row.get("seed", 0)) for row in left})
            right_seeds = sorted({int(row.get("seed", 0)) for row in right})
            if left_seeds != right_seeds:
                raise SignificanceError(
                    f"{comparison['id']}: seed sets differ ({left_seeds} vs {right_seeds})"
                )
            for seed in left_seeds:
                result = exact_mcnemar(_rows_of_seed(left, seed), _rows_of_seed(right, seed))
                rows.append({
                    "block": block["id"],
                    "comparison_id": comparison["id"],
                    "correction_family": correction_family_id(block["id"], comparison["id"]),
                    "dataset": comparison.get("dataset"),
                    "seed": seed,
                    "seeds_in_comparison": len(left_seeds),
                    "uncorrected_significant": result["p_value"] < alpha,
                    **result,
                })
    return rows


def write_table(rows: list[dict[str, Any]], csv_path: Path, metadata: dict[str, Any]) -> Path:
    """Write an export CSV plus its provenance sidecar; return the sidecar path."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["comparison_id"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metadata_path = csv_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def metric_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Uncorrected comparison × metric rows read from a written report payload.

    Every Holm column is dropped: the per-comparison p-value is the one the
    permutation test produced, with no family correction applied.
    """
    alpha = float(payload.get("alpha", 0.05))
    rows: list[dict[str, Any]] = []
    for block in payload["results"]["blocks"]:
        for comparison in block["comparisons"]:
            mcnemar = comparison.get("mcnemar", {})
            for metric in payload.get("metrics", PRIMARY_METRICS):
                result = comparison["metrics"][metric]
                permutation = result.get("permutation", {})
                bootstrap = result.get("bootstrap", {})
                p_value = permutation.get("p_value")
                rows.append({
                    "block": block["id"],
                    "comparison_id": comparison["id"],
                    "correction_family": comparison.get("correction_family"),
                    "dataset": comparison.get("dataset"),
                    "subjects": comparison.get("n_subjects"),
                    "seeds": comparison.get("n_seeds"),
                    "metric": metric,
                    "observed_delta": permutation.get("observed_delta"),
                    "bootstrap_ci_low": bootstrap.get("ci_low"),
                    "bootstrap_ci_high": bootstrap.get("ci_high"),
                    "permutation_method": permutation.get("method"),
                    "p_value": p_value,
                    "uncorrected_significant": p_value is not None and float(p_value) < alpha,
                    "mcnemar_p_value": mcnemar.get("p_value"),
                })
    return rows


def native_en_seed_map(report: dict[str, Any]) -> dict[str, int]:
    """Map every recorded hidden-head prediction file to the seed it was run under.

    Multi-seed sides keep their seed in the file path, not in the rows, so the
    export has to rebuild the same per-file seed the original resolver passed.
    """
    mapping: dict[str, int] = {}
    for item in report.get("seed_details", []):
        seed = int(item["seed"])
        for key, value in item.items():
            if not key.endswith("_provenance") or not isinstance(value, list):
                continue
            for fold in value:
                for artifact in fold.get("metrics_artifacts", []):
                    mapping[str(Path(artifact["prediction_path"]))] = seed
    if not mapping:
        raise SignificanceError("the native/English hidden-head report lists no prediction files")
    return mapping


def load_verified_side(
    files: list[str], hashes: dict[str, str], dataset: str | None, context: str,
    seed_map: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Load one side of a comparison from the files the report recorded.

    Every file is re-hashed against the report so a re-derived score cannot come
    from evidence that changed after the report was written. Files listed in
    ``seed_map`` keep their recorded seed, which multi-seed sides need because
    their rows do not carry one. An output that is not 0/1 counts as wrong for
    its true class, the strict convention used everywhere else in the repository.
    """
    if not files:
        raise SignificanceError(f"{context}: the report records no prediction files")
    entries = []
    for path_string in files:
        path = Path(path_string)
        if not path.is_file():
            raise SignificanceError(f"{context}: missing prediction file {path}")
        expected = hashes.get(path_string)
        if expected and sha256_file(path) != expected:
            raise SignificanceError(f"{context}: {path} changed since the report was written")
        seed = (seed_map or {}).get(path_string)
        entries.append((path, dataset) if seed is None else (path, dataset, seed))
    rows = normalize_subjects(pool(entries), dataset)
    for row in rows:
        row["invalid_output"] = int(row["prediction"]) not in (0, 1)
        if row["invalid_output"]:
            row["prediction"] = 1 - int(row["label"])
    return rows


def seed_metric_scores(rows: list[dict[str, Any]], metric: str) -> dict[int, float]:
    """One metric value per seed, the same per-seed view the report's delta averages."""
    scores: dict[int, float] = {}
    for seed in sorted({int(row.get("seed", 0)) for row in rows}):
        subset = _rows_of_seed(rows, seed)
        scores[seed] = float(classification_metrics(
            [int(row["label"]) for row in subset], [int(row["prediction"]) for row in subset]
        )[metric])
    return scores


def report_export_tables(
    payload: dict[str, Any], seed_map: dict[str, int] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Build the results table and the paired subject dump from one report payload.

    Absolute scores are re-derived from the prediction files the report lists, so
    ``delta`` is recomputed rather than copied. Every comparison is checked against
    the report's own ``observed_delta`` and any mismatch is returned as a note.
    """
    alpha = float(payload.get("alpha", 0.05))
    results: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    notes: list[str] = []
    for block in payload["results"]["blocks"]:
        for comparison in block["comparisons"]:
            context = comparison["id"]
            left = load_verified_side(
                comparison.get("baseline_files", []), comparison.get("baseline_file_sha256", {}),
                comparison.get("dataset"), f"{context} baseline", seed_map,
            )
            right = load_verified_side(
                comparison.get("comparison_files", []), comparison.get("comparison_file_sha256", {}),
                comparison.get("dataset"), f"{context} comparison", seed_map,
            )
            left_rows = {(row["subject_id"], int(row.get("seed", 0))): row for row in left}
            right_rows = {(row["subject_id"], int(row.get("seed", 0))): row for row in right}
            if set(left_rows) != set(right_rows):
                raise SignificanceError(f"{context}: subject and seed sets differ between the sides")
            for metric in payload.get("metrics", PRIMARY_METRICS):
                left_scores = seed_metric_scores(left, metric)
                right_scores = seed_metric_scores(right, metric)
                score_baseline = sum(left_scores.values()) / len(left_scores)
                score_comparison = sum(right_scores.values()) / len(right_scores)
                delta = score_comparison - score_baseline
                permutation = comparison["metrics"][metric].get("permutation", {})
                bootstrap = comparison["metrics"][metric].get("bootstrap", {})
                observed = permutation.get("observed_delta")
                matched = observed is not None and abs(delta - float(observed)) <= 1e-9
                if not matched:
                    notes.append(f"{context} / {metric}: recomputed delta {delta:.10f} vs report {observed}")
                holm_family = permutation.get("p_value_holm_family")
                results.append({
                    "block": block["id"],
                    "comparison_id": comparison["id"],
                    "correction_family": comparison.get("correction_family"),
                    "dataset": comparison.get("dataset"),
                    "subjects": comparison.get("n_subjects"),
                    "seeds": comparison.get("n_seeds"),
                    "metric": metric,
                    "score_baseline": score_baseline,
                    "score_comparison": score_comparison,
                    "delta": delta,
                    "delta_report": observed,
                    "delta_check": "matched" if matched else "mismatch",
                    "bootstrap_ci_low": bootstrap.get("ci_low"),
                    "bootstrap_ci_high": bootstrap.get("ci_high"),
                    "raw_p": permutation.get("p_value"),
                    "holm_family_p": holm_family,
                    "holm_primary_family_p": permutation.get("p_value_holm_primary_family"),
                    "adjusted_significant": holm_family is not None and float(holm_family) < alpha,
                })
            for key in sorted(left_rows):
                subject_id, seed = key
                baseline_row, comparison_row = left_rows[key], right_rows[key]
                paired.append({
                    "block": block["id"],
                    "comparison_id": comparison["id"],
                    "dataset": comparison.get("dataset"),
                    "seed": seed,
                    "subject_id": subject_id,
                    "label": int(baseline_row["label"]),
                    "baseline_prediction": int(baseline_row["prediction"]),
                    "comparison_prediction": int(comparison_row["prediction"]),
                    "baseline_correct": int(int(baseline_row["prediction"]) == int(baseline_row["label"])),
                    "comparison_correct": int(int(comparison_row["prediction"]) == int(comparison_row["label"])),
                    "baseline_invalid": int(bool(baseline_row.get("invalid_output"))),
                    "comparison_invalid": int(bool(comparison_row.get("invalid_output"))),
                })
    return results, paired, notes


def coverage_family_from_report(payload: dict[str, Any], family: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the coverage inventory from a report plus the family's excluded list."""
    return {
        "families": [
            {"id": block["id"], "comparisons": [{"id": comparison["id"]} for comparison in block["comparisons"]]}
            for block in payload["results"]["blocks"]
        ],
        "excluded": family.get("excluded", []),
    }


def family_audit_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """One row represents one family member × metric/test correction."""
    rows: list[dict[str, Any]] = []
    for block in payload["results"]["blocks"]:
        for comparison in block["comparisons"]:
            family_id = comparison["correction_family"]
            family_size = sum(
                other["correction_family"] == family_id
                for candidate_block in payload["results"]["blocks"]
                for other in candidate_block["comparisons"]
            )
            for metric in payload["metrics"]:
                permutation = comparison["metrics"][metric]["permutation"]
                rows.append({
                    "family_id": family_id,
                    "family_size": family_size,
                    "member": comparison["id"],
                    "test": "paired_prediction_swap",
                    "metric": metric,
                    "role": "primary" if metric == "macro_f1" else "secondary",
                    "raw_p": permutation["p_value"],
                    "holm_p": permutation["p_value_holm_family"],
                })
            mcnemar = comparison["mcnemar"]
            rows.append({
                "family_id": family_id,
                "family_size": family_size,
                "member": comparison["id"],
                "test": "exact_mcnemar",
                "metric": "correctness",
                "role": "separate",
                "raw_p": mcnemar.get("p_value"),
                "holm_p": mcnemar.get("p_value_holm_primary_family"),
            })
    return sorted(rows, key=lambda row: (row["family_id"], row["test"], row["metric"], row["member"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise SignificanceError("cannot write an empty significance table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def coverage_rows(family: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {"block": block["id"], "comparison_id": comparison["id"], "status": "tested", "reason": ""}
        for block in family["families"] for comparison in block["comparisons"]
    ]
    rows.extend(
        {"block": "excluded", "comparison_id": f"excluded-{index + 1}", "status": "not_testable", "reason": reason}
        for index, reason in enumerate(family.get("excluded", []))
    )
    ids = [row["comparison_id"] for row in rows if row["status"] == "tested"]
    if len(ids) != len(set(ids)):
        raise SignificanceError("comparison inventory contains duplicate tested IDs")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--evidence", type=Path, required=True,
                        help="three_route_evidence.json (presentation evidence)")
    parser.add_argument("--joint-evidence", type=Path, default=None,
                        help="joint_k4_evidence.json; required only for joint-K blocks")
    parser.add_argument("--native-en-head-report", type=Path, default=None,
                        help="native_en_report.json with seed/fold hidden-head prediction provenance")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/significance")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and validate every pair without running the tests")
    parser.add_argument("--mcnemar-table", type=Path, default=None,
                        help="write the uncorrected exact-McNemar table (CSV plus a JSON provenance sidecar at "
                             "the same stem) instead of running the permutation and bootstrap tests")
    parser.add_argument("--metric-table", type=Path, default=None,
                        help="write the uncorrected comparison x metric table (CSV plus sidecar) from an existing "
                             "report; requires --from-report")
    parser.add_argument("--from-report", type=Path, default=None,
                        help="existing significance_report.json to read instead of re-running the tests")
    parser.add_argument("--report-export", type=Path, default=None,
                        help="write the corrected results table, the full/audit/coverage tables and the paired "
                             "subject dump from an existing report; requires --from-report")
    args = parser.parse_args()

    if not args.family.is_file():
        raise SignificanceError(f"family file not found: {args.family}")
    family = yaml.safe_load(args.family.read_text(encoding="utf-8"))
    if family.get("schema_version") != "audiollm.significance_family.v1":
        raise SignificanceError("unsupported family schema version")
    if args.metric_table is not None:
        if args.from_report is None:
            raise SignificanceError("--metric-table reads an existing report; pass --from-report")
        payload = json.loads(args.from_report.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "audiollm.significance_report.v1":
            raise SignificanceError("unsupported report schema version")
        if payload.get("family_sha256") != sha256_file(args.family):
            raise SignificanceError(
                "report was produced from a different family file; rebuild the report instead of relabelling it"
            )
        alpha = float(payload.get("alpha", family.get("alpha", 0.05)))
        rows = metric_rows(payload)
        metadata = {
            "schema_version": "audiollm.metric_table.v1",
            "source_report_path": str(args.from_report),
            "source_report_sha256": sha256_file(args.from_report),
            "family_path": str(args.family),
            "family_sha256": payload["family_sha256"],
            "evidence_path": payload.get("evidence_path"),
            "evidence_sha256": payload.get("evidence_sha256"),
            "alpha": alpha,
            "analysis_status": "retrospective_exploratory",
            "correction": "none",
            "correction_note": "Uncorrected permutation p-values; no family-wise correction is applied. Read them "
                               "against the number of tests, never as a per-row verdict.",
            "aggregation": "pooled out-of-fold subject-level predictions; the deck headline cells are unweighted "
                           "fold means, which is a different aggregation",
            "metrics": payload.get("metrics"),
            "comparisons": len({row["comparison_id"] for row in rows}),
            "tests": len(rows),
            "expected_false_positives_at_alpha": round(len(rows) * alpha, 2),
        }
        metadata_path = write_table(rows, args.metric_table, metadata)
        significant = sum(1 for row in rows if row["uncorrected_significant"])
        print(f"metric table: {len(rows)} tests over {metadata['comparisons']} comparisons "
              f"({', '.join(payload.get('metrics', []))})")
        print(f"uncorrected p<{alpha}: {significant} | expected by chance: "
              f"{metadata['expected_false_positives_at_alpha']}")
        print(f"wrote {args.metric_table} and {metadata_path}")
        return 0
    if args.report_export is not None:
        if args.from_report is None:
            raise SignificanceError("--report-export reads an existing report; pass --from-report")
        payload = json.loads(args.from_report.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "audiollm.significance_report.v1":
            raise SignificanceError("unsupported report schema version")
        if payload.get("family_sha256") != sha256_file(args.family):
            raise SignificanceError(
                "report was produced from a different family file; rebuild the report instead of relabelling it"
            )
        alpha = float(payload.get("alpha", family.get("alpha", 0.05)))
        multi_seed = any(
            (comparison.get("n_seeds") or 1) > 1
            for block in payload["results"]["blocks"] for comparison in block["comparisons"]
        )
        seed_map = None
        if multi_seed:
            if args.native_en_head_report is None:
                raise SignificanceError(
                    "--report-export needs --native-en-head-report: multi-seed hidden-head sides keep their seed "
                    "in the file path, and the report does not record it"
                )
            seed_map = native_en_seed_map(json.loads(args.native_en_head_report.read_text(encoding="utf-8")))
        out_dir = args.report_export
        out_dir.mkdir(parents=True, exist_ok=True)
        results, paired, notes = report_export_tables(payload, seed_map)
        coverage = coverage_family_from_report(payload, family)
        write_csv(out_dir / "significance_full.csv", flat_rows(payload))
        write_csv(out_dir / "family_audit.csv", family_audit_rows(payload))
        write_csv(out_dir / "coverage_report.csv", coverage_rows(coverage))
        write_csv(out_dir / "results_table.csv", results)
        write_csv(out_dir / "paired_subjects.csv", paired)
        comparisons = len({row["comparison_id"] for row in results})
        metadata = {
            "schema_version": "audiollm.report_export.v1",
            "source_report_path": str(args.from_report),
            "source_report_sha256": sha256_file(args.from_report),
            "family_path": str(args.family),
            "family_sha256": payload["family_sha256"],
            "evidence_path": payload.get("evidence_path"),
            "evidence_sha256": payload.get("evidence_sha256"),
            "alpha": alpha,
            "analysis_status": "retrospective_exploratory",
            "metrics": payload.get("metrics"),
            "primary_metric": payload.get("primary_metric"),
            "comparisons": comparisons,
            "metric_tests": len(results),
            "paired_rows": len(paired),
            "delta_checks": {"matched": len(results) - len(notes), "mismatch": len(notes)},
            "delta_check_notes": notes[:20],
            "score_aggregation": "one metric value per seed over that seed's pooled subject rows, averaged across "
                                 "seeds; this is the aggregation behind the report's observed_delta",
            "headline_aggregation": "the deck and workbook headline cells are unweighted fold means, which is a "
                                    "different aggregation",
            "thresholds": "no decision threshold is tuned on the test set: the LLM backends decide by candidate-label "
                          "argmax or likelihood margin sign, hidden heads use a fixed 0.5 probability threshold",
            "correction": "Holm within each pre-specified scientific contrast family, per metric; exact McNemar is "
                          "corrected separately within the same families",
        }
        (out_dir / "report_export.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"report export: {comparisons} comparisons, {len(results)} metric rows, {len(paired)} paired subject rows")
        print(f"delta checks: {metadata['delta_checks']['matched']} matched, {metadata['delta_checks']['mismatch']} "
              f"mismatched against the report's observed_delta")
        print(f"wrote 5 tables and report_export.json under {out_dir}")
        if notes:
            for note in notes[:5]:
                print(f"  mismatch: {note}")
            return 1
        return 0
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    joint = json.loads(args.joint_evidence.read_text(encoding="utf-8")) if args.joint_evidence else None
    native_en_report = (json.loads(args.native_en_head_report.read_text(encoding="utf-8"))
                        if args.native_en_head_report else None)
    if native_en_report is not None:
        evidence["_native_en_head_report"] = native_en_report
    expand_generated_families(family, native_en_report)

    metrics = list(family.get("metrics", PRIMARY_METRICS))
    alpha = float(family.get("alpha", 0.05))
    if args.mcnemar_table is not None:
        rows = mcnemar_rows(family, evidence, joint, alpha=alpha)
        metadata = {
            "schema_version": "audiollm.mcnemar_table.v1",
            "family_path": str(args.family),
            "family_sha256": sha256_file(args.family),
            "evidence_path": str(args.evidence),
            "evidence_sha256": sha256_file(args.evidence),
            "native_en_head_report_path": str(args.native_en_head_report) if args.native_en_head_report else None,
            "native_en_head_report_sha256": (
                sha256_file(args.native_en_head_report) if args.native_en_head_report else None
            ),
            "alpha": alpha,
            "analysis_status": "retrospective_exploratory",
            "correction": "none",
            "correction_note": "Uncorrected exact McNemar p-values over the whole frozen family; exploratory only. "
                               "The family-wise Holm report stays in the --output-dir payload.",
            "multi_seed_policy": "one exact McNemar per seed; several seeds do not define one final subject "
                                 "correctness decision",
            "test": "exact McNemar, two-sided, on paired subject hard predictions (correctness view, not macro-F1)",
            "comparisons": len({row["comparison_id"] for row in rows}),
            "p_values": len(rows),
            "expected_false_positives_at_alpha": round(len(rows) * alpha, 2),
        }
        metadata_path = write_table(rows, args.mcnemar_table, metadata)
        significant = sum(1 for row in rows if row["uncorrected_significant"])
        per_seed = sum(1 for row in rows if row["seeds_in_comparison"] > 1)
        print(f"mcnemar table: {len(rows)} tests over {metadata['comparisons']} comparisons "
              f"({per_seed} from multi-seed comparisons)")
        print(f"uncorrected p<{alpha}: {significant} | expected by chance: "
              f"{metadata['expected_false_positives_at_alpha']}")
        print(f"wrote {args.mcnemar_table} and {metadata_path}")
        return 0
    if args.dry_run:
        checked = 0
        for block in family["families"]:
            for comparison in block["comparisons"]:
                left = resolve_side(comparison["baseline"], evidence, joint)
                right = resolve_side(comparison["comparison"], evidence, joint)
                left_rows = normalize_subjects(pool(left), comparison.get("dataset"))
                right_rows = normalize_subjects(pool(right), comparison.get("dataset"))
                if {(row["subject_id"], row.get("seed", 0)) for row in left_rows} != {
                    (row["subject_id"], row.get("seed", 0)) for row in right_rows
                }:
                    raise SignificanceError(f"{comparison['id']}: subject sets differ")
                checked += 1
                print(f"ok {comparison['id']}: {len({row['subject_id'] for row in left_rows})} subjects, "
                      f"{len(left)}x{len(right)} files")
        print(f"dry run: {checked} comparisons resolvable")
        return 0

    bootstrap_iterations = args.bootstrap_iterations or int(family.get("bootstrap_iterations", args.iterations))
    results = run_family(
        family, evidence, joint, iterations=args.iterations,
        bootstrap_iterations=bootstrap_iterations, seed=args.seed, metrics=metrics,
    )
    payload = {
        "schema_version": "audiollm.significance_report.v1",
        "family_path": str(args.family),
        "family_sha256": sha256_file(args.family),
        "evidence_path": str(args.evidence),
        "evidence_sha256": sha256_file(args.evidence),
        "native_en_head_report_path": str(args.native_en_head_report) if args.native_en_head_report else None,
        "native_en_head_report_sha256": sha256_file(args.native_en_head_report) if args.native_en_head_report else None,
        "iterations": args.iterations,
        "bootstrap_iterations": bootstrap_iterations,
        "seed": args.seed,
        "alpha": float(family.get("alpha", 0.05)),
        "analysis_status": "retrospective_exploratory",
        "metrics": metrics,
        "primary_metric": "macro_f1",
        "supporting_metrics": ["positive_f1", "macro_recall"],
        "primary_metric_field": "macro_recall is UAR",
        "deltas": "permutation.observed_delta is the exact observed difference; bootstrap.mean_delta is the mean "
                  "of the resampled differences (slightly biased for nonlinear metrics) and the CI is percentile",
        "correction": {
            "metric_primary": "Macro-F1 Holm within each pre-specified scientific contrast family",
            "metric_secondary": "separate Positive-F1 and UAR Holm corrections using the same family membership",
            "mcnemar_primary": "McNemar Holm within each pre-specified scientific contrast family",
            "sensitivity": ["joint comparison x metric Holm within broad block", "metric-specific Holm within broad block", "global Holm"],
        },
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "significance_report.json"
    md_path = args.output_dir / "significance_report.md"
    csv_path = args.output_dir / "significance_full.csv"
    family_audit_path = args.output_dir / "family_audit.csv"
    coverage_path = args.output_dir / "coverage_report.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(format_markdown(payload, family), encoding="utf-8")
    write_csv(csv_path, flat_rows(payload))
    audit_rows = family_audit_rows(payload)
    write_csv(family_audit_path, audit_rows)
    write_csv(coverage_path, coverage_rows(family))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {family_audit_path}")
    print(f"wrote {coverage_path}")
    for block in results["blocks"]:
        significant = [
            row["id"] for row in block["comparisons"]
            if row["metrics"]["macro_f1"]["permutation"].get("primary_significant", False)
        ]
        print(f"{block['id']}: {len(block['comparisons'])} comparisons, "
              f"{len(significant)} significant after Holm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
