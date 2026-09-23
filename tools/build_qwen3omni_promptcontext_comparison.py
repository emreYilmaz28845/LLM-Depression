#!/usr/bin/env python
"""Build the Qwen3-Omni standalone prompt-context comparison from verified local evidence.

One row per model x dataset x modality. Qwen3-Omni values come from the runs'
own local artifacts: D3TEC, Androids, CMDC and Turkish pooled t17 carry the
unweighted mean of their five verified folds (all five required), DAIC keeps its
official-test fold 0. The Qwen2-Audio references are the PR #255 derived
canonical likelihood values, whose recorded per-subject prediction files are
re-hashed here; a value that cannot be verified locally stays blank with its
reason instead of being inferred. Smoke runs are refused.

The Qwen3-Omni rows carry the promptcontext_v1 prompt while the Qwen2-Audio
references carry the older inline prompt, so a difference between them is a
model-plus-prompt difference and the table says so.

Outputs (deterministic, no timestamps):
  comparison.csv         one headline row per model x dataset x modality
  comparison_by_fold.csv one row per verified fold
  comparison.md          the short reader-facing report
  comparison.json        per-row provenance (paths, hashes, jobs, notes)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_yaml  # noqa: E402

PROJECT_ROOT = Path("/home/emre/Projects/AudioLLM/LLM-Depression")
DERIVED_REFERENCE = "outputs/experiment_reports/likelihood_canonical_values/derived_values.json"
DERIVED_REFERENCE_LABEL = (
    "PR #255 derived canonical likelihood evidence (argmax of mean dep/non candidate "
    "scores from the teacher-forced runs' saved per-subject scores)"
)
OMNI_MODEL = "Qwen3-Omni-30B-A3B Thinker"
QWEN2_MODEL = "Qwen2-Audio-7B-Instruct"

COLUMNS = [
    "model",
    "backend",
    "dataset",
    "modality",
    "endpoint",
    "folds_verified",
    "fold",
    "total_parameters",
    "active_parameters",
    "trainable_parameters",
    "gpu_shape",
    "activation_offload",
    "effective_batch",
    "selection_metric",
    "evaluation_view",
    "aggregation",
    "macro_f1",
    "positive_f1",
    "uar",
    "run_name",
    "attempt_id",
    "evidence_path",
    "jobs",
    "qualification",
    "notes",
]

# Declared five-fold campaign cells. ``run_dir_template`` is resolved with the
# fold and the run-name suffix; every fold must be present for a headline value.
FIVE_FOLD_CELLS = (
    ("d3tec", "D3TEC", "audio_only", "outer_fold_test_partition"),
    ("d3tec", "D3TEC", "audio_text", "outer_fold_test_partition"),
    ("androids_interview", "Androids Interview", "audio_only", "outer_fold_test_partition"),
    ("androids_interview", "Androids Interview", "audio_text", "outer_fold_test_partition"),
    ("cmdc", "CMDC", "audio_only", "selected_validation_view"),
    ("cmdc", "CMDC", "audio_text", "selected_validation_view"),
    ("turkish", "Turkish", "audio_only", "selected_validation_view"),
    ("turkish", "Turkish", "audio_text", "selected_validation_view"),
)

# DAIC was produced by PR #261 and is carried into the campaign comparison
# without being rerun; its run names do not carry the campaign suffix.
DAIC_CELLS = (
    ("daic", "DAIC", "audio_only", "official_test_partition"),
    ("daic", "DAIC", "audio_text", "official_test_partition"),
)

REFERENCE_MODALITY_LABELS = {"turkish": {"audio_only": "Audio only", "audio_text": "Audio + Text"}}


def _rows_for(suffix: str, daic_suffix: str) -> list[dict]:
    rows: list[dict] = []
    for dataset, reference_dataset, modality, endpoint in FIVE_FOLD_CELLS:
        reference_modality = REFERENCE_MODALITY_LABELS.get(dataset, {}).get(modality, modality)
        rows.append(
            {
                "kind": "derived_reference",
                "model": QWEN2_MODEL,
                "dataset": dataset,
                "modality": modality,
                "reference_dataset": reference_dataset,
                "reference_modality": reference_modality,
                "transcript_condition": "native" if dataset == "turkish" else None,
                "endpoint": endpoint,
                "project_root": str(PROJECT_ROOT),
            }
        )
        rows.append(
            {
                "kind": "run",
                "model": OMNI_MODEL,
                "dataset": dataset,
                "modality": modality,
                "endpoint": endpoint,
                "mode": "fold_mean",
                "run_dir_template": (
                    "output_model/promptcontext_v1_qwen3omni_likelihood/"
                    f"{modality}/{dataset}/qwen3omni_{dataset}_"
                    f"{modality}_f{{fold}}_{suffix}/fold_{{fold}}"
                ),
                "backend": "likelihood",
                "project_root": str(PROJECT_ROOT),
            }
        )
    for dataset, reference_dataset, modality, endpoint in DAIC_CELLS:
        rows.append(
            {
                "kind": "derived_reference",
                "model": QWEN2_MODEL,
                "dataset": dataset,
                "modality": modality,
                "reference_dataset": reference_dataset,
                "reference_modality": modality,
                "transcript_condition": None,
                "endpoint": endpoint,
                "project_root": str(PROJECT_ROOT),
            }
        )
        rows.append(
            {
                "kind": "run",
                "model": OMNI_MODEL,
                "dataset": dataset,
                "modality": modality,
                "endpoint": endpoint,
                "mode": "single_fold",
                "run_dir_template": (
                    "output_model/promptcontext_v1_qwen3omni_likelihood/"
                    f"{modality}/daic/qwen3omni_daic_{modality}_{daic_suffix}/fold_0"
                ),
                "backend": "likelihood",
                "project_root": str(PROJECT_ROOT),
            }
        )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uar_from_confusion(confusion: list[list[int]] | None) -> float | None:
    if not confusion or len(confusion) != 2:
        return None
    true_negative, false_positive = confusion[0][0], confusion[0][1]
    false_negative, true_positive = confusion[1][0], confusion[1][1]
    negatives = true_negative + false_positive
    positives = true_positive + false_negative
    if negatives == 0 or positives == 0:
        return None
    return round((true_positive / positives + true_negative / negatives) / 2.0, 6)


def _job_ids(run_dir: Path) -> str:
    jobs_path = run_dir / "jobs.jsonl"
    if not jobs_path.is_file():
        return ""
    identifiers: list[str] = []
    for line in jobs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        job_id = str(event.get("slurm_job_id") or "").strip()
        if job_id and job_id not in identifiers:
            identifiers.append(job_id)
    return ",".join(identifiers)


def _active_parameters_estimate(structure: dict) -> tuple[int | None, str]:
    """MoE active-parameter estimate: every expert's share scaled by the routing top-k.

    ``active = total - routed_expert_total + routed_expert_total * top_k / num_experts``
    with the routed expert total computed from the checkpoint's own declared
    dimensions, so the formula and its inputs are both recorded.
    """
    layers = structure.get("num_hidden_layers")
    experts = structure.get("num_experts")
    top_k = structure.get("num_experts_per_tok")
    intermediate = structure.get("moe_intermediate_size")
    hidden = structure.get("hidden_size")
    total = structure.get("total_parameters")
    if None in (layers, experts, top_k, intermediate, hidden, total):
        return None, "active-parameter estimate unavailable: checkpoint dimensions not recorded"
    expert_total = int(layers) * int(experts) * (3 * int(intermediate) * int(hidden))
    active = int(total) - expert_total + expert_total * int(top_k) / int(experts)
    formula = (
        f"active = total - routed_experts + routed_experts*top_k/num_experts = "
        f"{total} - {expert_total} + {expert_total}*{top_k}/{experts}"
    )
    return int(round(active)), formula


def _derived_row(declared: dict, project_root: Path) -> tuple[dict, dict]:
    row = {column: "" for column in COLUMNS}
    row["model"] = declared["model"]
    row["backend"] = "likelihood"
    row["dataset"] = declared["dataset"]
    row["modality"] = declared["modality"]
    row["endpoint"] = declared["endpoint"]
    notes: list[str] = []
    root = Path(declared.get("project_root") or project_root)
    reference_path = root / DERIVED_REFERENCE
    if not reference_path.is_file():
        row["notes"] = f"derived reference file missing: {DERIVED_REFERENCE}"
        return row, {"fold": "", "values": {}}
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    entry = None
    for candidate in payload.get("cells", []):
        if str(candidate.get("dataset")) != declared["reference_dataset"]:
            continue
        if str(candidate.get("modality")) != declared["reference_modality"]:
            continue
        if declared.get("transcript_condition") and str(
            candidate.get("transcript_condition") or ""
        ) != declared["transcript_condition"]:
            continue
        entry = candidate
        break
    if entry is None:
        row["notes"] = "no matching cell in the derived reference file"
        return row, {"fold": "", "values": {}}
    metrics = entry.get("likelihood") or {}
    for key, column in (("macro_f1", "macro_f1"), ("positive_f1", "positive_f1"), ("uar", "uar")):
        value = metrics.get(key)
        row[column] = "" if value is None else f"{float(value):.6f}"
    row["evaluation_view"] = "harmonized_all_windows_full_coverage"
    # The derived file records the endpoint (single test fold or fold mean); the
    # decision is subject-level, which is what a comparison against the Omni rows needs.
    row["aggregation"] = "subject_level"
    notes.append(f"endpoint: {entry.get('aggregation') or 'not recorded'}")
    row["selection_metric"] = "inner_val_macro_f1 (max)"
    row["qualification"] = DERIVED_REFERENCE_LABEL + "; older inline prompt (not promptcontext_v1)"
    evidence_files = []
    for recorded in entry.get("files") or []:
        recorded_path = Path(recorded.get("path", ""))
        local_path = Path(
            str(recorded_path).replace(
                "/home/emre/Projects/AudioLLM/worktrees/LLM-Depression-feat-canonical-likelihood/",
                str(root) + "/",
            )
        )
        if not local_path.is_file():
            notes.append(f"prediction artifact missing locally: {local_path}")
            continue
        local_sha = _sha256(local_path)
        if local_sha != recorded.get("sha256"):
            raise ValueError(
                f"derived reference hash mismatch for {local_path}: {local_sha} != {recorded.get('sha256')}"
            )
        evidence_files.append(str(local_path))
    row["evidence_path"] = ";".join(evidence_files)
    row["notes"] = "; ".join(notes) or "reference values verified against the sha256-recorded local artifacts"
    return row, {"fold": "", "values": {}}


def _fold_metrics(run_dir: Path, backend: str, declared: dict, root: Path) -> dict:
    """One fold's verified metrics and provenance, or the reason they are missing."""
    result: dict = {"run_dir": str(run_dir), "macro_f1": None, "positive_f1": None, "uar": None, "notes": []}
    notes = result["notes"]
    if not run_dir.is_dir():
        notes.append(f"run directory not present: {run_dir}")
        return result
    if "smoke" in run_dir.parent.name.lower() or run_dir.parent.name.endswith("_smoke"):
        raise ValueError(f"Refusing to include smoke run {run_dir.parent.name} in the comparison table.")
    run_config_path = run_dir / "run_config.yaml"
    if run_config_path.is_file():
        run_config = load_yaml(run_config_path)
        selection = run_config.get("selection_protocol", {}) or {}
        result["selection_metric"] = str(selection.get("metric_name") or "")
        result["aggregation"] = str((run_config.get("evaluation", {}) or {}).get("aggregation_level") or "")
        result["fold"] = str(run_config.get("fold", ""))
        tracking = run_config.get("tracking", {}) or {}
        result["attempt_id"] = str(tracking.get("attempt_id") or "")
        training = run_config.get("training_strategy", {}) or {}
        result["gpu_shape"] = (
            f"{training.get('world_size', '?')} ranks, per-rank batch "
            f"{training.get('per_device_train_batch_size', '?')}"
        )
        result["activation_offload"] = str(training.get("activation_offload") or "")
        result["effective_batch"] = str(training.get("effective_global_batch_size") or "")
        audit = run_config.get("model_load_audit", {}) or {}
        result["total_parameters"] = str(audit.get("total_parameters") or "")
        result["trainable_parameters"] = str(audit.get("lora_trainable_params") or "")
        active, formula = _active_parameters_estimate(dict(audit))
        result["active_parameters"] = "" if active is None else str(active)
        result["active_parameters_formula"] = formula if active is not None else ""
    else:
        notes.append("run_config.yaml missing")

    evaluations_path = run_dir / "evaluations.json"
    if evaluations_path.is_file():
        payload = json.loads(evaluations_path.read_text(encoding="utf-8"))
        for evaluation in payload.get("evaluations", []):
            if str(evaluation.get("backend")) != backend:
                continue
            result["evaluation_view"] = str(evaluation.get("evaluation_view") or "")
            result["aggregation"] = result.get("aggregation") or str(evaluation.get("aggregation") or "")
            if evaluation.get("locally_verified") is False:
                warnings = "; ".join(str(item) for item in evaluation.get("warnings") or [])
                notes.append(f"evaluation not locally verified: {warnings or 'no reason recorded'}")
            if evaluation.get("qualification"):
                result["qualification"] = str(evaluation["qualification"])
    else:
        notes.append("evaluations.json missing")

    metrics_path = run_dir / "best_model" / "standalone_eval" / f"metrics_{backend}.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        headline = metrics.get("headline_metrics", metrics) or {}
        macro_f1 = headline.get("binary_strict_macro_f1", headline.get("macro_f1"))
        positive_f1 = headline.get("binary_strict_positive_f1", headline.get("positive_f1"))
        result["macro_f1"] = None if macro_f1 is None else float(macro_f1)
        result["positive_f1"] = None if positive_f1 is None else float(positive_f1)
        uar = _uar_from_confusion(headline.get("binary_strict_confusion_matrix"))
        recorded_uar = headline.get("binary_strict_uar", headline.get("macro_recall"))
        if uar is None and recorded_uar is not None:
            uar = round(float(recorded_uar), 6)
            notes.append("uar taken from the recorded value (confusion matrix unavailable)")
        elif uar is not None and recorded_uar is not None and abs(uar - float(recorded_uar)) > 1e-6:
            notes.append(
                "uar mismatch: confusion-derived %s vs recorded %s"
                % (uar, round(float(recorded_uar), 6))
            )
        result["uar"] = uar
        try:
            result["evidence_path"] = str(metrics_path.relative_to(root))
        except ValueError:
            result["evidence_path"] = str(metrics_path)
    else:
        notes.append(f"metrics_{backend}.json missing")
    return result


def _mean_or_blank(values: list[float | None]) -> tuple[str, bool]:
    if any(value is None for value in values):
        return "", False
    return f"{sum(values) / len(values):.6f}", True


def _run_row(declared: dict, project_root: Path, suffix: str, daic_suffix: str) -> tuple[dict, list[dict]]:
    root = Path(declared.get("project_root") or project_root)
    folds = [0, 1, 2, 3, 4] if declared["mode"] == "fold_mean" else [0]
    template = (
        declared["run_dir_template"]
        .replace("{suffix}", suffix)
        .replace("{daic_suffix}", daic_suffix)
    )
    per_fold: list[dict] = []
    headline = {column: "" for column in COLUMNS}
    headline["model"] = declared["model"]
    headline["backend"] = declared["backend"]
    headline["dataset"] = declared["dataset"]
    headline["modality"] = declared["modality"]
    headline["endpoint"] = declared["endpoint"]
    notes: list[str] = []
    for fold in folds:
        run_dir = Path(template.format(fold=fold))
        if not run_dir.is_absolute():
            run_dir = root / run_dir
        result = _fold_metrics(run_dir, declared["backend"], declared, root)
        result["fold"] = str(fold)
        per_fold.append(result)
    present = [result for result in per_fold if result["macro_f1"] is not None]
    headline["folds_verified"] = f"{len(present)}/{len(per_fold)}"
    if declared["mode"] == "fold_mean":
        if len(present) != len(per_fold):
            missing = [result["fold"] for result in per_fold if result["macro_f1"] is None]
            notes.append(f"headline blank: folds without verified metrics {missing}")
            for result in per_fold:
                for note in result["notes"]:
                    if note and note not in notes:
                        notes.append(note)
            headline["notes"] = "; ".join(notes)
            headline["run_name"] = Path(template).parent.name
            return headline, per_fold
        for key, column in (("macro_f1", "macro_f1"), ("positive_f1", "positive_f1"), ("uar", "uar")):
            value, ok = _mean_or_blank([result[key] for result in present])
            headline[column] = value
            if not ok:
                notes.append(f"{column} blank: a verified fold has no recorded value")
        headline["qualification"] = (
            "promptcontext_v1 prompt; unweighted mean of the five verified folds; locally validated runs"
        )
    else:
        result = per_fold[0]
        for key in ("macro_f1", "positive_f1", "uar"):
            value = result[key]
            headline[key] = "" if value is None else f"{float(value):.6f}"
        headline["fold"] = "0"
        headline["qualification"] = result.get("qualification") or (
            "promptcontext_v1 prompt; locally validated run"
        )
    last = per_fold[-1]
    headline["selection_metric"] = last.get("selection_metric", "")
    headline["aggregation"] = last.get("aggregation", "")
    headline["evaluation_view"] = last.get("evaluation_view", "")
    headline["gpu_shape"] = last.get("gpu_shape", "")
    headline["activation_offload"] = last.get("activation_offload", "")
    headline["effective_batch"] = last.get("effective_batch", "")
    headline["total_parameters"] = last.get("total_parameters", "")
    headline["trainable_parameters"] = last.get("trainable_parameters", "")
    headline["active_parameters"] = last.get("active_parameters", "")
    headline["run_name"] = Path(template).parent.name
    headline["attempt_id"] = ";".join(
        sorted({result.get("attempt_id", "") for result in per_fold if result.get("attempt_id")})
    )
    headline["evidence_path"] = ";".join(
        sorted({result.get("evidence_path", "") for result in per_fold if result.get("evidence_path")})
    )
    headline["jobs"] = ",".join(
        sorted({job for result in per_fold for job in _job_ids(Path(result["run_dir"])).split(",") if job})
    )
    fold_note = "; ".join(
        f"fold {result['fold']}: {result['macro_f1']:.6f}" if result["macro_f1"] is not None
        else f"fold {result['fold']}: no verified metrics"
        for result in per_fold
    )
    if fold_note:
        notes.append(fold_note)
    formula = last.get("active_parameters_formula")
    if formula:
        notes.append(formula)
    for result in per_fold:
        for note in result["notes"]:
            if note and note not in notes:
                notes.append(note)
    headline["notes"] = "; ".join(notes)
    return headline, per_fold


def build(
    project_root: Path, suffix: str, daic_suffix: str
) -> tuple[list[dict], list[dict], dict]:
    headline_rows: list[dict] = []
    fold_rows: list[dict] = []
    for declared in _rows_for(suffix, daic_suffix):
        if declared["kind"] == "derived_reference":
            row, _ = _derived_row(declared, project_root)
            headline_rows.append(row)
            continue
        row, per_fold = _run_row(declared, project_root, suffix, daic_suffix)
        headline_rows.append(row)
        for result in per_fold:
            fold_row = {column: "" for column in COLUMNS}
            fold_row.update(
                {
                    "model": declared["model"],
                    "backend": declared["backend"],
                    "dataset": declared["dataset"],
                    "modality": declared["modality"],
                    "endpoint": declared["endpoint"],
                    "fold": result.get("fold", ""),
                    "run_name": Path(result["run_dir"]).parent.name,
                    "attempt_id": result.get("attempt_id", ""),
                    "evidence_path": result.get("evidence_path", ""),
                    "jobs": _job_ids(Path(result["run_dir"])),
                    "evaluation_view": result.get("evaluation_view", ""),
                    "aggregation": result.get("aggregation", ""),
                    "macro_f1": "" if result["macro_f1"] is None else f"{result['macro_f1']:.6f}",
                    "positive_f1": "" if result["positive_f1"] is None else f"{result['positive_f1']:.6f}",
                    "uar": "" if result["uar"] is None else f"{float(result['uar']):.6f}",
                    "folds_verified": "1/1" if result["macro_f1"] is not None else "0/1",
                    "notes": "; ".join(result["notes"]),
                }
            )
            fold_rows.append(fold_row)
    provenance = {
        "schema": "audiollm.qwen3omni_promptcontext_comparison.v1",
        "project_root": str(project_root),
        "run_suffix": suffix,
        "daic_run_suffix": daic_suffix,
        "reference": {
            "path": DERIVED_REFERENCE,
            "label": DERIVED_REFERENCE_LABEL,
            "note": (
                "the reference rows carry the older inline prompt, so a difference against the "
                "Qwen3-Omni rows is a model-plus-prompt difference, not a prompt-only effect"
            ),
        },
        "rows": headline_rows,
        "by_fold": fold_rows,
    }
    return headline_rows, fold_rows, provenance


def _markdown(rows: list[dict]) -> str:
    header = (
        "| Dataset | Modality | Model | Backend | View | Aggregation | Endpoint | Folds | Macro-F1 | "
        "Positive-F1 | UAR | Run / attempt | Qualification |"
    )
    separator = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, separator]
    for row in rows:
        run = row["run_name"] + (f" / {row['attempt_id']}" if row["attempt_id"] else "")
        lines.append(
            "| {dataset} | {modality} | {model} | {backend} | {view} | {aggregation} | {endpoint} | "
            "{folds} | {macro} | {positive} | {uar} | {run} | {qualification} |".format(
                dataset=row["dataset"],
                modality=row["modality"],
                model=row["model"],
                backend=row["backend"],
                view=row["evaluation_view"] or "not recorded",
                aggregation=row["aggregation"] or "",
                endpoint=row["endpoint"],
                folds=row["folds_verified"] or "",
                macro=row["macro_f1"] or "",
                positive=row["positive_f1"] or "",
                uar=row["uar"] or "",
                run=run,
                qualification=row["qualification"] or "",
            )
        )
    body = "\n".join(lines)
    return (
        "# Qwen3-Omni Thinker standalone prompt-context campaign: qualified comparison\n\n"
        "One row per model per dataset and modality. Qwen3-Omni values come from each production run's\n"
        "`best_model/standalone_eval` artifacts; D3TEC, Androids, CMDC and Turkish pooled t17 are\n"
        "unweighted means of the five verified folds, DAIC is its official-test fold 0. The Qwen2-Audio\n"
        "rows are the PR #255 derived canonical likelihood evidence, whose recorded per-subject artifacts\n"
        "are re-hashed here. Every pair shares its dataset recipe, split, seed, evaluation view and\n"
        "subject-level aggregation.\n\n"
        "The Qwen3-Omni rows use the `promptcontext_v1` prompt while the references use the older inline\n"
        "prompt, so the comparison is a model-plus-prompt comparison. Blanks mean the value is not recorded\n"
        "in local evidence. The table is descriptive: it carries no winner claim and no significance test.\n\n"
        f"{body}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output-dir", default="outputs/qwen3omni_promptcontext_comparison/report")
    parser.add_argument(
        "--run-suffix",
        default="prod_20260923",
        help="run-name suffix of the five-fold campaign runs",
    )
    parser.add_argument(
        "--daic-run-suffix",
        default="fold0_prod_20260923",
        help="run-name suffix of the PR #261 DAIC runs",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    rows, by_fold, provenance = build(project_root, args.run_suffix, args.daic_run_suffix)
    output_dir = (
        project_root / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    by_fold_path = output_dir / "comparison_by_fold.csv"
    with by_fold_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(by_fold)
    (output_dir / "comparison.md").write_text(_markdown(rows), encoding="utf-8")
    (output_dir / "comparison.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {csv_path}, {by_fold_path}, {output_dir / 'comparison.md'}, {output_dir / 'comparison.json'}")
    for row in rows:
        print(
            f"  {row['dataset']:<20} {row['modality']:<11} {row['model']:<30} "
            f"macro_f1={row['macro_f1'] or '-':<10} folds={row['folds_verified'] or '-'} "
            f"notes={row['notes'][:120]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
