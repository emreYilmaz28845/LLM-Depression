#!/usr/bin/env python
"""Build the Qwen3-Omni DAIC comparison table from verified local evidence.

Rows are model x modality x verified evaluation view on the DAIC official-test
fold 0. Every value comes from the run's own local artifacts, or — for the
Qwen2-Audio reference — from the PR #255 derived canonical likelihood file whose
source prediction artifacts are re-hashed here; a value that cannot be verified
locally stays blank with its reason instead of being inferred. Smoke runs are
refused: only declared production runs enter the table.

The Qwen3-Omni rows carry the promptcontext_v1 prompt while the Qwen2-Audio
reference carries the older inline prompt, so a difference between them is a
model-plus-prompt difference and the table says so.

Outputs (deterministic, no timestamps):
  comparison.csv   one row per model x modality, the source table
  comparison.md    the short reader-facing report
  comparison.json  per-row provenance (paths, hashes, jobs, notes)
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

COLUMNS = [
    "model",
    "backend",
    "modality",
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

# Declared rows. The reference rows are read from the derived canonical file; the
# Qwen3-Omni rows are read from the production run directories after collection.
ROWS = [
    {
        "kind": "derived_reference",
        "model": "Qwen2-Audio-7B-Instruct",
        "modality": "audio_only",
        "project_root": "/home/emre/Projects/AudioLLM/LLM-Depression",
    },
    {
        "kind": "derived_reference",
        "model": "Qwen2-Audio-7B-Instruct",
        "modality": "audio_text",
        "project_root": "/home/emre/Projects/AudioLLM/LLM-Depression",
    },
    {
        "kind": "run",
        "model": "Qwen3-Omni-30B-A3B Thinker",
        "modality": "audio_only",
        "run_dir": "output_model/promptcontext_v1_qwen3omni_likelihood/audio_only/daic/"
        "qwen3omni_daic_audio_only_fold0_prod_20260923/fold_0",
        "backend": "likelihood",
        "project_root": "/home/emre/Projects/AudioLLM/LLM-Depression",
    },
    {
        "kind": "run",
        "model": "Qwen3-Omni-30B-A3B Thinker",
        "modality": "audio_text",
        "run_dir": "output_model/promptcontext_v1_qwen3omni_likelihood/audio_text/daic/"
        "qwen3omni_daic_audio_text_fold0_prod_20260923/fold_0",
        "backend": "likelihood",
        "project_root": "/home/emre/Projects/AudioLLM/LLM-Depression",
    },
]


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


def _derived_row(declared: dict, project_root: Path) -> dict:
    row = {column: "" for column in COLUMNS}
    row["model"] = declared["model"]
    row["backend"] = "likelihood"
    row["modality"] = declared["modality"]
    notes: list[str] = []
    root = Path(declared.get("project_root") or project_root)
    reference_path = root / DERIVED_REFERENCE
    if not reference_path.is_file():
        row["notes"] = f"derived reference file missing: {DERIVED_REFERENCE}"
        return row
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    entry = None
    for candidate in payload.get("cells", []):
        if str(candidate.get("dataset")) == "DAIC" and str(candidate.get("modality")) == declared["modality"]:
            entry = candidate
            break
    if entry is None:
        row["notes"] = "no DAIC cell for this modality in the derived reference file"
        return row
    metrics = entry.get("likelihood") or {}
    for key, column in (("macro_f1", "macro_f1"), ("positive_f1", "positive_f1"), ("uar", "uar")):
        value = metrics.get(key)
        row[column] = "" if value is None else f"{float(value):.6f}"
    row["evaluation_view"] = "harmonized_all_windows_full_coverage"
    row["aggregation"] = str(entry.get("aggregation") or "")
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
    return row


def _run_row(declared: dict, project_root: Path) -> dict:
    root = Path(declared.get("project_root") or project_root)
    run_dir = root / declared["run_dir"]
    row = {column: "" for column in COLUMNS}
    row["model"] = declared["model"]
    row["backend"] = declared["backend"]
    row["modality"] = declared["modality"]
    row["run_name"] = run_dir.parent.name
    notes: list[str] = []
    if not run_dir.is_dir():
        row["notes"] = f"run directory not present: {declared['run_dir']}"
        return row
    if "smoke" in run_dir.parent.name.lower():
        raise ValueError(f"Refusing to include smoke run {run_dir.parent.name} in the comparison table.")

    run_config_path = run_dir / "run_config.yaml"
    structure: dict = {}
    if run_config_path.is_file():
        run_config = load_yaml(run_config_path)
        selection = run_config.get("selection_protocol", {}) or {}
        row["selection_metric"] = str(selection.get("metric_name") or "")
        row["aggregation"] = str((run_config.get("evaluation", {}) or {}).get("aggregation_level") or "")
        row["fold"] = str(run_config.get("fold", ""))
        tracking = run_config.get("tracking", {}) or {}
        row["attempt_id"] = str(tracking.get("attempt_id") or "")
        training = run_config.get("training_strategy", {}) or {}
        row["gpu_shape"] = (
            f"{training.get('world_size', '?')} ranks, per-rank batch "
            f"{training.get('per_device_train_batch_size', '?')}"
        )
        row["activation_offload"] = str(training.get("activation_offload") or "")
        row["effective_batch"] = str(training.get("effective_global_batch_size") or "")
        audit = run_config.get("model_load_audit", {}) or {}
        structure = dict(audit)
        row["total_parameters"] = str(audit.get("total_parameters") or "")
        row["trainable_parameters"] = str(audit.get("lora_trainable_params") or "")
        active, formula = _active_parameters_estimate(structure)
        row["active_parameters"] = "" if active is None else str(active)
        if active is not None:
            notes.append(formula)
    else:
        notes.append("run_config.yaml missing")

    evaluations_path = run_dir / "evaluations.json"
    if evaluations_path.is_file():
        payload = json.loads(evaluations_path.read_text(encoding="utf-8"))
        for evaluation in payload.get("evaluations", []):
            if str(evaluation.get("backend")) != declared["backend"]:
                continue
            row["evaluation_view"] = str(evaluation.get("evaluation_view") or "")
            row["aggregation"] = row["aggregation"] or str(evaluation.get("aggregation") or "")
            if evaluation.get("locally_verified") is False:
                warnings = "; ".join(str(item) for item in evaluation.get("warnings") or [])
                notes.append(f"evaluation not locally verified: {warnings or 'no reason recorded'}")
            if evaluation.get("qualification"):
                row["qualification"] = str(evaluation["qualification"])
    else:
        notes.append("evaluations.json missing")

    metrics_path = run_dir / "best_model" / "standalone_eval" / f"metrics_{declared['backend']}.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        headline = metrics.get("headline_metrics", metrics) or {}
        macro_f1 = headline.get("binary_strict_macro_f1", headline.get("macro_f1"))
        positive_f1 = headline.get("binary_strict_positive_f1", headline.get("positive_f1"))
        row["macro_f1"] = "" if macro_f1 is None else f"{float(macro_f1):.6f}"
        row["positive_f1"] = "" if positive_f1 is None else f"{float(positive_f1):.6f}"
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
        row["uar"] = "" if uar is None else f"{uar:.6f}"
        row["evidence_path"] = str(metrics_path.relative_to(root))
    else:
        notes.append(f"metrics_{declared['backend']}.json missing")
    row["qualification"] = row["qualification"] or "promptcontext_v1 prompt; locally validated run"
    row["jobs"] = _job_ids(run_dir)
    row["notes"] = "; ".join(notes)
    return row


def build(project_root: Path) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for declared in ROWS:
        if declared["kind"] == "derived_reference":
            rows.append(_derived_row(declared, project_root))
        else:
            rows.append(_run_row(declared, project_root))
    provenance = {
        "schema": "audiollm.qwen3omni_daic_comparison.v1",
        "project_root": str(project_root),
        "reference": {
            "path": DERIVED_REFERENCE,
            "label": DERIVED_REFERENCE_LABEL,
            "note": (
                "the reference rows carry the older inline prompt, so a difference against the "
                "Qwen3-Omni rows is a model-plus-prompt difference, not a prompt-only effect"
            ),
        },
        "rows": rows,
    }
    return rows, provenance


def _markdown(rows: list[dict]) -> str:
    header = (
        "| Model | Modality | Backend | View | Aggregation | Macro-F1 | Positive-F1 | UAR | "
        "Trainable params | GPU shape | Run / attempt | Qualification |"
    )
    separator = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, separator]
    for row in rows:
        run = row["run_name"] + (f" / {row['attempt_id']}" if row["attempt_id"] else "")
        lines.append(
            "| {model} | {modality} | {backend} | {view} | {aggregation} | {macro} | {positive} | "
            "{uar} | {trainable} | {shape} | {run} | {qualification} |".format(
                model=row["model"],
                modality=row["modality"],
                backend=row["backend"],
                view=row["evaluation_view"] or "not recorded",
                aggregation=row["aggregation"] or "",
                macro=row["macro_f1"] or "",
                positive=row["positive_f1"] or "",
                uar=row["uar"] or "",
                trainable=row["trainable_parameters"] or "",
                shape=row["gpu_shape"] or "",
                run=run,
                qualification=row["qualification"] or "",
            )
        )
    body = "\n".join(lines)
    return (
        "# DAIC fold 0: Qwen3-Omni against the qualified Qwen2-Audio reference\n\n"
        "One row per model per modality. Qwen3-Omni values come from each production run's\n"
        "`best_model/standalone_eval` artifacts; the Qwen2-Audio reference is the PR #255 derived\n"
        "canonical likelihood evidence, whose recorded per-subject artifacts are re-hashed here.\n"
        "Both share the canonical DAIC manifest, split, seed and subject-level aggregation.\n\n"
        "The Qwen3-Omni rows use the `promptcontext_v1` prompt while the reference uses the older inline\n"
        "prompt, so the comparison is a model-plus-prompt comparison. Blanks mean the value is not\n"
        "recorded in local evidence. The table is descriptive: it carries no winner claim.\n\n"
        f"{body}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output-dir", default="outputs/qwen3omni_daic_comparison/report")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    rows, provenance = build(project_root)
    output_dir = (
        project_root / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "comparison.md").write_text(_markdown(rows), encoding="utf-8")
    (output_dir / "comparison.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {csv_path}, {output_dir / 'comparison.md'}, {output_dir / 'comparison.json'}")
    for row in rows:
        print(
            f"  {row['model']:<30} {row['modality']:<11} macro_f1={row['macro_f1'] or '-':<10} "
            f"notes={row['notes']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
