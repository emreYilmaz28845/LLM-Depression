"""Paired significance tests for the deck's within-corpus comparisons.

Reads a pre-declared family file (default:
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
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.daic_statistics import (  # noqa: E402
    exact_mcnemar,
    holm_adjust,
    paired_prediction_swap_permutation,
    stratified_paired_bootstrap,
)

DEFAULT_FAMILY = PROJECT_ROOT / "experiments/definitions/significance_family.yaml"
PRIMARY_METRICS = ("macro_f1", "macro_recall")  # macro_recall is UAR
METRIC_LABELS = {"macro_f1": "Macro-F1", "macro_recall": "UAR", "positive_f1": "Positive-F1"}


class SignificanceError(RuntimeError):
    """Raised when the family or its evidence cannot be resolved."""


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
            }
        )
    return out


def pool(entries: list[tuple[Path, str | None]]) -> list[dict[str, Any]]:
    """Pool per-fold subject predictions; a subject must appear exactly once."""
    merged: dict[str, dict[str, Any]] = {}
    for path, dataset in entries:
        rows = read_rows(path, dataset)
        if not rows:
            raise SignificanceError(f"no subject rows in {path} for dataset {dataset!r}")
        for row in rows:
            if row["subject_id"] in merged:
                raise SignificanceError(f"subject {row['subject_id']} appears in more than one fold file")
            merged[row["subject_id"]] = row
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
    raise SignificanceError(f"unknown side kind: {kind!r}")


def normalize_subjects(rows: list[dict[str, Any]], dataset: str | None) -> list[dict[str, Any]]:
    """Strip a leading ``<dataset>::`` namespace from subject ids.

    Merged campaign artifacts namespace subject ids by dataset while standalone
    runs use the bare id; both name the same people within one corpus.
    """
    out: dict[str, dict[str, Any]] = {}
    prefix = f"{dataset}::" if dataset else None
    for row in rows:
        subject_id = row["subject_id"]
        if prefix and subject_id.startswith(prefix):
            subject_id = subject_id[len(prefix):]
        if subject_id in out:
            raise SignificanceError(f"duplicate subject id after normalization: {subject_id}")
        out[subject_id] = {**row, "subject_id": subject_id}
    return list(out.values())


def run_family(family: dict[str, Any], evidence: dict[str, Any], joint: dict[str, Any] | None,
               *, iterations: int, seed: int, metrics: list[str]) -> dict[str, Any]:
    blocks_out: list[dict[str, Any]] = []
    for block in family["families"]:
        rows_out: list[dict[str, Any]] = []
        for comparison in block["comparisons"]:
            left_files = resolve_side(comparison["baseline"], evidence, joint)
            right_files = resolve_side(comparison["comparison"], evidence, joint)
            left_rows = normalize_subjects(pool(left_files), comparison.get("dataset"))
            right_rows = normalize_subjects(pool(right_files), comparison.get("dataset"))
            left_keys = {row["subject_id"] for row in left_rows}
            right_keys = {row["subject_id"] for row in right_rows}
            if left_keys != right_keys:
                raise SignificanceError(
                    f"{comparison['id']}: subject sets differ ({len(left_keys)} vs {len(right_keys)})"
                )
            entry: dict[str, Any] = {
                "id": comparison["id"],
                "description": comparison.get("description"),
                "dataset": comparison.get("dataset"),
                "n_subjects": len(left_keys),
                "baseline_files": [str(path) for path, _ in left_files],
                "comparison_files": [str(path) for path, _ in right_files],
                "baseline_file_sha256": {str(path): sha256_file(path) for path, _ in left_files},
                "comparison_file_sha256": {str(path): sha256_file(path) for path, _ in right_files},
                "mcnemar": exact_mcnemar(left_rows, right_rows),
                "metrics": {},
            }
            for metric in metrics:
                entry["metrics"][metric] = {
                    "permutation": paired_prediction_swap_permutation(
                        left_rows, right_rows, metric=metric, iterations=iterations, seed=seed
                    ),
                    "bootstrap": stratified_paired_bootstrap(
                        left_rows, right_rows, metric=metric, iterations=iterations, seed=seed
                    ),
                }
            rows_out.append(entry)
        blocks_out.append({
            "id": block["id"],
            "description": block.get("description"),
            "comparisons": rows_out,
        })

    # Holm inside every block: permutation p-values per metric, and McNemar p.
    for block in blocks_out:
        for metric in metrics:
            p_values = [row["metrics"][metric]["permutation"]["p_value"] for row in block["comparisons"]]
            adjusted = holm_adjust(p_values)
            for row, value in zip(block["comparisons"], adjusted):
                row["metrics"][metric]["permutation"]["p_value_holm"] = value
        mcnemar_p = [row["mcnemar"]["p_value"] for row in block["comparisons"]]
        for row, value in zip(block["comparisons"], holm_adjust(mcnemar_p)):
            row["mcnemar"]["p_value_holm"] = value
    return {"blocks": blocks_out}


def format_markdown(payload: dict[str, Any], family: dict[str, Any]) -> str:
    lines = [
        "# Paired significance report",
        "",
        f"Family file: `{payload['family_path']}` (sha256 `{payload['family_sha256'][:16]}…`)",
        f"Evidence: `{payload['evidence_path']}` (sha256 `{payload['evidence_sha256'][:16]}…`)",
        f"Permutation: {payload['iterations']} iterations, seed {payload['seed']}; "
        f"Holm correction inside every block; p-values are two-sided.",
        "The permutation section reports the exact observed delta; the bootstrap section reports the mean of "
        "the resampled deltas (slightly biased for nonlinear metrics) with its 95% CI.",
        "Pooled out-of-fold subject predictions; the decks' headline cells are unweighted fold means.",
        "",
    ]
    for block in payload["results"]["blocks"]:
        lines.append(f"## {block['id']}")
        if block.get("description"):
            lines.append(block["description"])
        lines += ["", "| Comparison | n | Δ Macro-F1 [95% CI] | p | p_Holm | Δ UAR [95% CI] | p | p_Holm | McNemar b/c | p | p_Holm |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in block["comparisons"]:
            f1 = row["metrics"]["macro_f1"]
            uar = row["metrics"]["macro_recall"]
            lines.append(
                f"| {row['id']} | {row['n_subjects']} | "
                f"{f1['bootstrap']['mean_delta']:+.4f} [{f1['bootstrap']['ci_low']:+.4f}, {f1['bootstrap']['ci_high']:+.4f}] | "
                f"{f1['permutation']['p_value']:.4f} | {f1['permutation']['p_value_holm']:.4f} | "
                f"{uar['bootstrap']['mean_delta']:+.4f} [{uar['bootstrap']['ci_low']:+.4f}, {uar['bootstrap']['ci_high']:+.4f}] | "
                f"{uar['permutation']['p_value']:.4f} | {uar['permutation']['p_value_holm']:.4f} | "
                f"{row['mcnemar']['baseline_only_correct']}/{row['mcnemar']['comparison_only_correct']} | "
                f"{row['mcnemar']['p_value']:.4f} | {row['mcnemar']['p_value_holm']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--evidence", type=Path, required=True,
                        help="three_route_evidence.json (presentation evidence)")
    parser.add_argument("--joint-evidence", type=Path, default=None,
                        help="joint_k4_evidence.json; required only for joint-K blocks")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/significance")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and validate every pair without running the tests")
    args = parser.parse_args()

    if not args.family.is_file():
        raise SignificanceError(f"family file not found: {args.family}")
    family = yaml.safe_load(args.family.read_text(encoding="utf-8"))
    if family.get("schema_version") != "audiollm.significance_family.v1":
        raise SignificanceError("unsupported family schema version")
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    joint = json.loads(args.joint_evidence.read_text(encoding="utf-8")) if args.joint_evidence else None

    metrics = list(family.get("metrics", PRIMARY_METRICS))
    if args.dry_run:
        checked = 0
        for block in family["families"]:
            for comparison in block["comparisons"]:
                left = resolve_side(comparison["baseline"], evidence, joint)
                right = resolve_side(comparison["comparison"], evidence, joint)
                left_rows = normalize_subjects(pool(left), comparison.get("dataset"))
                right_rows = normalize_subjects(pool(right), comparison.get("dataset"))
                if {row["subject_id"] for row in left_rows} != {row["subject_id"] for row in right_rows}:
                    raise SignificanceError(f"{comparison['id']}: subject sets differ")
                checked += 1
                print(f"ok {comparison['id']}: {len(left_rows)} subjects, {len(left)}x{len(right)} files")
        print(f"dry run: {checked} comparisons resolvable")
        return 0

    results = run_family(family, evidence, joint, iterations=args.iterations, seed=args.seed, metrics=metrics)
    payload = {
        "schema_version": "audiollm.significance_report.v1",
        "family_path": str(args.family),
        "family_sha256": sha256_file(args.family),
        "evidence_path": str(args.evidence),
        "evidence_sha256": sha256_file(args.evidence),
        "iterations": args.iterations,
        "seed": args.seed,
        "metrics": metrics,
        "primary_metric_field": "macro_recall is UAR",
        "deltas": "permutation.observed_delta is the exact observed difference; bootstrap.mean_delta is the mean "
                  "of the resampled differences (slightly biased for nonlinear metrics) and the CI is percentile",
        "correction": "holm within each block and metric",
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "significance_report.json"
    md_path = args.output_dir / "significance_report.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(format_markdown(payload, family), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    for block in results["blocks"]:
        significant = [
            row["id"] for row in block["comparisons"]
            if any(row["metrics"][metric]["permutation"]["p_value_holm"] <= family.get("alpha", 0.05)
                   for metric in metrics)
        ]
        print(f"{block['id']}: {len(block['comparisons'])} comparisons, "
              f"{len(significant)} significant after Holm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
