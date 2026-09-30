#!/usr/bin/env python3
"""Build a deterministic isolated-smoke subject selection for a Qwen3 checkpoint.

The smoke extraction runs on a tiny, predeclared subset of the checkpoint's own
saved split. This builder resolves that split exactly the way the extractor does
(``_load_saved_run`` + ``_resolve_subject_partitions``), then picks, per label
stratum, the first subjects (sorted) that qualify.

For the pooled Turkish family (``dataset_variant: pooled_t17``) a subject
qualifies only when it carries every required question condition, so the pooled
pairing and the two-condition identity are preserved. Any other cell has no
question-condition contract and the selection is by label only.

The output JSON is what ``src/features/extract_qwen_hidden.py
--subject-selection`` accepts. Its sha256 becomes part of the cache identity, so
a smoke cache can never collide with a production cache. ``--preview`` prints
the same selection without writing anything.

This tool is smoke-only: the production head orchestration resolves its parents
from recorded attempts and never uses this small selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extract_qwen_hidden import (  # noqa: E402
    _load_saved_run,
    _resolve_subject_partitions,
    _saved_path,
)
from src.utils import read_json, read_jsonl, save_json  # noqa: E402

DEFAULT_CONDITIONS = ("pos_only_t17", "negative_only_t17")


class SelectionError(RuntimeError):
    """Raised when the requested smoke selection cannot be built."""


def _subject_conditions(manifest_rows: list[dict]) -> tuple[dict[str, set[str]], dict[str, int]]:
    conditions: dict[str, set[str]] = {}
    labels: dict[str, int] = {}
    for row in manifest_rows:
        subject_id = str(row["subject_id"])
        conditions.setdefault(subject_id, set()).add(str(row.get("dataset_variant", "")))
        label = int(row["label"])
        if subject_id in labels and labels[subject_id] != label:
            raise SelectionError(f"Subject {subject_id} has inconsistent labels in the manifest.")
        labels[subject_id] = label
    return conditions, labels


def _select_per_label(
    candidates: list[str],
    labels: dict[str, int],
    conditions: dict[str, set[str]],
    required: set[str],
    per_label: int,
    partition: str,
) -> list[str]:
    chosen: list[str] = []
    for label_value in (0, 1):
        taken = 0
        for subject_id in sorted(candidates):
            if labels.get(subject_id) != label_value:
                continue
            if required and conditions.get(subject_id, set()) != required:
                continue
            chosen.append(subject_id)
            taken += 1
            if taken >= per_label:
                break
        if taken < per_label:
            raise SelectionError(
                f"{partition}: only {taken} subject(s) with label {label_value} carry "
                f"exactly the conditions {sorted(required)}; {per_label} required."
            )
    return sorted(chosen)


def build_selection(
    *,
    checkpoint_dir: Path,
    manifest_path: Path | None,
    train_per_label: int,
    eval_per_label: int,
    conditions: tuple[str, ...],
) -> dict:
    if train_per_label < 1 or eval_per_label < 1:
        raise SelectionError("Train/eval subjects per label must be positive.")
    saved, config, run_config_path, split_path = _load_saved_run(checkpoint_dir)
    pooled = str(config.get("dataset_variant", "")) == "pooled_t17"
    if pooled:
        required = set(conditions)
        if not required:
            raise SelectionError("At least one required question condition is needed.")
    else:
        # A non-pooled cell has no question-condition contract, so the smoke
        # selection is by label only; the pooled family keeps its exact pairing.
        required = set()
    resolved_manifest = manifest_path or _saved_path(saved["manifest_path"])
    if not resolved_manifest.is_file():
        raise SelectionError(f"Manifest is unavailable: {resolved_manifest}")
    split_payload = read_json(split_path)
    partitions, provenance = _resolve_subject_partitions(saved, config, split_payload)
    condition_map, labels = _subject_conditions(read_jsonl(resolved_manifest))
    selection = {
        "outer_train": _select_per_label(
            partitions["outer_train"], labels, condition_map, required,
            train_per_label, "outer_train",
        ),
        "final_eval": _select_per_label(
            partitions["final_eval"], labels, condition_map, required,
            eval_per_label, "final_eval",
        ),
    }
    overlap = sorted(set(selection["outer_train"]) & set(selection["final_eval"]))
    if overlap:
        raise SelectionError(f"Smoke fit and eval subject sets overlap: {overlap}")
    return {
        "schema_version": "audiollm.qwen3_smoke_subject_selection.v1",
        "rule": "sorted_first_per_label_with_all_required_conditions",
        "checkpoint_dir": str(checkpoint_dir),
        "run_config": str(run_config_path),
        "saved_split": str(split_path),
        "manifest": str(resolved_manifest),
        "fold": int(saved["fold"]),
        "evaluation_protocol": provenance.get("evaluation_protocol"),
        "required_conditions": sorted(required),
        "per_label": {"outer_train": train_per_label, "final_eval": eval_per_label},
        "outer_train": selection["outer_train"],
        "final_eval": selection["final_eval"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--manifest-path", type=Path, help="Manifest override; defaults to the saved run manifest.")
    parser.add_argument("--output", type=Path, help="Where to write the selection JSON (submit mode).")
    parser.add_argument("--train-per-label", type=int, default=1)
    parser.add_argument("--eval-per-label", type=int, default=1)
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--preview", action="store_true", help="Print the selection without writing it.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Fail unless the existing --output file equals the computed selection.",
    )
    parser.add_argument("--print-json", action="store_true", help="Print the full selection payload.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_selection(
        checkpoint_dir=args.checkpoint_dir.resolve(),
        manifest_path=args.manifest_path.resolve() if args.manifest_path else None,
        train_per_label=args.train_per_label,
        eval_per_label=args.eval_per_label,
        conditions=tuple(args.conditions),
    )
    if args.verify:
        if args.output is None or not args.output.is_file():
            raise SelectionError(f"--verify requires an existing --output file: {args.output}")
        existing = read_json(args.output)
        if existing != payload:
            raise SelectionError(
                f"Existing subject selection differs from the computed one: {args.output}"
            )
    if args.output is not None and not args.preview and not args.verify:
        save_json(payload, args.output)
    if args.print_json:
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    else:
        if args.verify:
            status = "verified"
        elif args.preview or args.output is None:
            status = "preview"
        else:
            status = "written"
        print(
            json.dumps(
                {
                    "status": status,
                    "output": str(args.output) if args.output and status != "preview" else None,
                    "checkpoint_dir": payload["checkpoint_dir"],
                    "fold": payload["fold"],
                    "evaluation_protocol": payload["evaluation_protocol"],
                    "conditions": payload["required_conditions"],
                    "outer_train": payload["outer_train"],
                    "final_eval": payload["final_eval"],
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
