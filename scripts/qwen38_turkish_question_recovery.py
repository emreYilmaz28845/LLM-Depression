#!/usr/bin/env python3
"""Qwen3.8 Turkish question-recovery pipeline CLI.

Subcommands:

- ``prepare``: parse and verify the private ASR transcript, group the 1,186
  windows into 135 subject sequences with opaque IDs, and write owner-only
  prepared packets.
- ``infer-subjects``: one deterministic request per sequence, resumable by
  sequence ID, refusing resume when any provenance hash differs.
- ``consolidate``: five fixed batches plus one final merge; enforces exact
  candidate/cluster assignment.
- ``render``: deterministic CSV, JSON, and Markdown tables from the final
  families.

All JSON writes are atomic. The condition tag, subject identifiers, and raw
transcripts never enter model input beyond the prepared prompts, and the
compact outputs never contain them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qwen38.contracts import MODEL_ID, MODEL_REVISION, SERVED_MODEL
from src.qwen38.turkish_questions import (
    aggregate_families,
    consolidate as consolidate_stage,
    infer_subjects as infer_subjects_stage,
    prepare_sequences,
    render_tables,
)

RUN_SUBDIR = "outputs/turkish_question_recovery"


def _atomic_write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _restrict_run_dir(run_dir: Path) -> None:
    restricted = run_dir / "restricted"
    if restricted.exists():
        os.chmod(run_dir, 0o700)
        os.chmod(restricted, 0o700)
        for child in restricted.rglob("*"):
            if child.is_file():
                os.chmod(child, 0o600)


def cmd_prepare(args: argparse.Namespace) -> int:
    summary = prepare_sequences(
        args.transcript,
        run_dir=args.run_dir,
        deployment_id=args.deployment_id,
        model_revision=args.model_revision,
        source_commit=args.source_commit,
    )
    _restrict_run_dir(Path(args.run_dir))
    _atomic_write_json(summary, Path(args.run_dir) / "prepare_summary.json")
    os.chmod(Path(args.run_dir) / "prepare_summary.json", 0o600)
    print(
        f"prepared: sequences={summary['sequences']} windows={summary['windows']} "
        f"source_sha256={summary['source_sha256']}"
    )
    return 0


def cmd_infer_subjects(args: argparse.Namespace) -> int:
    summary = infer_subjects_stage(
        args.prepared,
        args.inferences_dir,
        base_url=args.base_url,
        model=args.model,
        concurrency=args.concurrency,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )
    _atomic_write_json(summary, Path(args.run_dir) / "inference_summary.json")
    os.chmod(Path(args.run_dir) / "inference_summary.json", 0o600)
    print(
        f"inferred: total={summary['sequences_total']} completed={summary['completed_total']} "
        f"failed={len(summary['failed'])}"
    )
    for failure in summary["failed"]:
        print(f"  failed: {failure}", file=sys.stderr)
    if not summary["complete"]:
        print("inference incomplete", file=sys.stderr)
        return 1
    return 0


def cmd_consolidate(args: argparse.Namespace) -> int:
    summary = consolidate_stage(
        args.inferences_dir,
        args.consolidation_dir,
        base_url=args.base_url,
        model=args.model,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )
    _restrict_run_dir(Path(args.run_dir))
    _atomic_write_json(summary, Path(args.run_dir) / "consolidation_summary.json")
    os.chmod(Path(args.run_dir) / "consolidation_summary.json", 0o600)
    print(
        f"consolidated: sequences={summary['sequences']} candidates={summary['candidates']} "
        f"clusters={summary['clusters']} families={summary['families']}"
    )
    return 0


def _load_records(inferences_dir: str | Path) -> list[dict[str, Any]]:
    inferences_dir = Path(inferences_dir)
    records: list[dict[str, Any]] = []
    for path in sorted(inferences_dir.glob("S*.json")):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("status") != "completed":
            raise ValueError(f"{path.name}: not completed")
        records.append(record)
    return records


def cmd_render(args: argparse.Namespace) -> int:
    from src.qwen38.turkish_questions import collect_candidates

    records = _load_records(args.inferences_dir)
    candidates = collect_candidates(records)
    final_merge = _read_json(args.final_merge)
    families = final_merge["families"]
    cluster_assignment = final_merge["cluster_assignment"]
    cluster_to_candidate: dict[str, list[str]] = {}
    for batch_path in sorted(Path(args.consolidation_dir).glob("batch_*.json")):
        batch = _read_json(batch_path)
        for cluster in batch["clusters"]:
            cluster_to_candidate[cluster["cluster_id"]] = cluster["member_candidate_ids"]
    rows = aggregate_families(
        candidates,
        records,
        families,
        cluster_to_candidate,
        cluster_assignment,
    )
    result = render_tables(
        rows,
        run_dir=args.run_dir,
        deployment_id=args.deployment_id,
        model_id=MODEL_ID,
        model_revision=args.model_revision,
        source_commit=args.source_commit,
    )
    _atomic_write_json(result, Path(args.run_dir) / "render_summary.json")
    print(
        f"rendered: rows={result['rows']} rows_sha256={result['rows_sha256']} "
        f"csv={result['csv']}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3.8 Turkish question-recovery pipeline (prepare / infer-subjects / consolidate / render)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--deployment-id", required=True)
    common.add_argument("--run-dir", required=True, help="run output directory")
    common.add_argument("--source-commit", default="")
    common.add_argument("--model-revision", default=MODEL_REVISION)

    prepare = subparsers.add_parser("prepare", parents=[common], help="prepare subject sequences")
    prepare.add_argument("--transcript", required=True, help="private ASR transcript JSONL")
    prepare.set_defaults(func=cmd_prepare)

    infer = subparsers.add_parser("infer-subjects", parents=[common], help="infer question families per subject sequence")
    infer.add_argument("--prepared", required=True, help="prepared_sequences.jsonl")
    infer.add_argument("--inferences-dir", required=True)
    infer.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    infer.add_argument("--model", default=SERVED_MODEL)
    infer.add_argument("--concurrency", type=int, default=8)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--max-tokens", type=int, default=2048)
    infer.set_defaults(func=cmd_infer_subjects)

    consolidate = subparsers.add_parser("consolidate", parents=[common], help="two-level consolidation")
    consolidate.add_argument("--inferences-dir", required=True)
    consolidate.add_argument("--consolidation-dir", required=True)
    consolidate.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    consolidate.add_argument("--model", default=SERVED_MODEL)
    consolidate.add_argument("--seed", type=int, default=42)
    consolidate.add_argument("--max-tokens", type=int, default=2048)
    consolidate.set_defaults(func=cmd_consolidate)

    render = subparsers.add_parser("render", parents=[common], help="render compact tables")
    render.add_argument("--inferences-dir", required=True)
    render.add_argument("--consolidation-dir", required=True)
    render.add_argument("--final-merge", required=True, help="final_merge.json")
    render.set_defaults(func=cmd_render)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
