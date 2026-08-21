from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.identity import sanitize_logical_run_name

SBATCH_PARSABLE_PATTERN = None  # placeholder; validation is numeric-only in parse_sbatch_parsable

SQUEUE_COLUMNS = ("job_id", "state", "time", "nodes", "job_name", "reason")
SACCT_COLUMNS = ("job_id", "job_name", "state", "exit_code", "elapsed", "max_rss", "alloc_cpus", "node_list")

COMPACT_EVIDENCE_EXCLUDES = ("best_model/", "last_model/")


def parse_sbatch_parsable(text: str) -> str:
    line = text.strip().splitlines()[0].strip()
    job_id = line.split(";", 1)[0]
    if not job_id.isdigit():
        raise ValueError(f"unexpected sbatch --parsable output: {line!r}")
    return job_id


def parse_squeue_output(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].split()
    if len(header) != len(SQUEUE_COLUMNS):
        raise ValueError(f"unexpected squeue header: {lines[0]!r}")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        tokens = line.split()
        if len(tokens) < len(SQUEUE_COLUMNS) - 1:
            continue
        row = dict(zip(SQUEUE_COLUMNS, tokens))
        row.setdefault("reason", "")
        rows.append(row)
    return rows


def parse_sacct_output(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].split()
    if len(header) != len(SACCT_COLUMNS):
        raise ValueError(f"unexpected sacct header: {lines[0]!r}")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        tokens = line.split()
        if len(tokens) != len(SACCT_COLUMNS):
            continue
        rows.append(dict(zip(SACCT_COLUMNS, tokens)))
    return rows


def terminal_job_states(rows: Iterable[dict[str, str]]) -> list[str]:
    return sorted(
        {
            row["state"]
            for row in rows
            if row["state"] in (
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "PREEMPTED",
                "BOOT_FAIL",
            )
        }
    )


def deployed_source_manifest(repo_root: str | Path) -> list[dict[str, Any]]:
    root = Path(repo_root)
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    records: list[dict[str, Any]] = []
    for relative in sorted(line for line in result.stdout.splitlines() if line.strip()):
        path = root / relative
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            size = path.stat().st_size
        except OSError:
            continue
        records.append({"path": relative, "sha256": digest, "size_bytes": size})
    return records


def source_manifest_sha256(records: list[dict[str, Any]]) -> str:
    ordered = sorted(records, key=lambda record: record["path"])
    payload = json.dumps(
        ordered,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def expected_compact_evidence(fold_dir: str | Path) -> list[str]:
    root = Path(fold_dir)
    patterns = (
        "run_config.yaml",
        "metadata.json",
        "status.json",
        "jobs.jsonl",
        "artifacts.json",
        "evaluations.json",
        "final_summary.json",
        "final_summary.csv",
        "best_vs_last_checkpoint_metrics.json",
        "logs/*.json",
        "logs/*.jsonl",
        "logs/training_history.json",
        "best_model/standalone_eval/*.json",
        "best_model/standalone_eval/*.csv",
        "best_model/standalone_eval/*.jsonl",
        "best_model/standalone_eval/eval_config.yaml",
    )
    matches: list[str] = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                matches.append(path.relative_to(root).as_posix())
    return matches


def build_collect_command(
    fold_dir: str | Path,
    destination: str | Path,
    *,
    dry_run: bool = True,
    transfer_host: str = "ozu647717@transfer1.bsc.es",
    remote_project: str = "/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression",
) -> list[str]:
    relative = Path(fold_dir).relative_to(remote_project) if str(fold_dir).startswith(remote_project) else Path(fold_dir).name
    remote_source = f"{transfer_host}:{remote_project}/{relative}/"
    excludes: list[str] = []
    for pattern in COMPACT_EVIDENCE_EXCLUDES:
        excludes.extend(["--exclude", pattern])
    command = ["rsync", "-avz"]
    if dry_run:
        command.append("-n")
    command.extend(excludes)
    command.extend(["--include", "*/", remote_source, str(destination)])
    return command


def plan_matrix(
    *,
    logical_run_names: list[str],
    seeds: list[int],
    folds: list[int],
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for seed in seeds:
        for fold in folds:
            name = logical_run_names[0] if len(logical_run_names) == 1 else f"{logical_run_names[0]}_s{seed}"
            cells.append(
                {
                    "logical_run_name": name,
                    "seed": seed,
                    "fold": fold,
                    "jobs": ["train", "evaluation"],
                    "job_count": 2,
                }
            )
    return cells


def format_plan(cells: list[dict[str, Any]], *, git_commit: str, git_dirty: bool) -> dict[str, Any]:
    return {
        "source": {"git_commit": git_commit, "git_dirty": git_dirty},
        "matrix": cells,
        "total_jobs": sum(cell["job_count"] for cell in cells),
        "endpoint_split": {
            "transfer": "ozu647717@transfer1.bsc.es",
            "scheduler": "ozu647717@alogin2.bsc.es",
        },
        "rsync_policy": "no --delete; dry-run first; review every destination change",
        "checkpoint_policy": (
            "harmonized configs select best_model by inner_val_macro_f1; "
            "E-DAIC selposf1 configs are explicit legacy exceptions; "
            "last_model is never substituted"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor parsers and collection dry-run helpers (local only).")
    parser.add_argument("--parse-sbatch", default=None, help="Path to a file with sbatch --parsable output")
    parser.add_argument("--parse-squeue", default=None, help="Path to a file with squeue output")
    parser.add_argument("--parse-sacct", default=None, help="Path to a file with sacct output")
    parser.add_argument("--collect-dry-run", default=None, help="Fold directory to print the collection dry-run for")
    parser.add_argument("--destination", default="output_model", help="Destination root for --collect-dry-run")
    args = parser.parse_args()

    if args.parse_sbatch is not None:
        print(parse_sbatch_parsable(Path(args.parse_sbatch).read_text(encoding="utf-8")))
    if args.parse_squeue is not None:
        print(json.dumps(parse_squeue_output(Path(args.parse_squeue).read_text(encoding="utf-8")), indent=2))
    if args.parse_sacct is not None:
        print(json.dumps(parse_sacct_output(Path(args.parse_sacct).read_text(encoding="utf-8")), indent=2))
    if args.collect_dry_run is not None:
        command = build_collect_command(args.collect_dry_run, args.destination, dry_run=True)
        print(" ".join(command))
    return 0


if __name__ == "__main__":
    sys.exit(main())
