#!/usr/bin/env python3
"""Training diagnostics for finished runs: overfit, collapse, best epoch, fold stability.

  python tools/training_diagnostics.py run <run_dir> [--output DIR]
  python tools/training_diagnostics.py scan <output_model_root> [--output DIR]

`run` writes one run's "Training observations" (JSON + markdown). `scan` goes over every run under a
root and writes a summary table (CSV + markdown) with one row per run. Read-only on the inputs;
runs without a local `logs/training_history.json` are listed as n/a, never guessed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking import diagnostics

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "training_diagnostics"
SUMMARY_FIELDS = [
    "campaign", "modality", "dataset", "run", "folds", "epochs_min", "epochs_max", "best_epochs",
    "best_macro_f1_mean", "last_macro_f1_mean", "flags", "run_dir", "layout",
]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_run(run_dir: Path, scan_root: Path, fold_dirs: list[Path]) -> dict:
    report = diagnostics.diagnose_run(run_dir, fold_dirs)
    usable = [r for r in report["folds"].values() if r.get("status") in ("ok", "short_history")]
    best_epochs = {name: r.get("best_epoch") for name, r in report["folds"].items() if r.get("best_epoch") is not None}
    mean = lambda key: (round(sum(r[key] for r in usable if r.get(key) is not None) / max(1, sum(1 for r in usable if r.get(key) is not None)), 4)
                        if any(r.get(key) is not None for r in usable) else "")
    row = {
        **diagnostics.describe_run_path(run_dir, scan_root),
        "folds": f"{len(usable)}/{len(report['folds'])}",
        "epochs_min": min((r["n_epochs"] for r in usable), default=""),
        "epochs_max": max((r["n_epochs"] for r in usable), default=""),
        "best_epochs": ",".join(f"{k.removeprefix('fold_')}:{v}" for k, v in sorted(best_epochs.items())),
        "best_macro_f1_mean": mean("macro_f1_best"),
        "last_macro_f1_mean": mean("macro_f1_last"),
        "flags": ";".join(report["flag_names"]) or ("n/a" if not usable else "none"),
        "run_dir": str(run_dir),
    }
    return {"row": row, "report": report}


def render_summary_markdown(rows: list[dict]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        for flag in row["flags"].split(";"):
            counts[flag] = counts.get(flag, 0) + 1
    head = ["# Training diagnostics summary", "", f"Runs: {len(rows)}. Flag counts: "
            + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), "",
            "Best epochs are `fold:epoch`. Macro-F1 is validation (inner_val / selection split), averaged over folds with history.",
            "Diagnostics only; flags never change a run's lifecycle state.", "",
            "| Campaign | Modality | Dataset | Run | Folds | Epochs | Best epochs | Best / last val macro-F1 | Flags |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        epochs = f"{r['epochs_min']}–{r['epochs_max']}" if r["epochs_min"] != "" else "n/a"
        f1 = f"{r['best_macro_f1_mean']} / {r['last_macro_f1_mean']}" if r["best_macro_f1_mean"] != "" else "n/a"
        head.append(f"| {r['campaign']} | {r['modality']} | {r['dataset']} | {r['run']} | {r['folds']} | {epochs} | {r['best_epochs'] or 'n/a'} | {f1} | {r['flags']} |")
    return "\n".join(head) + "\n"


def cmd_run(args) -> int:
    run_dir = Path(args.run_dir).resolve()
    report = diagnostics.diagnose_run(run_dir)
    out = Path(args.output) / run_dir.name
    _write_json(out / "diagnostics.json", report)
    md = diagnostics.render_run_markdown(report)
    (out / "diagnostics.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out / 'diagnostics.json'}\nwrote {out / 'diagnostics.md'}")
    return 0


def cmd_scan(args) -> int:
    root = Path(args.root).resolve()
    runs = diagnostics.discover_runs(root)
    if not runs:
        print(f"no runs with logs/training_history.json under {root}", file=sys.stderr)
        return 1
    rows, details = [], {}
    for run_dir, fold_dirs in sorted(runs.items()):
        result = summarize_run(run_dir, root, fold_dirs)
        rows.append(result["row"])
        details[str(run_dir)] = result["report"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.md").write_text(render_summary_markdown(rows), encoding="utf-8")
    _write_json(out / "details.json", {"schema_version": diagnostics.SCHEMA_VERSION, "scan_root": str(root), "thresholds": diagnostics.DIAGNOSTIC_THRESHOLDS, "runs": details})
    flagged = sum(1 for r in rows if r["flags"] not in ("none", "n/a"))
    print(f"{len(rows)} runs, {flagged} with at least one flag")
    for name in ("summary.csv", "summary.md", "details.json"):
        print(f"wrote {out / name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="diagnose one run directory (contains fold_*/)")
    run.add_argument("run_dir")
    run.add_argument("--output", default=str(DEFAULT_OUTPUT / "runs"))
    run.set_defaults(func=cmd_run)
    scan = sub.add_parser("scan", help="diagnose every run under a root and write a summary table")
    scan.add_argument("root")
    scan.add_argument("--output", default=str(DEFAULT_OUTPUT / "scan"))
    scan.set_defaults(func=cmd_scan)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
