"""Side-by-side metrics of Claude DAIC runs (e.g. first-4000-chars vs full-transcript reruns).

Recomputes every number from each run's responses.jsonl (binary_strict: INVALID counts as wrong) and prints a
markdown table: macro-F1, positive-F1, UAR, confusion matrix, INVALID count, and, for each pair, how many
subjects get the same answer.

Usage:
  python experiments/manyshot_claude/compare_runs.py \
      --pair "Many-shot v2=OLD_DIR,NEW_DIR" --pair "Zero-shot=OLD_DIR,NEW_DIR" [--single "v1=DIR"]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("runner", Path(__file__).with_name("run_claude_manyshot.py"))
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def load(run_dir: Path) -> dict[str, dict]:
    recs = {}
    for line in (run_dir / "responses.jsonl").read_text().splitlines():
        r = json.loads(line)
        recs[r["subject_id"]] = r
    return recs


def row(name: str, text: str, recs: dict[str, dict], run_dir: Path, agree: str = "") -> str:
    vals = list(recs.values())
    m = runner.metrics(vals)
    cm = {k: 0 for k in ("TP", "FN", "FP", "TN")}
    for r in vals:
        pred = r["pred"] if r["pred"] is not None else 1 - r["label"]
        cm[{(1, 1): "TP", (1, 0): "FN", (0, 1): "FP", (0, 0): "TN"}[(r["label"], pred)]] += 1
    info = json.loads((run_dir / "run_info.json").read_text())
    return (f"| {name} | {text} | {m['n_subjects']} | {m['macro_f1']:.3f} | {m['positive_f1']:.3f} | {m['uar']:.3f} | "
            f"{cm['TP']} / {cm['FN']} / {cm['FP']} / {cm['TN']} | {m['invalid']} | {agree} | "
            f"`{run_dir.name}` ({info['manifest_sha256'][:12]}) |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pair", action="append", default=[], help="'Name=OLD_DIR,NEW_DIR'")
    p.add_argument("--single", action="append", default=[], help="'Name=DIR' (first 4000 chars only)")
    args = p.parse_args()
    print("| Run | Text | n | macro-F1 | pos-F1 | UAR | TP / FN / FP / TN | INVALID | same answer old vs new "
          "| Artifacts (manifest sha256) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for spec in args.pair:
        name, dirs = spec.split("=", 1)
        old_dir, new_dir = (Path(d) for d in dirs.split(","))
        old, new = load(old_dir), load(new_dir)
        if set(old) != set(new):
            raise ValueError(f"{name}: old and new runs cover different subjects.")
        same = sum(old[s]["pred"] == new[s]["pred"] for s in old)
        print(row(name, "first 4000 chars", old, old_dir))
        print(row(name, "full transcript", new, new_dir, f"{same}/{len(old)}"))
    for spec in args.single:
        name, d = spec.split("=", 1)
        print(row(name, "first 4000 chars", load(Path(d)), Path(d)))


if __name__ == "__main__":
    main()
