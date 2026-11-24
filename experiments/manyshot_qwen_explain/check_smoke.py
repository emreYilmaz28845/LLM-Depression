"""Exit 0 only if a smoke run of run_qwen_manyshot_explain.py looks usable (gate for queued full runs).

Checks: at least --min-targets answers, every answer has a valid label (no INVALID) and finished
before max_new_tokens, and every recorded determinism check is identical. The cache-vs-fresh-prefill
comparison is reported but does not gate (different chunk boundaries can change bf16 rounding).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check(out_dir: Path, min_targets: int) -> list[str]:
    problems = []
    recs = [json.loads(line) for p in sorted(out_dir.glob("responses*.jsonl"))
            for line in p.read_text().splitlines() if line.strip()]
    if len(recs) < min_targets:
        problems.append(f"{len(recs)} answers < {min_targets}")
    for r in recs:
        if r.get("pred") is None:
            problems.append(f"{r['subject_id']}: INVALID answer")
        if not r.get("finished"):
            problems.append(f"{r['subject_id']}: answer hit max_new_tokens")
    checks = [json.loads(line) for p in sorted(out_dir.glob("checks*.jsonl"))
              for line in p.read_text().splitlines() if line.strip()]
    det = [c for c in checks if "determinism_identical" in c]
    if not det:
        problems.append("no determinism check recorded")
    problems += [f"{c['subject_id']}: determinism differs" for c in det if not c["determinism_identical"]]
    for c in checks:
        if "uncached_identical" in c:
            print(f"uncached check {c['subject_id']}: identical={c['uncached_identical']} "
                  f"first_divergence={c.get('uncached_first_divergence')}", flush=True)
    return problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out_dir", type=Path)
    p.add_argument("--min-targets", type=int, default=3)
    args = p.parse_args(argv)
    problems = check(args.out_dir, args.min_targets)
    if problems:
        print("SMOKE GATE FAILED: " + "; ".join(problems), flush=True)
        return 1
    print("SMOKE GATE OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
