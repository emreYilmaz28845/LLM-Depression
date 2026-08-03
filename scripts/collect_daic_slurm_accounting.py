from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def indices(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    submission = json.loads(args.submission.read_text())
    job_map: dict[str, tuple[str, list[int]]] = {}
    train = submission["arrays"]["train"]
    train_jobs = list(train["job_ids"])
    regular = indices(train.get("regular_indices", ""))
    mil = indices(train.get("mil_indices", ""))
    cursor = 0
    if regular:
        job_map[train_jobs[cursor]] = ("train", regular); cursor += 1
    if mil:
        job_map[train_jobs[cursor]] = ("train", mil)
    for kind in ("evaluation", "hidden", "classical"):
        job_id = submission["arrays"][kind]["job_ids"][0]
        task_count = sum(task["kind"] == kind for task in matrix["tasks"])
        job_map[job_id] = (kind, list(range(task_count)))
    job_ids = ",".join(job_map)
    result = subprocess.run(
        ["sacct", "-j", job_ids, "--noheader", "--parsable2", "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList,AllocCPUS,AllocTRES"],
        text=True, stdout=subprocess.PIPE, check=True,
    )
    task_lists = {kind: [task for task in matrix["tasks"] if task["kind"] == kind] for kind in ("train", "evaluation", "hidden", "classical")}
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 7:
            continue
        raw_id = fields[0]
        matched = next(((job, re.fullmatch(re.escape(job) + r"_(\d+)", raw_id)) for job in job_map if re.fullmatch(re.escape(job) + r"_(\d+)", raw_id)), None)
        if not matched:
            continue
        job, match = matched
        kind, submitted_indices = job_map[job]
        array_index = int(match.group(1))
        if array_index not in submitted_indices:
            continue
        task = task_lists[kind][array_index]
        rows.append({
            "task_id": task["task_id"], "cell_id": task["cell_id"], "kind": kind,
            "job_id_raw": raw_id, "state": fields[1].split()[0], "exit_code": fields[2],
            "elapsed": fields[3], "node_list": fields[4], "allocated_cpus": fields[5], "allocated_tres": fields[6],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "expected": len(matrix["tasks"])}, sort_keys=True))


if __name__ == "__main__":
    main()
