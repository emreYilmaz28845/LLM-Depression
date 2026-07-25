from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.utils import read_json, read_jsonl, save_json


def audit(matrix_path: Path, results_root: Path) -> dict[str, Any]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    experiment_id = matrix["experiment_id"]
    jobs = [
        {**item, "fold": fold}
        for item in matrix["experiments"]
        for fold in item["folds"]
    ]
    findings = []
    failures = []
    for job in jobs:
        run_name = Path(job["run_dir"]).name
        output = (
            results_root
            / job["dataset"]
            / job["condition"]
            / run_name
            / f"fold_{job['fold']}"
            / experiment_id
        )
        try:
            completion = read_json(output / "completion.json")
            config = read_json(output / "screen_config.json")
            summaries = read_json(output / "screen_summary.json")
            subject_rows = read_jsonl(output / "inner_oof_subject_predictions.jsonl")
            audits = list((output / "sampling_audits").glob("*/*/*.json"))
            assert completion["status"] == "complete"
            assert completion["final_eval_loaded"] is False
            assert len(summaries) == 14
            assert len(audits) == 42
            assert all(read_json(path)["validation_indices_untouched"] for path in audits)
            assert all(read_json(path)["evaluation_indices_untouched"] for path in audits)
            grouped = {}
            for row in subject_rows:
                key = (row["profile_id"], row["head"])
                grouped.setdefault(key, []).append(row["subject_id"])
            expected_subjects = int(summaries[0]["inner_subject_count"])
            assert len(grouped) == 14
            assert all(
                len(ids) == expected_subjects and len(ids) == len(set(ids))
                for ids in grouped.values()
            )
            findings.append(
                {
                    "condition": job["condition"],
                    "fold": int(job["fold"]),
                    "status": "pass",
                    "configuration_sha256": config["configuration_sha256"],
                    "summary_rows": len(summaries),
                    "sampling_audits": len(audits),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "condition": job["condition"],
                    "fold": int(job["fold"]),
                    "output": str(output),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "schema_version": "turkish_oversampling_screen_audit.v1",
        "expected_jobs": int(matrix["expected_jobs"]),
        "observed_jobs": len(findings),
        "failed_jobs": len(failures),
        "status": "pass" if not failures and len(findings) == int(matrix["expected_jobs"]) else "fail",
        "findings": findings,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.matrix, args.results_root)
    save_json(payload, args.output)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
