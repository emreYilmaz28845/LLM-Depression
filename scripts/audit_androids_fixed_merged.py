#!/usr/bin/env python3
"""Refresh and audit the corrected merged baseline chains for the campaign.

For every requested route/seed and stage this:
1. refreshes the run registry with ``monitor_symmetric_merged.py`` (squeue/sacct);
2. runs ``audit_symmetric_merged.py`` with the same resolved override tokens the
   submission used (merged root + component input repointing);
3. stores each acceptance audit JSON under ``--out-dir`` and summarises.

Exit code is non-zero when any audited chain does not pass. Run from the
immutable deployment code on the MN5 scheduler login.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.submit_symmetric_merged import (  # noqa: E402
    _runtime_override_tokens,
    load_merged_config,
)

ROUTES = {
    "native_text_only": ("native", "text_only"),
    "native_audio_only": ("native", "audio_only"),
    "native_audio_text": ("native", "audio_text"),
    "english_text_only": ("english", "text_only"),
    "english_audio_text": ("english", "audio_text"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["smoke", "cv", "final"])
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--pooled-runtime-root", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--campaign-base", default="qwen3_androids_official_folds_20261008")
    parser.add_argument("--only", default="")
    parser.add_argument("--seeds", nargs="*", type=int, default=[7, 1337, 2024])
    parser.add_argument("--expected-folds", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    for route, (campaign_suffix, modality) in ROUTES.items():
        if args.only and args.only != route:
            continue
        config_path = REPO_ROOT / f"configs/experiments/merged/symmetric_merged_qwen3_pooled_{route}.yaml"
        declared = load_merged_config(config_path, [])
        merged_root = Path(args.input_root) / "outputs/symmetric_merged" / f"{args.campaign_base}_{campaign_suffix}" / modality
        run_root = Path(args.input_root) / "output_model/symmetric_merged" / f"{args.campaign_base}_{campaign_suffix}_likelihood" / modality
        tokens = [
            f"--set=output_dirs.merged_root={merged_root}",
            f"--set=output_dirs.run_root={run_root}",
        ]
        tokens += _runtime_override_tokens(
            declared,
            input_root=args.input_root,
            pooled_runtime_root=args.pooled_runtime_root,
        )
        for seed in args.seeds:
            run_id = f"qmsm_{route}_s{seed}"
            registry = Path(args.runtime) / "registries" / f"{run_id}.json"
            if not registry.is_file():
                results.append({"run_id": run_id, "stage": args.stage, "status": "registry_missing"})
                continue
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/monitor_symmetric_merged.py"),
                    "--registry",
                    str(registry),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts/audit_symmetric_merged.py"),
                "--config",
                str(config_path),
                "--stage",
                args.stage,
                "--run-id",
                run_id,
                "--registry",
                str(registry),
                "--allow-omitted-heavy-artifacts",
            ]
            if args.expected_folds:
                command += ["--expected-folds", str(args.expected_folds)]
            for token in tokens:
                command.append(f"--override={token}")
            proc = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
            payload = None
            try:
                payload = json.loads(proc.stdout)
            except ValueError:
                payload = {"status": "unparsed", "stdout_tail": proc.stdout[-800:], "stderr_tail": proc.stderr[-800:]}
            audit_path = out_dir / f"{run_id}.{args.stage}.acceptance_audit.json"
            audit_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            status = payload.get("status") if isinstance(payload, dict) else "unparsed"
            results.append(
                {
                    "run_id": run_id,
                    "stage": args.stage,
                    "status": status,
                    "audit_path": str(audit_path),
                    "returncode": proc.returncode,
                }
            )
            if status != "passed":
                failures.append(f"{run_id}: audit status {status} (rc={proc.returncode})")
    summary = {"status": "passed" if not failures else "failed", "results": results, "failures": failures}
    (out_dir / f"audit_summary.{args.stage}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
