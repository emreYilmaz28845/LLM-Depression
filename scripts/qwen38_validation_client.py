#!/usr/bin/env python3
"""Qwen3.8 validation client: run, summarize, select.

Subcommands:

- ``run``: execute the synthetic-case workload against the localhost server
  (concurrency 1, 8, 16, 32, a determinism re-run at concurrency 1, or the
  post-restart 8-case subset) and write a results payload.
- ``summarize``: evaluate the runbook section 11.4 acceptance gates and write
  ``acceptance.json``; exits non-zero when the gate fails.
- ``select``: apply the runbook section 17 rules to the TP acceptance
  records and write ``serving_selection.json``.

All JSON writes are atomic (same-directory temporary file plus ``os.replace``).
Prompts and responses never reach stdout/stderr; only counts and hashes are
printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qwen38.contracts import (
    MODEL_ID,
    MODEL_REVISION,
    SERVED_MODEL,
    CandidateResult,
    SelectionResult,
    select_serving_configuration,
)
from src.qwen38.validation import (
    load_synthetic_cases,
    run_validation,
    summarize_acceptance,
)


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


def cmd_run(args: argparse.Namespace) -> int:
    cases = load_synthetic_cases(args.cases)
    results = run_validation(
        base_url=args.base_url,
        model=args.model,
        cases=cases,
        concurrency_levels=tuple(int(level) for level in args.concurrency_levels.split(",")),
        max_tokens=args.max_tokens,
        seed=args.seed,
        restart_subset=args.restart_subset,
        server_startup_timeout=args.server_startup_timeout,
    )
    results["model_id"] = MODEL_ID
    results["model_revision"] = MODEL_REVISION
    results["case_count"] = len(cases)
    _atomic_write_json(results, args.out)
    if args.restart_subset:
        print(f"restart subset results written to {args.out}")
    else:
        print(f"validation results written to {args.out}")
    return 0


def _environment_versions() -> dict[str, Any]:
    import torch
    import transformers
    import vllm

    try:
        import huggingface_hub
    except ImportError:
        huggingface_hub = None
    import openai

    return {
        "python_major": sys.version_info.major,
        "python_minor": sys.version_info.minor,
        "vllm": vllm.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__.split("+")[0],
        "torchvision": __import__("torchvision").__version__,
        "torchaudio": __import__("torchaudio").__version__,
        "openai": openai.__version__,
        "huggingface_hub": getattr(huggingface_hub, "__version__", ""),
    }


def cmd_summarize(args: argparse.Namespace) -> int:
    results = _read_json(args.results)
    restart_results = _read_json(args.restart_results) if args.restart_results else None
    env = _environment_versions()
    deployment_env: dict[str, Any] = {}
    if args.deployment_env:
        deployment_env = _read_json(args.deployment_env)
    expected_env_keys = set(env)
    deployment_env_subset = {key: deployment_env.get(key) for key in expected_env_keys}
    acceptance = summarize_acceptance(
        results,
        restart_results,
        model_revision=results.get("model_revision", ""),
        deployment_model_revision=deployment_env.get("model_revision", ""),
        environment_versions=env,
        deployment_environment_versions=deployment_env_subset,
        model_manifest_sha256=args.model_manifest_sha256,
        deployment_model_manifest_sha256=args.deployment_model_manifest_sha256,
        wheelhouse_manifest_sha256=args.wheelhouse_manifest_sha256,
        deployment_wheelhouse_manifest_sha256=args.deployment_wheelhouse_manifest_sha256,
    )
    acceptance["results_path"] = str(args.results)
    acceptance["restart_results_path"] = str(args.restart_results) if args.restart_results else None
    _atomic_write_json(acceptance, args.out)
    print(f"acceptance={acceptance['passed']} checks={len(acceptance['checks'])}")
    if not acceptance["passed"]:
        for check in acceptance["checks"]:
            if not check["passed"]:
                print(f"  failed: {check['check_id']}: {check['description']}", file=sys.stderr)
        return 1
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    deploy_dir = Path(args.deploy_dir)
    deployment_id = args.deployment_id
    source_commit = args.source_commit

    candidates: dict[int, CandidateResult] = {}
    measured_paths: dict[int, str] = {}
    for tp in (1, 2, 4):
        tp_dir = deploy_dir / deployment_id / "validation" / f"tp{tp}"
        attempts = sorted(p.name for p in tp_dir.glob("attempt*")) if tp_dir.is_dir() else []
        if not attempts:
            candidates[tp] = CandidateResult(tp=tp, passed=False)
            continue
        acceptance_path = tp_dir / attempts[-1] / "acceptance.json"
        metrics_path = tp_dir / attempts[-1] / "results.json"
        if not acceptance_path.is_file():
            candidates[tp] = CandidateResult(tp=tp, passed=False)
            continue
        acceptance = _read_json(acceptance_path)
        metrics = _read_json(metrics_path) if metrics_path.is_file() else {}
        rate_c1 = metrics.get("levels", {}).get("c1_pass_a", {}).get("aggregate_requests_per_second")
        rate_c8 = metrics.get("levels", {}).get("c8", {}).get("aggregate_requests_per_second")
        passed = bool(acceptance.get("passed"))
        candidates[tp] = CandidateResult(
            tp=tp,
            passed=passed,
            request_rate_c1=rate_c1 if rate_c1 is not None else None,
            request_rate_c8=rate_c8 if rate_c8 is not None else None,
            metrics_path=str(metrics_path),
        )
        if passed:
            measured_paths[tp] = str(metrics_path)

    selection: SelectionResult = select_serving_configuration(candidates)
    if selection.selected_tp is None:
        print("no serving configuration selected (TP=2 did not pass)", file=sys.stderr)
        return 1

    payload = {
        "deployment_id": deployment_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "selected_tp": selection.selected_tp,
        "decision_rule": selection.decision_rule,
        "candidate_results": {
            str(tp): {
                "passed": candidate.passed,
                "request_rate_c1": candidate.request_rate_c1,
                "request_rate_c8": candidate.request_rate_c8,
                "metrics_path": candidate.metrics_path,
            }
            for tp, candidate in sorted(selection.candidate_results.items())
        },
        "projected_requests": selection.projected_requests,
        "projected_wall_seconds": {
            str(tp): round(value, 3) for tp, value in sorted(selection.projected_wall_seconds.items())
        },
        "measured_metrics_paths": {str(tp): path for tp, path in sorted(measured_paths.items())},
        "created_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "source_commit": source_commit,
    }
    out_path = deploy_dir / deployment_id / "serving_selection.json"
    _atomic_write_json(payload, out_path)
    print(f"selected_tp={payload['selected_tp']} rule={payload['decision_rule']}")
    print(f"serving_selection.json written to {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3.8 validation client (run / summarize / select)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the synthetic-case workload")
    run_parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    run_parser.add_argument("--model", default=SERVED_MODEL)
    run_parser.add_argument("--cases", required=True, help="synthetic fixture JSONL")
    run_parser.add_argument("--out", required=True, help="results JSON")
    run_parser.add_argument("--concurrency-levels", default="1,8,16,32")
    run_parser.add_argument("--max-tokens", type=int, default=1024)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--restart-subset", action="store_true", help="run only the fixed 8-case restart subset")
    run_parser.add_argument("--server-startup-timeout", type=int, default=600)
    run_parser.set_defaults(func=cmd_run)

    summarize_parser = subparsers.add_parser("summarize", help="evaluate acceptance gates")
    summarize_parser.add_argument("--results", required=True, help="results JSON from run")
    summarize_parser.add_argument("--restart-results", default=None, help="post-restart subset results JSON")
    summarize_parser.add_argument("--out", required=True, help="acceptance JSON")
    summarize_parser.add_argument("--deployment-env", default=None, help="deployment runtime_versions.json")
    summarize_parser.add_argument("--model-manifest-sha256", default=None, help="computed model manifest hash")
    summarize_parser.add_argument("--wheelhouse-manifest-sha256", default=None, help="computed wheelhouse manifest hash")
    summarize_parser.add_argument("--deployment-model-manifest-sha256", default=None, help="recorded model manifest hash")
    summarize_parser.add_argument("--deployment-wheelhouse-manifest-sha256", default=None, help="recorded wheelhouse manifest hash")
    summarize_parser.set_defaults(func=cmd_summarize)

    select_parser = subparsers.add_parser("select", help="select the serving configuration")
    select_parser.add_argument("--deploy-dir", required=True, help="REMOTE_DEPLOY root")
    select_parser.add_argument("--deployment-id", required=True)
    select_parser.add_argument("--source-commit", required=True)
    select_parser.set_defaults(func=cmd_select)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
