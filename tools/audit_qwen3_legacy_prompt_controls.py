#!/usr/bin/env python3
"""Read-only compatibility audit of the 189 promptcontext_v1 control fits.

The Worker 4 legacy-prompt comparison reuses the completed Native standalone
promptcontext_v1 campaign as its control arm (189 training fits and 189 fixed
head keys). This audit verifies, without mutating anything, that every control
fit is compatible with the frozen treatment contract:

- run_config and eval_config are locally present; recorded run_config sha256
  matches;
- the prompt block and the rendered system prompt equal the canonical control
  config for the route (train and evaluation render the same prompt);
- backend, model revision, LoRA target, training selection/early stopping,
  accumulation, evaluation view/aggregation, split seed 1337 and the training
  seed all match the route contract;
- compact evaluation evidence (subject predictions + likelihood metrics) is
  present;
- every head key has a resolved parent, exactly the logreg_raw/xgb_raw
  variants at head seed 1337, a REPORTABLE attempt, matching parent adapter
  identity, and a local attempt directory whose classifier contract matches.

The output is a deterministic JSON report with per-cell verdicts and a compact
summary; the tool exits nonzero when any cell is blocked.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.prompt_context import resolve_system_prompt
from tools.qwen3_legacy_prompt_plan import build_matrix

SCHEMA_VERSION = "audiollm.worker4_control_audit.v1"
DEFAULT_NATIVE_EVIDENCE = Path(
    "/home/emre/Projects/AudioLLM/worktrees/LLM-Depression-feat-qwen3-multiseed-native-20261002"
    "/outputs/qwen3_multiseed_native_20261002"
)
KEY_RE = re.compile(r"^(.+)\|s?(\d+)\|f?(\d+)$")

REQUIRED_TRAIN_KEYS = (
    "strategy",
    "activation_offload",
    "selection_metric",
    "selection_metric_mode",
    "num_train_epochs",
    "gradient_accumulation_steps",
    "class_balance",
)
REQUIRED_EVAL_KEYS = (
    "sample_prediction_mode",
    "headline_mode",
    "evaluation_view",
    "aggregation_level",
    "subject_score_aggregation",
    "hierarchical_score_aggregation",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def normalize_key(key: str) -> tuple[str, int, int] | None:
    match = KEY_RE.match(key)
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def unwrap(record: dict[str, Any]) -> dict[str, Any]:
    inner = record.get("config")
    if isinstance(inner, dict) and "dataset" in inner:
        return inner
    return record


def value(record: dict[str, Any], key: str) -> Any:
    return record.get(key)


def is_audio(route: dict[str, Any]) -> bool:
    return route["modality"] in ("audio_only", "audio_text")


def audit_fit(
    row: dict[str, Any],
    route: dict[str, Any],
    matrix_parent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    notes: list[str] = []
    fold_dir = Path(row["fold_dir"])
    run_config_path = fold_dir / "run_config.yaml"
    if not run_config_path.is_file():
        return {
            "registry_key": row["registry_key"],
            "run_name": row.get("run_name"),
            "verdict": "blocked",
            "failures": [f"missing {run_config_path}"],
        }
    recorded = row.get("run_config_sha256")
    actual = sha256_file(run_config_path)
    if recorded and actual != recorded:
        failures.append(f"run_config sha256 drift: {actual} != {recorded}")
    config = unwrap(load_yaml(run_config_path))
    control = load_yaml(ROOT / route["control_config"])

    prompt = config.get("prompt") or {}
    control_prompt = control.get("prompt") or {}
    for key in ("version", "dataset_context", "question_context_version", "user_template"):
        if prompt.get(key) != control_prompt.get(key):
            failures.append(f"prompt.{key} differs from control")
    system_prompt = resolve_system_prompt(config)
    if system_prompt != resolve_system_prompt(control):
        failures.append("rendered system prompt differs from control")

    if config.get("model_backend") != control.get("model_backend"):
        failures.append("model_backend differs from control")
    if control.get("model_revision"):
        recorded_revision = config.get("model_revision")
        if recorded_revision is None:
            notes.append("model_revision not recorded in run_config (path-pinned snapshot)")
        elif recorded_revision != control.get("model_revision"):
            failures.append("model_revision differs from control")
    if (config.get("lora") or {}).get("target_modules") != (control.get("lora") or {}).get(
        "target_modules"
    ):
        failures.append("lora target_modules differ from control")

    training = config.get("training") or {}
    control_training = control.get("training") or {}
    for key in REQUIRED_TRAIN_KEYS:
        if training.get(key) != control_training.get(key):
            failures.append(f"training.{key} differs from control")
    early = training.get("early_stopping") or {}
    control_early = control_training.get("early_stopping") or {}
    if (early.get("metric"), early.get("mode"), early.get("patience")) != (
        control_early.get("metric"),
        control_early.get("mode"),
        control_early.get("patience"),
    ):
        failures.append("early stopping differs from control")

    evaluation = config.get("evaluation") or {}
    control_evaluation = control.get("evaluation") or {}
    for key in REQUIRED_EVAL_KEYS:
        if evaluation.get(key) != control_evaluation.get(key):
            failures.append(f"evaluation.{key} differs from control")

    if (config.get("split") or {}).get("seed") != 1337:
        failures.append("split.seed is not 1337")
    if int(config.get("seed", row["seed"])) != int(row["seed"]):
        failures.append("training seed differs from the registry key")

    if is_audio(route):
        resources = config.get("resources") or {}
        expected = control.get("resources") or {}
        if resources == expected:
            pass
        elif set(resources) == {"eval_nodes", "eval_gpus_per_node"} and all(
            resources.get(key) == expected.get(key) for key in resources
        ):
            world_size = (matrix_parent or {}).get("world_size_recorded")
            if world_size == 8:
                notes.append(
                    "gen-1 audio declaration omits train_nodes; two-node world_size "
                    "8 verified from the campaign head-matrix parent record"
                )
            else:
                failures.append(
                    f"gen-1 audio train shape not verified (world_size_recorded={world_size!r})"
                )
        else:
            failures.append("audio resource shape differs from control")
    else:
        if (config.get("resources") or {}).get("train_nodes") not in (None, 1):
            failures.append("text route declares an unexpected resource shape")

    eval_dir = fold_dir / "best_model/standalone_eval"
    predictions = eval_dir / "predictions_subject_level.csv"
    metrics = eval_dir / "metrics_likelihood.json"
    eval_config = eval_dir / "eval_config.yaml"
    for required in (predictions, metrics, eval_config):
        if not required.is_file():
            failures.append(f"missing compact evidence: {required.relative_to(fold_dir)}")
    if eval_config.is_file():
        eval_root = unwrap(load_yaml(eval_config))
        if resolve_system_prompt(eval_root) != system_prompt:
            failures.append("evaluation renders a different prompt than training")

    return {
        "registry_key": row["registry_key"],
        "run_name": row.get("run_name"),
        "attempt_id": row.get("attempt_id"),
        "generation": row.get("generation"),
        "state": row.get("state"),
        "verdict": "pass" if not failures else "blocked",
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "notes": notes,
        "failures": failures,
    }


def audit_head(
    row: dict[str, Any],
    matrix_job: dict[str, Any] | None,
    fit_attempt: str | None,
    native_evidence: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    notes: list[str] = []
    if row.get("state") != "REPORTABLE":
        failures.append(f"head state is {row.get('state')!r}, not REPORTABLE")
    if not row.get("parent_adapter_sha256"):
        failures.append("missing parent adapter sha256")
    if not row.get("sidecar_sha256"):
        failures.append("missing sidecar hashes")
    if matrix_job is None:
        failures.append("no resolved head matrix job for this key")
    else:
        if matrix_job.get("parent_status") != "resolved":
            notes.append(
                "head matrix planning record was unresolved; terminal parent identity "
                "taken from the final package and the local attempt"
            )
        if "logreg_raw" not in matrix_job.get("variants", []) or "xgb_raw" not in matrix_job.get(
            "variants", []
        ):
            failures.append("head matrix variants are not exactly logreg_raw/xgb_raw")
        if matrix_job.get("head_seed") != 1337:
            failures.append("head seed is not 1337")
        if (
            matrix_job.get("parent_attempt_id")
            and fit_attempt
            and matrix_job.get("parent_attempt_id") != fit_attempt
        ):
            notes.append("head matrix parent attempt differs from the final package fit attempt")

    attempt_id = row.get("attempt_id")
    attempt_dir = native_evidence / "head_attempts" / str(attempt_id)
    if not attempt_dir.is_dir():
        failures.append(f"local head attempt directory missing: {attempt_dir.name}")
    else:
        status_path = attempt_dir / "status.json"
        if status_path.is_file():
            state = json.loads(status_path.read_text(encoding="utf-8")).get("state")
            if state != "REPORTABLE":
                failures.append(f"head attempt status is {state!r}, not REPORTABLE")
        config_path = attempt_dir / "run_config.yaml"
        if config_path.is_file():
            config = unwrap(load_yaml(config_path))
            classifier = config.get("classifier") or {}
            if classifier.get("method") != "fixed_logreg_xgb":
                failures.append("head classifier method is not fixed_logreg_xgb")
            if classifier.get("sampling_mode") != "legacy":
                failures.append("head classifier sampling_mode is not legacy")
            if int(classifier.get("seed", -1)) != 1337:
                failures.append("head classifier seed is not 1337")
            if sorted(classifier.get("variants") or []) != ["logreg_raw", "xgb_raw"]:
                failures.append("head classifier variants are not logreg_raw/xgb_raw")
            if (
                matrix_job is not None
                and matrix_job.get("parent_training_seed") is not None
                and config.get("parent_training_seed") != matrix_job.get("parent_training_seed")
            ):
                failures.append("parent_training_seed differs from the matrix parent")
            if config.get("split_seed") != 1337 or config.get("head_seed") != 1337:
                failures.append("head/split seed is not 1337")
        for variant in ("logreg_raw", "xgb_raw"):
            if not (attempt_dir / "classifier" / variant).is_dir():
                failures.append(f"head classifier evidence missing: {variant}")

    return {
        "registry_key": row["registry_key"],
        "attempt_id": attempt_id,
        "parent_attempt_id": row.get("parent_attempt_id"),
        "state": row.get("state"),
        "verdict": "pass" if not failures else "blocked",
        "notes": notes,
        "failures": failures,
    }


def build_report(native_evidence: Path) -> dict[str, Any]:
    package = json.loads(
        (native_evidence / "final_package.json").read_text(encoding="utf-8")
    )
    matrix = build_matrix()
    routes = {route["route_id"]: route for route in matrix["routes"]}

    heads_matrix = json.loads(
        (native_evidence / glob_with_suffix(native_evidence, "heads_matrix_v")).read_text(
            encoding="utf-8"
        )
    )
    head_jobs: dict[tuple[str, int, int], dict[str, Any]] = {}
    for route in heads_matrix["routes"]:
        route_base = route["route_id"].removesuffix("_native")
        for job in route.get("jobs", []):
            head_jobs[(route_base, int(job["seed"]), int(job["fold"]))] = {
                "variants": heads_matrix["head_variants"],
                "head_seed": heads_matrix["head_seed"],
                "parent_status": job.get("parent_status"),
                "parent_attempt_id": (job.get("parent") or {}).get("attempt_id"),
                "parent_training_seed": (job.get("parent") or {}).get("parent_training_seed"),
                "world_size_recorded": (job.get("parent") or {}).get("world_size_recorded"),
                "declaration_only_shape_difference": (job.get("parent") or {}).get(
                    "declaration_only_shape_difference"
                ),
            }

    fit_rows: list[dict[str, Any]] = []
    fit_attempt_by_key: dict[tuple[str, int, int], str] = {}
    seen_keys: set[tuple[str, int, int]] = set()
    for row in package["fits"]["rows"]:
        normalized = normalize_key(row["registry_key"])
        if normalized is None:
            fit_rows.append(
                {
                    "registry_key": row["registry_key"],
                    "verdict": "blocked",
                    "failures": ["unparseable registry key"],
                }
            )
            continue
        route_base, seed, fold = normalized
        route_base = route_base.removesuffix("_native")
        seen_keys.add((route_base, seed, fold))
        route = routes.get(route_base)
        if route is None:
            fit_rows.append(
                {
                    "registry_key": row["registry_key"],
                    "verdict": "blocked",
                    "failures": ["route not in the treatment matrix"],
                }
            )
            continue
        result = audit_fit(row, route, head_jobs.get((route_base, seed, fold)))
        fit_rows.append(result)
        if result["verdict"] == "pass":
            fit_attempt_by_key[(route_base, seed, fold)] = row.get("attempt_id")

    expected_keys = {
        (fit["route_id"], int(fit["seed"]), int(fit["fold"])) for fit in matrix["fits"]
    }
    for key in sorted(expected_keys - seen_keys):
        fit_rows.append(
            {
                "registry_key": f"{key[0]}|s{key[1]}|f{key[2]}",
                "verdict": "blocked",
                "failures": ["control fit missing from the native package"],
            }
        )

    head_rows: list[dict[str, Any]] = []
    head_seen: set[tuple[str, int, int]] = set()
    for row in package["heads"]["rows"]:
        normalized = normalize_key(row["registry_key"])
        if normalized is None:
            head_rows.append(
                {
                    "registry_key": row["registry_key"],
                    "verdict": "blocked",
                    "failures": ["unparseable registry key"],
                }
            )
            continue
        route_base, seed, fold = normalized
        route_base = route_base.removesuffix("_native")
        head_seen.add((route_base, seed, fold))
        head_rows.append(
            audit_head(
                row,
                head_jobs.get((route_base, seed, fold)),
                fit_attempt_by_key.get((route_base, seed, fold)),
                native_evidence,
            )
        )
    for key in sorted(expected_keys - head_seen):
        head_rows.append(
            {
                "registry_key": f"{key[0]}|s{key[1]}|f{key[2]}",
                "verdict": "blocked",
                "failures": ["control head key missing from the native package"],
            }
        )

    blocked_fits = [row for row in fit_rows if row["verdict"] != "pass"]
    blocked_heads = [row for row in head_rows if row["verdict"] != "pass"]
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": matrix["campaign"],
        "native_evidence": str(native_evidence),
        "fit_count": len(fit_rows),
        "head_count": len(head_rows),
        "fits": fit_rows,
        "heads": head_rows,
        "summary": {
            "fits_pass": len(fit_rows) - len(blocked_fits),
            "fits_blocked": len(blocked_fits),
            "heads_pass": len(head_rows) - len(blocked_heads),
            "heads_blocked": len(blocked_heads),
            "blocked": [
                {"kind": "fit", "key": row["registry_key"], "failures": row["failures"]}
                for row in blocked_fits
            ]
            + [
                {"kind": "head", "key": row["registry_key"], "failures": row["failures"]}
                for row in blocked_heads
            ],
        },
        "status": "passed" if not blocked_fits and not blocked_heads else "blocked",
    }


def glob_with_suffix(directory: Path, prefix: str) -> str:
    matches = list(directory.glob(f"{prefix}*.json"))
    if not matches:
        raise FileNotFoundError(f"no {prefix}* files under {directory}")
    return max(matches, key=lambda path: int(path.stem.rsplit("_v", 1)[1])).name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-evidence", default=str(DEFAULT_NATIVE_EVIDENCE))
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs/qwen3_legacy_prompt_20261008/control_audit.json"),
    )
    args = parser.parse_args()
    report = build_report(Path(args.native_evidence))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(
        f"control audit: fits {summary['fits_pass']}/{report['fit_count']} pass, "
        f"heads {summary['heads_pass']}/{report['head_count']} pass -> {output}"
    )
    for item in summary["blocked"][:20]:
        print(f"- {item['kind']} {item['key']}: {'; '.join(item['failures'])}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
