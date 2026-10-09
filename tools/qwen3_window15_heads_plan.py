#!/usr/bin/env python3
"""Build the window15 treatment parent map and eligibility audit for the head stage.

The 126 treatment fits each get one logical head chain (both approved variants
``logreg_raw`` and ``xgb_raw``). This planner binds every key to its **exact
treatment attempt** from the dispatch ledger and the submission contract: the
fold directory, the treatment config under ``configs/experiments/window15/``
and the recorded attempt id. Control-30 runs are never substituted: an entry
whose attempt or config does not belong to the window15 treatment matrix is
refused, and the dataset/modality of every key comes from the treatment config
itself.

Eligibility (audit only; the remote planner independently re-checks lifecycle,
terminal jobs, config/prompt hashes and adapter files):

- ``eligible``: local fold evidence shows LOCALLY_VALIDATED or REPORTABLE;
- ``waiting_validation``: fold evidence shows COMPLETED_ON_MN5 or SYNCED_LOCALLY;
- ``waiting_training``: attempt submitted, no local fold evidence yet;
- ``blocked_failed``: terminal failed/cancelled attempt.

``--check`` recomputes and compares against the written outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

PARENT_MAP_SCHEMA = "audiollm.qwen3_heads_parent_map.v1"
AUDIT_SCHEMA = "audiollm.qwen3_window15_heads_plan_audit.v1"
CONTRACT = LANE / "outputs/qwen3_window15_20261008/contracts/treatment_contract.json"
LEDGER = LANE / "outputs/qwen3_window15_20261008/submissions.jsonl"
EXP_SUBMIT = LANE / "outputs/exp_submit"
OUT_DIR = LANE / "outputs/qwen3_window15_20261008/heads"
RUN_ROOT = LANE / "output_model/qwen3_window15_20261008"
TREATMENT_CONFIG_PREFIX = "configs/experiments/window15/"
ELIGIBLE_STATES = {"LOCALLY_VALIDATED", "REPORTABLE"}
WAITING_VALIDATION_STATES = {"COMPLETED_ON_MN5", "SYNCED_LOCALLY"}
FAILED_STATES = {"FAILED", "CANCELLED", "SUPERSEDED", "REJECTED"}


class PlanError(RuntimeError):
    pass


def _read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"unreadable JSON: {path}: {exc}") from exc


def _ledger_by_key(ledger_path: Path) -> dict[str, dict]:
    records = {}
    if not ledger_path.is_file():
        return records
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "submitted" and record.get("key"):
            records[record["key"]] = record
    return records


def _fold_evidence_state(fold_dir: Path) -> str | None:
    status = _read_json(fold_dir / "status.json")
    if isinstance(status, dict) and status.get("state"):
        return str(status["state"])
    return None


def _config_identity(config_path: str) -> dict[str, str]:
    payload = yaml.safe_load((LANE / config_path).read_text(encoding="utf-8")) or {}
    data = payload.get("data") or {}
    use_audio = bool(data.get("use_audio", False))
    use_text = bool(data.get("use_text", False))
    modality = "audio_text" if (use_audio and use_text) else "audio_only"
    return {
        "dataset": str(payload["dataset"]),
        "modality": modality,
        "manifest_variant": str(payload.get("manifest_variant") or ""),
        "segment_seconds": str(data.get("segment_seconds") or ""),
        "processor_min_audio_samples": str(data.get("processor_min_audio_samples") or ""),
    }


def build_plan(
    contract_path: Path = CONTRACT,
    ledger_path: Path = LEDGER,
    exp_submit_dir: Path = EXP_SUBMIT,
) -> tuple[dict, dict]:
    contract = _read_json(contract_path)
    if not contract:
        raise PlanError(f"treatment contract missing: {contract_path}")
    rows = contract.get("rows") or []
    if len(rows) != 126:
        raise PlanError(f"treatment contract must carry 126 rows, found {len(rows)}")
    submitted = _ledger_by_key(ledger_path)

    entries: list[dict] = []
    audit_keys: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row["registry_key"])
        if key in seen:
            raise PlanError(f"duplicate treatment key: {key}")
        seen.add(key)
        config = str(row["treatment_config"])
        if not config.startswith(TREATMENT_CONFIG_PREFIX):
            raise PlanError(f"{key}: config {config!r} is not a window15 treatment config")
        identity = _config_identity(config)
        if identity["processor_min_audio_samples"] != "201":
            raise PlanError(f"{key}: treatment config is missing processor_min_audio_samples=201")
        record = submitted.get(key)
        attempt_id = str(record.get("attempt_id")) if record and record.get("attempt_id") else None
        contract_payload = _read_json(exp_submit_dir / attempt_id / "contract.json") if attempt_id else None
        remote_fold_dir = None
        if contract_payload is not None:
            if str(contract_payload.get("dataset")) != identity["dataset"]:
                raise PlanError(f"{key}: submission contract dataset disagrees with the treatment config")
            remote_config = str(contract_payload.get("config_path_remote") or "")
            if remote_config and f"/{TREATMENT_CONFIG_PREFIX}" not in remote_config:
                raise PlanError(f"{key}: submission contract config is not a window15 treatment config")
            if str(contract_payload.get("deployment_id", "")).startswith("feat-qwen3-window15-"):
                remote_fold_dir = str(contract_payload.get("fold_dir") or "") or None
            if remote_fold_dir is None:
                raise PlanError(f"{key}: submission contract has no treatment fold dir")

        run_name = str(row.get("planned_run_name") or "")
        local_candidates = sorted(
            RUN_ROOT.glob(
                f"{identity['modality']}/{identity['dataset']}/{run_name}/fold_{int(row['fold'])}"
            )
        )
        local_fold = local_candidates[0] if len(local_candidates) == 1 else None
        state = _fold_evidence_state(local_fold) if local_fold is not None else None
        if attempt_id is None:
            status = "waiting_training"
        elif state in ELIGIBLE_STATES:
            status = "eligible"
        elif state in WAITING_VALIDATION_STATES:
            status = "waiting_validation"
        elif state in FAILED_STATES:
            status = "blocked_failed"
        else:
            status = "waiting_training"
        audit_keys.append(
            {
                "key": key,
                "route_id": str(row["route_id"]),
                "seed": int(row["seed"]),
                "fold": int(row["fold"]),
                "config": config,
                "dataset": identity["dataset"],
                "modality": identity["modality"],
                "run_name": run_name,
                "attempt_id": attempt_id,
                "remote_fold_dir": remote_fold_dir,
                "local_fold_dir": str(local_fold) if local_fold is not None else None,
                "evidence_state": state,
                "status": status,
            }
        )
        if attempt_id is not None and remote_fold_dir:
            entries.append(
                {
                    "route_id": str(row["route_id"]),
                    "parent_training_seed": int(row["seed"]),
                    "fold": int(row["fold"]),
                    "config": config,
                    "fold_dir": remote_fold_dir,
                    "parent_attempt_id": attempt_id,
                }
            )
    parent_map = {
        "schema_version": PARENT_MAP_SCHEMA,
        "campaign": "qwen3_window15_20261008",
        "entries": entries,
    }
    audit = {
        "schema_version": AUDIT_SCHEMA,
        "campaign": "qwen3_window15_20261008",
        "expected_keys": 126,
        "keys_total": len(audit_keys),
        "entries_in_parent_map": len(entries),
        "status_counts": {
            status: sum(1 for item in audit_keys if item["status"] == status)
            for status in ("eligible", "waiting_validation", "waiting_training", "blocked_failed")
        },
        "variant_policy": ["logreg_raw", "xgb_raw"],
        "control_substitution": "refused: entries bind only to exact treatment attempts and window15 configs",
        "keys": sorted(audit_keys, key=lambda item: item["key"]),
    }
    return parent_map, audit


def _treatment_routes(contract_path: Path = CONTRACT) -> list[dict]:
    """Lane route source for the generic planner: treatment configs only."""
    contract = _read_json(contract_path)
    if not contract:
        raise PlanError(f"treatment contract missing: {contract_path}")
    rows = contract.get("rows") or []
    routes: dict[str, dict] = {}
    for row in rows:
        route_id = str(row["route_id"])
        config = str(row["treatment_config"])
        if not config.startswith(TREATMENT_CONFIG_PREFIX):
            raise PlanError(f"{route_id}: config {config!r} is not a window15 treatment config")
        if route_id in routes:
            routes[route_id]["folds"].add(int(row["fold"]))
            continue
        identity = _config_identity(config)
        payload = yaml.safe_load((LANE / config).read_text(encoding="utf-8")) or {}
        routes[route_id] = {
            "route_id": route_id,
            "config": config,
            "dataset": identity["dataset"],
            "modality": identity["modality"],
            "language": "native",
            "model_backend": str(payload.get("model_backend") or "qwen3omni"),
            "folds": {int(row["fold"])},
        }
    return [
        {**route, "folds": sorted(route["folds"])}
        for route in (routes[key] for key in sorted(routes))
    ]


def eligible_parent_map(parent_map: dict, audit: dict) -> dict:
    """Keep only entries whose key is audit-eligible (validated and collectable).

    The generic planner raises on an explicit parent that is not yet eligible;
    in-flight fits are therefore excluded here and appear in the matrix as
    waiting_for_checkpoint until their evidence is collected and validated.
    """
    eligible_keys = {
        item["key"] for item in audit.get("keys", []) if item.get("status") == "eligible"
    }
    entries = [
        entry
        for entry in parent_map.get("entries", [])
        if f"{entry['route_id']}|{entry['parent_training_seed']}|{entry['fold']}" in eligible_keys
    ]
    return {**parent_map, "entries": entries}


def build_matrix_mode(
    parent_map_path: Path,
    out_path: Path,
    cache_root: Path,
    seeds: list[int],
    contract_path: Path = CONTRACT,
    campaign_root: Path | None = None,
    ledger_path: Path = LEDGER,
) -> dict:
    """Run the generic planner with the lane's treatment route source.

    The generic planner hardcodes the control selection map; here it is given
    the window15 treatment routes so the resolved parents are treatment checkpoints
    only. The parent map is first filtered to audit-eligible keys (validated
    parents) because the generic planner fails closed on an explicit ineligible
    parent. Everything else (eligibility, prompt/config matching, adapter
    hashing, fail-closed waiting states) is the generic implementation.
    """
    import tools.qwen3_heads_matrix as heads_matrix  # noqa: PLC0415

    parent_map = _read_json(parent_map_path) or {"entries": []}
    _, audit = build_plan(contract_path, ledger_path)
    filtered = eligible_parent_map(parent_map, audit)
    filtered_path = out_path.with_suffix(".parent_map.eligible.json")
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_path.write_text(json.dumps(filtered, indent=1, sort_keys=True), encoding="utf-8")

    routes = _treatment_routes(contract_path)
    heads_matrix.build_selection_map = lambda: {"routes": routes}
    matrix = heads_matrix.build_matrix(
        seeds=seeds,
        scan_roots=[],
        campaign_root=campaign_root or RUN_ROOT,
        parent_map_path=filtered_path,
        cache_root=cache_root,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(matrix, indent=1, sort_keys=True), encoding="utf-8")
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--build-matrix",
        action="store_true",
        help="run the generic planner with the treatment route source (cluster-side paths)",
    )
    parser.add_argument("--parent-map", type=Path, default=None)
    parser.add_argument("--matrix-out", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--campaign-root", type=Path, default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    args = parser.parse_args()

    if args.build_matrix:
        parent_map_path = args.parent_map or (args.out_dir / "parent_map.json")
        matrix_out = args.matrix_out or (args.out_dir / "window15_heads_matrix.json")
        cache_root = args.cache_root or (LANE / "head_cache")
        seeds = args.seed or [7, 1337, 2024]
        try:
            matrix = build_matrix_mode(
                parent_map_path,
                matrix_out,
                cache_root,
                seeds,
                contract_path=args.contract,
                campaign_root=args.campaign_root,
            )
        except Exception as exc:  # planner errors are fail-closed
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        summary = matrix.get("summary") or {}
        print(json.dumps({"status": "ok", "matrix": str(matrix_out), "summary": summary}, sort_keys=True))
        return 0

    try:
        parent_map, audit = build_plan(args.contract, args.ledger)
    except PlanError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if audit["keys_total"] != 126:
        print(f"REFUSED: expected 126 treatment keys, found {audit['keys_total']}", file=sys.stderr)
        return 2

    out_dir = args.out_dir
    map_path = out_dir / "parent_map.json"
    audit_path = out_dir / "heads_plan_audit.json"
    if args.check:
        if not map_path.is_file() or not audit_path.is_file():
            print("REFUSED: no written plan to check", file=sys.stderr)
            return 2
        if json.loads(map_path.read_text(encoding="utf-8")) != parent_map:
            print("REFUSED: parent map drifted since write", file=sys.stderr)
            return 2
        if json.loads(audit_path.read_text(encoding="utf-8")) != audit:
            print("REFUSED: plan audit drifted since write", file=sys.stderr)
            return 2
        print(json.dumps({"status": "ok", "checked": True, "entries": len(parent_map["entries"]), "status_counts": audit["status_counts"]}))
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(parent_map, indent=1, sort_keys=True), encoding="utf-8")
    audit_path.write_text(json.dumps(audit, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "parent_map": str(map_path), "audit": str(audit_path), "entries": len(parent_map["entries"]), "status_counts": audit["status_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
