#!/usr/bin/env python
"""Audit the 126 reusable 30-second Native audio controls for the window15 lane.

Reads the completed campaign's evidence (read-only) and the current selection
map, then verifies for every control key: the route config matches the current
canonical contract, the fit is REPORTABLE with a recorded run_config hash, and
the matching logreg_raw/xgb_raw head key is REPORTABLE. Writes a machine
readable control audit plus the frozen control contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

from tools.qwen3_multiseed_plan import build_selection_map  # noqa: E402

SEEDS = (7, 1337, 2024)


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-evidence-root", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=LANE / "outputs/qwen3_window15_20261008/contracts/control_audit.json"
    )
    args = parser.parse_args()

    evidence = args.control_evidence_root
    package = read_json(evidence / "final_package.json")
    parent_map = read_json(evidence / "parent_map_v1.json")
    recompute = read_json(evidence / "strict_recompute_verification.json")
    training_jobs = read_json(evidence / "training_jobs.json").get("jobs", [])

    selection = build_selection_map()
    audio_routes = {
        str(route["route_id"]): str(route["config"])
        for route in selection["routes"]
        if route.get("language") == "native"
        and route.get("modality") in ("audio_only", "audio_text")
    }
    if len(audio_routes) != 10:
        raise SystemExit(f"expected 10 native audio routes, found {len(audio_routes)}")

    fits = {
        str(row["registry_key"]): row
        for row in package.get("fits", {}).get("rows", [])
        if str(row.get("route_id")) in audio_routes
    }
    heads = {
        str(row["registry_key"]): row
        for row in package.get("heads", {}).get("rows", [])
        if str(row.get("registry_key", "")).split("|")[0] in audio_routes
    }
    head_recompute = [
        row
        for row in recompute.get("heads", {}).get("rows", [])
        if str(row.get("registry_key", "")).split("|")[0] in audio_routes
    ]
    parents = {
        (str(entry["route_id"]), int(entry["fold"])): entry
        for entry in parent_map.get("entries", [])
        if str(entry.get("route_id")) in audio_routes
    }
    # Exact-parent config evidence: seed 1337 from the parent map, seeds 7/2024
    # from the completed campaign's training jobs.
    config_by_key: dict[str, str] = {}
    for (route_id, fold), entry in parents.items():
        config_by_key[f"{route_id}|1337|{fold}"] = str(entry.get("config") or "")
    for job in training_jobs:
        if str(job.get("route_id")) in audio_routes:
            config_by_key[str(job["key"])] = str(job.get("config") or "")

    rows = []
    blocked = []

    def fit_lookup(route_id: str, seed: int, fold: int) -> dict | None:
        for candidate in (f"{route_id}|s{seed}|f{fold}", f"{route_id}|{seed}|{fold}"):
            if candidate in fits:
                return fits[candidate]
        return None

    def config_lookup(route_id: str, seed: int, fold: int) -> str | None:
        for candidate in (f"{route_id}|s{seed}|f{fold}", f"{route_id}|{seed}|{fold}"):
            if candidate in config_by_key:
                return config_by_key[candidate]
        return None

    for route_id, config in sorted(audio_routes.items()):
        folds = sorted(
            {
                int(str(key).split("|")[2].lstrip("sf"))
                for key in fits
                if str(key).split("|")[0] == route_id
            }
            | {fold for (r, fold) in parents if r == route_id}
        )
        for seed in SEEDS:
            for fold in folds:
                key = f"{route_id}|{seed}|{fold}"
                fit = fit_lookup(route_id, seed, fold)
                if fit is None:
                    blocked.append({"registry_key": key, "reason": "no completed-campaign fit evidence"})
                    continue
                issues = []
                if fit.get("state") != "REPORTABLE":
                    issues.append(f"fit state {fit.get('state')!r} is not REPORTABLE")
                recorded_config = config_lookup(route_id, seed, fold)
                if not recorded_config:
                    issues.append("no exact-parent config evidence for this key")
                elif recorded_config != config:
                    issues.append(f"recorded config {recorded_config!r} != current contract {config!r}")
                if not fit.get("run_config_sha256"):
                    issues.append("missing run_config sha256")
                head = heads.get(key)
                if head is None:
                    issues.append("no head evidence for this key")
                elif head.get("state") != "REPORTABLE":
                    issues.append(f"head state {head.get('state')!r} is not REPORTABLE")
                variants = {
                    str(row.get("variant"))
                    for row in head_recompute
                    if str(row.get("registry_key")) == key and row.get("variant")
                }
                missing_variants = sorted({"logreg_raw", "xgb_raw"} - variants)
                if missing_variants:
                    issues.append(f"head variants missing: {missing_variants}")
                entry = {
                    "registry_key": key,
                    "route_id": route_id,
                    "seed": seed,
                    "fold": fold,
                    "config": config,
                    "attempt_id": fit.get("attempt_id"),
                    "run_name": fit.get("run_name"),
                    "fold_dir": fit.get("fold_dir"),
                    "run_config_sha256": fit.get("run_config_sha256"),
                    "fit_state": fit.get("state"),
                    "head_attempt_id": (head or {}).get("attempt_id"),
                    "head_variants": sorted(variants),
                    "issues": issues,
                }
                rows.append(entry)
                if issues:
                    blocked.append({"registry_key": key, "issues": issues})

    payload = {
        "schema_version": "audiollm.qwen3_window15_control_audit.v1",
        "campaign": "qwen3_window15_20261008",
        "control_evidence_root": str(evidence),
        "routes": audio_routes,
        "expected_controls": 126,
        "audited_controls": len(rows),
        "blocked": blocked,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "audited": len(rows),
                "blocked": len(blocked),
                "routes": len(audio_routes),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
