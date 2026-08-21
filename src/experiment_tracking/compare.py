"""Group-scoped qualified comparison and integration planning.

Comparison reads local REPORTABLE evidence only, requires every qualifier,
refuses mixed protocols/folds/seeds/views/aggregations, and emits a
deterministic audit (sorted keys, no timestamps). Integration planning checks
branch ancestry, Git conflicts (read-only merge-tree), shared contract files,
and semantic-conflict candidates.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Files whose changes can alter scientific/runtime semantics across lanes.
CONTRACT_FILE_PREFIXES = (
    "configs/",
    "src/data/",
    "src/evaluate.py",
    "src/train.py",
    "src/utils.py",
    "src/experiment_tracking/",
    "scripts/run_train_slurm.sh",
    "scripts/run_eval_slurm.sh",
    "scripts/submit_train_and_eval.sh",
)

REQUIRED_QUALIFIERS = (
    "dataset", "metric", "namespace", "backend", "view", "aggregation", "tie_rule",
)


class ComparisonError(RuntimeError):
    """Raised when a comparison or integration plan must fail closed."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_attempt_evidence(project_root: Path, attempt_id: str) -> dict[str, Any]:
    """Resolve an attempt's local fold evidence via its recorded contract."""
    contract_path = project_root / "outputs" / "exp_submit" / attempt_id / "contract.json"
    if not contract_path.is_file():
        raise ComparisonError(f"no recorded submission contract for attempt {attempt_id}")
    contract = _read_json(contract_path)
    fold_dir = project_root / contract["local_fold_rel"]
    status_path = fold_dir / "status.json"
    evals_path = fold_dir / "evaluations.json"
    jobs_path = fold_dir / "jobs.jsonl"
    for p in (status_path, evals_path, jobs_path):
        if not p.is_file():
            raise ComparisonError(f"attempt {attempt_id} missing {p.name} at {fold_dir}")
    status = _read_json(status_path)
    if status.get("state") != "REPORTABLE":
        raise ComparisonError(
            f"attempt {attempt_id} is {status.get('state')!r}, not REPORTABLE; "
            "only reportable attempts may enter a comparison"
        )
    evaluations = _read_json(evals_path).get("evaluations", [])
    return {
        "attempt_id": attempt_id,
        "group_id": contract.get("group_id") or contract.get("experiment_id"),
        "fold": contract.get("fold"),
        "seed": contract.get("seed"),
        "run_name": contract.get("run_name"),
        "state": status.get("state"),
        "evaluations": evaluations,
        "fold_dir": str(fold_dir),
    }


def _metric_value(evidence: dict[str, Any], metric: str, namespace: str) -> float:
    for evaluation in evidence["evaluations"]:
        if evaluation.get("metric_namespace") != namespace:
            continue
        for m in evaluation.get("metrics", []):
            if m.get("name") == metric and m.get("value") is not None:
                return float(m["value"])
    raise ComparisonError(
        f"attempt {evidence['attempt_id']} has no {namespace}/{metric} value"
    )


def compare_group(
    project_root: Path,
    *,
    group_id: str,
    attempt_ids: list[str],
    dataset: str,
    metric: str,
    namespace: str,
    backend: str,
    view: str,
    aggregation: str,
    tie_rule: str,
    tie_tolerance: float = 0.0,
) -> dict[str, Any]:
    if len(attempt_ids) < 2:
        raise ComparisonError("comparison requires at least two explicit attempts")
    if tie_rule not in ("max", "min"):
        raise ComparisonError("tie_rule must be 'max' or 'min'")

    evidences = [load_attempt_evidence(project_root, a) for a in sorted(attempt_ids)]

    for ev in evidences:
        if group_id and ev["group_id"] != group_id:
            raise ComparisonError(
                f"attempt {ev['attempt_id']} belongs to group {ev['group_id']!r}, not {group_id!r}"
            )
    # Homogeneity requirements.
    for key in ("fold", "seed"):
        values = {json.dumps(ev.get(key), sort_keys=True) for ev in evidences}
        if len(values) > 1:
            raise ComparisonError(f"mixed {key} across attempts: {sorted(values)}")
    qualifier_sets = []
    for ev in evidences:
        found = []
        for evaluation in ev["evaluations"]:
            found.append((
                evaluation.get("dataset"), evaluation.get("backend"),
                evaluation.get("evaluation_view"), evaluation.get("aggregation"),
                evaluation.get("metric_namespace"), evaluation.get("checkpoint_role"),
                evaluation.get("split_protocol"),
            ))
        qualifier_sets.append({json.dumps(t, sort_keys=True) for t in found})
    expected = {
        json.dumps((dataset, backend, view, aggregation, namespace, "best_model",
                    evidences[0]["evaluations"][0].get("split_protocol")), sort_keys=True)
    }
    for ev, qs in zip(evidences, qualifier_sets):
        if not qs:
            raise ComparisonError(f"attempt {ev['attempt_id']} has no evaluations")
        if qs != expected:
            raise ComparisonError(
                f"attempt {ev['attempt_id']} qualifiers {qs} do not match the "
                f"comparison contract {expected}"
            )

    scores = {
        ev["attempt_id"]: _metric_value(ev, metric, namespace) for ev in evidences
    }
    best = max(scores.values()) if tie_rule == "max" else min(scores.values())
    tied = sorted(a for a, v in scores.items() if abs(v - best) <= tie_tolerance)
    unambiguous = len(tied) == 1
    audit = {
        "schema_version": "audiollm.comparison_audit.v1",
        "group_id": group_id,
        "attempts": sorted(attempt_ids),
        "qualifiers": {
            "dataset": dataset, "metric": metric, "namespace": namespace,
            "backend": backend, "view": view, "aggregation": aggregation,
            "checkpoint_role": "best_model",
        },
        "tie_rule": tie_rule,
        "tie_tolerance": tie_tolerance,
        "scores": {k: scores[k] for k in sorted(scores)},
        "tied_winners": tied,
        "unambiguous_winner": tied[0] if unambiguous else None,
        "automatic_selection_authorized": unambiguous,
    }
    return audit


def write_comparison_audit(audit: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    output_path.write_text(payload, encoding="utf-8")
    return output_path


def _git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + args, cwd=str(cwd) if cwd else None,
                          capture_output=True, text=True)


def _changed_files(base: str, branch: str, cwd: Path | None = None) -> set[str]:
    proc = _git(["diff", "--name-only", f"{base}...{branch}"], cwd=cwd)
    if proc.returncode != 0:
        raise ComparisonError(f"git diff failed: {proc.stderr.strip()}")
    return {l for l in proc.stdout.splitlines() if l.strip()}


def plan_integration(
    repo: Path,
    *,
    branch_a: str,
    branch_b: str,
    base: str = "origin/main",
) -> dict[str, Any]:
    """Read-only integration plan: ancestry, Git conflicts, semantic candidates."""
    for ref in (branch_a, branch_b, base):
        if _git(["rev-parse", "--verify", ref], cwd=repo).returncode != 0:
            raise ComparisonError(f"reference not found: {ref}")

    merge_base_ab = _git(["merge-base", branch_a, branch_b], cwd=repo).stdout.strip()
    base_a = _git(["merge-base", base, branch_a], cwd=repo).stdout.strip()
    base_b = _git(["merge-base", base, branch_b], cwd=repo).stdout.strip()

    # Read-only Git conflict detection via merge-tree (version-agnostic).
    git_conflicts: list[str] = []
    merge_tree = _git(["merge-tree", "--write-tree", branch_a, branch_b], cwd=repo)
    stderr_l = (merge_tree.stderr or "").lower()
    unsupported = (
        merge_tree.returncode == 129
        or "unknown option" in stderr_l
        or "unknown rev" in stderr_l
    )
    if unsupported:
        base_ref = _git(["merge-base", branch_a, branch_b], cwd=repo).stdout.strip()
        legacy = _git(["merge-tree", base_ref, branch_a, branch_b], cwd=repo)
        if legacy.returncode == 0:
            for line in legacy.stdout.splitlines():
                if "<<<<<<<" in line:
                    git_conflicts.append(line.strip())
    elif merge_tree.returncode != 0:
        for line in merge_tree.stdout.splitlines():
            line = line.strip()
            if line and "CONFLICT" in line.upper():
                git_conflicts.append(line)
        if not git_conflicts:
            git_conflicts = [merge_tree.stderr.strip() or "merge-tree reported conflicts"]

    files_a = _changed_files(base, branch_a, cwd=repo)
    files_b = _changed_files(base, branch_b, cwd=repo)
    overlapping = sorted(files_a & files_b)
    semantic_candidates = sorted(
        f for f in overlapping
        if any(f == p or f.startswith(p) for p in CONTRACT_FILE_PREFIXES)
    )

    return {
        "schema_version": "audiollm.integration_plan.v1",
        "branch_a": branch_a,
        "branch_b": branch_b,
        "base": base,
        "merge_base_ab": merge_base_ab,
        "stacked": bool(merge_base_ab and merge_base_ab in (
            _git(["rev-parse", branch_a], cwd=repo).stdout.strip(),
            _git(["rev-parse", branch_b], cwd=repo).stdout.strip(),
        )),
        "shares_base": base_a == base_b,
        "git_conflicts": git_conflicts,
        "overlapping_files": overlapping,
        "semantic_conflict_candidates": semantic_candidates,
        "cross_feature_tests_required": bool(semantic_candidates),
        "automatic_merge_authorized": False,
        "decision": (
            "human/orchestrator decision required before merging complementary lanes"
            if semantic_candidates or git_conflicts
            else "no conflicts detected; normal review still applies"
        ),
    }
