#!/usr/bin/env python3
"""Lane-owned treatment-only planner for the 189 Native standalone head chains.

This is the Worker 4 replacement for the shared generic head inventory
(``tools/qwen3_heads_matrix.py`` + ``tools/qwen3_heads_dispatch.py plan``).
The generic inventory scans control-config run roots and includes English
cells; it must never drive this lane's head dispatch. This planner is bound to
the frozen treatment matrix (``outputs/<campaign>/matrix.json``, the approved
189 Native fits) and emits a compact
``audiollm.qwen3_heads_matrix.v1`` payload that the shared dispatch
``plan``/``submit`` commands consume unchanged.

For every cell whose training leg has durable local validation (exact attempt
in the append-only submission ledger plus a validation receipt or a
LOCALLY_VALIDATED lifecycle proof), the planner proves the parent chain
*in place on MN5 GPFS* (one batched SSH read; adapters are never downloaded):

- ``run_config.yaml`` / ``metadata.json`` / ``status.json`` carry the exact
  validated attempt id (and the expected fold);
- the recorded ``prompt_context.system_prompt_sha256`` equals a fresh local
  render of the tracked treatment config's inline legacy system prompt (and
  the route's pinned ``prompt_system_sha256``);
- the recorded ``split_metadata_hash`` equals the pinned manifest-contract
  ``folds_json_sha256`` for the dataset, and the recorded ``manifest_hash``
  equals the pinned ``recorded_manifest_hash``; the manifest file actually
  referenced by ``run_config.yaml`` is hashed in place and must equal the
  pinned ``manifest_jsonl_sha256``;
- ``best_model/adapter_config.json`` and ``best_model/adapter_model.safetensors``
  are hashed in place (these hashes travel into the head attempt context and
  are re-verified by the submission path);
- ``logs/split_used.json`` is hashed in place as the split fingerprint.

Cells without a validated parent are emitted as ``waiting_for_checkpoint``;
a cell that fails any identity check is emitted as waiting with an explicit
reason and is never silently promoted. The two released epoch-1 parents keep
their zero-delta qualifier in ``selection``/``selection_reason``.

The emitted payload always contains exactly the approved 189 Native keys (no
English, control, or peer cells) and is validated here before writing
(``--check`` re-runs the same validation on an existing file). Nothing is
written on the cluster and no adapter bytes are fetched locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

LANE = Path(__file__).resolve().parents[1]
if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

from src.data.prompt_context import resolve_system_prompt  # noqa: E402
from src.experiment_tracking.deployment import DEFAULT_TRANSFER_HOST  # noqa: E402
from tools import collect_legacy_prompt_cells as collector  # noqa: E402
from tools import qwen3_legacy_prompt_heads_guard as guard  # noqa: E402

SCHEMA_VERSION = "audiollm.qwen3_heads_matrix.v1"
CAMPAIGN = "qwen3_legacy_prompt_20261008"
EVIDENCE = LANE / "outputs/qwen3_legacy_prompt_20261008"
MATRIX = EVIDENCE / "matrix.json"
LEDGER = EVIDENCE / "submissions.jsonl"
RECEIPTS = EVIDENCE / "validation_receipts.jsonl"
CONTRACT = LANE / "experiments/definitions/qwen3_legacy_prompt_manifest_contract.json"
RUN_ROOT = LANE / "output_model/qwen3_legacy_prompt_20261008"
RUNTIME_ROOT = Path(
    "/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime/"
    "feat-qwen3-legacy-prompt-20261008"
)
DEFAULT_CACHE_ROOT = str(RUNTIME_ROOT / "heads_cache")
PERMANENT_OUTPUT_ROOT = Path(
    "/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/output_model"
)

# Released epoch-1 parents whose saved adapter provably carries no fine-tuning
# delta: the first optimizer step ran at LR 0 (warmup_steps=1), so both seed-7
# folds saved the same initial LoRA draw with an all-zero ``lora_B``. The
# qualifier must travel with every head chain built on them so no downstream
# report can claim a nonzero fine-tuning effect for these two cells.
ZERO_DELTA_EPOCH1_QUALIFIERS: dict[str, str] = {
    "d3tec_text_only|s7|f1": (
        "epoch-1 checkpoint; zero LoRA delta (first optimizer step ran at LR 0, "
        "warmup_steps=1); read as the untuned-base condition, not as fine-tuning"
    ),
    "d3tec_text_only|s7|f2": (
        "epoch-1 checkpoint; zero LoRA delta (first optimizer step ran at LR 0, "
        "warmup_steps=1); read as the untuned-base condition, not as fine-tuning"
    ),
}

# Per-dataset split-file pins that are not the folds file pinned in the
# manifest contract. DAIC packed30 training reads the subject-partition file
# ``daic_subject_partitions.json`` (the contract additionally pins the derived
# ``daic_folds.json``). The lane value below was verified in place on MN5 GPFS
# on 2026-10-08 and was recorded identically by the three independent DAIC
# seed runs (7/1337/2024); it is never taken from a run sidecar.
SPLIT_FILE_PINS: dict[tuple[str, str], str] = {
    ("daic", "daic_subject_partitions.json"): (
        "441333e0c88845eeacba9ea5355a8920cdd1f70e8cf7a7c15b9547b46da51473"
    ),
}
# Remote status sidecars on GPFS are known to be stale at ``RUNNING`` even for
# attempts whose Slurm jobs and evaluation later completed; the completion
# authority for this lane is the ledger job pair plus the validation
# receipt/lifecycle proof that gated the cell. The sidecar must still carry
# the exact attempt/fold identity and a known lifecycle state; FAILED,
# CANCELLED, or unknown states contradict the validated attempt and wait.
ACCEPTED_REMOTE_STATES = {
    "RUNNING",
    "COMPLETED_ON_MN5",
    "SYNCED_LOCALLY",
    "LOCALLY_VALIDATED",
    "REPORTABLE",
}


class PlannerError(RuntimeError):
    """A planner contract was violated or a proof could not be assembled."""


@dataclass
class RemoteCellEvidence:
    """Raw in-place evidence read from one remote fold directory."""

    exists: bool = False
    run_config_text: str = ""
    metadata_text: str = ""
    status_text: str = ""
    adapter_config_sha256: str = ""
    adapter_model_sha256: str = ""
    split_metadata_sha256: str = ""
    split_used_sha256: str = ""
    manifest_sha256: str = ""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fold_dir_for(fit: dict[str, Any], permanent_root: Path = PERMANENT_OUTPUT_ROOT) -> Path:
    return (
        permanent_root
        / CAMPAIGN
        / str(fit["modality"])
        / str(fit["dataset_dir"])
        / str(fit["run_name"])
        / f"fold_{int(fit['fold'])}"
    )


def expected_prompt_sha(config_path: Path) -> str:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return sha256_text(resolve_system_prompt(config))


def build_remote_script(cells: list[tuple[str, Path]]) -> str:
    """One batched read; every hash is computed on the remote host."""
    lines = ["set -u"]
    for key, fold in cells:
        quoted = shlex.quote(str(fold))
        lines += [
            f"echo '===CELL {key}==='",
            f"d={quoted}",
            'if [ -d "$d" ]; then echo DIR=1; else echo DIR=0; fi',
            'if [ -f "$d/run_config.yaml" ]; then',
            '  echo RUNCONFIG_BEGIN; cat "$d/run_config.yaml"; echo RUNCONFIG_END',
            '  mp=$(grep -m1 "^manifest_path:[[:space:]]*" "$d/run_config.yaml" | '
            "sed -e 's/^manifest_path:[[:space:]]*//' -e 's/^\"//' -e 's/\"$//')",
            '  if [ -n "$mp" ] && [ -f "$mp" ]; then',
            '    echo MANIFEST_SHA=$(sha256sum "$mp" | cut -d" " -f1)',
            "  fi",
            '  sp=$(grep -m1 "^split_metadata_path:[[:space:]]*" "$d/run_config.yaml" | '
            "sed -e 's/^split_metadata_path:[[:space:]]*//' -e 's/^\"//' -e 's/\"$//')",
            '  if [ -n "$sp" ] && [ -f "$sp" ]; then',
            '    echo SPLIT_METADATA_SHA=$(sha256sum "$sp" | cut -d" " -f1)',
            "  fi",
            "fi",
            'if [ -f "$d/metadata.json" ]; then',
            '  echo METADATA_BEGIN; cat "$d/metadata.json"; echo METADATA_END',
            "fi",
            'if [ -f "$d/status.json" ]; then',
            '  echo STATUS_BEGIN; cat "$d/status.json"; echo STATUS_END',
            "fi",
            'if [ -f "$d/best_model/adapter_config.json" ]; then',
            '  echo ADAPTER_CONFIG_SHA=$(sha256sum "$d/best_model/adapter_config.json" | cut -d" " -f1)',
            "fi",
            'if [ -f "$d/best_model/adapter_model.safetensors" ]; then',
            '  echo ADAPTER_MODEL_SHA=$(sha256sum "$d/best_model/adapter_model.safetensors" | cut -d" " -f1)',
            "fi",
            'if [ -f "$d/logs/split_used.json" ]; then',
            '  echo SPLIT_USED_SHA=$(sha256sum "$d/logs/split_used.json" | cut -d" " -f1)',
            "fi",
        ]
    return "\n".join(lines) + "\n"


def parse_remote_output(text: str) -> dict[str, RemoteCellEvidence]:
    """Parse the ``===CELL ... ===`` blocks of :func:`build_remote_script`."""
    evidence: dict[str, RemoteCellEvidence] = {}
    current: RemoteCellEvidence | None = None
    mode: str | None = None
    buffer: list[str] = []
    prefixes = (
        ("ADAPTER_CONFIG_SHA=", "adapter_config_sha256"),
        ("ADAPTER_MODEL_SHA=", "adapter_model_sha256"),
        ("SPLIT_METADATA_SHA=", "split_metadata_sha256"),
        ("SPLIT_USED_SHA=", "split_used_sha256"),
        ("MANIFEST_SHA=", "manifest_sha256"),
    )
    for line in text.splitlines():
        if line.startswith("===CELL ") and line.endswith("==="):
            key = line[len("===CELL ") : -3]
            current = RemoteCellEvidence()
            evidence[key] = current
            mode = None
            buffer = []
            continue
        if current is None:
            continue
        if line == "DIR=1":
            current.exists = True
            continue
        if line.endswith("_BEGIN"):
            mode = line[: -len("_BEGIN")]
            buffer = []
            continue
        if line.endswith("_END"):
            content = "\n".join(buffer)
            if mode == "RUNCONFIG":
                current.run_config_text = content
            elif mode == "METADATA":
                current.metadata_text = content
            elif mode == "STATUS":
                current.status_text = content
            mode = None
            continue
        if mode is not None:
            buffer.append(line)
            continue
        for prefix, attribute in prefixes:
            if line.startswith(prefix):
                setattr(current, attribute, line.split("=", 1)[1].strip())
                break
    return evidence


def ssh_remote_reader(host: str = DEFAULT_TRANSFER_HOST) -> Callable[[str], str]:
    """Default reader: stream the batched script through one SSH call."""

    def reader(script: str) -> str:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, "bash -s"],
            input=script,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode != 0:
            raise PlannerError(
                f"remote read failed rc={result.returncode}: {result.stderr[-400:]}"
            )
        return result.stdout

    return reader


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_gpus(run_config: dict[str, Any], modality: str) -> int:
    shape = run_config.get("evaluation_resource_shape") or {}
    gpus = shape.get("gpus_per_node")
    if isinstance(gpus, int) and gpus > 0:
        return gpus
    return 1 if modality == "text_only" else 4


def expected_split_pin(
    dataset: str, split_path: str, dataset_contract: dict[str, Any]
) -> str:
    """Pinned sha256 of the split file a run is allowed to reference."""
    basename = Path(split_path).name
    pinned = SPLIT_FILE_PINS.get((dataset, basename))
    if pinned:
        return pinned
    return str(dataset_contract["folds_json_sha256"])


def resolve_cell(
    fit: dict[str, Any],
    route: dict[str, Any],
    attempt: str,
    evidence: RemoteCellEvidence | None,
    dataset_contract: dict[str, Any],
    prompt_sha_expected: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Assemble the exact parent entry or an explicit waiting reason.

    Every identity that the run sidecar schema defines is required and must be
    exact: attempt/fold/seed in ``metadata.json`` and attempt/fold/state in
    ``status.json``, attempt/fold/modality/dataset/seed/backend/config identity
    in ``run_config.yaml``. Missing evidence fails closed.
    """
    if evidence is None or not evidence.exists:
        return None, "remote fold dir missing"
    if not evidence.run_config_text:
        return None, "remote run_config.yaml missing"
    try:
        run_config = yaml.safe_load(evidence.run_config_text) or {}
    except yaml.YAMLError:
        return None, "remote run_config.yaml unreadable"
    remote_attempt = str(((run_config.get("tracking") or {}).get("attempt_id")) or "")
    if remote_attempt != attempt:
        return None, f"remote attempt {remote_attempt or 'missing'} != validated {attempt}"

    if not evidence.metadata_text:
        return None, "remote metadata.json missing"
    try:
        metadata = json.loads(evidence.metadata_text)
    except ValueError:
        return None, "remote metadata.json unreadable"
    if str(metadata.get("attempt_id") or "") != attempt:
        return None, "remote metadata attempt id mismatch"
    if _int_or_none(metadata.get("fold")) != int(fit["fold"]):
        return None, "remote metadata fold mismatch"
    if _int_or_none(metadata.get("seed")) != int(fit["seed"]):
        return None, "remote metadata seed mismatch"

    if not evidence.status_text:
        return None, "remote status.json missing"
    try:
        status = json.loads(evidence.status_text)
    except ValueError:
        return None, "remote status.json unreadable"
    if str(status.get("attempt_id") or "") != attempt:
        return None, "remote status attempt id mismatch"
    if _int_or_none(status.get("fold")) != int(fit["fold"]):
        return None, "remote status fold mismatch"
    state = str(status.get("state") or "")
    if state not in ACCEPTED_REMOTE_STATES:
        return None, (
            f"remote lifecycle state {state or 'missing'} contradicts the validated attempt"
        )

    # run_config identity: fold, modality, dataset, seed, backend, config file.
    if _int_or_none(run_config.get("fold")) != int(fit["fold"]):
        return None, "remote run_config fold mismatch"
    if str(run_config.get("input_modality") or "") != str(fit["modality"]):
        return None, "remote run_config modality mismatch"
    config_block = run_config.get("config") or {}
    if str(config_block.get("dataset") or "") != str(fit["dataset"]):
        return None, "remote run_config dataset mismatch"
    if _int_or_none(config_block.get("seed")) != int(fit["seed"]):
        return None, "remote run_config seed mismatch"
    if str(config_block.get("model_backend") or "") != str(route["backend"]):
        return None, "remote run_config backend mismatch"
    base_config = str(run_config.get("base_config_path") or "")
    if not base_config or Path(base_config).name != Path(str(route["config"])).name:
        return None, "remote run used a different config than the treatment route"

    prompt_context = run_config.get("prompt_context") or {}
    if prompt_context.get("version") is not None:
        return None, "remote run resolves a prompt context version (not the legacy inline prompt)"
    recorded_prompt = str(prompt_context.get("system_prompt_sha256") or "")
    if recorded_prompt != prompt_sha_expected:
        return None, (
            "prompt sha mismatch: remote "
            f"{recorded_prompt[:12] or 'missing'} != expected {prompt_sha_expected[:12]}"
        )

    # Split and manifest chain, all proven from in-place remote hashes.
    split_path = str(run_config.get("split_metadata_path") or "")
    if not split_path:
        return None, "remote run_config has no split_metadata_path"
    recorded_split = str(run_config.get("split_metadata_hash") or "")
    if not evidence.split_metadata_sha256:
        return None, "referenced split file was not hashed in place on GPFS"
    if evidence.split_metadata_sha256 != recorded_split:
        return None, "in-place split file hash does not equal the recorded split_metadata_hash"
    if recorded_split != expected_split_pin(str(fit["dataset"]), split_path, dataset_contract):
        return None, "split hash does not match the pinned contract"
    if str(run_config.get("manifest_hash") or "") != str(
        dataset_contract["recorded_manifest_hash"]
    ):
        return None, "manifest hash does not match the pinned manifest contract"
    if not evidence.manifest_sha256:
        return None, "referenced manifest file was not hashed in place on GPFS"
    if evidence.manifest_sha256 != str(dataset_contract["manifest_jsonl_sha256"]):
        return None, "remote manifest file hash does not match the pinned manifest contract"

    if not evidence.split_used_sha256:
        return None, "remote logs/split_used.json missing (split fingerprint required)"
    if not evidence.adapter_config_sha256 or not evidence.adapter_model_sha256:
        return None, "remote adapter files missing (adapter_config/adapter_model)"

    key = str(fit["key"])
    qualifier = ZERO_DELTA_EPOCH1_QUALIFIERS.get(key)
    fold_dir = fold_dir_for(fit)
    parent: dict[str, Any] = {
        "attempt_id": attempt,
        "run_name": str(fit["run_name"]),
        "fold_dir": str(fold_dir),
        "checkpoint_dir": str(fold_dir / "best_model"),
        "checkpoint_adapter_config_sha256": evidence.adapter_config_sha256,
        "checkpoint_adapter_model_sha256": evidence.adapter_model_sha256,
        "split_fingerprint": {
            "sha256": evidence.split_used_sha256,
            "source": "remote logs/split_used.json in-place sha256",
        },
        "manifest_hash_recorded": str(run_config.get("manifest_hash") or ""),
        "extract_gpus": extract_gpus(run_config, str(fit["modality"])),
        "selection": "validated_receipt_epoch1_zero_delta" if qualifier else "validated_receipt",
        "selection_reason": qualifier
        or "exact validated parent attempt; standalone logreg_raw+xgb_raw head chain (seed 1337)",
        "excluded_attempts": [],
        "state": state,
        "verification": {
            "prompt_sha256": prompt_sha_expected,
            "split_metadata_sha256": evidence.split_metadata_sha256,
            "folds_json_sha256": str(dataset_contract["folds_json_sha256"]),
            "manifest_jsonl_sha256": evidence.manifest_sha256,
            "hashed_in_place": "MN5 GPFS",
            "completion_authority": "ledger job pair + validation receipt/lifecycle proof",
        },
    }
    return parent, None


def build_matrix(
    *,
    matrix_path: Path,
    contract_path: Path,
    ledger_path: Path,
    receipts_path: Path,
    run_root: Path,
    cache_root: str,
    remote_reader: Callable[[str], str],
    permanent_root: Path = PERMANENT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Plan the treatment-only head matrix; only validated parents resolve."""
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    routes = {str(route["route_id"]): route for route in matrix.get("routes") or []}
    fits = list(matrix.get("fits") or [])
    fit_keys = [str(fit["key"]) for fit in fits]
    if len(set(fit_keys)) != len(fit_keys):
        raise PlannerError("frozen matrix contains duplicate fit keys")
    approved = guard.approved_head_keys(matrix_path)
    if len(approved) != len(fits):
        raise PlannerError(
            f"approved key set ({len(approved)}) does not cover the frozen matrix ({len(fits)})"
        )

    # Expected prompt sha per route: fresh local render cross-checked against
    # the route's pinned value; a drift refuses the whole plan.
    prompt_sha: dict[str, str] = {}
    for route_id, route in routes.items():
        rendered = expected_prompt_sha(LANE / str(route["config"]))
        pinned = str(route.get("prompt_system_sha256") or "")
        if pinned and pinned != rendered:
            raise PlannerError(
                f"route {route_id}: config render {rendered[:12]} != pinned {pinned[:12]}"
            )
        prompt_sha[route_id] = rendered

    validated = guard.validated_cells(
        receipts_path,
        lifecycle=collector.lifecycle_validated(run_root),
        ledger_path=ledger_path,
    )
    cells = []
    for fit in fits:
        cell = (str(fit["route_id"]), int(fit["seed"]), int(fit["fold"]))
        attempt = validated.get(cell)
        if attempt:
            cells.append((str(fit["key"]), attempt, fit))

    evidence: dict[str, RemoteCellEvidence] = {}
    if cells:
        script = build_remote_script(
            [(fit["key"], fold_dir_for(fit, permanent_root)) for _, _, fit in cells]
        )
        evidence = parse_remote_output(remote_reader(script))

    route_jobs: dict[str, list[dict[str, Any]]] = {route_id: [] for route_id in routes}
    resolved = waiting = 0
    for fit in fits:
        route_id = str(fit["route_id"])
        cell = (route_id, int(fit["seed"]), int(fit["fold"]))
        attempt = validated.get(cell)
        if not attempt:
            job = {
                "seed": int(fit["seed"]),
                "fold": int(fit["fold"]),
                "parent_status": "waiting_for_checkpoint",
                "reason": "no validated training parent for this cell yet",
            }
            waiting += 1
        else:
            dataset_contract = (contract.get("datasets") or {}).get(str(fit["dataset"]))
            if not dataset_contract:
                parent, reason = None, f"dataset {fit['dataset']} missing from the manifest contract"
            else:
                parent, reason = resolve_cell(
                    fit,
                    routes[route_id],
                    attempt,
                    evidence.get(str(fit["key"])),
                    dataset_contract,
                    prompt_sha[route_id],
                )
            if parent is None:
                job = {
                    "seed": int(fit["seed"]),
                    "fold": int(fit["fold"]),
                    "parent_status": "waiting_for_checkpoint",
                    "reason": reason,
                }
                waiting += 1
            else:
                job = {
                    "seed": int(fit["seed"]),
                    "fold": int(fit["fold"]),
                    "parent_status": "resolved",
                    "parent": parent,
                    "extract": {
                        "cache_dir": f"{cache_root}/{route_id}/s{int(fit['seed'])}_f{int(fit['fold'])}",
                        "gpus": int(parent.get("extract_gpus") or 1),
                    },
                }
                resolved += 1
        route_jobs[route_id].append(job)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "planned_seeds": [int(seed) for seed in matrix.get("training_seeds") or []],
        "head_seed": int(matrix.get("head_seed") or 1337),
        "cache_root": str(cache_root),
        "source": {
            "matrix_path": str(matrix_path),
            "matrix_sha256": sha256_text(matrix_path.read_text(encoding="utf-8")),
            "contract_path": str(contract_path),
            "planner": "tools/qwen3_legacy_prompt_head_plan.py",
            "adapter_hashing": "in place on MN5 GPFS; no local adapter download",
        },
        "routes": [],
        "summary": {
            "routes": len(routes),
            "jobs": len(fits),
            "resolved": resolved,
            "waiting_for_checkpoint": waiting,
            "approved_keys": len(approved),
        },
    }
    for route_id, route in routes.items():
        payload["routes"].append(
            {
                "route_id": route_id,
                "config": str(route["config"]),
                "dataset": str(route["dataset"]),
                "modality": str(route["modality"]),
                "language": "native",
                "backend": str(route["backend"]),
                "jobs": route_jobs[route_id],
            }
        )
    return payload


def validate_matrix(matrix: dict[str, Any], approved: set[str], cache_root_prefix: str) -> list[str]:
    """Acceptance checks for the emitted treatment-only payload."""
    failures: list[str] = []
    if matrix.get("schema_version") != SCHEMA_VERSION:
        failures.append(f"unexpected schema {matrix.get('schema_version')!r}")
    if not str(matrix.get("cache_root") or "").startswith(cache_root_prefix):
        failures.append("cache_root is not under the lane runtime heads cache")
    seen: set[str] = set()
    keys: list[str] = []
    for route in matrix.get("routes") or []:
        if route.get("language") != "native":
            failures.append(f"non-native route {route.get('route_id')}")
        for job in route.get("jobs") or []:
            key = f"{route['route_id']}|{int(job['seed'])}|{int(job['fold'])}"
            keys.append(key)
            seen.add(key)
            if job.get("parent_status") == "resolved":
                parent = job.get("parent") or {}
                for field in (
                    "attempt_id",
                    "checkpoint_adapter_config_sha256",
                    "checkpoint_adapter_model_sha256",
                    "manifest_hash_recorded",
                ):
                    if not parent.get(field):
                        failures.append(f"{key}: resolved parent missing {field}")
                if not (parent.get("checkpoint_dir") or "").startswith(str(PERMANENT_OUTPUT_ROOT)):
                    failures.append(f"{key}: checkpoint_dir is not a GPFS output path")
                cache_dir = str((job.get("extract") or {}).get("cache_dir") or "")
                if not cache_dir.startswith(str(matrix.get("cache_root") or "")):
                    failures.append(f"{key}: cache_dir outside cache_root")
            elif job.get("parent_status") != "waiting_for_checkpoint":
                failures.append(f"{key}: unexpected parent_status {job.get('parent_status')!r}")
            elif not job.get("reason"):
                failures.append(f"{key}: waiting cell without a reason")
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        failures.append(f"duplicate job keys: {duplicates[:5]}")
    if set(keys) != approved or len(keys) != len(approved):
        missing = sorted(approved - seen)[:5]
        extra = sorted(seen - approved)[:5]
        failures.append(
            f"key set drift: missing={missing} extra={extra} "
            f"count={len(keys)} approved={len(approved)}"
        )
    return failures


def check_existing(path: Path, matrix_path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = validate_matrix(payload, guard.approved_head_keys(matrix_path), str(RUNTIME_ROOT))
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"OK: {path} matches the approved Native treatment contract")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-matrix", type=Path, required=False)
    parser.add_argument("--check", type=Path, default=None)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--receipts", type=Path, default=RECEIPTS)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--remote-host", default=DEFAULT_TRANSFER_HOST)
    parser.add_argument("--permanent-root", type=Path, default=PERMANENT_OUTPUT_ROOT)
    args = parser.parse_args()

    if args.check is not None:
        return check_existing(args.check, args.matrix)
    if args.emit_matrix is None:
        parser.error("--emit-matrix or --check is required")

    payload = build_matrix(
        matrix_path=args.matrix,
        contract_path=args.contract,
        ledger_path=args.ledger,
        receipts_path=args.receipts,
        run_root=args.run_root,
        cache_root=args.cache_root,
        remote_reader=ssh_remote_reader(args.remote_host),
        permanent_root=args.permanent_root,
    )
    if str(args.cache_root) != DEFAULT_CACHE_ROOT:
        if not str(args.cache_root).startswith(str(RUNTIME_ROOT)):
            raise PlannerError("cache_root must live under the lane runtime root")
    failures = validate_matrix(
        payload, guard.approved_head_keys(args.matrix), str(RUNTIME_ROOT)
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    args.emit_matrix.parent.mkdir(parents=True, exist_ok=True)
    args.emit_matrix.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.emit_matrix}")
    print("head matrix:", json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
