from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the hash-recorded gate required before DAIC official-test evaluation."
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-commit", default=None)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing authorization marker: {args.output}")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    spec = __import__("yaml").safe_load(args.spec.read_text(encoding="utf-8"))
    winner = str(selection.get("winner", "")).strip()
    epoch_count = int(selection.get("final_epoch_count", 0) or 0)
    if not winner:
        raise SystemExit("Selection artifact has no winner.")
    if not 1 <= epoch_count <= 20:
        raise SystemExit("Selection artifact final_epoch_count must be in the range 1..20.")
    selection_hash = hashlib.sha256(args.selection.read_bytes()).hexdigest()
    winner_protocol = selection.get("winner_protocol") or (spec.get("protocols") or {}).get(winner) or {}
    base_config = winner_protocol.get("base_config") if isinstance(winner_protocol, dict) else None
    config_hashes = dict(selection.get("config_hashes") or {})
    if base_config:
        config_path = ROOT / str(base_config)
        if config_path.is_file():
            config_hashes.setdefault(winner, hashlib.sha256(config_path.read_bytes()).hexdigest())
    if not config_hashes:
        raise SystemExit("Selection artifact/spec did not provide a resolvable winner config hash.")
    if not selection.get("aggregation_view"):
        raise SystemExit("Selection artifact has no locked aggregation_view.")
    payload = {
        "schema_version": "daic_final_test_authorization.v1",
        "winner": winner,
        "aggregation_view": selection.get("aggregation_view"),
        "final_epoch_count": epoch_count,
        "selection_artifact": str(args.selection),
        "selection_artifact_sha256": selection_hash,
        "implementation_commit": args.implementation_commit or git_commit(),
        "spec_hash": canonical_hash(spec),
        "config_hashes": config_hashes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "historical_test_exposure": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "winner": winner, "selection_artifact_sha256": selection_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
