#!/usr/bin/env python3
"""Environment audit for the offline Qwen3-Omni MN5 runtime.

Records the interpreter, CUDA visibility, library versions, the importability of
the Qwen3Omni processor and model classes, and the identity of the snapshot the
config resolves (path, config hash, shard count). Writes ``env_audit.json`` into
the given output directory and exits non-zero when a required class is missing.

Runs inside the project tree: the config is resolved with the repository's own
loader, so the audited snapshot is the one the run will load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION = "audiollm.qwen3omni_env_audit.v1"
REQUIRED_CLASSES = (
    "Qwen3OmniMoeProcessor",
    "Qwen3OmniMoeForConditionalGeneration",
    "Qwen3OmniMoeThinkerForConditionalGeneration",
)


def build_audit(config_path: Path) -> dict:
    import accelerate
    import peft
    import torch
    import transformers

    from src.utils import load_yaml_with_overrides, resolve_model_name_or_path

    config = load_yaml_with_overrides(config_path, [])
    model_dir = Path(str(resolve_model_name_or_path(None, config)))
    snapshot_files = sorted(path.name for path in model_dir.glob("*.safetensors"))
    snapshot_config = model_dir / "config.json"
    audit = {
        "schema_version": SCHEMA_VERSION,
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "peft": peft.__version__,
        "gpu_total_gib": [
            round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 2)
            for index in range(torch.cuda.device_count())
        ],
        "imports": {},
        "model_dir": str(model_dir),
        "model_config_sha256": (
            hashlib.sha256(snapshot_config.read_bytes()).hexdigest()
            if snapshot_config.is_file()
            else None
        ),
        "model_shard_count": len(snapshot_files),
        "model_shard_names_head": snapshot_files[:3],
    }
    for name in REQUIRED_CLASSES:
        try:
            module = __import__("transformers", fromlist=[name])
            getattr(module, name)
            audit["imports"][name] = "ok"
        except Exception as exc:  # noqa: BLE001 - report the exact import failure
            audit["imports"][name] = f"failed: {exc}"
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="directory that receives env_audit.json",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="config path the audited model snapshot resolves from",
    )
    args = parser.parse_args(argv)

    audit = build_audit(Path(args.config))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "env_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    for key, value in audit["imports"].items():
        if value != "ok":
            print(f"required class {key} is not importable: {value}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
