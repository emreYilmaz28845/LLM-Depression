#!/bin/bash
#SBATCH -J qwen3omni-prep
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

# Qwen3-Omni preflight: environment audit, module-tree audit, risk inventory.
#
#   PROJECT_ROOT=<deployed code path>   (required)
#   CONFIG=<config path>                (required; relative to PROJECT_ROOT)
#   PREP_OUTPUT=<directory>             (required; probe runtime directory)
#   OVERRIDES_JSON_B64 / EXTRA_PREP_ARGS: manifest/split dir overrides
#
# One GPU so CUDA visibility is recorded; no training, no weight loading beyond
# the processor, and nothing written outside PREP_OUTPUT.

set -e
set -o pipefail

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen3omni/bin/activate}"
if [ -f "$ENV_ACTIVATE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
else
    echo "Environment activate script not found: $ENV_ACTIVATE"
    exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the deployed code path}"
export PROJECT_ROOT
CONFIG="${CONFIG:?Set CONFIG}"
PREP_OUTPUT="${PREP_OUTPUT:?Set PREP_OUTPUT}"
EXTRA_PREP_ARGS="${EXTRA_PREP_ARGS:-}"
OVERRIDES_JSON_B64="${OVERRIDES_JSON_B64:-}"

cd "$PROJECT_ROOT"
mkdir -p "$PREP_OUTPUT"

DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
export DAIC_DATASET_ROOT="${DAIC_DATASET_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/preprocessed}"
export DAIC_UNPROCESSED_ROOT="${DAIC_UNPROCESSED_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/unprocessed}"
export DAIC_LABEL_ROOT="${DAIC_LABEL_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/minimal_zips}"

if [ -n "$OVERRIDES_JSON_B64" ]; then
    mapfile -t -d '' OVERRIDE_ARGS < <(python - "$OVERRIDES_JSON_B64" <<'PY'
import base64, json, sys
tokens = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
sys.stdout.write("\0".join(tokens))
PY
)
else
    read -r -a OVERRIDE_ARGS <<< "$EXTRA_PREP_ARGS"
fi

TIMESTAMP="$(date +%Y-%m-%d_%H:%M:%S)"
RUN_LOG="$PREP_OUTPUT/prep-${SLURM_JOB_ID:-local}-${TIMESTAMP}.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "========================================"
echo "Qwen3-Omni preflight | job=${SLURM_JOB_ID:-} | host=$(hostname)"
echo "project_root=$PROJECT_ROOT"
echo "config=$CONFIG"
echo "prep_output=$PREP_OUTPUT"
echo "overrides=${OVERRIDE_ARGS[*]:-<none>}"
echo "========================================"
nvidia-smi
python -V

echo "=== pip check ==="
python -m pip check
echo "=== environment freeze ==="
python -m pip freeze > "$PREP_OUTPUT/pip_freeze.txt"
wc -l "$PREP_OUTPUT/pip_freeze.txt"

echo "=== environment audit ==="
python - "$PREP_OUTPUT" "$PROJECT_ROOT" "$CONFIG" <<'PY'
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

out_dir, project_root, config_name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
import torch
import transformers
import accelerate
import peft

from src.utils import load_yaml_with_overrides, resolve_model_name_or_path

config = load_yaml_with_overrides(project_root / config_name, [])
model_dir = Path(str(resolve_model_name_or_path(None, config)))
snapshot_files = sorted(path.name for path in model_dir.glob("*.safetensors"))
config_path = model_dir / "config.json"
audit = {
    "schema_version": "audiollm.qwen3omni_env_audit.v1",
    "hostname": platform.node(),
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "peft": peft.__version__,
    "gpu_total_gib": [
        round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)
        for i in range(torch.cuda.device_count())
    ],
    "imports": {},
    "model_dir": str(model_dir),
    "model_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.is_file() else None,
    "model_shard_count": len(snapshot_files),
    "model_shard_names_head": snapshot_files[:3],
}
for name in ("Qwen3OmniMoeProcessor", "Qwen3OmniMoeForConditionalGeneration", "Qwen3OmniMoeThinkerForConditionalGeneration"):
    try:
        module = __import__("transformers", fromlist=[name])
        getattr(module, name)
        audit["imports"][name] = "ok"
    except Exception as exc:  # noqa: BLE001 - report the exact import failure
        audit["imports"][name] = f"failed: {exc}"
(out_dir / "env_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(audit, indent=2, sort_keys=True))
for key, value in audit["imports"].items():
    if value != "ok":
        raise SystemExit(f"required class {key} is not importable: {value}")
PY

echo "=== module-tree audit ==="
python scripts/qwen3omni_backend_probe.py tree \
    --config "$CONFIG" \
    --output "$PREP_OUTPUT/module_tree.json" \
    "${OVERRIDE_ARGS[@]}"

echo "=== DAIC risk inventory (model-free plus processor) ==="
python scripts/qwen3omni_daic_risk_inventory.py \
    --config "$CONFIG" \
    --output "$PREP_OUTPUT/risk_inventory.json" \
    --with-processor \
    "${OVERRIDE_ARGS[@]}"

echo "preflight complete: $PREP_OUTPUT"
