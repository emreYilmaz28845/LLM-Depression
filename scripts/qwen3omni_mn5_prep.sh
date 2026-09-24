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
python scripts/qwen3omni_env_audit.py --output "$PREP_OUTPUT" --config "$CONFIG"

echo "=== module-tree audit ==="
python scripts/qwen3omni_backend_probe.py tree \
    --config "$CONFIG" \
    --output "$PREP_OUTPUT/module_tree.json" \
    "${OVERRIDE_ARGS[@]}"

echo "=== risk inventory (model-free plus processor) ==="
python scripts/qwen3omni_risk_inventory.py \
    --config "$CONFIG" \
    --output "$PREP_OUTPUT/risk_inventory.json" \
    --with-processor \
    "${OVERRIDE_ARGS[@]}"

echo "preflight complete: $PREP_OUTPUT"
