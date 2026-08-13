#!/bin/bash
#SBATCH -J daic-odv-preflight
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
BUILD_MANIFEST="${BUILD_MANIFEST:-1}"

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
export PROJECT_ROOT DATASET_BASE_ROOT
export DAIC_UNPROCESSED_ROOT="${DAIC_UNPROCESSED_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/unprocessed}"
export DAIC_LABEL_ROOT="${DAIC_LABEL_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/minimal_zips}"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/daic_officialdev_preflight/$RUN_ID}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/preflight-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/preflight-${SLURM_JOB_ID}.err" >&2)

BUILD_ARGS=()
if [ "$BUILD_MANIFEST" = "1" ]; then
    BUILD_ARGS+=(--build)
fi
python "$PROJECT_ROOT/scripts/prepare_daic_officialdev_mn5.py" \
    --run-id "$RUN_ID" \
    --required-path-prefix "$DATASET_BASE_ROOT" \
    "${BUILD_ARGS[@]}"
