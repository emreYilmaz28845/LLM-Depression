#!/bin/bash
#SBATCH -J sym-merged-train
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
# MN5 allocates 20 CPUs per requested GPU.
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
CONFIG="${CONFIG:?CONFIG is required}"
STAGE="${STAGE:?STAGE is required}"
FOLD="${FOLD:-0}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
EPOCHS="${EPOCHS:-}"
SUBJECTS_PER_CLASS="${SUBJECTS_PER_CLASS:-}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/symmetric_merged}"

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/train-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/train-${SLURM_JOB_ID}.err" >&2)
export PROJECT_ROOT CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export PYTHONPATH="$PROJECT_ROOT/.deps/qwen_hidden:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CMD=(python -m src.merged.train --config "$CONFIG" --stage "$STAGE" --fold "$FOLD" --run-id "$RUN_ID")
if [ -n "$EPOCHS" ]; then CMD+=(--epochs "$EPOCHS"); fi
if [ -n "$SUBJECTS_PER_CLASS" ]; then CMD+=(--subjects-per-class "$SUBJECTS_PER_CLASS"); fi
"${CMD[@]}"
