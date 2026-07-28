#!/bin/bash
#SBATCH -J d3tec-worker
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
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
FOLD="${FOLD:?FOLD is required}"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
export PROJECT_ROOT
export D3TEC_DATASET_ROOT="${D3TEC_DATASET_ROOT:-$DATASET_BASE_ROOT/D3TEC DATASET/D3TEC DATASET}"
export D3TEC_FULL_TRANSCRIPTS="${D3TEC_FULL_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish.jsonl}"
export D3TEC_SEGMENT_TRANSCRIPTS="${D3TEC_SEGMENT_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish_segments.jsonl}"
export NPROC_PER_NODE=1

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"

LOG_ROOT="$PROJECT_ROOT/logs/slurm_d3tec"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/train-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/train-${SLURM_JOB_ID}.err" >&2)

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

CMD=(
    python "$PROJECT_ROOT/src/train.py"
    --config "$CONFIG"
    --fold "$FOLD"
    --run_name "$RUN_NAME"
    --save_strategy best_only
)
if [ -n "$EXTRA_TRAIN_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=($EXTRA_TRAIN_ARGS)
    CMD+=("${EXTRA_ARGS[@]}")
fi
"${CMD[@]}"
