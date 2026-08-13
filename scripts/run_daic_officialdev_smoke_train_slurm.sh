#!/bin/bash
#SBATCH -J daic-odv-smoke-train
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
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
ENV_ACTIVATE="${ENV_ACTIVATE:?ENV_ACTIVATE is required}"
CONFIG="${CONFIG:?Set CONFIG}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME}"
SMOKE_RUN_ROOT="${SMOKE_RUN_ROOT:?Set SMOKE_RUN_ROOT}"
SMOKE_SUBJECT_LIMIT="${SMOKE_SUBJECT_LIMIT:-6}"
MODEL_PATH="${MODEL_PATH:-}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-}"
GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH:-}"
DAIC_UNPROCESSED_ROOT="${DAIC_UNPROCESSED_ROOT:?Set DAIC_UNPROCESSED_ROOT}"
DAIC_LABEL_ROOT="${DAIC_LABEL_ROOT:?Set DAIC_LABEL_ROOT}"
export DAIC_UNPROCESSED_ROOT DAIC_LABEL_ROOT MODEL_PATH TEXT_MODEL_PATH GEMMA4_MODEL_PATH

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
export PROJECT_ROOT

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

LOG_ROOT="$PROJECT_ROOT/logs/daic_officialdev_smokes"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/smoke-train-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/smoke-train-${SLURM_JOB_ID}.err" >&2)

# Smoke training: one epoch, official-train subjects only (smoke cap keeps
# both classes), inner validation only, final development eval disabled by
# the officialdev config. The run root is isolated so smoke checkpoints can
# never collide with production. class_balance stays "none": the locked
# subject-normalized weighting requires it. The master port is derived from
# the job id so concurrent torchrun jobs on a shared node cannot collide.
MASTER_PORT="${MASTER_PORT:-$(( 29000 + (${SLURM_JOB_ID:-0} % 1000) ))}"
CMD=(
  torchrun --nproc_per_node=1 --master_port="$MASTER_PORT"
  "$PROJECT_ROOT/src/train.py"
  --config "$CONFIG"
  --fold 0
  --run_name "$RUN_NAME"
  --set "training.num_train_epochs=1"
  --set "split.smoke_subject_limit=$SMOKE_SUBJECT_LIMIT"
  --set "output_dirs.run_root=$SMOKE_RUN_ROOT"
)
echo "== officialdev smoke training (1 GPU, 1 epoch, cap $SMOKE_SUBJECT_LIMIT) =="
printf 'Launch: '; printf '%q ' "${CMD[@]}"; printf '\n'
"${CMD[@]}"
echo "== officialdev smoke training finished =="
