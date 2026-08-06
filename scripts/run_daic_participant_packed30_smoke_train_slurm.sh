#!/bin/bash
#SBATCH -J p30-smoke-train
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

ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
source "$ENV_ACTIVATE"
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:?Set CONFIG}"
FOLD="${FOLD:-0}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME}"
SEED="${SEED:-1337}"
SMOKE_SUBJECT_LIMIT="${SMOKE_SUBJECT_LIMIT:-6}"
MODEL_PATH="${MODEL_PATH:-}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-}"
DAIC_UNPROCESSED_ROOT="${DAIC_UNPROCESSED_ROOT:?Set DAIC_UNPROCESSED_ROOT}"
DAIC_LABEL_ROOT="${DAIC_LABEL_ROOT:?Set DAIC_LABEL_ROOT}"
export DAIC_UNPROCESSED_ROOT DAIC_LABEL_ROOT

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_daic_participant_packed30}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/smoke-train-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/smoke-train-${SLURM_JOB_ID}.err" >&2)

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

CMD=(
  torchrun --nproc_per_node=1
  "$PROJECT_ROOT/src/train.py"
  --config "$CONFIG"
  --fold "$FOLD"
  --run_name "$RUN_NAME"
  --set "seed=$SEED"
  --set "split.seed=1337"
  --set "training.num_train_epochs=1"
  --set "split.smoke_subject_limit=$SMOKE_SUBJECT_LIMIT"
)
# The config YAML already resolves the correct base model via the exported
# MODEL_PATH / TEXT_MODEL_PATH env vars; never pass model flags here, because
# --model_name_or_path would override the text-only YAML default.

echo "== packed30 smoke training (1 GPU, 1 epoch, $SMOKE_SUBJECT_LIMIT subjects) =="
echo "config=$CONFIG fold=$FOLD run_name=$RUN_NAME seed=$SEED"
printf 'Launch: '; printf '%q ' "${CMD[@]}"; printf '\n'
"${CMD[@]}"
echo "== packed30 smoke training finished =="
