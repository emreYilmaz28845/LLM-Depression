#!/bin/bash
#SBATCH -J jk4-train
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
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
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MODEL_PATH="${MODEL_PATH:-}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-}"
DAIC_UNPROCESSED_ROOT="${DAIC_UNPROCESSED_ROOT:?Set DAIC_UNPROCESSED_ROOT}"
DAIC_LABEL_ROOT="${DAIC_LABEL_ROOT:?Set DAIC_LABEL_ROOT}"
export DAIC_UNPROCESSED_ROOT DAIC_LABEL_ROOT

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_daic_participant_packed30_jointk4}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/train-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/train-${SLURM_JOB_ID}.err" >&2)

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

CMD=(
  torchrun --nproc_per_node="$NPROC_PER_NODE"
  "$PROJECT_ROOT/src/train.py"
  --config "$CONFIG"
  --fold "$FOLD"
  --run_name "$RUN_NAME"
  --set "seed=$SEED"
  --set "split.seed=1337"
)
# The config YAML already resolves the correct base model via the exported
# MODEL_PATH env var; never pass model flags here.

echo "== packed30 jointk4 production training =="
echo "config=$CONFIG fold=$FOLD run_name=$RUN_NAME seed=$SEED"
printf 'Launch: '; printf '%q ' "${CMD[@]}"; printf '\n'
"${CMD[@]}"
echo "== packed30 jointk4 training finished =="
