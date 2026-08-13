#!/bin/bash
#SBATCH -J daic-odv-smoke-ext
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 08:00:00
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
ATTEMPT_DIR="${ATTEMPT_DIR:?ATTEMPT_DIR is required}"
PARENT_FOLD_DIR="${PARENT_FOLD_DIR:?PARENT_FOLD_DIR is required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
CONDITION="${CONDITION:-daic_officialdev_smoke}"
SUBJECT_SELECTION="${SUBJECT_SELECTION:?SUBJECT_SELECTION is required for smoke extraction}"

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_ROOT="$PROJECT_ROOT/logs/daic_officialdev_smokes"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/smoke-extract-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/smoke-extract-${SLURM_JOB_ID}.err" >&2)

mkdir -p "$ATTEMPT_DIR"
python src/features/extract_qwen_hidden.py \
    --checkpoint-dir "$PARENT_FOLD_DIR/best_model" \
    --output-dir "$ATTEMPT_DIR/hidden_features" \
    --model-name-or-path "$MODEL_PATH" \
    --condition "$CONDITION" \
    --subject-selection "$SUBJECT_SELECTION"
